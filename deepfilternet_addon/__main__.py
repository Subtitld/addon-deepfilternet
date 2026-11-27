"""Separator add-on for Subtitld — JSON-line protocol entry point.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License"); you may not
use this file except in compliance with the License. You may obtain a copy of
the License at http://www.apache.org/licenses/LICENSE-2.0. Unless required by
applicable law or agreed to in writing, software distributed under the License
is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.

What this add-on is
-------------------
Serves `audio.separate`: the sound of a video split into the voice and the
background, with the model spec.py names, run by audio.cpp on the CPU (see
engine.py and separate.py). Nothing leaves the machine; the weights are in
the archive or, for some models, downloaded once on the first separation.

Wire format
-----------
One JSON object per line on stdin/stdout. Stderr is free-form logging, which
the host prefixes with `[addon:<id>]` and surfaces.

    request   {input_path, output_dir, options}
    progress  {value, message}
    result    {vocals, background}: two 48 kHz mono FLAC files in output_dir

Options
-------
threads, from the request's `options` or, as the host sets them from
Configure, <PREFIX>_THREADS.

Limits
------
One separation at a time; a second waits for the first. A cancel stops it
between two pieces of the sound (spec.CHUNK_SEC), or stops the download,
which resumes on the next request.
"""

from __future__ import annotations

import errno
import json
import logging
import os
import platform
import shutil
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

from deepfilternet_addon import ADDON_ID, PROTOCOL_VERSION, __version__
from deepfilternet_addon import engine, separate, spec, tls

ADDON_VERSION = __version__
TASK = 'audio.separate'

ERR_MODEL_MISSING = 'model_missing'
ERR_BAD_PARAMS = 'bad_params'
ERR_INTERNAL = 'internal'
ERR_CANCELLED = 'cancelled'
ERR_OOM = 'oom'
ERR_DISK_FULL = 'disk_full'
ERR_NETWORK = 'network_unavailable'

log = logging.getLogger(f'{ADDON_ID}_addon')


# ---------------------------------------------------------------------------
# stdio ownership + protocol frames
# ---------------------------------------------------------------------------
_PROTO_IN = None   # binary reader for host frames
_PROTO_OUT = None  # binary writer for our frames
_proto_lock = threading.Lock()


def _claim_protocol_stdio() -> None:
    """Give the protocol exclusive use of the real stdin/stdout.

    fd-level, not just `sys.stdout = sys.stderr`: native code (audio.cpp,
    ggml) may write to fd 1 directly. Falls back to the
    Python-level swap if the fd dance fails.
    """
    global _PROTO_IN, _PROTO_OUT
    try:
        sys.stdout.flush()
        in_fd = os.dup(0)
        out_fd = os.dup(1)
        devnull = os.open(os.devnull, os.O_RDONLY)
        try:
            os.dup2(devnull, 0)
        finally:
            os.close(devnull)
        os.dup2(2, 1)
        _PROTO_IN = os.fdopen(in_fd, 'rb')
        _PROTO_OUT = os.fdopen(out_fd, 'wb')
    except (OSError, ValueError, AttributeError):
        _PROTO_IN = sys.stdin.buffer
        _PROTO_OUT = sys.stdout.buffer
    sys.stdin = open(os.devnull, encoding='utf-8')  # stray input() -> EOFError
    sys.stdout = sys.stderr                        # stray print() -> the log


def _emit(frame: dict) -> None:
    data = (json.dumps(frame, ensure_ascii=False, separators=(',', ':')) + '\n').encode('utf-8')
    out = _PROTO_OUT if _PROTO_OUT is not None else sys.stdout.buffer
    with _proto_lock:
        try:
            out.write(data)
            out.flush()
        except (BrokenPipeError, OSError, ValueError):
            pass  # host went away; the frame loop exits on the next empty read


def _emit_progress(req_id: str, value: float, message: str = '') -> None:
    _emit({'id': req_id, 'type': 'progress',
           'value': max(0.0, min(1.0, float(value))), 'message': message})


def _emit_result(req_id: str, data: dict) -> None:
    _emit({'id': req_id, 'type': 'result', 'data': data})


def _emit_error(req_id: str, code: str, message: str, retryable: bool = False) -> None:
    _emit({'id': req_id, 'type': 'error', 'code': code,
           'message': message, 'retryable': retryable})


class _AddonError(Exception):
    """An expected failure that already knows its protocol code."""

    def __init__(self, code: str, message: str, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
def _threads(options: dict) -> int:
    raw = options.get('threads')
    if raw in (None, ''):
        raw = (os.environ.get(f'{spec.ENV_PREFIX}_THREADS') or '').strip() or 0
    try:
        return max(0, min(256, int(float(raw))))
    except (TypeError, ValueError):
        log.warning('threads=%r is not a number; picking the cores', raw)
        return 0


# ---------------------------------------------------------------------------
# The model: loaded on first use, shared by every request
# ---------------------------------------------------------------------------
_loaded: engine.Loaded | None = None
_loaded_threads = -1
_model_lock = threading.Lock()   # held for the whole separation: one session


def _load_model(state: '_Request', threads: int) -> engine.Loaded:
    """Download (once) and load. The caller holds _model_lock."""
    global _loaded, _loaded_threads
    if _loaded is not None and _loaded_threads == threads:
        return _loaded
    if _loaded is not None:
        engine.close(_loaded)
        _loaded = None
    folder = engine.model_dir()
    if not engine.weights_present(folder):
        if not engine.downloads():
            raise _AddonError(ERR_MODEL_MISSING, f'The {spec.MODEL_NAME} weights are missing from the add-on '
                                                 f'({folder}). Reinstall it.')
        last = [0.0]

        def progress(done: int, total: int) -> None:
            now = time.monotonic()
            if now - last[0] < 1.0 and done < total:
                return
            last[0] = now
            share = done / total if total else 1.0
            _emit_progress(state.req_id, 0.02 + 0.08 * share,
                           f'Downloading {spec.MODEL_NAME}, once: {done / 1e6:.0f} of {total / 1e6:.0f} MB')

        log.info('weights missing in %s; downloading', folder)
        try:
            engine.ensure_weights(folder, progress, state.cancel.is_set)
        except engine.Cancelled:
            raise
        except engine.DownloadError as exc:
            raise _AddonError(ERR_NETWORK, f'Could not download {spec.MODEL_NAME}: {exc}', retryable=True) from exc
        except OSError as exc:
            if getattr(exc, 'errno', None) == errno.ENOSPC:
                raise _AddonError(ERR_DISK_FULL, f'Out of disk space downloading {spec.MODEL_NAME} to {folder} '
                                                 f'(it needs {engine.MODEL_BYTES / 1e6:.0f} MB).') from exc
            raise
    _emit_progress(state.req_id, 0.1, f'Loading {spec.MODEL_NAME}...')
    started = time.monotonic()
    try:
        _loaded = engine.load(engine.model_path(folder), threads)
    except engine.EngineError as exc:
        raise _AddonError(ERR_INTERNAL, f'{spec.MODEL_NAME} could not be loaded: {exc}') from exc
    _loaded_threads = threads
    log.info('%s loaded in %.1fs, %d threads%s', spec.MODEL_NAME, time.monotonic() - started, _loaded.threads,
             f' on CPUs {_loaded.cpus[0]}-{_loaded.cpus[-1]} (the performance cores)' if _loaded.cpus else '')
    return _loaded


# ---------------------------------------------------------------------------
# The request
# ---------------------------------------------------------------------------
class _Request:
    def __init__(self, req_id: str):
        self.req_id = req_id
        self.cancel = threading.Event()
        self.thread: threading.Thread | None = None


_active: dict[str, _Request] = {}
_active_lock = threading.Lock()


def _handle_separate(state: _Request, params: dict) -> None:
    req_id = state.req_id
    input_path = params.get('input_path')
    output_dir = params.get('output_dir')
    options = params.get('options') if isinstance(params.get('options'), dict) else {}
    if not isinstance(input_path, str) or not input_path.strip():
        _emit_error(req_id, ERR_BAD_PARAMS, 'input_path is required')
        return
    if not os.path.isfile(input_path):
        _emit_error(req_id, ERR_BAD_PARAMS, f'The file to separate is missing: {input_path}')
        return
    if not isinstance(output_dir, str) or not output_dir.strip():
        _emit_error(req_id, ERR_BAD_PARAMS, 'output_dir is required')
        return
    threads = _threads(options)

    out_dir = Path(output_dir)
    stem = f'{Path(input_path).stem}_{ADDON_ID}'
    vocals, background = out_dir / f'{stem}_vocals.flac', out_dir / f'{stem}_background.flac'
    parts = [vocals.with_name(vocals.name + '.part'), background.with_name(background.name + '.part')]
    outcome: tuple = ('error', ERR_INTERNAL, 'no outcome', False)
    scratch = None
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        scratch = Path(tempfile.mkdtemp(prefix=f'{ADDON_ID}-', dir=os.environ.get('SUBTITLD_TEMP_DIR') or None))
        cancelled = state.cancel.is_set
        _emit_progress(req_id, 0.0, 'Reading the sound...')
        mix = scratch / 'mix.raw'
        frames = separate.decode(input_path, mix, cancelled)
        seconds = frames / spec.RATE
        _emit_progress(req_id, 0.02, f'Waiting for {spec.MODEL_NAME}...')
        with _model_lock:
            if cancelled():
                raise engine.Cancelled()
            loaded = _load_model(state, threads)
            log.info('separate %s: %s, %.1f s of sound, %d threads', req_id, Path(input_path).name, seconds,
                     loaded.threads)
            started = time.monotonic()
            shown = [0.0]

            def on_piece(share: float) -> None:
                value = 0.12 + 0.83 * share
                if value - shown[0] >= 0.01 or share >= 1.0:
                    shown[0] = value
                    _emit_progress(req_id, value, 'Separating the voice...')

            separate.voice(loaded, mix, frames, scratch / 'voice.raw', cancelled, on_piece)
            took = time.monotonic() - started
        _emit_progress(req_id, 0.96, 'Writing the voice and the background...')
        separate.encode(mix, scratch / 'voice.raw', parts[0], parts[1], cancelled)
        os.replace(parts[0], vocals)
        os.replace(parts[1], background)
        log.info('separate %s: %.1f s of sound in %.1f s (real-time factor %.2f)', req_id, seconds, took,
                 took / seconds if seconds else 0.0)
        outcome = ('result', {'vocals': str(vocals), 'background': str(background)})
    except engine.Cancelled:
        log.info('separate %s: cancelled', req_id)
        outcome = ('error', ERR_CANCELLED, 'cancelled', False)
    except separate.Failed as exc:
        log.warning('separate %s failed: %s', req_id, exc)
        outcome = ('error', ERR_BAD_PARAMS, str(exc), False)
    except _AddonError as exc:
        log.warning('separate %s failed: [%s] %s', req_id, exc.code, exc.message)
        outcome = ('error', exc.code, exc.message, exc.retryable)
    except MemoryError:
        outcome = ('error', ERR_OOM, f'Out of memory. {spec.MODEL_NAME} needs about {spec.RAM_GB:g} GB free.', False)
    except OSError as exc:
        log.exception('separate %s failed', req_id)
        if exc.errno == errno.ENOSPC:
            outcome = ('error', ERR_DISK_FULL, f'Out of disk space: {exc}', False)
        else:
            outcome = ('error', ERR_INTERNAL, f'{type(exc).__name__}: {exc}', False)
    except Exception as exc:  # noqa: BLE001 — every path must end in one terminal frame
        log.exception('separate %s failed', req_id)
        if 'out of memory' in str(exc).lower():
            outcome = ('error', ERR_OOM, f'Out of memory: {exc}', False)
        else:
            outcome = ('error', ERR_INTERNAL, f'{type(exc).__name__}: {exc}', False)
    finally:
        if scratch is not None:
            shutil.rmtree(scratch, ignore_errors=True)
        for part in parts:
            try:
                part.unlink()
            except OSError:
                pass

    if outcome[0] == 'result':
        _emit_progress(req_id, 1.0, '')
        _emit_result(req_id, outcome[1])
    else:
        _emit_error(req_id, *outcome[1:])


def _spawn(req_id: str, target, *args) -> None:
    state = _Request(req_id)
    with _active_lock:
        _active[req_id] = state

    def _run():
        try:
            target(state, *args)
        except BaseException:  # noqa: BLE001
            log.exception('worker crashed')
            tail = traceback.format_exc().strip().splitlines()[-1]
            _emit_error(req_id, ERR_INTERNAL, f'worker crashed: {tail}')
        finally:
            with _active_lock:
                _active.pop(req_id, None)

    thread = threading.Thread(target=_run, name=f'{ADDON_ID}-{req_id}', daemon=True)
    state.thread = thread
    thread.start()


# ---------------------------------------------------------------------------
# Handshake + frame loop
# ---------------------------------------------------------------------------
def _send_hello() -> None:
    _emit({
        'type': 'hello',
        'protocol': PROTOCOL_VERSION,
        'addon': ADDON_ID,
        'version': ADDON_VERSION,
        'capabilities': [{'task': TASK}],
    })


def _read_frame() -> dict | None:
    source = _PROTO_IN if _PROTO_IN is not None else sys.stdin.buffer
    line = source.readline()
    if not line:
        return None  # stdin closed: the host went away
    text = line.decode('utf-8', errors='replace').strip().lstrip('﻿')
    if not text:
        return {}
    try:
        frame = json.loads(text)
    except ValueError as exc:
        log.warning('malformed frame %r: %s', line[:200], exc)
        return {}
    return frame if isinstance(frame, dict) else {}


def _main_loop() -> int:
    while True:
        frame = _read_frame()
        if frame is None:
            log.info('stdin closed; exiting')
            return 0
        if not frame:
            continue
        ftype = frame.get('type')

        if ftype == 'shutdown':
            log.info('shutdown requested')
            return 0

        if ftype == 'ready':
            log.info('handshake complete; host=%s host_version=%s',
                     frame.get('host'), frame.get('host_version'))
            continue

        if ftype == 'cancel':
            target = frame.get('target')
            with _active_lock:
                state = _active.get(target) if isinstance(target, str) else None
            if state is not None:
                log.info('cancel %s', target)
                state.cancel.set()
            continue

        if ftype == TASK:
            req_id = frame.get('id')
            params = frame.get('params')
            if not isinstance(req_id, str) or not req_id:
                log.warning('%s without id', TASK)
                continue
            if params is None:
                params = {}
            if not isinstance(params, dict):
                _emit_error(req_id, ERR_BAD_PARAMS, 'params must be an object')
                continue
            _spawn(req_id, _handle_separate, params)
            continue

        # Unknown task: still answer, so the host's future resolves instead of
        # hanging until its timeout.
        req_id = frame.get('id')
        if isinstance(req_id, str) and req_id:
            _emit_error(req_id, ERR_BAD_PARAMS, f'unsupported frame type {ftype!r}')


def _stop_workers(budget_sec: float = 3.5) -> bool:
    """Cancel and join every worker. True when none is left running.

    The host allows 5 s between `shutdown` and SIGTERM, so stay under it.
    """
    with _active_lock:
        states = list(_active.values())
    for state in states:
        state.cancel.set()
    deadline = time.monotonic() + budget_sec
    for state in states:
        if state.thread is not None:
            state.thread.join(timeout=max(0.0, deadline - time.monotonic()))
    return not any(s.thread is not None and s.thread.is_alive() for s in states)


def _self_test() -> int:
    """Exercise what the frozen bundle needs to separate, short of a video.

    Loads audio.cpp and finds the model's family in it; when the weights are
    there (in the archive, or already downloaded) loads them and runs them on
    a second of tone; and makes the TLS context a download uses. A library or
    module left out of the bundle then fails the build instead of the user's
    first separation.
    """
    import math
    import ssl
    from array import array

    families = engine.families()
    if spec.FAMILY not in families:
        raise SystemExit(f'audio.cpp has no {spec.FAMILY} family, only {families}')
    report = {
        'ok': True,
        'version': ADDON_VERSION,
        'runtime': engine.runtime_version(),
        'families': families,
        'threads': engine.default_threads(),
        'performance_cpus': engine.performance_cpus(),
        'openssl': ssl.OPENSSL_VERSION,
    }
    report['model'] = engine.weights_present(engine.model_dir())
    if report['model']:
        loaded = engine.load(engine.model_path(engine.model_dir()), 2)
        try:
            frames = int(spec.RATE * max(1.0, spec.MIN_PIECE_SEC))
            tone = array('f', (0.3 * math.sin(2 * math.pi * 220 * i / spec.RATE)
                               for i in range(frames) for _c in range(spec.CHANNELS)))
            report['frames'] = engine.run(loaded, tone, spec.RATE, spec.CHANNELS).frames
        finally:
            engine.close(loaded)
    engine._ssl_context()
    print(json.dumps(report), file=sys.__stdout__, flush=True)
    return 0


def main() -> int:
    if '--version' in sys.argv[1:]:
        print(f'{ADDON_ID}-addon {ADDON_VERSION}')
        return 0
    logging.basicConfig(
        level=(os.environ.get(f'{spec.ENV_PREFIX}_ADDON_LOG_LEVEL') or 'INFO').strip().upper(),
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
        stream=sys.stderr,
    )
    tls.configure_ca_bundle()
    if '--self-test' in sys.argv[1:]:
        return _self_test()
    _claim_protocol_stdio()
    log.info('%s-addon %s on %s %s, python %s', ADDON_ID,
             ADDON_VERSION, platform.system(), platform.machine(), platform.python_version())

    _send_hello()
    rc = 1
    try:
        rc = _main_loop()
    except KeyboardInterrupt:
        rc = 130
    except Exception:  # noqa: BLE001
        log.exception('main loop crashed')
        rc = 1
    finally:
        stopped = _stop_workers()
    if not stopped:
        # A worker is still inside audio.cpp (a piece, or loading the model).
        # Normal teardown could free what it is using, so leave without it.
        log.warning('a separation is still running; exiting without cleanup')
        try:
            sys.stderr.flush()
        finally:
            os._exit(rc)
    return rc


if __name__ == '__main__':
    sys.exit(main())

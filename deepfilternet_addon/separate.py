"""One separation: the video's sound in, the voice and the background out.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

1. ffmpeg decodes the first audio track of the input (a video or a sound
   file) to raw float samples at the model's rate and channel count, into
   the request's scratch folder.
2. The model runs on it in pieces of spec.CHUNK_SEC, each overlapping the one
   before by spec.OVERLAP_SEC; the overlaps are crossfaded, so no seam is
   heard. Between two pieces the request can be cancelled.
3. ffmpeg writes the voice, and the mix minus the voice as the background,
   both 48 kHz mono FLAC, as Subtitld's own separator does.

Memory stays the size of one piece, whatever the length of the video. The
scratch folder holds the decoded mix and the voice: about 21 MB a minute at
44.1 kHz stereo, 11 MB at 48 kHz mono.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from array import array
from pathlib import Path
from typing import Callable

from . import engine, spec

OUTPUT_RATE = 48000
_FLOAT = 4


class Failed(Exception):
    """A failure with a message for the user: no audio track, ffmpeg missing, ..."""


def find_ffmpeg() -> str | None:
    for name in ('SUBTITLD_FFMPEG_EXECUTABLE', 'FFMPEG_EXECUTABLE'):
        candidate = (os.environ.get(name) or '').strip()
        if candidate and (os.path.isfile(candidate) or shutil.which(candidate)):
            return candidate
    return shutil.which('ffmpeg')


def _no_window_startupinfo():
    """Keep ffmpeg from flashing a console window on Windows GUI hosts."""
    if sys.platform != 'win32':
        return None
    si = subprocess.STARTUPINFO()  # type: ignore[attr-defined]
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW  # type: ignore[attr-defined]
    si.wShowWindow = subprocess.SW_HIDE  # type: ignore[attr-defined]
    return si


def _child_env() -> dict | None:
    """Environment for ffmpeg, or None to inherit ours unchanged.

    The frozen Linux bootloader prepends the bundle's `_internal/` to
    LD_LIBRARY_PATH, and a system ffmpeg that inherits it loads the bundle's
    libraries instead of its own. PyInstaller saves the host value in
    LD_LIBRARY_PATH_ORIG, so restore it for the child.
    """
    if not (getattr(sys, 'frozen', False) and sys.platform.startswith('linux')):
        return None
    env = dict(os.environ)
    original = env.get('LD_LIBRARY_PATH_ORIG')
    if original is not None:
        env['LD_LIBRARY_PATH'] = original
    else:
        env.pop('LD_LIBRARY_PATH', None)
    return env


# Haiku's fork() fails with "Out of memory" once the model has run (it
# reserves room to copy what it wrote), and its posix_spawn forks too as soon
# as it has file actions to carry out; the encode comes after the model. So on
# Haiku a shell is spawned with none, sets up ffmpeg's files itself and becomes
# ffmpeg.
HAIKU = sys.platform.startswith('haiku')
_HAIKU_SCRIPT = 'err=$1; shift; exec "$@" </dev/null >/dev/null 2>"$err"'


def _ffmpeg_haiku(command: list[str], cancel: Callable[[], bool]) -> tuple[int, str]:
    """Run `command` to the end (Haiku): its exit code and what it wrote on stderr."""
    with tempfile.TemporaryDirectory(prefix='ffmpeg-') as folder:
        err = os.path.join(folder, 'errors.txt')
        pid = os.posix_spawn('/bin/sh', ['sh', '-c', _HAIKU_SCRIPT, 'sh', err, *command], os.environ)
        try:
            while True:
                done, status = os.waitpid(pid, os.WNOHANG)
                if done:
                    break
                if cancel():
                    raise engine.Cancelled()
                time.sleep(0.1)
        except BaseException:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
            raise
        errors = Path(err).read_text(encoding='utf-8', errors='replace').strip() if os.path.exists(err) else ''
    return os.waitstatus_to_exitcode(status), errors


def _ffmpeg(args: list[str], cancel: Callable[[], bool], what: str) -> None:
    """Run ffmpeg to the end, killing it on a cancel. Raises Failed with its last lines."""
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise Failed('ffmpeg was not found. Subtitld passes the one it uses; set FFMPEG_EXECUTABLE otherwise.')
    if HAIKU:
        command = [shutil.which(ffmpeg) or ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error', '-y', *args]
        returncode, errors = _ffmpeg_haiku(command, cancel)
        if returncode != 0:
            tail = ' / '.join(errors.splitlines()[-3:]) or f'exit {returncode}'
            raise Failed(f'{what}: {tail}')
        return
    process = subprocess.Popen([ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error', '-y', *args],
                               stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                               env=_child_env(), startupinfo=_no_window_startupinfo())
    try:
        while process.poll() is None:
            if cancel():
                process.kill()
                process.wait()
                raise engine.Cancelled()
            time.sleep(0.1)
        errors = process.stderr.read().decode('utf-8', 'replace').strip()
    finally:
        process.stderr.close()
    if process.returncode != 0:
        tail = ' / '.join(errors.splitlines()[-3:]) or f'exit {process.returncode}'
        raise Failed(f'{what}: {tail}')


def decode(input_path: str, out: Path, cancel: Callable[[], bool]) -> int:
    """The first audio track of `input_path` as raw float32 at spec.RATE x spec.CHANNELS. Returns frames."""
    _ffmpeg(['-i', input_path, '-vn', '-sn', '-dn', '-map', '0:a:0', '-ac', str(spec.CHANNELS),
             '-ar', str(spec.RATE), '-f', 'f32le', '-c:a', 'pcm_f32le', str(out)],
            cancel, f'Could not read the sound of {Path(input_path).name}')
    frames = out.stat().st_size // (_FLOAT * spec.CHANNELS) if out.is_file() else 0
    if frames == 0:
        raise Failed(f'{Path(input_path).name} has no sound to separate.')
    return frames


def pieces(frames: int, chunk: int, overlap: int) -> list[tuple[int, int]]:
    """[start, end) of each piece: `chunk` frames long, each starting `overlap` before the last ended."""
    if frames <= chunk:
        return [(0, frames)]
    step = chunk - overlap
    spans, start = [], 0
    while True:
        end = min(start + chunk, frames)
        spans.append((start, end))
        if end == frames:
            return spans
        start += step


def _read(handle, start: int, end: int) -> array:
    handle.seek(start * spec.CHANNELS * _FLOAT)
    samples = array('f')
    samples.frombytes(handle.read((end - start) * spec.CHANNELS * _FLOAT))
    if sys.byteorder != 'little':
        samples.byteswap()
    return samples


def _write(handle, samples: array) -> None:
    if sys.byteorder != 'little':
        samples = array('f', samples)
        samples.byteswap()
    handle.write(samples.tobytes())


def _crossfade(tail: array, head: array, channels: int) -> array:
    """`tail` fading out under `head` fading in, both the same length."""
    frames = len(head) // channels
    out = array('f', head)
    for frame in range(frames):
        w = (frame + 0.5) / frames
        base = frame * channels
        for c in range(base, base + channels):
            out[c] = tail[c] * (1.0 - w) + head[c] * w
    return out


def voice(loaded: engine.Loaded, mix: Path, frames: int, out: Path, cancel: Callable[[], bool],
          progress: Callable[[float], None]) -> None:
    """The model's voice for the whole of `mix`, stitched from pieces, into `out` (raw float32)."""
    channels = spec.CHANNELS
    chunk = max(1, int(spec.CHUNK_SEC * spec.RATE))
    overlap = min(int(spec.OVERLAP_SEC * spec.RATE), chunk // 2)
    delay = spec.DELAY_FRAMES
    shortest = int(spec.MIN_PIECE_SEC * spec.RATE)
    spans = pieces(frames, chunk, overlap)
    tail = array('f')
    with mix.open('rb') as source, out.open('wb') as sink:
        for index, (start, end) in enumerate(spans):
            if cancel():
                raise engine.Cancelled()
            samples = _read(source, start, end)
            # Silence after the piece: the model's lag, and enough for the
            # shortest piece it takes. The output is cut back to the piece.
            padding = delay + max(0, shortest - (end - start) - delay)
            if padding:
                samples.extend(array('f', [0.0]) * (padding * channels))
            result = engine.run(loaded, samples, spec.RATE, channels)
            if result.channels != channels or result.rate != spec.RATE:
                raise engine.EngineError(f'{spec.MODEL_NAME} gave {result.channels} channel(s) at {result.rate} Hz, '
                                         f'not {channels} at {spec.RATE}')
            got = result.samples[delay * channels:(delay + end - start) * channels]
            if len(got) < (end - start) * channels:
                got.extend(array('f', [0.0]) * ((end - start) * channels - len(got)))
            if tail:
                _write(sink, _crossfade(tail, got[:len(tail)], channels))
                got = got[len(tail):]
            last = index == len(spans) - 1
            keep = 0 if last else overlap * channels
            _write(sink, got[:len(got) - keep] if keep else got)
            tail = got[len(got) - keep:] if keep else array('f')
            progress((index + 1) / len(spans))


def _pan(gains: str) -> str:
    return f'pan=mono|c0={gains},aresample={OUTPUT_RATE}'


def encode(mix: Path, voice_raw: Path, vocals: Path, background: Path, cancel: Callable[[], bool]) -> None:
    """The voice, and the mix minus the voice, as 48 kHz mono FLAC."""
    raw = ['-f', 'f32le', '-ar', str(spec.RATE), '-ac', str(spec.CHANNELS)]
    if spec.CHANNELS == 2:
        speech, rest = '0.5*c2+0.5*c3', '0.5*c0+0.5*c1-0.5*c2-0.5*c3'
    else:
        speech, rest = 'c1', 'c0-c1'
    graph = (f'[0:a][1:a]amerge=inputs=2,asplit=2[a][b];'
             f'[a]{_pan(speech)}[voice];[b]{_pan(rest)}[background]')
    flac = ['-c:a', 'flac', '-sample_fmt', 's16', '-f', 'flac']
    _ffmpeg([*raw, '-i', str(mix), *raw, '-i', str(voice_raw), '-filter_complex', graph,
             '-map', '[voice]', *flac, str(vocals), '-map', '[background]', *flac, str(background)],
            cancel, 'Could not write the separated sound')

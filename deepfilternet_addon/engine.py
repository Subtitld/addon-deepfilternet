"""The model for the add-on: fetch its weights, load it in audio.cpp, run it on a piece of audio.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

Runtime
-------
The model is run by audio.cpp (Apache-2.0), a ggml runtime, through its C API
with ctypes: build/assets/audiocpp/ from source, `audiocpp/` inside the frozen
bundle. tools/build_native.py builds it with only this add-on's model family
compiled in. ggml picks the CPU backend for the processor it runs on.

Weights
-------
spec.py says where they come from. Either they are in the archive
(build/assets/weights/, `weights/` inside the frozen bundle), or they are
fetched on the first separation from Hugging Face at a pinned revision of
audio.cpp's model repository, checked against their sha256, into the host's
model cache ($SUBTITLD_MODELS_DIR/<add-on id>/<folder>). An interrupted
download resumes. $HF_ENDPOINT points it at a mirror; $<PREFIX>_MODEL_DIR at
a folder that already holds the files, which is then used as it is.

Threads
-------
On a CPU with performance and efficiency cores, the slower cores hold every
step back: the model uses one thread per performance core, and on Linux the
process is kept on those cores. Elsewhere it uses one thread per core.
"""

from __future__ import annotations

import ctypes
import hashlib
import logging
import os
import ssl
import subprocess
import sys
import urllib.error
import urllib.request
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import ADDON_ID, __version__, spec

log = logging.getLogger(f'{ADDON_ID}_addon.engine')

_PACKAGE = Path(__file__).resolve().parent

MODEL_BYTES = sum(size for _name, size, _sha in spec.MODEL_FILES)

_CHUNK = 1 << 20
_VERIFIED = '.verified'


class Cancelled(Exception):
    """The request was cancelled."""


class DownloadError(Exception):
    """The weights could not be fetched or did not check out."""


class EngineError(Exception):
    """audio.cpp failed, or could not be loaded."""


# ---------------------------------------------------------------------------
# Where the weights live
# ---------------------------------------------------------------------------
def downloads() -> bool:
    """Whether the weights are downloaded (else they are in the archive)."""
    return bool(spec.MODEL_FILES)


def cache_root() -> Path:
    """$SUBTITLD_MODELS_DIR/<add-on id>."""
    base = (os.environ.get('SUBTITLD_MODELS_DIR') or '').strip()
    root = Path(os.path.expanduser(base)) if base else Path.home() / '.cache' / 'subtitld' / 'models'
    return root / ADDON_ID


def _user_folder() -> str:
    return (os.environ.get(f'{spec.ENV_PREFIX}_MODEL_DIR') or '').strip()


def model_dir() -> Path:
    if _user_folder():
        return Path(os.path.expanduser(_user_folder()))
    if not downloads():
        return assets_dir() / 'weights' / spec.BUNDLED_WEIGHTS[0]
    return cache_root() / spec.MODEL_DIRNAME


def model_path(folder: Path) -> Path:
    """What audio.cpp loads: the GGUF file, or the folder of a bundled utility's weights."""
    return folder / spec.MODEL_FILE if downloads() else folder


def _local(name: str) -> str:
    return name.rsplit('/', 1)[-1]


def weights_present(folder: Path) -> bool:
    """True when every file is there at its expected size, checked once by sha256."""
    if not downloads():
        return (folder / spec.BUNDLED_WEIGHTS[1]).is_file()
    if _user_folder():
        return all((folder / _local(name)).is_file() for name, _size, _sha in spec.MODEL_FILES)
    marker = folder / _VERIFIED
    try:
        if marker.read_text(encoding='utf-8').strip() != spec.MODEL_REVISION:
            return False
    except OSError:
        return False
    return all((folder / _local(name)).is_file() and (folder / _local(name)).stat().st_size == size
               for name, size, _sha in spec.MODEL_FILES)


def missing_bytes(folder: Path) -> int:
    """Bytes still to download, counting partial files as progress."""
    total = 0
    for name, size, _sha in spec.MODEL_FILES:
        path = folder / _local(name)
        if path.is_file() and path.stat().st_size == size:
            continue
        part = folder / (_local(name) + '.part')
        have = part.stat().st_size if part.is_file() else 0
        total += max(0, size - have)
    return total


def _ssl_context() -> ssl.SSLContext:
    # tls.configure_ca_bundle() has set SSL_CERT_FILE for the frozen Linux and
    # macOS builds; elsewhere this is the platform's own trust store.
    return ssl.create_default_context()


def _sha256(path: Path, cancel: Callable[[], bool]) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(_CHUNK * 8):
            if cancel():
                raise Cancelled()
            digest.update(chunk)
    return digest.hexdigest()


def _fetch(url: str, dest: Path, size: int, sha256: str,
           on_bytes: Callable[[int], None], cancel: Callable[[], bool]) -> None:
    part = dest.with_name(dest.name + '.part')
    have = part.stat().st_size if part.is_file() else 0
    if have > size:
        part.unlink()
        have = 0
    if have < size:
        headers = {'User-Agent': f'subtitld-{ADDON_ID}-addon/{__version__}'}
        if have:
            headers['Range'] = f'bytes={have}-'
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=60, context=_ssl_context()) as response:
                if have and response.status != 206:
                    have = 0  # the server ignored the range: start over
                on_bytes(have)
                with part.open('ab' if have else 'wb') as out:
                    while chunk := response.read(_CHUNK):
                        if cancel():
                            raise Cancelled()
                        out.write(chunk)
                        on_bytes(len(chunk))
        except urllib.error.HTTPError as exc:
            raise DownloadError(f'{dest.name}: HTTP {exc.code} from {url}') from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            raise DownloadError(f'{dest.name}: {getattr(exc, "reason", exc)}') from exc
    got = part.stat().st_size
    if got != size:
        raise DownloadError(f'{dest.name}: got {got:,} bytes, expected {size:,}. Try again to resume.')
    if _sha256(part, cancel) != sha256:
        part.unlink()
        raise DownloadError(f'{dest.name} does not match its pinned sha256; the download was discarded.')
    os.replace(part, dest)


def ensure_weights(folder: Path, progress: Callable[[int, int], None],
                   cancel: Callable[[], bool]) -> None:
    """Download whatever is missing. `progress(done, total)` is in bytes."""
    if weights_present(folder):
        return
    if not downloads():
        raise EngineError(f'the bundled weights are missing: no {folder / spec.BUNDLED_WEIGHTS[1]}')
    if _user_folder():
        missing = [_local(name) for name, _s, _h in spec.MODEL_FILES if not (folder / _local(name)).is_file()]
        raise DownloadError(f'{spec.ENV_PREFIX}_MODEL_DIR={folder} lacks {", ".join(missing)}')
    folder.mkdir(parents=True, exist_ok=True)
    endpoint = (os.environ.get('HF_ENDPOINT') or 'https://huggingface.co').rstrip('/')
    total = missing_bytes(folder)
    done = 0

    def on_bytes(count: int) -> None:
        nonlocal done
        done += count
        progress(min(done, total), total)

    for name, size, sha256 in spec.MODEL_FILES:
        dest = folder / _local(name)
        if dest.is_file() and dest.stat().st_size == size and _sha256(dest, cancel) == sha256:
            continue
        part = folder / (_local(name) + '.part')
        before = part.stat().st_size if part.is_file() else 0
        done -= before  # _fetch reports the bytes already on disk again
        url = f'{endpoint}/{spec.MODEL_REPO}/resolve/{spec.MODEL_REVISION}/{name}'
        log.info('fetching %s (%s bytes)', url, f'{size:,}')
        _fetch(url, dest, size, sha256, on_bytes, cancel)
    (folder / _VERIFIED).write_text(spec.MODEL_REVISION + '\n', encoding='utf-8')


# ---------------------------------------------------------------------------
# Threads
# ---------------------------------------------------------------------------
def _cpu_list(text: str) -> list[int]:
    """'0-3,8,10-11' -> [0, 1, 2, 3, 8, 10, 11]"""
    cpus: list[int] = []
    for part in text.strip().split(','):
        if '-' in part:
            low, high = part.split('-', 1)
            cpus.extend(range(int(low), int(high) + 1))
        elif part:
            cpus.append(int(part))
    return cpus


def performance_cpus() -> list[int]:
    """The logical CPUs of a hybrid processor's performance cores, on Linux; [] otherwise."""
    if not sys.platform.startswith('linux'):
        return []
    try:
        return _cpu_list(Path('/sys/devices/cpu_core/cpus').read_text())
    except (OSError, ValueError):
        return []


def _physical(cpus: list[int]) -> int:
    """How many physical cores the logical CPUs `cpus` are, on Linux."""
    cores = set()
    for cpu in cpus:
        topology = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        try:
            cores.add(((topology / 'physical_package_id').read_text().strip(),
                       (topology / 'core_id').read_text().strip()))
        except OSError:
            cores.add(('cpu', str(cpu)))
    return len(cores)


def _sysctl(*names: str) -> list[int]:
    out = subprocess.run(['sysctl', '-n', *names], capture_output=True, text=True, timeout=5)
    return [int(line) for line in out.stdout.split()]


def default_threads() -> int:
    """One thread per performance core of a hybrid CPU, else one per core."""
    try:
        if sys.platform.startswith('linux'):
            cpus = performance_cpus() or sorted(os.sched_getaffinity(0))
            return max(1, _physical(cpus))
        if sys.platform == 'darwin':
            counts = _sysctl('hw.perflevel0.physicalcpu')
            return max(1, counts[0]) if counts else max(1, _sysctl('hw.physicalcpu')[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        pass
    logical = os.cpu_count() or 4
    return max(1, logical // 2 if logical >= 4 else logical)


def keep_to_performance_cores() -> list[int]:
    """On Linux with a hybrid CPU, keep this process on its performance cores. Returns them, or []."""
    cpus = performance_cpus()
    if not cpus:
        return []
    try:
        allowed = os.sched_getaffinity(0)
        chosen = [cpu for cpu in cpus if cpu in allowed]
        if chosen and set(chosen) != allowed:
            os.sched_setaffinity(0, chosen)
        return chosen
    except OSError:
        return []


# ---------------------------------------------------------------------------
# audio.cpp's C API (include/audiocpp.h)
# ---------------------------------------------------------------------------
LIBRARY = {'linux': 'libaudiocpp.so.0', 'darwin': 'libaudiocpp.0.dylib', 'win32': 'audiocpp.dll'}
ABI_MAJOR = 0

OK = 0
ERR_OUT_OF_MEMORY = 5
ERR_CANCELLED = 8

_LOG_LEVELS = {1: logging.DEBUG, 2: logging.DEBUG, 3: logging.WARNING, 4: logging.ERROR}


class _BackendConfig(ctypes.Structure):
    _fields_ = [('backend', ctypes.c_char_p), ('device', ctypes.c_int), ('threads', ctypes.c_int)]


class _ModelConfig(ctypes.Structure):
    _fields_ = [('family_hint', ctypes.c_char_p), ('config_id', ctypes.c_char_p),
                ('weight_id', ctypes.c_char_p), ('model_spec_override', ctypes.c_char_p)]


_LOG_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)


def assets_dir() -> Path:
    """The frozen bundle's _internal/, or build/assets/ from source."""
    if getattr(sys, 'frozen', False):
        return Path(getattr(sys, '_MEIPASS', Path(sys.executable).parent))
    return _PACKAGE.parent / 'build' / 'assets'


def runtime_dir() -> Path:
    """Where audio.cpp is: $<PREFIX>_AUDIOCPP_DIR, or audiocpp/ in assets_dir()."""
    override = (os.environ.get(f'{spec.ENV_PREFIX}_AUDIOCPP_DIR') or '').strip()
    if override:
        return Path(os.path.expanduser(override))
    return assets_dir() / 'audiocpp'


_lib = None
_log_callback = None  # kept alive for as long as the library may call it


def _on_log(level: int, text: bytes, _user) -> None:
    message = (text or b'').decode('utf-8', 'replace').rstrip()
    if message:
        log.log(_LOG_LEVELS.get(level, logging.DEBUG), 'audio.cpp: %s', message)


def load_library():
    """audio.cpp's C library, loaded once, with its functions typed."""
    global _lib, _log_callback
    if _lib is not None:
        return _lib
    folder = runtime_dir()
    path = folder / LIBRARY.get(sys.platform, LIBRARY['linux'])
    if not path.is_file():
        raise EngineError(f'audio.cpp is missing: no {path}')
    if sys.platform == 'win32':
        os.add_dll_directory(str(folder))
    try:
        lib = ctypes.CDLL(str(path))
    except OSError as exc:
        raise EngineError(f'could not load {path.name}: {exc}') from exc

    c = ctypes
    handle = c.POINTER(c.c_void_p)
    floats = c.POINTER(c.POINTER(c.c_float))
    signatures = {
        'audiocpp_abi_version': (c.c_uint32, []),
        'audiocpp_build_version': (c.c_char_p, []),
        'audiocpp_last_error': (c.c_char_p, []),
        'audiocpp_log_set': (None, [_LOG_CALLBACK, c.c_void_p]),
        'audiocpp_registry_create': (c.c_int, [c.c_char_p, handle]),
        'audiocpp_registry_free': (None, [c.c_void_p]),
        'audiocpp_registry_family_count': (c.c_size_t, [c.c_void_p]),
        'audiocpp_registry_family': (c.c_int, [c.c_void_p, c.c_size_t, c.POINTER(c.c_char_p)]),
        'audiocpp_options_create': (c.c_void_p, []),
        'audiocpp_options_set': (c.c_int, [c.c_void_p, c.c_char_p, c.c_char_p]),
        'audiocpp_options_free': (None, [c.c_void_p]),
        'audiocpp_model_load': (c.c_int, [c.c_void_p, c.c_char_p, c.POINTER(_ModelConfig), c.c_void_p, handle]),
        'audiocpp_model_free': (None, [c.c_void_p]),
        'audiocpp_session_create': (c.c_int, [c.c_void_p, c.c_char_p, c.c_char_p, c.POINTER(_BackendConfig),
                                              c.c_void_p, handle]),
        'audiocpp_session_free': (None, [c.c_void_p]),
        'audiocpp_session_run': (c.c_int, [c.c_void_p, c.c_void_p, handle]),
        'audiocpp_request_create': (c.c_void_p, []),
        'audiocpp_request_free': (None, [c.c_void_p]),
        'audiocpp_request_set_audio': (c.c_int, [c.c_void_p, c.POINTER(c.c_float), c.c_size_t, c.c_int, c.c_int]),
        'audiocpp_result_free': (None, [c.c_void_p]),
        'audiocpp_result_audio': (c.c_int, [c.c_void_p, floats, c.POINTER(c.c_size_t),
                                            c.POINTER(c.c_int), c.POINTER(c.c_int)]),
        'audiocpp_result_named_audio_count': (c.c_size_t, [c.c_void_p]),
        'audiocpp_result_named_audio': (c.c_int, [c.c_void_p, c.c_size_t, c.POINTER(c.c_char_p), floats,
                                                  c.POINTER(c.c_size_t), c.POINTER(c.c_int), c.POINTER(c.c_int)]),
    }
    for name, (restype, argtypes) in signatures.items():
        try:
            function = getattr(lib, name)
        except AttributeError as exc:
            raise EngineError(f'{path.name} has no {name}: not the build tools/build_native.py makes') from exc
        function.restype = restype
        function.argtypes = argtypes
    abi = lib.audiocpp_abi_version()
    if abi >> 16 != ABI_MAJOR:
        raise EngineError(f'{path.name} has C ABI {abi >> 16}.{(abi >> 8) & 0xFF}, the add-on needs {ABI_MAJOR}.x')
    _log_callback = _LOG_CALLBACK(_on_log)
    lib.audiocpp_log_set(_log_callback, None)
    _lib = lib
    return lib


def runtime_version() -> str:
    lib = load_library()
    abi = lib.audiocpp_abi_version()
    return f'audio.cpp {lib.audiocpp_build_version().decode()} (C ABI {abi >> 16}.{(abi >> 8) & 0xFF})'


def families() -> list[str]:
    """The model families this build of audio.cpp has: this add-on's, for ours."""
    lib = load_library()
    registry = ctypes.c_void_p()
    _check(lib, lib.audiocpp_registry_create(None, ctypes.byref(registry)), 'registry')
    try:
        names = []
        for index in range(lib.audiocpp_registry_family_count(registry)):
            name = ctypes.c_char_p()
            _check(lib, lib.audiocpp_registry_family(registry, index, ctypes.byref(name)), 'registry')
            names.append(name.value.decode())
        return names
    finally:
        lib.audiocpp_registry_free(registry)


def _check(lib, status: int, what: str) -> None:
    if status == OK:
        return
    detail = (lib.audiocpp_last_error() or b'').decode('utf-8', 'replace')
    if status == ERR_CANCELLED:
        raise Cancelled()
    if status == ERR_OUT_OF_MEMORY:
        raise MemoryError(f'{what}: {detail}')
    raise EngineError(f'{what}: {detail}')


# ---------------------------------------------------------------------------
# Loading and running
# ---------------------------------------------------------------------------
@dataclass
class Loaded:
    lib: object
    registry: ctypes.c_void_p
    model: ctypes.c_void_p
    session: ctypes.c_void_p
    threads: int
    cpus: list[int] = field(default_factory=list)


def _options(lib, values: dict[str, str]):
    options = lib.audiocpp_options_create()
    for key, value in values.items():
        _check(lib, lib.audiocpp_options_set(options, key.encode(), value.encode()), f'option {key}')
    return options


def load(path: Path, threads: int = 0) -> Loaded:
    """The model at `path`, in a CPU session for spec.TASK."""
    lib = load_library()
    cpus = keep_to_performance_cores()
    threads = threads if threads > 0 else default_threads()
    registry, model, session = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
    load_options = _options(lib, spec.LOAD_OPTIONS)
    try:
        _check(lib, lib.audiocpp_registry_create(None, ctypes.byref(registry)), 'registry')
        _check(lib, lib.audiocpp_model_load(registry, str(path).encode(),
                                            ctypes.byref(_ModelConfig(spec.FAMILY.encode(), None, None, None)),
                                            load_options, ctypes.byref(model)), f'loading {spec.MODEL_NAME}')
        _check(lib, lib.audiocpp_session_create(model, spec.TASK.encode(), b'offline',
                                                ctypes.byref(_BackendConfig(b'cpu', 0, threads)),
                                                None, ctypes.byref(session)), 'session')
    except BaseException:
        lib.audiocpp_session_free(session)
        lib.audiocpp_model_free(model)
        lib.audiocpp_registry_free(registry)
        raise
    finally:
        lib.audiocpp_options_free(load_options)
    return Loaded(lib=lib, registry=registry, model=model, session=session, threads=threads, cpus=cpus)


def close(loaded: Loaded) -> None:
    loaded.lib.audiocpp_session_free(loaded.session)
    loaded.lib.audiocpp_model_free(loaded.model)
    loaded.lib.audiocpp_registry_free(loaded.registry)


@dataclass
class Output:
    samples: array      # float32, channels interleaved
    rate: int
    channels: int

    @property
    def frames(self) -> int:
        return len(self.samples) // self.channels


def run(loaded: Loaded, samples: array, rate: int, channels: int) -> Output:
    """spec.OUTPUT (the voice) of the float32 interleaved `samples`."""
    lib = loaded.lib
    request = lib.audiocpp_request_create()
    result = ctypes.c_void_p()
    try:
        buffer = (ctypes.c_float * len(samples)).from_buffer(samples)
        _check(lib, lib.audiocpp_request_set_audio(request, buffer, len(samples) // channels, rate, channels),
               'audio')
        _check(lib, lib.audiocpp_session_run(loaded.session, request, ctypes.byref(result)), spec.MODEL_NAME)
        data, frames = ctypes.POINTER(ctypes.c_float)(), ctypes.c_size_t()
        out_rate, out_channels = ctypes.c_int(), ctypes.c_int()
        if spec.OUTPUT is None:
            _check(lib, lib.audiocpp_result_audio(result, ctypes.byref(data), ctypes.byref(frames),
                                                  ctypes.byref(out_rate), ctypes.byref(out_channels)), spec.MODEL_NAME)
        else:
            names = []
            for index in range(lib.audiocpp_result_named_audio_count(result)):
                name = ctypes.c_char_p()
                _check(lib, lib.audiocpp_result_named_audio(result, index, ctypes.byref(name), ctypes.byref(data),
                                                            ctypes.byref(frames), ctypes.byref(out_rate),
                                                            ctypes.byref(out_channels)), spec.MODEL_NAME)
                names.append(name.value.decode())
                if names[-1] == spec.OUTPUT:
                    break
            else:
                raise EngineError(f'{spec.MODEL_NAME} gave no {spec.OUTPUT!r} output, only {names}')
        out = array('f')
        out.frombytes(ctypes.string_at(data, frames.value * out_channels.value * 4))
        return Output(out, out_rate.value, out_channels.value)
    finally:
        lib.audiocpp_result_free(result)
        lib.audiocpp_request_free(request)

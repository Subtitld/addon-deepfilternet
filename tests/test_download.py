"""The weights download: pinned files, resumed after an interruption, refused when they do not check out.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

The download is tested whether or not this add-on's model downloads: the
fixture gives the spec a model that does.
"""

from __future__ import annotations

import hashlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from deepfilternet_addon import ADDON_ID, engine, spec

REPO, REVISION, FOLDER = 'audio-cpp/test-repo', 'a' * 40, 'Model-GGUF'
# Paths in the repository, as MODEL_FILES gives them; each is saved under its file name.
SMALL, BIG = f'{FOLDER}/small.bin', f'{FOLDER}/model.gguf'
FILES = {SMALL: b'{"architecture": "test"}', BIG: bytes(range(256)) * 400}


def _name(path: str) -> str:
    return path.rsplit('/', 1)[-1]


class _Hub(BaseHTTPRequestHandler):
    served: dict[str, bytes] = {}
    ignore_range = False
    requests: list[tuple[str, str | None]] = []

    def do_GET(self):  # noqa: N802
        prefix = f'/{REPO}/resolve/{REVISION}/'
        path = self.path[len(prefix):] if self.path.startswith(prefix) else None
        data = self.served.get(path)
        self.requests.append((path, self.headers.get('Range')))
        if data is None:
            self.send_error(404)
            return
        start = 0
        wanted = self.headers.get('Range')
        if wanted and not self.ignore_range:
            start = int(wanted.split('=', 1)[1].split('-', 1)[0])
            self.send_response(206)
            self.send_header('Content-Range', f'bytes {start}-{len(data) - 1}/{len(data)}')
        else:
            self.send_response(200)
        body = data[start:]
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def downloading(monkeypatch):
    """A spec whose model downloads FILES."""
    monkeypatch.setattr(spec, 'MODEL_REPO', REPO)
    monkeypatch.setattr(spec, 'MODEL_REVISION', REVISION)
    monkeypatch.setattr(spec, 'MODEL_DIRNAME', FOLDER)
    monkeypatch.setattr(spec, 'MODEL_FILE', _name(BIG))
    monkeypatch.setattr(spec, 'BUNDLED_WEIGHTS', None)
    monkeypatch.setattr(spec, 'MODEL_FILES', tuple(
        (name, len(data), hashlib.sha256(data).hexdigest()) for name, data in FILES.items()))
    monkeypatch.delenv(f'{spec.ENV_PREFIX}_MODEL_DIR', raising=False)


@pytest.fixture
def hub(downloading, monkeypatch, tmp_path):
    _Hub.served = dict(FILES)
    _Hub.ignore_range = False
    _Hub.requests = []
    server = ThreadingHTTPServer(('127.0.0.1', 0), _Hub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv('HF_ENDPOINT', f'http://127.0.0.1:{server.server_address[1]}')
    yield tmp_path / FOLDER
    server.shutdown()


def _never():
    return False


def test_downloads_every_file_and_remembers_it_checked_out(hub):
    seen = []
    engine.ensure_weights(hub, lambda done, total: seen.append((done, total)), _never)
    for name, data in FILES.items():
        assert (hub / _name(name)).read_bytes() == data
    assert engine.weights_present(hub)
    total = sum(len(d) for d in FILES.values())
    assert seen[-1] == (total, total)
    assert [done for done, _ in seen] == sorted(done for done, _ in seen)

    _Hub.requests.clear()
    engine.ensure_weights(hub, lambda *a: None, _never)
    assert _Hub.requests == []  # nothing fetched again


def test_an_interrupted_download_resumes(hub):
    hub.mkdir(parents=True)
    data = FILES[BIG]
    (hub / 'model.gguf.part').write_bytes(data[:30000])
    assert engine.missing_bytes(hub) == len(FILES[SMALL]) + len(data) - 30000
    engine.ensure_weights(hub, lambda *a: None, _never)
    assert (hub / 'model.gguf').read_bytes() == data
    assert (BIG, 'bytes=30000-') in _Hub.requests


def test_a_server_ignoring_the_range_still_gives_the_whole_file(hub):
    hub.mkdir(parents=True)
    (hub / 'model.gguf.part').write_bytes(FILES[BIG][:500])
    _Hub.ignore_range = True
    engine.ensure_weights(hub, lambda *a: None, _never)
    assert (hub / 'model.gguf').read_bytes() == FILES[BIG]


def test_a_file_that_does_not_match_its_sha256_is_discarded(hub):
    _Hub.served[BIG] = bytes(len(FILES[BIG]))
    with pytest.raises(engine.DownloadError, match='sha256'):
        engine.ensure_weights(hub, lambda *a: None, _never)
    assert not (hub / 'model.gguf').exists()
    assert not (hub / 'model.gguf.part').exists()
    assert not engine.weights_present(hub)


def test_a_missing_file_on_the_server_is_an_error(hub):
    del _Hub.served[SMALL]
    with pytest.raises(engine.DownloadError, match='HTTP 404'):
        engine.ensure_weights(hub, lambda *a: None, _never)


def test_cancel_keeps_what_was_downloaded(hub):
    calls = []

    def cancel():
        calls.append(1)
        return len(calls) > 3

    with pytest.raises(engine.Cancelled):
        engine.ensure_weights(hub, lambda *a: None, cancel)
    assert not engine.weights_present(hub)


def test_a_folder_given_by_the_user_is_used_as_it_is(downloading, monkeypatch, tmp_path):
    monkeypatch.setenv(f'{spec.ENV_PREFIX}_MODEL_DIR', str(tmp_path))
    assert engine.model_dir() == tmp_path
    with pytest.raises(engine.DownloadError, match='lacks'):
        engine.ensure_weights(tmp_path, lambda *a: None, _never)
    for name, _size, _sha in spec.MODEL_FILES:
        (tmp_path / _name(name)).write_bytes(b'x')
    assert engine.weights_present(tmp_path)


def test_the_weights_go_to_the_hosts_model_cache(downloading, monkeypatch, tmp_path):
    monkeypatch.setenv('SUBTITLD_MODELS_DIR', str(tmp_path))
    assert engine.model_dir() == tmp_path / ADDON_ID / FOLDER
    assert engine.model_path(engine.model_dir()) == tmp_path / ADDON_ID / FOLDER / 'model.gguf'


def test_bundled_weights_are_found_in_the_bundle(monkeypatch):
    monkeypatch.setattr(spec, 'MODEL_FILES', ())
    monkeypatch.setattr(spec, 'BUNDLED_WEIGHTS', ('utility', 'weights.safetensors'))
    monkeypatch.delenv(f'{spec.ENV_PREFIX}_MODEL_DIR', raising=False)
    folder = engine.model_dir()
    assert folder == engine.assets_dir() / 'weights' / 'utility'
    assert engine.model_path(folder) == folder

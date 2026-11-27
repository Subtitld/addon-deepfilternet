"""audio.separate through the request handler and the frame loop, with a stand-in for the model.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import io
import json


from conftest import make_media, needs_ffmpeg
from deepfilternet_addon import __main__ as addon
from deepfilternet_addon import ADDON_ID, engine, spec


def _terminal(frames):
    final = [f for f in frames if f['type'] in ('result', 'error')]
    assert len(final) == 1, frames
    return final[0]


def test_missing_paths_are_refused(run, tmp_path):
    assert _terminal(run({}))['code'] == 'bad_params'
    assert _terminal(run({'input_path': str(tmp_path / 'nope.mkv'), 'output_dir': str(tmp_path)}))['code'] \
        == 'bad_params'
    video = tmp_path / 'there.mkv'
    video.write_bytes(b'x')
    assert _terminal(run({'input_path': str(video)}))['code'] == 'bad_params'


@needs_ffmpeg
def test_separates_a_video_into_two_files(run, fake, tmp_path):
    video = make_media(tmp_path / 'clip.mkv', 1.5)
    out = tmp_path / 'out'
    frames = run({'input_path': str(video), 'output_dir': str(out), 'options': {}})
    final = _terminal(frames)
    assert final['type'] == 'result', final
    for kind in ('vocals', 'background'):
        path = out / f'clip_{ADDON_ID}_{kind}.flac'
        assert final['data'][kind] == str(path) and path.stat().st_size > 1000
    assert sorted(p.name for p in out.iterdir()) == [f'clip_{ADDON_ID}_background.flac',
                                                      f'clip_{ADDON_ID}_vocals.flac']
    values = [f['value'] for f in frames if f['type'] == 'progress']
    assert values == sorted(values) and values[-1] == 1.0
    assert fake.loads == 1
    # The scratch folder is gone.
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(f'{ADDON_ID}-')]


@needs_ffmpeg
def test_the_model_is_loaded_once_and_again_for_other_threads(run, fake, tmp_path):
    video = make_media(tmp_path / 'clip.mkv', 0.5)
    for threads in (None, None, 3):
        options = {} if threads is None else {'threads': threads}
        assert _terminal(run({'input_path': str(video), 'output_dir': str(tmp_path / 'o'),
                              'options': options}))['type'] == 'result'
    assert fake.loads == 2


@needs_ffmpeg
def test_a_cancel_leaves_no_files(run, fake, monkeypatch, tmp_path):
    monkeypatch.setattr(spec, 'CHUNK_SEC', 0.3)
    monkeypatch.setattr(spec, 'OVERLAP_SEC', 0.05)
    video = make_media(tmp_path / 'clip.mkv', 2.0)
    state = addon._Request('req-1')
    fake.on_call = lambda calls: state.cancel.set() if calls == 2 else None
    out = tmp_path / 'out'
    final = _terminal(run({'input_path': str(video), 'output_dir': str(out)}, state))
    assert final['code'] == 'cancelled'
    assert list(out.iterdir()) == []


@needs_ffmpeg
def test_a_failing_model_is_one_error(run, fake, monkeypatch, tmp_path):
    def broken(*args):
        raise engine.EngineError('boom')
    monkeypatch.setattr(engine, 'run', broken)
    video = make_media(tmp_path / 'clip.mkv', 0.5)
    final = _terminal(run({'input_path': str(video), 'output_dir': str(tmp_path / 'out')}))
    assert final['code'] == 'internal' and 'boom' in final['message']


@needs_ffmpeg
def test_a_download_failure_is_retryable(run, fake, monkeypatch, tmp_path):
    monkeypatch.setattr(engine, 'downloads', lambda: True)
    monkeypatch.setattr(engine, 'model_dir', lambda: tmp_path / 'models')
    monkeypatch.setattr(engine, 'weights_present', lambda folder: False)

    def offline(folder, progress, cancel):
        raise engine.DownloadError('no network')
    monkeypatch.setattr(engine, 'ensure_weights', offline)
    video = make_media(tmp_path / 'clip.mkv', 0.5)
    final = _terminal(run({'input_path': str(video), 'output_dir': str(tmp_path / 'out')}))
    assert final['code'] == 'network_unavailable' and final['retryable'] is True


def test_threads_come_from_the_request_or_the_environment(monkeypatch):
    monkeypatch.delenv(f'{spec.ENV_PREFIX}_THREADS', raising=False)
    assert addon._threads({}) == 0
    assert addon._threads({'threads': '6'}) == 6
    assert addon._threads({'threads': 1000}) == 256
    assert addon._threads({'threads': 'many'}) == 0
    monkeypatch.setenv(f'{spec.ENV_PREFIX}_THREADS', '5')
    assert addon._threads({}) == 5
    assert addon._threads({'threads': 2}) == 2


def test_the_frame_loop_answers_every_request(monkeypatch):
    frames: list[dict] = []
    monkeypatch.setattr(addon, '_emit', frames.append)
    started = []
    monkeypatch.setattr(addon, '_spawn', lambda req_id, target, params: started.append((req_id, params)))
    lines = [{'type': 'ready', 'host': 'test'},
             {'id': 'a', 'type': 'audio.separate', 'params': {'input_path': 'x'}},
             {'id': 'b', 'type': 'audio.separate', 'params': []},
             {'id': 'c', 'type': 'tts.synthesize', 'params': {}},
             {'type': 'cancel', 'target': 'zzz'},
             {'type': 'shutdown'}]
    stream = io.BytesIO(b'not json\n' + b''.join(json.dumps(line).encode() + b'\n' for line in lines))
    monkeypatch.setattr(addon, '_PROTO_IN', stream)
    assert addon._main_loop() == 0
    assert started == [('a', {'input_path': 'x'})]
    assert [(f['id'], f['code']) for f in frames] == [('b', 'bad_params'), ('c', 'bad_params')]


def test_hello_names_the_task(monkeypatch):
    frames: list[dict] = []
    monkeypatch.setattr(addon, '_emit', frames.append)
    addon._send_hello()
    assert frames == [{'type': 'hello', 'protocol': 1, 'addon': ADDON_ID, 'version': addon.ADDON_VERSION,
                       'capabilities': [{'task': 'audio.separate'}]}]

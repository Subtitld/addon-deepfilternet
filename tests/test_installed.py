"""The frozen binary, unpacked from its release archive the way Subtitld installs it.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

$DEEPFILTERNET_TEST_BINARY names the binary; without it these tests skip. The
release workflow sets it on every platform. The self-test loads audio.cpp
from the bundle, and runs the model when its weights are there. The tests
marked `weights` separate a real video and have one cancelled, through the
protocol, with ffmpeg on PATH; $DEEPFILTERNET_TEST_MODELS_DIR is the model cache
(as $SUBTITLD_MODELS_DIR would be), where a downloaded model is fetched once:

    DEEPFILTERNET_TEST_BINARY=... DEEPFILTERNET_TEST_MODELS_DIR=~/.cache/subtitld/models pytest tests/test_installed.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deepfilternet_addon import spec  # noqa: E402

ADDON_ID = 'deepfilternet'
BINARY = os.environ.get('DEEPFILTERNET_TEST_BINARY', '')
MODELS = os.environ.get('DEEPFILTERNET_TEST_MODELS_DIR', '')
pytestmark = pytest.mark.skipif(not BINARY, reason='DEEPFILTERNET_TEST_BINARY is not set')
# A model that downloads needs the cache to download into; one in the archive does not.
needs_models = pytest.mark.skipif(not shutil.which('ffmpeg') or (bool(spec.MODEL_FILES) and not MODELS),
                                  reason='no ffmpeg, or DEEPFILTERNET_TEST_MODELS_DIR is not set')


class Addon:
    """The binary on the protocol, with a watchdog so a hang fails instead of stalling CI."""

    def __init__(self, tmp_path: Path, timeout: float):
        env = dict(os.environ, SUBTITLD_TEMP_DIR=str(tmp_path),
                   SUBTITLD_MODELS_DIR=MODELS or str(tmp_path / 'models'))
        self.proc = subprocess.Popen([BINARY], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, env=env)
        self.timer = threading.Timer(timeout, self.proc.kill)
        self.timer.start()

    def send(self, frame: dict) -> None:
        self.proc.stdin.write((json.dumps(frame) + '\n').encode())
        self.proc.stdin.flush()

    def read(self) -> dict:
        line = self.proc.stdout.readline()
        assert line, 'the add-on closed its output (crashed or timed out)'
        return json.loads(line)

    def until_terminal(self, req_id: str, on_progress=None) -> dict:
        while True:
            frame = self.read()
            if frame.get('id') != req_id:
                continue
            if frame['type'] in ('result', 'error'):
                return frame
            if on_progress is not None:
                on_progress(frame)

    def close(self) -> int:
        try:
            self.send({'type': 'shutdown'})
        except (BrokenPipeError, OSError):
            pass
        try:
            return self.proc.wait(timeout=15)
        finally:
            self.timer.cancel()
            self.proc.kill()


def test_version():
    out = subprocess.run([BINARY, '--version'], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0 and out.stdout.startswith(f'{ADDON_ID}-addon ')


def test_self_test_loads_audiocpp():
    env = dict(os.environ, SUBTITLD_MODELS_DIR=MODELS) if MODELS else None
    out = subprocess.run([BINARY, '--self-test'], capture_output=True, text=True, timeout=300, env=env)
    assert out.returncode == 0, out.stderr[-2000:]
    report = json.loads(out.stdout.strip().splitlines()[-1])
    assert report['ok'] is True and report['runtime'].startswith('audio.cpp ')
    if report['model']:
        assert report['frames'] > 0


def test_handshake_and_refusals(tmp_path):
    addon = Addon(tmp_path, timeout=120)
    try:
        hello = addon.read()
        assert hello['type'] == 'hello' and hello['addon'] == ADDON_ID
        assert hello['capabilities'] == [{'task': 'audio.separate'}]
        addon.send({'type': 'ready', 'host': 'test'})
        addon.send({'id': 'a', 'type': 'audio.separate',
                    'params': {'input_path': str(tmp_path / 'none.mkv'), 'output_dir': str(tmp_path)}})
        assert addon.until_terminal('a')['code'] == 'bad_params'
    finally:
        assert addon.close() == 0


@pytest.mark.weights
@needs_models
def test_separates_a_video(tmp_path):
    from conftest import make_media
    video = make_media(tmp_path / 'clip.mkv', 40.0)  # two pieces
    out = tmp_path / 'out'
    addon = Addon(tmp_path, timeout=1800)
    try:
        addon.read()
        addon.send({'id': 'a', 'type': 'audio.separate',
                    'params': {'input_path': str(video), 'output_dir': str(out), 'options': {}}})
        result = addon.until_terminal('a')
        assert result['type'] == 'result', result
        for kind in ('vocals', 'background'):
            path = Path(result['data'][kind])
            assert path.parent == out and path.stat().st_size > 10000
            probe = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'stream=sample_rate,channels',
                                    '-show_entries', 'format=duration', '-of', 'json', str(path)],
                                   capture_output=True, text=True, check=True)
            info = json.loads(probe.stdout)
            assert info['streams'][0]['sample_rate'] == '48000' and info['streams'][0]['channels'] == 1
            assert abs(float(info['format']['duration']) - 40.0) < 0.2
    finally:
        assert addon.close() == 0


@pytest.mark.weights
@needs_models
def test_cancel_stops_a_separation(tmp_path):
    from conftest import make_media
    video = make_media(tmp_path / 'clip.mkv', 100.0)
    out = tmp_path / 'out'
    addon = Addon(tmp_path, timeout=1800)
    try:
        addon.read()
        cancelled = []

        def on_progress(frame):
            if frame.get('message') == 'Separating the voice...' and not cancelled:
                cancelled.append(frame['value'])
                addon.send({'type': 'cancel', 'target': 'a'})

        addon.send({'id': 'a', 'type': 'audio.separate',
                    'params': {'input_path': str(video), 'output_dir': str(out)}})
        error = addon.until_terminal('a', on_progress)
        assert cancelled and error['type'] == 'error' and error['code'] == 'cancelled'
        assert not out.exists() or list(out.iterdir()) == []
    finally:
        assert addon.close() == 0

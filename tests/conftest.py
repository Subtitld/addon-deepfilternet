"""Shared fixtures: test media made by ffmpeg, and a stand-in for the model.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from array import array
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deepfilternet_addon import __main__ as addon  # noqa: E402
from deepfilternet_addon import engine, spec  # noqa: E402

FFMPEG = shutil.which('ffmpeg')


def make_media(path: Path, seconds: float, video: bool = True) -> Path:
    """A video (or a sound file) of a tone over noise, made by ffmpeg."""
    inputs = ['-f', 'lavfi', '-i', f'sine=frequency=330:duration={seconds}',
              '-f', 'lavfi', '-i', f'anoisesrc=color=pink:amplitude=0.05:duration={seconds}']
    graph = ['-filter_complex', '[0:a][1:a]amix=inputs=2:normalize=0,aformat=channel_layouts=stereo[a]',
             '-map', '[a]']
    if video:
        inputs += ['-f', 'lavfi', '-i', f'color=c=black:s=64x64:r=10:d={seconds}']
        graph += ['-map', '2:v', '-c:v', 'mpeg4']
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y', *inputs, *graph, '-ar', '48000', str(path)],
                   check=True)
    return path


class FakeModel:
    """Gives back its input as the voice: the stitching must then rebuild the mix exactly.

    `delay` frames of lag, like RNNoise, and `cancel_after` pieces, like a
    user pressing Cancel, are optional.
    """

    def __init__(self):
        self.loads = 0
        self.calls = 0
        self.delay = 0
        self.on_call = None

    def load(self, path, threads=0):
        self.loads += 1
        return engine.Loaded(lib=None, registry=None, model=None, session=None, threads=threads or 4)

    def run(self, loaded, samples, rate, channels):
        self.calls += 1
        if self.on_call is not None:
            self.on_call(self.calls)
        out = array('f', [0.0]) * (self.delay * channels)
        out.extend(samples[:len(samples) - self.delay * channels])
        return engine.Output(out, rate, channels)


@pytest.fixture
def fake(monkeypatch, tmp_path):
    model = FakeModel()
    monkeypatch.setattr(engine, 'load', model.load)
    monkeypatch.setattr(engine, 'run', model.run)
    monkeypatch.setattr(engine, 'close', lambda loaded: None)
    monkeypatch.setattr(engine, 'weights_present', lambda folder: True)
    monkeypatch.setattr(addon, '_loaded', None)
    monkeypatch.setattr(addon, '_loaded_threads', -1)
    monkeypatch.setenv('SUBTITLD_MODELS_DIR', str(tmp_path / 'models'))
    monkeypatch.setenv('SUBTITLD_TEMP_DIR', str(tmp_path))
    monkeypatch.delenv(f'{spec.ENV_PREFIX}_THREADS', raising=False)
    return model


@pytest.fixture
def run(fake, monkeypatch):
    """run(params) -> frames: one audio.separate request through the handler."""
    frames: list[dict] = []
    monkeypatch.setattr(addon, '_emit', frames.append)

    def _run(params: dict, state=None):
        state = state or addon._Request('req-1')
        frames.clear()
        addon._handle_separate(state, params)
        return list(frames)
    return _run


needs_ffmpeg = pytest.mark.skipif(FFMPEG is None, reason='ffmpeg is not on PATH')


def pytest_configure(config):
    # Also declared in pyproject.toml, which tools/test_archive.sh does not mount.
    config.addinivalue_line('markers', "weights: needs the model's weights and audio.cpp")

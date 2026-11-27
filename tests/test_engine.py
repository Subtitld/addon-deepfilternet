"""audio.cpp as built by tools/build_native.py: loaded, typed, and running the model.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

Needs build/assets/audiocpp/ (run tools/build_native.py first). The tests
marked `weights` also need the model's weights: in the archive, or, for a
model that downloads them, already in $SUBTITLD_MODELS_DIR (or the folder
$<PREFIX>_MODEL_DIR names).
"""

from __future__ import annotations

import math
from array import array

import pytest

from deepfilternet_addon import engine, spec

needs_weights = pytest.mark.skipif(not engine.weights_present(engine.model_dir()),
                                   reason=f'no {spec.MODEL_NAME} weights in {engine.model_dir()}')


def test_the_library_has_the_family():
    assert engine.runtime_version().startswith('audio.cpp 0.9.5 (C ABI 0.')
    assert spec.FAMILY in engine.families()


def test_a_missing_library_is_reported(monkeypatch, tmp_path):
    monkeypatch.setattr(engine, '_lib', None)
    monkeypatch.setenv(f'{spec.ENV_PREFIX}_AUDIOCPP_DIR', str(tmp_path))
    with pytest.raises(engine.EngineError, match='missing'):
        engine.load_library()


def test_a_model_that_is_not_there_fails_cleanly(tmp_path):
    with pytest.raises(engine.EngineError):
        engine.load(tmp_path / 'nothing.gguf', threads=1)


def test_threads():
    assert engine.default_threads() >= 1
    cpus = engine.performance_cpus()
    assert cpus == sorted(cpus)


@pytest.mark.weights
@needs_weights
def test_the_model_gives_a_voice_as_long_as_its_input():
    loaded = engine.load(engine.model_path(engine.model_dir()), threads=2)
    try:
        frames = int(spec.RATE * max(2.0, spec.MIN_PIECE_SEC + 0.5))
        tone = array('f', (0.3 * math.sin(2 * math.pi * 220 * i / spec.RATE)
                           for i in range(frames) for _c in range(spec.CHANNELS)))
        out = engine.run(loaded, tone, spec.RATE, spec.CHANNELS)
        assert (out.rate, out.channels, out.frames) == (spec.RATE, spec.CHANNELS, frames)
        assert all(math.isfinite(x) for x in out.samples)
    finally:
        engine.close(loaded)

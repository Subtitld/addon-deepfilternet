"""The sound in pieces: decoded, stitched back without a seam, and written as voice and background.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import subprocess
import wave
from array import array

import pytest

from conftest import make_media, needs_ffmpeg
from deepfilternet_addon import engine, separate, spec


def test_pieces_cover_the_sound_and_overlap():
    assert separate.pieces(100, 30, 5) == [(0, 30), (25, 55), (50, 80), (75, 100)]
    assert separate.pieces(30, 30, 5) == [(0, 30)]
    assert separate.pieces(10, 30, 5) == [(0, 10)]
    spans = separate.pieces(1_000_003, 300_000, 10_000)
    assert spans[0][0] == 0 and spans[-1][1] == 1_000_003
    for (_a, end), (start, _b) in zip(spans, spans[1:]):
        assert end - start == 10_000


def test_the_last_piece_is_longer_than_the_overlap():
    for frames in range(31, 200):
        spans = separate.pieces(frames, 30, 5)
        assert all(end - start > 5 for start, end in spans[1:]), frames


def test_a_crossfade_of_equal_sounds_is_that_sound():
    head = array('f', [0.5, -0.5] * 10)
    assert list(separate._crossfade(array('f', head), head, 2)) == pytest.approx(list(head))
    faded = separate._crossfade(array('f', [1.0] * 4), array('f', [0.0] * 4), 1)
    assert list(faded) == pytest.approx([0.875, 0.625, 0.375, 0.125])


def _ramp(tmp_path, frames):
    mix = tmp_path / 'mix.raw'
    samples = array('f', ((i % 997) / 997.0 - 0.5 for i in range(frames * spec.CHANNELS)))
    mix.write_bytes(samples.tobytes())
    return mix, samples


@pytest.mark.parametrize('delay, shortest', [(0, 0.0), (960, 0.0), (0, 0.3), (960, 0.3)])
def test_stitching_rebuilds_the_voice_exactly(fake, monkeypatch, tmp_path, delay, shortest):
    monkeypatch.setattr(spec, 'CHUNK_SEC', 0.25)
    monkeypatch.setattr(spec, 'OVERLAP_SEC', 0.05)
    monkeypatch.setattr(spec, 'DELAY_FRAMES', delay)
    monkeypatch.setattr(spec, 'MIN_PIECE_SEC', shortest)
    fake.delay = delay
    lengths = []
    run = fake.run
    fake.run = lambda loaded, samples, rate, channels: (lengths.append(len(samples) // channels),
                                                        run(loaded, samples, rate, channels))[1]
    monkeypatch.setattr(engine, 'run', fake.run)
    frames = int(spec.RATE * 1.3)
    mix, samples = _ramp(tmp_path, frames)
    shares = []
    separate.voice(fake.load(None), mix, frames, tmp_path / 'voice.raw', lambda: False, shares.append)
    out = array('f')
    out.frombytes((tmp_path / 'voice.raw').read_bytes())
    assert len(out) == len(samples)
    assert list(out) == pytest.approx(list(samples), abs=1e-6)
    assert fake.calls == len(shares) == len(separate.pieces(frames, int(0.25 * spec.RATE), int(0.05 * spec.RATE)))
    assert shares[-1] == 1.0
    assert min(lengths) >= int(shortest * spec.RATE)


def test_a_cancel_stops_between_two_pieces(fake, monkeypatch, tmp_path):
    monkeypatch.setattr(spec, 'CHUNK_SEC', 0.25)
    monkeypatch.setattr(spec, 'OVERLAP_SEC', 0.05)
    frames = spec.RATE * 2
    mix, _ = _ramp(tmp_path, frames)
    with pytest.raises(engine.Cancelled):
        separate.voice(fake.load(None), mix, frames, tmp_path / 'voice.raw', lambda: fake.calls >= 2,
                       lambda share: None)
    assert fake.calls == 2


def test_a_model_that_changes_the_rate_is_refused(fake, monkeypatch, tmp_path):
    monkeypatch.setattr(engine, 'run', lambda loaded, samples, rate, channels: engine.Output(samples, rate // 2,
                                                                                            channels))
    mix, _ = _ramp(tmp_path, 1000)
    with pytest.raises(engine.EngineError, match='Hz'):
        separate.voice(fake.load(None), mix, 1000, tmp_path / 'voice.raw', lambda: False, lambda share: None)


def _flac(path):
    """(rate, channels, samples as float) of a FLAC file, decoded by ffmpeg."""
    probe = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'stream=sample_rate,channels',
                            '-of', 'csv=p=0', str(path)], capture_output=True, text=True, check=True)
    rate, channels = (int(x) for x in probe.stdout.strip().split(','))
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-f', 'f32le', '-'], capture_output=True,
                         check=True).stdout
    samples = array('f')
    samples.frombytes(raw)
    return rate, channels, samples


@needs_ffmpeg
def test_a_video_is_decoded_at_the_models_rate(tmp_path):
    video = make_media(tmp_path / 'clip.mkv', 2.0)
    frames = separate.decode(str(video), tmp_path / 'mix.raw', lambda: False)
    assert frames == pytest.approx(2.0 * spec.RATE, rel=0.01)
    assert (tmp_path / 'mix.raw').stat().st_size == frames * spec.CHANNELS * 4


@needs_ffmpeg
def test_a_file_without_sound_is_refused(tmp_path):
    silent = tmp_path / 'silent.mkv'
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i', 'color=c=black:s=64x64:d=1',
                    '-c:v', 'mpeg4', str(silent)], check=True)
    with pytest.raises(separate.Failed, match='silent.mkv'):
        separate.decode(str(silent), tmp_path / 'mix.raw', lambda: False)


@needs_ffmpeg
def test_the_voice_and_the_rest_of_the_mix_are_written(tmp_path):
    video = make_media(tmp_path / 'clip.mkv', 1.0)
    mix = tmp_path / 'mix.raw'
    frames = separate.decode(str(video), mix, lambda: False)
    # Half of the mix as the voice: the background is then the other half.
    samples = array('f')
    samples.frombytes(mix.read_bytes())
    half = array('f', (x * 0.5 for x in samples))
    (tmp_path / 'voice.raw').write_bytes(half.tobytes())
    vocals, background = tmp_path / 'v.flac', tmp_path / 'b.flac'
    separate.encode(mix, tmp_path / 'voice.raw', vocals, background, lambda: False)
    rate_v, channels_v, voice = _flac(vocals)
    rate_b, channels_b, rest = _flac(background)
    assert (rate_v, channels_v, rate_b, channels_b) == (48000, 1, 48000, 1)
    assert len(voice) == pytest.approx(frames * 48000 / spec.RATE, abs=64)
    # A tone and noise at 16 bits: the two halves agree to within a few steps.
    middle = slice(1000, len(voice) - 1000)
    assert max(abs(a - b) for a, b in zip(voice[middle], rest[middle])) < 0.002
    assert max(abs(x) for x in voice[middle]) > 0.03  # ffmpeg's sine is at 1/8, halved here


@needs_ffmpeg
def test_writing_is_a_flac_ffmpeg_reads_back(tmp_path):
    video = make_media(tmp_path / 'clip.wav', 0.5, video=False)
    mix = tmp_path / 'mix.raw'
    separate.decode(str(video), mix, lambda: False)
    (tmp_path / 'voice.raw').write_bytes(mix.read_bytes())
    separate.encode(mix, tmp_path / 'voice.raw', tmp_path / 'v.flac', tmp_path / 'b.flac', lambda: False)
    with pytest.raises(wave.Error):
        wave.open(str(tmp_path / 'v.flac'))
    _rate, _channels, rest = _flac(tmp_path / 'b.flac')
    assert max(abs(x) for x in rest) < 0.001  # the voice was the whole mix

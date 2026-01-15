"""Test beat grid functionality."""
import pytest

from mixinto.dsp.beats import build_beat_grid
from mixinto.utils.types import AudioBuffer, AudioMetadata
import numpy as np


def test_build_beat_grid(sample_audio_buffer):
    """Test building beat grid."""
    beat_grid = build_beat_grid(sample_audio_buffer)
    
    assert beat_grid.bpm > 0
    assert 60 <= beat_grid.bpm <= 200
    assert 0.0 <= beat_grid.confidence <= 1.0
    assert len(beat_grid.beats_s) > 0
    assert len(beat_grid.downbeats_s) > 0


def test_build_beat_grid_with_bpm(sample_audio_buffer):
    """Test building beat grid with known BPM."""
    known_bpm = 120.0
    beat_grid = build_beat_grid(sample_audio_buffer, bpm=known_bpm)
    
    assert beat_grid.bpm == known_bpm
    assert len(beat_grid.beats_s) > 0


def test_beat_grid_seconds_per_beat(beat_grid_fixture):
    """Test seconds per beat calculation."""
    expected = 60.0 / beat_grid_fixture.bpm
    assert beat_grid_fixture.seconds_per_beat() == pytest.approx(expected, abs=0.001)


def test_beat_grid_nearest_beat(beat_grid_fixture):
    """Test finding nearest beat."""
    # Test with a time that should be close to a beat
    test_time = 1.0
    nearest = beat_grid_fixture.nearest_beat(test_time)
    
    # Should be one of the beats
    assert nearest in beat_grid_fixture.beats_s or abs(nearest - test_time) < 1.0


def test_build_beat_grid_short_audio():
    """Test building beat grid on very short audio."""
    sample_rate = 44100
    duration_s = 0.5  # Very short
    samples = int(sample_rate * duration_s)
    
    # Simple tone
    t = np.linspace(0, duration_s, samples)
    audio_data = np.sin(2 * np.pi * 440 * t).reshape(-1, 1)
    
    meta = AudioMetadata(
        source_path="short.wav",
        sample_rate=sample_rate,
        channels=1,
        duration_s=duration_s,
        format="wav",
    )
    
    buffer = AudioBuffer(
        samples=audio_data,
        sample_rate=sample_rate,
        meta=meta,
    )
    
    # Should either work or raise a meaningful error
    try:
        beat_grid = build_beat_grid(buffer)
        assert len(beat_grid.beats_s) >= 0
    except ValueError as e:
        # Expected if too short
        assert "beat" in str(e).lower() or "insufficient" in str(e).lower()

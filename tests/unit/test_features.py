"""Test feature extraction functionality."""
import pytest
import numpy as np

from mixinto.features.stability import (
    calculate_energy_features,
    calculate_rhythm_stability,
    calculate_spectral_stability,
)
from mixinto.features.vocals import detect_vocal_presence
from mixinto.utils.types import AudioBuffer, AudioMetadata, BeatGrid


def test_calculate_spectral_stability(sample_audio_buffer):
    """Test spectral stability calculation."""
    stability = calculate_spectral_stability(sample_audio_buffer)
    
    assert 0.0 <= stability <= 1.0


def test_calculate_rhythm_stability(sample_audio_buffer, beat_grid_fixture):
    """Test rhythm stability calculation."""
    stability = calculate_rhythm_stability(sample_audio_buffer, beat_grid_fixture)
    
    assert 0.0 <= stability <= 1.0


def test_calculate_energy_features(sample_audio_buffer):
    """Test energy feature calculation."""
    mean_energy, std_energy = calculate_energy_features(sample_audio_buffer)
    
    assert mean_energy >= 0
    assert std_energy >= 0


def test_detect_vocal_presence(sample_audio_buffer):
    """Test vocal presence detection."""
    vocal_score = detect_vocal_presence(sample_audio_buffer)
    
    assert 0.0 <= vocal_score <= 1.0


def test_spectral_stability_consistency():
    """Test that spectral stability is consistent for similar audio."""
    sample_rate = 44100
    duration_s = 2.0
    samples = int(sample_rate * duration_s)
    
    # Create two similar sine waves
    t = np.linspace(0, duration_s, samples)
    audio1 = np.sin(2 * np.pi * 440 * t).reshape(-1, 1)
    audio2 = np.sin(2 * np.pi * 440 * t).reshape(-1, 1)
    
    meta = AudioMetadata(
        source_path="test.wav",
        sample_rate=sample_rate,
        channels=1,
        duration_s=duration_s,
        format="wav",
    )
    
    buffer1 = AudioBuffer(samples=audio1, sample_rate=sample_rate, meta=meta)
    buffer2 = AudioBuffer(samples=audio2, sample_rate=sample_rate, meta=meta)
    
    stability1 = calculate_spectral_stability(buffer1)
    stability2 = calculate_spectral_stability(buffer2)
    
    # Should be similar (within reasonable tolerance)
    assert abs(stability1 - stability2) < 0.2


def test_rhythm_stability_with_consistent_beats():
    """Test rhythm stability with very consistent beats."""
    # Create a beat grid with perfect timing
    bpm = 120.0
    beats_per_second = bpm / 60.0
    beats_s = [i / beats_per_second for i in range(20)]
    downbeats_s = beats_s[::4]
    
    beat_grid = BeatGrid(
        bpm=bpm,
        beats_s=beats_s,
        downbeats_s=downbeats_s,
        confidence=1.0,
    )
    
    # Create simple audio buffer
    sample_rate = 44100
    duration_s = 5.0
    samples = int(sample_rate * duration_s)
    t = np.linspace(0, duration_s, samples)
    audio_data = np.sin(2 * np.pi * 440 * t).reshape(-1, 1)
    
    meta = AudioMetadata(
        source_path="test.wav",
        sample_rate=sample_rate,
        channels=1,
        duration_s=duration_s,
        format="wav",
    )
    
    buffer = AudioBuffer(samples=audio_data, sample_rate=sample_rate, meta=meta)
    
    stability = calculate_rhythm_stability(buffer, beat_grid)
    
    # Perfect timing should give high stability
    assert stability > 0.5

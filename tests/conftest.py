"""Shared test fixtures."""
import pytest
import numpy as np
from pathlib import Path
import tempfile
import soundfile as sf

from mixinto.utils.types import AudioBuffer, AudioMetadata


@pytest.fixture
def sample_audio_buffer():
    """Create a simple test audio buffer."""
    sample_rate = 44100
    duration_s = 5.0
    samples = int(sample_rate * duration_s)
    
    # Generate a simple sine wave (440 Hz)
    t = np.linspace(0, duration_s, samples)
    audio_data = np.sin(2 * np.pi * 440 * t)
    audio_data = audio_data.reshape(-1, 1)  # Mono, 2D shape
    
    meta = AudioMetadata(
        source_path="test.wav",
        sample_rate=sample_rate,
        channels=1,
        duration_s=duration_s,
        format="wav",
    )
    
    return AudioBuffer(
        samples=audio_data,
        sample_rate=sample_rate,
        meta=meta,
    )


@pytest.fixture
def stereo_audio_buffer():
    """Create a stereo test audio buffer."""
    sample_rate = 44100
    duration_s = 5.0
    samples = int(sample_rate * duration_s)
    
    # Generate stereo audio (different frequencies for L/R)
    t = np.linspace(0, duration_s, samples)
    left = np.sin(2 * np.pi * 440 * t)
    right = np.sin(2 * np.pi * 880 * t)
    audio_data = np.column_stack([left, right])
    
    meta = AudioMetadata(
        source_path="test_stereo.wav",
        sample_rate=sample_rate,
        channels=2,
        duration_s=duration_s,
        format="wav",
    )
    
    return AudioBuffer(
        samples=audio_data,
        sample_rate=sample_rate,
        meta=meta,
    )


@pytest.fixture
def temp_audio_file(sample_audio_buffer):
    """Create a temporary audio file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = Path(f.name)
    
    # Write test audio
    sf.write(
        str(temp_path),
        sample_audio_buffer.samples,
        sample_audio_buffer.sample_rate,
    )
    
    yield temp_path
    
    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def beat_grid_fixture():
    """Create a mock beat grid for testing."""
    from mixinto.utils.types import BeatGrid
    
    # Create a beat grid with 120 BPM
    bpm = 120.0
    duration_s = 10.0
    beats_per_second = bpm / 60.0
    
    # Generate beats
    beats_s = []
    current_time = 0.0
    while current_time < duration_s:
        beats_s.append(current_time)
        current_time += 1.0 / beats_per_second
    
    # Generate downbeats (every 4 beats)
    downbeats_s = beats_s[::4]
    
    return BeatGrid(
        bpm=bpm,
        beats_s=beats_s,
        downbeats_s=downbeats_s,
        confidence=0.9,
    )


@pytest.fixture
def intro_profile_fixture():
    """Create a mock intro profile for testing."""
    from mixinto.utils.types import IntroProfile
    
    return IntroProfile(
        start_s=0.0,
        end_s=16.0,
        energy_mean=0.5,
        energy_std=0.1,
        spectral_stability=0.8,
        rhythm_stability=0.9,
        vocal_presence=0.2,
        mix_safety_score=0.85,
        flags=[],
    )

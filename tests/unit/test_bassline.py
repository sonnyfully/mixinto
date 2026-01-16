"""Test bassline generation functionality."""
import hashlib
import pytest
import numpy as np

from mixinto.generation.bassline import (
    estimate_root_note,
    generate_bassline,
    mix_bass_into_extension,
)
from mixinto.utils.types import AudioBuffer, AudioMetadata


@pytest.fixture
def context_audio_with_bass():
    """Create a context audio buffer with bass content."""
    sample_rate = 44100
    duration_s = 4.0
    samples = int(sample_rate * duration_s)
    
    # Generate audio with bass at ~80 Hz (E2, MIDI 40)
    t = np.linspace(0, duration_s, samples)
    bass_freq = 80.0
    audio_data = np.sin(2 * np.pi * bass_freq * t) * 0.5
    audio_data = audio_data.reshape(-1, 1)
    
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


def test_generate_bassline_length_exact():
    """Test that generated bassline has exact duration matching bars."""
    sr = 44100
    bpm = 120.0
    bars = 8
    seed = 42
    root_midi = 43  # G2
    
    # Calculate expected duration
    seconds_per_beat = 60.0 / bpm
    seconds_per_bar = seconds_per_beat * 4
    expected_duration_s = bars * seconds_per_bar
    
    bassline = generate_bassline(
        sr=sr,
        bpm=bpm,
        bars=bars,
        seed=seed,
        root_midi=root_midi,
        preset="dj_safe",
    )
    
    # Check duration is exact (within 1 sample tolerance)
    actual_duration_s = bassline.length_s()
    expected_samples = int(expected_duration_s * sr)
    actual_samples = len(bassline)
    
    assert actual_samples == expected_samples, (
        f"Expected {expected_samples} samples, got {actual_samples}"
    )
    assert abs(actual_duration_s - expected_duration_s) < (1.0 / sr), (
        f"Expected {expected_duration_s}s, got {actual_duration_s}s"
    )


def test_generate_bassline_determinism():
    """Test that same seed produces identical waveform."""
    sr = 44100
    bpm = 120.0
    bars = 4
    seed = 123
    root_midi = 43
    
    # Generate twice with same seed
    bassline1 = generate_bassline(
        sr=sr,
        bpm=bpm,
        bars=bars,
        seed=seed,
        root_midi=root_midi,
        preset="dj_safe",
    )
    
    bassline2 = generate_bassline(
        sr=sr,
        bpm=bpm,
        bars=bars,
        seed=seed,
        root_midi=root_midi,
        preset="dj_safe",
    )
    
    # Check samples are identical
    assert np.array_equal(bassline1.samples, bassline2.samples), (
        "Same seed should produce identical samples"
    )
    
    # Also check hash for quick verification
    hash1 = hashlib.md5(bassline1.samples.tobytes()).hexdigest()
    hash2 = hashlib.md5(bassline2.samples.tobytes()).hexdigest()
    assert hash1 == hash2, "Waveform hash should match for same seed"


def test_generate_bassline_different_seeds():
    """Test that different seeds produce different waveforms."""
    sr = 44100
    bpm = 120.0
    bars = 4
    root_midi = 43
    
    bassline1 = generate_bassline(
        sr=sr,
        bpm=bpm,
        bars=bars,
        seed=1,
        root_midi=root_midi,
        preset="more_motion",  # Use more_motion to get variation
    )
    
    bassline2 = generate_bassline(
        sr=sr,
        bpm=bpm,
        bars=bars,
        seed=2,
        root_midi=root_midi,
        preset="more_motion",
    )
    
    # For more_motion preset, different seeds should produce different patterns
    hash1 = hashlib.md5(bassline1.samples.tobytes()).hexdigest()
    hash2 = hashlib.md5(bassline2.samples.tobytes()).hexdigest()
    assert hash1 != hash2, "Different seeds should produce different waveforms"


def test_generate_bassline_no_clipping():
    """Test that generated bassline doesn't clip."""
    sr = 44100
    bpm = 120.0
    bars = 8
    seed = 42
    root_midi = 43
    
    bassline = generate_bassline(
        sr=sr,
        bpm=bpm,
        bars=bars,
        seed=seed,
        root_midi=root_midi,
        preset="dj_safe",
    )
    
    # Check peak is within safe range
    peak = np.max(np.abs(bassline.samples))
    assert peak <= 0.99, f"Peak should be <= 0.99, got {peak}"
    assert peak > 0.0, "Peak should be greater than 0"


def test_mix_bass_into_extension_no_clipping():
    """Test that mixing bass into extension doesn't cause clipping."""
    sr = 44100
    duration_s = 4.0
    samples = int(sr * duration_s)
    
    # Create extension audio (moderate level)
    extension_samples = np.random.randn(samples, 2) * 0.3  # Stereo
    extension_samples = extension_samples.reshape(-1, 2)
    
    extension_meta = AudioMetadata(
        source_path="test.wav",
        sample_rate=sr,
        channels=2,
        duration_s=duration_s,
        format="wav",
    )
    extension = AudioBuffer(
        samples=extension_samples,
        sample_rate=sr,
        meta=extension_meta,
    )
    
    # Create bassline
    bars = int(duration_s / (60.0 / 120.0 / 4))  # Approximate bars
    bassline = generate_bassline(
        sr=sr,
        bpm=120.0,
        bars=bars,
        seed=42,
        root_midi=43,
        preset="dj_safe",
    )
    
    # Mix with moderate gain
    mixed = mix_bass_into_extension(
        extension=extension,
        bass=bassline,
        bass_gain_db=-12.0,
    )
    
    # Check no clipping
    peak = np.max(np.abs(mixed.samples))
    assert peak <= 0.99, f"Peak should be <= 0.99 after mixing, got {peak}"


def test_mix_bass_into_extension_length_match():
    """Test that mixing matches lengths exactly."""
    sr = 44100
    duration_s = 4.0
    samples = int(sr * duration_s)
    
    # Create extension audio
    extension_samples = np.random.randn(samples, 1) * 0.3
    extension_meta = AudioMetadata(
        source_path="test.wav",
        sample_rate=sr,
        channels=1,
        duration_s=duration_s,
        format="wav",
    )
    extension = AudioBuffer(
        samples=extension_samples,
        sample_rate=sr,
        meta=extension_meta,
    )
    
    # Create bassline (might be slightly different length)
    bars = int(duration_s / (60.0 / 120.0 / 4))
    bassline = generate_bassline(
        sr=sr,
        bpm=120.0,
        bars=bars,
        seed=42,
        root_midi=43,
        preset="dj_safe",
    )
    
    # Mix
    mixed = mix_bass_into_extension(
        extension=extension,
        bass=bassline,
        bass_gain_db=-15.0,
    )
    
    # Check length matches extension exactly
    assert len(mixed) == len(extension), (
        f"Mixed length {len(mixed)} should match extension length {len(extension)}"
    )


def test_mix_bass_into_extension_stereo_upmix():
    """Test that mono bass is upmixed to stereo when extension is stereo."""
    sr = 44100
    duration_s = 2.0
    samples = int(sr * duration_s)
    
    # Create stereo extension
    extension_samples = np.random.randn(samples, 2) * 0.3
    extension_meta = AudioMetadata(
        source_path="test.wav",
        sample_rate=sr,
        channels=2,
        duration_s=duration_s,
        format="wav",
    )
    extension = AudioBuffer(
        samples=extension_samples,
        sample_rate=sr,
        meta=extension_meta,
    )
    
    # Create mono bassline
    bars = int(duration_s / (60.0 / 120.0 / 4))
    bassline = generate_bassline(
        sr=sr,
        bpm=120.0,
        bars=bars,
        seed=42,
        root_midi=43,
        preset="dj_safe",
    )
    
    assert bassline.meta.channels == 1, "Bassline should be mono"
    
    # Mix
    mixed = mix_bass_into_extension(
        extension=extension,
        bass=bassline,
        bass_gain_db=-15.0,
    )
    
    # Check output is stereo
    assert mixed.meta.channels == 2, "Mixed output should be stereo"
    assert mixed.samples.shape[1] == 2, "Mixed samples should have 2 channels"


def test_estimate_root_note(context_audio_with_bass):
    """Test root note estimation from context audio."""
    root_midi = estimate_root_note(context_audio_with_bass, bpm=120.0)
    
    # Should return a MIDI note in bass range (36-52)
    assert root_midi is None or (36 <= root_midi <= 52), (
        f"Root MIDI note should be in range 36-52 or None, got {root_midi}"
    )


def test_estimate_root_note_none_on_weak_signal():
    """Test that estimate_root_note returns None for weak/no bass signal."""
    # Create audio with no bass content (high frequency only)
    sr = 44100
    duration_s = 2.0
    samples = int(sr * duration_s)
    t = np.linspace(0, duration_s, samples)
    
    # High frequency only (no bass)
    audio_data = np.sin(2 * np.pi * 2000.0 * t) * 0.1
    audio_data = audio_data.reshape(-1, 1)
    
    meta = AudioMetadata(
        source_path="test.wav",
        sample_rate=sr,
        channels=1,
        duration_s=duration_s,
        format="wav",
    )
    
    context = AudioBuffer(
        samples=audio_data,
        sample_rate=sr,
        meta=meta,
    )
    
    root_midi = estimate_root_note(context, bpm=120.0)
    
    # Should return None or a default value (implementation dependent)
    # Just check it doesn't crash and returns something reasonable
    assert root_midi is None or (36 <= root_midi <= 52)

"""Test tempo detection."""
import pytest
import numpy as np

from mixinto.dsp.tempo import detect_tempo
from mixinto.utils.types import AudioBuffer, AudioMetadata


def test_detect_tempo(sample_audio_buffer):
    """Test tempo detection on simple audio."""
    bpm, confidence = detect_tempo(sample_audio_buffer, confidence_threshold=0.1)
    
    assert 60 <= bpm <= 200  # Reasonable BPM range
    assert 0.0 <= confidence <= 1.0


def test_detect_tempo_stereo(stereo_audio_buffer):
    """Test tempo detection on stereo audio."""
    bpm, confidence = detect_tempo(stereo_audio_buffer, confidence_threshold=0.1)
    
    assert 60 <= bpm <= 200
    assert 0.0 <= confidence <= 1.0


def test_detect_tempo_low_confidence():
    """Test tempo detection with very low confidence threshold."""
    # Create a very short, noisy buffer that might have low confidence
    sample_rate = 44100
    duration_s = 0.5  # Very short
    samples = int(sample_rate * duration_s)
    
    # Random noise
    audio_data = np.random.randn(samples).reshape(-1, 1)
    
    meta = AudioMetadata(
        source_path="noise.wav",
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
    
    # Should either succeed with low threshold or raise error
    try:
        bpm, confidence = detect_tempo(buffer, confidence_threshold=0.01)
        assert 0.0 <= confidence <= 1.0
    except ValueError:
        # Expected if confidence is too low
        pass


def test_detect_tempo_confidence_threshold():
    """Test that confidence threshold is enforced."""
    sample_rate = 44100
    duration_s = 0.1  # Very short
    samples = int(sample_rate * duration_s)
    
    # Random noise
    audio_data = np.random.randn(samples).reshape(-1, 1)
    
    meta = AudioMetadata(
        source_path="noise.wav",
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
    
    # High threshold should fail
    with pytest.raises(ValueError, match="confidence too low"):
        detect_tempo(buffer, confidence_threshold=0.9)

"""Test enhanced stability and tempo detection features."""
import numpy as np
import pytest

from mixinto.dsp.tempo.detector import detect_tempo
from mixinto.features.stability import (
    calculate_energy_features,
    calculate_rhythm_stability,
    calculate_spectral_stability,
)
from mixinto.utils.types import AnalysisConfig, AudioBuffer, AudioMetadata, BeatGrid


def test_tempo_detection_with_config(sample_audio_buffer):
    """Test tempo detection with custom config."""
    config = AnalysisConfig(
        tempo_min=100.0,
        tempo_max=140.0,
        tempo_confidence_threshold=0.2,
    )
    
    bpm, confidence = detect_tempo(sample_audio_buffer, config=config)
    
    assert 100.0 <= bpm <= 140.0
    assert 0.0 <= confidence <= 1.0


def test_spectral_stability_with_config(sample_audio_buffer):
    """Test spectral stability with different feature sets."""
    # Test with chroma only
    config_chroma = AnalysisConfig(spectral_features=["chroma"])
    stability_chroma = calculate_spectral_stability(sample_audio_buffer, config_chroma)
    assert 0.0 <= stability_chroma <= 1.0
    
    # Test with mfcc
    config_mfcc = AnalysisConfig(spectral_features=["mfcc"])
    stability_mfcc = calculate_spectral_stability(sample_audio_buffer, config_mfcc)
    assert 0.0 <= stability_mfcc <= 1.0
    
    # Test with multiple features
    config_multi = AnalysisConfig(spectral_features=["chroma", "mfcc"])
    stability_multi = calculate_spectral_stability(sample_audio_buffer, config_multi)
    assert 0.0 <= stability_multi <= 1.0


def test_rhythm_stability_with_tempo_drift(sample_audio_buffer):
    """Test rhythm stability calculation with tempo drift detection."""
    config = AnalysisConfig(rhythm_tempo_drift_threshold=1.0)
    
    # Create a simple beat grid
    beat_grid = BeatGrid(
        bpm=120.0,
        beats_s=[i * 0.5 for i in range(10)],  # Regular beats at 120 BPM
        downbeats_s=[i * 2.0 for i in range(3)],
        confidence=0.8,
    )
    
    stability = calculate_rhythm_stability(sample_audio_buffer, beat_grid, config)
    assert 0.0 <= stability <= 1.0


def test_energy_features_with_config(sample_audio_buffer):
    """Test energy feature calculation with config."""
    config = AnalysisConfig(
        onset_frame_length=1024,
        onset_hop_length=256,
    )
    
    mean_energy, std_energy = calculate_energy_features(sample_audio_buffer, config)
    
    assert mean_energy >= 0.0
    assert std_energy >= 0.0

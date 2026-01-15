"""Vocal presence detection functionality."""
import numpy as np
import librosa

from mixinto.utils.types import AudioBuffer


def detect_vocal_presence(buffer: AudioBuffer) -> float:
    """
    Detect vocal presence in an audio buffer.
    
    Uses spectral features (centroid, rolloff) as heuristics.
    More sophisticated methods could be added later.
    
    Args:
        buffer: AudioBuffer to analyze
    
    Returns:
        Vocal presence score in [0, 1] where 1 indicates strong vocal presence
    """
    mono_buffer = buffer.to_mono()
    samples = mono_buffer.samples.flatten()
    
    # Compute spectral features
    # Spectral centroid: higher values often indicate vocals
    spectral_centroid = librosa.feature.spectral_centroid(
        y=samples,
        sr=buffer.sample_rate,
    )[0]
    
    # Spectral rolloff: frequency below which a percentage of energy is contained
    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=samples,
        sr=buffer.sample_rate,
    )[0]
    
    # Mean values
    mean_centroid = np.mean(spectral_centroid)
    mean_rolloff = np.mean(spectral_rolloff)
    
    # Heuristic: vocals typically have:
    # - Higher spectral centroid (more energy in mid-high frequencies)
    # - Moderate to high spectral rolloff
    
    # Normalize centroid (typical range for music: 1000-4000 Hz)
    # Higher centroid = more likely to have vocals
    centroid_score = min(1.0, (mean_centroid - 1000) / 3000)
    centroid_score = max(0.0, centroid_score)
    
    # Normalize rolloff (typical range: 2000-8000 Hz)
    # Moderate to high rolloff = more likely vocals
    rolloff_score = min(1.0, (mean_rolloff - 2000) / 6000)
    rolloff_score = max(0.0, rolloff_score)
    
    # Combine scores (weighted average)
    # Centroid is more indicative of vocals
    vocal_score = 0.7 * centroid_score + 0.3 * rolloff_score
    
    return float(np.clip(vocal_score, 0.0, 1.0))

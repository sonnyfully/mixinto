"""Tempo (BPM) detection functionality."""
import warnings

import numpy as np
import librosa

from mixinto.utils.types import AudioBuffer, AnalysisConfig


def detect_tempo(
    buffer: AudioBuffer,
    confidence_threshold: float = 0.3,
    config: AnalysisConfig | None = None,
) -> tuple[float, float]:
    """
    Detect the tempo (BPM) of an audio buffer.
    
    Args:
        buffer: AudioBuffer to analyze
        confidence_threshold: Minimum confidence for valid detection
        config: AnalysisConfig with configurable parameters (uses defaults if None)
    
    Returns:
        Tuple of (bpm, confidence) where confidence is in [0, 1]
    
    Raises:
        ValueError: If tempo cannot be detected
    """
    if config is None:
        config = AnalysisConfig()
    
    # Convert to mono if needed
    mono_buffer = buffer.to_mono()
    samples = mono_buffer.samples
    
    # Flatten if needed
    if samples.ndim > 1:
        samples = samples.flatten()
    
    # Calculate onset strength with configurable parameters
    onset_strength = librosa.onset.onset_strength(
        y=samples,
        sr=buffer.sample_rate,
        hop_length=config.onset_hop_length,
        frame_length=config.onset_frame_length,
    )
    
    # Use librosa's tempo estimation with onset strength
    # Check for new API first (librosa >= 0.10.0), fall back to old API
    if hasattr(librosa.feature, 'rhythm') and hasattr(librosa.feature.rhythm, 'tempo'):
        # New API (librosa >= 0.10.0)
        tempo_est = librosa.feature.rhythm.tempo(
            onset_envelope=onset_strength,
            sr=buffer.sample_rate,
            aggregate=np.mean,
        )
    else:
        # Old API (librosa < 0.10.0) - suppress deprecation warning
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            tempo_est = librosa.beat.tempo(
                onset_envelope=onset_strength,
                sr=buffer.sample_rate,
                aggregate=np.mean,
            )
    
    if len(tempo_est) == 0:
        raise ValueError("Could not detect tempo")
    
    tempo = float(tempo_est[0])
    
    # Calculate confidence based on multiple factors
    if len(onset_strength) > 0:
        mean_strength = np.mean(onset_strength)
        std_strength = np.std(onset_strength)
        
        # Factor 1: Onset strength consistency (lower CV = higher confidence)
        if std_strength > 0 and mean_strength > 0:
            cv = std_strength / (mean_strength + 0.001)
            consistency_score = 1.0 / (1.0 + cv * 2.0)
        else:
            consistency_score = 0.5
        
        # Factor 2: Overall onset strength (stronger onsets = more reliable)
        # Normalize by typical range (0-1, but can be higher)
        strength_score = min(1.0, mean_strength / 0.5)  # 0.5 is a reasonable threshold
        
        # Factor 3: Tempo reasonableness (tempos in common range are more likely correct)
        tempo_score = 1.0
        if tempo < 80 or tempo > 180:
            # Slightly penalize very slow or very fast tempos
            if tempo < 80:
                tempo_score = 0.8
            elif tempo > 180:
                tempo_score = 0.8
        
        # Combine factors (weighted average)
        confidence = 0.4 * consistency_score + 0.4 * strength_score + 0.2 * tempo_score
        confidence = float(np.clip(confidence, 0.0, 1.0))
    else:
        confidence = 0.3  # Minimum confidence if we have some data
    
    # Validate tempo is within configured range and apply octave correction
    if config.use_tempo_hints:
        # Check if tempo is outside expected range (might be doubled/halved)
        if tempo < config.tempo_min:
            # Try doubling (might be half-tempo)
            tempo_candidate = tempo * 2
            if config.tempo_min <= tempo_candidate <= config.tempo_max:
                tempo = tempo_candidate
        elif tempo > config.tempo_max:
            # Try halving (might be double-tempo)
            tempo_candidate = tempo / 2
            if config.tempo_min <= tempo_candidate <= config.tempo_max:
                tempo = tempo_candidate
        
        # Final range check - clamp to valid range if still outside
        if tempo < config.tempo_min:
            tempo = config.tempo_min
        elif tempo > config.tempo_max:
            tempo = config.tempo_max
    else:
        # Legacy behavior: simple octave correction
        if tempo < 60 or tempo > 200:
            if tempo < 60:
                tempo *= 2
            elif tempo > 200:
                tempo /= 2
    
    # Final confidence check (use config threshold if provided, otherwise use parameter)
    threshold = config.tempo_confidence_threshold if config.use_tempo_hints else confidence_threshold
    if confidence < threshold:
        raise ValueError(
            f"Tempo detection confidence too low: {confidence:.2f} "
            f"(threshold: {threshold:.2f})"
        )
    
    return float(tempo), float(confidence)

"""Stability feature extraction for intro analysis."""
import numpy as np
import librosa

from mixinto.utils.types import AudioBuffer, AnalysisConfig


def calculate_spectral_stability(
    buffer: AudioBuffer,
    config: AnalysisConfig | None = None,
) -> float:
    """
    Calculate spectral stability of an audio buffer.
    
    Higher values indicate more stable spectral content (good for mixing).
    
    Args:
        buffer: AudioBuffer to analyze
        config: AnalysisConfig with configurable parameters (uses defaults if None)
    
    Returns:
        Stability score in [0, 1] where 1 is most stable
    """
    if config is None:
        config = AnalysisConfig()
    
    mono_buffer = buffer.to_mono()
    samples = mono_buffer.samples.flatten()
    
    # Use consistent hop_length from config (ensure it's valid)
    hop_length = max(64, config.onset_hop_length)  # Ensure minimum hop_length
    n_fft = config.onset_frame_length
    
    stability_scores = []
    
    if "chroma" in config.spectral_features:
        chroma = librosa.feature.chroma_stft(
            y=samples,
            sr=buffer.sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
        )
        # Calculate stability as inverse of variance across time
        chroma_std = np.std(chroma, axis=1)
        mean_std = np.mean(chroma_std)
        chroma_stability = 1.0 / (1.0 + mean_std * 10.0)
        stability_scores.append(chroma_stability)
    
    if "mfcc" in config.spectral_features:
        # mfcc accepts n_fft and hop_length via **kwargs to melspectrogram
        # But we need to be careful - only pass valid melspectrogram parameters
        mfcc = librosa.feature.mfcc(
            y=samples,
            sr=buffer.sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mfcc=13,
            # Explicitly don't pass frame_length - mfcc/melspectrogram use n_fft
        )
        # Use first few MFCCs (more stable) for stability calculation
        mfcc_stable = mfcc[:6, :]  # First 6 MFCCs
        mfcc_std = np.std(mfcc_stable, axis=1)
        mean_std = np.mean(mfcc_std)
        mfcc_stability = 1.0 / (1.0 + mean_std * 5.0)
        stability_scores.append(mfcc_stability)
    
    if "tonnetz" in config.spectral_features:
        # tonnetz uses chroma_cqt internally, which doesn't use n_fft
        # It accepts hop_length and other CQT parameters via **kwargs
        tonnetz = librosa.feature.tonnetz(
            y=samples,
            sr=buffer.sample_rate,
            hop_length=hop_length,
        )
        tonnetz_std = np.std(tonnetz, axis=1)
        mean_std = np.mean(tonnetz_std)
        tonnetz_stability = 1.0 / (1.0 + mean_std * 8.0)
        stability_scores.append(tonnetz_stability)
    
    # Apply time-weighted analysis if enabled
    if config.time_weight_decay < 1.0 and len(stability_scores) > 0:
        # For now, use simple average (time-weighting would require frame-by-frame analysis)
        # This is a placeholder for future enhancement
        pass
    
    # Combine stability scores from different features
    if len(stability_scores) > 0:
        stability = float(np.mean(stability_scores))
    else:
        # Fallback to chroma if no features selected
        chroma = librosa.feature.chroma_stft(
            y=samples,
            sr=buffer.sample_rate,
        )
        chroma_std = np.std(chroma, axis=1)
        mean_std = np.mean(chroma_std)
        stability = 1.0 / (1.0 + mean_std * 10.0)
    
    return float(np.clip(stability, 0.0, 1.0))


def calculate_rhythm_stability(
    buffer: AudioBuffer,
    beat_grid,
    config: AnalysisConfig | None = None,
) -> float:
    """
    Calculate rhythm stability (consistency of beat intervals).
    
    Args:
        buffer: AudioBuffer to analyze
        beat_grid: BeatGrid with beat information
        config: AnalysisConfig with configurable parameters (uses defaults if None)
    
    Returns:
        Stability score in [0, 1] where 1 is most stable
    """
    if config is None:
        config = AnalysisConfig()
    
    if len(beat_grid.beats_s) < 2:
        return 0.0
    
    # Calculate inter-beat intervals
    intervals = np.diff(beat_grid.beats_s)
    
    if len(intervals) == 0:
        return 0.0
    
    # Expected interval based on BPM
    expected_interval = beat_grid.seconds_per_beat()
    
    # Calculate coefficient of variation (std / mean)
    # Lower CV = higher stability
    mean_interval = np.mean(intervals)
    std_interval = np.std(intervals)
    
    if mean_interval == 0:
        return 0.0
    
    cv = std_interval / mean_interval
    
    # Check for tempo drift over time
    # Split intervals into segments and check for drift
    if len(intervals) >= 8:
        segment_size = len(intervals) // 4
        segment_means = []
        for i in range(0, len(intervals), segment_size):
            segment = intervals[i:i+segment_size]
            if len(segment) > 0:
                segment_means.append(np.mean(segment))
        
        if len(segment_means) >= 2:
            # Calculate drift as max difference between segments
            max_drift = max(segment_means) - min(segment_means)
            drift_bpm = (max_drift / expected_interval) * beat_grid.bpm
            
            # Penalize if drift exceeds threshold
            if drift_bpm > config.rhythm_tempo_drift_threshold:
                # Reduce stability score based on drift
                drift_penalty = min(1.0, drift_bpm / (config.rhythm_tempo_drift_threshold * 2))
                cv *= (1.0 + drift_penalty)
    
    # Convert to stability score (inverse relationship)
    # Lower CV = higher stability
    stability = 1.0 / (1.0 + cv * 5.0)
    
    return float(np.clip(stability, 0.0, 1.0))


def calculate_energy_features(
    buffer: AudioBuffer,
    config: AnalysisConfig | None = None,
) -> tuple[float, float]:
    """
    Calculate energy statistics (mean and std).
    
    Args:
        buffer: AudioBuffer to analyze
        config: AnalysisConfig with configurable parameters (uses defaults if None)
    
    Returns:
        Tuple of (mean_energy, std_energy)
    """
    if config is None:
        config = AnalysisConfig()
    
    mono_buffer = buffer.to_mono()
    samples = mono_buffer.samples.flatten()
    
    # Calculate RMS energy with configurable frame parameters
    rms = librosa.feature.rms(
        y=samples,
        frame_length=config.onset_frame_length,
        hop_length=config.onset_hop_length,
    )[0]
    
    mean_energy = float(np.mean(rms))
    std_energy = float(np.std(rms))
    
    # Detect energy ramps (builds) - significant increases in energy
    if len(rms) > 10:
        # Use rolling window to detect significant energy increases
        window_size = min(10, len(rms) // 4)
        energy_changes = []
        for i in range(window_size, len(rms)):
            prev_mean = np.mean(rms[i-window_size:i])
            curr_mean = np.mean(rms[i-window_size//2:i+window_size//2])
            if prev_mean > 0:
                change = (curr_mean - prev_mean) / prev_mean
                energy_changes.append(abs(change))
        
        # If there are significant energy ramps, increase variance estimate
        if len(energy_changes) > 0:
            max_change = max(energy_changes)
            if max_change > config.energy_variance_threshold:
                # Energy is changing significantly - adjust std to reflect this
                std_energy *= (1.0 + max_change)
    
    return (mean_energy, std_energy)

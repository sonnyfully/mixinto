"""Per-bar/per-window feature timeline extraction."""
import numpy as np

from mixinto.features.stability import (
    calculate_energy_features,
    calculate_rhythm_stability,
    calculate_spectral_stability,
)
from mixinto.features.vocals import detect_vocal_presence
from mixinto.utils.types import AudioBuffer, BeatGrid, FeatureTimeline, AnalysisConfig


def extract_feature_timeline(
    buffer: AudioBuffer,
    beat_grid: BeatGrid,
    config: AnalysisConfig | None = None,
) -> FeatureTimeline:
    """
    Extract per-bar feature timeline across the entire track.
    
    This computes features on bar-aligned windows, creating a timeline
    of suitability metrics that can be used to identify the best segments.
    
    Args:
        buffer: AudioBuffer to analyze
        beat_grid: BeatGrid with beat and downbeat information
        config: AnalysisConfig with parameters (uses defaults if None)
    
    Returns:
        FeatureTimeline with per-bar features
    """
    if config is None:
        from mixinto.utils.types import AnalysisConfig
        config = AnalysisConfig()
    
    # Get bar boundaries from downbeats
    if len(beat_grid.downbeats_s) < 2:
        raise ValueError("Need at least 2 downbeats to create feature timeline")
    
    bar_start_times_s = beat_grid.downbeats_s.copy()
    # Add end of track as final bar boundary if needed
    track_end_s = buffer.length_s()
    if bar_start_times_s[-1] < track_end_s:
        bar_start_times_s.append(track_end_s)
    
    bar_count = len(bar_start_times_s) - 1
    
    # Initialize feature arrays
    tempo_confidence_list = []
    rhythm_stability_list = []
    spectral_stability_list = []
    energy_consistency_list = []
    vocal_presence_list = []
    loop_seam_score_list = []
    
    # Compute features for each bar (or bar-aligned window)
    base_resolution = config.segment_base_resolution_bars
    scoring_window = config.segment_scoring_window_bars
    
    for bar_idx in range(bar_count):
        # Determine window boundaries
        # Use rolling window of 'scoring_window' bars centered around current bar
        window_start_bar = max(0, bar_idx - scoring_window // 2)
        window_end_bar = min(bar_count, bar_idx + scoring_window // 2 + 1)
        
        start_s = bar_start_times_s[window_start_bar]
        end_s = bar_start_times_s[window_end_bar]
        
        # Extract window buffer
        window_buffer = buffer.trim(start_s, end_s)
        
        # Create beat grid for this window
        window_beats = [b for b in beat_grid.beats_s if start_s <= b <= end_s]
        window_downbeats = [b for b in beat_grid.downbeats_s if start_s <= b <= end_s]
        
        if len(window_beats) < 2:
            # Not enough beats - assign low scores
            tempo_confidence_list.append(beat_grid.confidence * 0.5)
            rhythm_stability_list.append(0.0)
            spectral_stability_list.append(0.0)
            energy_consistency_list.append(0.0)
            vocal_presence_list.append(0.5)  # Unknown
            loop_seam_score_list.append(0.0)
            continue
        
        from mixinto.utils.types import BeatGrid
        window_beat_grid = BeatGrid(
            bpm=beat_grid.bpm,
            beats_s=window_beats,
            downbeats_s=window_downbeats,
            confidence=beat_grid.confidence,
        )
        
        # Extract features for this window
        # Tempo confidence (use global confidence, could be refined per-window)
        tempo_confidence_list.append(beat_grid.confidence)
        
        # Rhythm stability
        rhythm_stability = calculate_rhythm_stability(window_buffer, window_beat_grid, config)
        rhythm_stability_list.append(rhythm_stability)
        
        # Spectral stability
        spectral_stability = calculate_spectral_stability(window_buffer, config)
        spectral_stability_list.append(spectral_stability)
        
        # Energy consistency
        energy_mean, energy_std = calculate_energy_features(window_buffer, config)
        if energy_mean > 0:
            energy_cv = energy_std / energy_mean
            energy_consistency = 1.0 / (1.0 + energy_cv * 2.0)
        else:
            energy_consistency = 0.0
        energy_consistency_list.append(energy_consistency)
        
        # Vocal presence
        vocal_presence = detect_vocal_presence(window_buffer)
        vocal_presence_list.append(vocal_presence)
        
        # Loop seam score (loopability)
        loop_seam_score = calculate_loop_seam_score(window_buffer, window_beat_grid, config)
        loop_seam_score_list.append(loop_seam_score)
    
    return FeatureTimeline(
        bar_count=bar_count,
        bar_start_times_s=bar_start_times_s,
        tempo_confidence=tempo_confidence_list,
        rhythm_stability=rhythm_stability_list,
        spectral_stability=spectral_stability_list,
        energy_consistency=energy_consistency_list,
        vocal_presence=vocal_presence_list,
        loop_seam_score=loop_seam_score_list,
    )


def calculate_loop_seam_score(
    buffer: AudioBuffer,
    beat_grid: BeatGrid,
    config: AnalysisConfig | None = None,
) -> float:
    """
    Calculate loopability score for a segment.
    
    This measures how well the segment can be looped by checking:
    1. How similar the start and end are (spectral/timbral)
    2. How well the segment aligns to bar boundaries
    3. Energy consistency at boundaries
    
    Args:
        buffer: AudioBuffer to analyze
        beat_grid: BeatGrid for alignment
        config: AnalysisConfig with parameters
    
    Returns:
        Loopability score in [0, 1] where 1 is most loopable
    """
    if config is None:
        from mixinto.utils.types import AnalysisConfig
        config = AnalysisConfig()
    
    duration_s = buffer.length_s()
    if duration_s < 0.1:  # Too short
        return 0.0
    
    # Check 1: Bar alignment
    # Segment should start and end on or near bar boundaries
    # Since buffer is already trimmed, check if boundaries align to downbeats
    segment_start_s = 0.0
    segment_end_s = duration_s
    
    # Find alignment errors relative to downbeats in the beat grid
    # The beat_grid passed here should have downbeats relative to the segment
    if len(beat_grid.downbeats_s) >= 2:
        # Check if start aligns with first downbeat and end aligns with last downbeat
        first_downbeat = beat_grid.downbeats_s[0]
        last_downbeat = beat_grid.downbeats_s[-1]
        
        start_alignment_error = abs(segment_start_s - first_downbeat)
        end_alignment_error = abs(segment_end_s - last_downbeat)
    else:
        # Fallback: check against all downbeats
        start_alignment_error = min([abs(segment_start_s - db) for db in beat_grid.downbeats_s], default=1.0)
        end_alignment_error = min([abs(segment_end_s - db) for db in beat_grid.downbeats_s], default=1.0)
    
    # Convert to bar-aligned error (normalize by bar length)
    seconds_per_bar = beat_grid.seconds_per_beat() * 4
    start_bar_error = start_alignment_error / max(seconds_per_bar, 0.001)
    end_bar_error = end_alignment_error / max(seconds_per_bar, 0.001)
    
    # Alignment score (penalize misalignment)
    # Perfect alignment gets 1.0, small errors are acceptable
    alignment_score = 1.0 / (1.0 + (start_bar_error + end_bar_error) * 10.0)
    
    # Check 2: Start-end similarity (spectral)
    # Compare spectral features at start and end of segment
    window_duration_s = min(0.5, duration_s * 0.1)  # 10% of segment or 0.5s, whichever is smaller
    
    start_window = buffer.trim(0.0, window_duration_s)
    end_window = buffer.trim(duration_s - window_duration_s, duration_s)
    
    # Calculate spectral features for start and end
    import librosa
    
    mono_start = start_window.to_mono()
    mono_end = end_window.to_mono()
    samples_start = mono_start.samples.flatten()
    samples_end = mono_end.samples.flatten()
    
    # Use chroma for timbral similarity
    chroma_start = librosa.feature.chroma_stft(
        y=samples_start,
        sr=buffer.sample_rate,
        hop_length=config.onset_hop_length,
        n_fft=config.onset_frame_length,
    )
    chroma_end = librosa.feature.chroma_stft(
        y=samples_end,
        sr=buffer.sample_rate,
        hop_length=config.onset_hop_length,
        n_fft=config.onset_frame_length,
    )
    
    # Average across time to get single vector per window
    chroma_start_mean = np.mean(chroma_start, axis=1)
    chroma_end_mean = np.mean(chroma_end, axis=1)
    
    # Cosine similarity
    dot_product = np.dot(chroma_start_mean, chroma_end_mean)
    norm_start = np.linalg.norm(chroma_start_mean)
    norm_end = np.linalg.norm(chroma_end_mean)
    
    if norm_start > 0 and norm_end > 0:
        similarity = dot_product / (norm_start * norm_end)
    else:
        similarity = 0.0
    
    # Check 3: Energy consistency at boundaries
    rms_start = librosa.feature.rms(
        y=samples_start,
        frame_length=config.onset_frame_length,
        hop_length=config.onset_hop_length,
    )[0]
    rms_end = librosa.feature.rms(
        y=samples_end,
        frame_length=config.onset_frame_length,
        hop_length=config.onset_hop_length,
    )[0]
    
    energy_start = np.mean(rms_start) if len(rms_start) > 0 else 0.0
    energy_end = np.mean(rms_end) if len(rms_end) > 0 else 0.0
    
    if energy_start > 0 and energy_end > 0:
        energy_ratio = min(energy_start, energy_end) / max(energy_start, energy_end)
    else:
        energy_ratio = 0.0
    
    # Combine scores (weighted average)
    # Alignment is most important, then similarity, then energy
    loopability = (
        0.5 * alignment_score +
        0.3 * similarity +
        0.2 * energy_ratio
    )
    
    return float(np.clip(loopability, 0.0, 1.0))

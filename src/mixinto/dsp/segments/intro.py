"""Intro segment detection functionality."""
import numpy as np

from mixinto.features.stability import (
    calculate_energy_features,
    calculate_rhythm_stability,
    calculate_spectral_stability,
)
from mixinto.utils.types import AudioBuffer, BeatGrid, AnalysisConfig


def detect_intro_window(
    buffer: AudioBuffer,
    beat_grid: BeatGrid,
    min_bars: int = 8,
    max_bars: int = 16,
    config: AnalysisConfig | None = None,
) -> tuple[float, float]:
    """
    Detect a stable intro window suitable for extension.
    
    Analyzes multiple candidate windows and selects the most stable one based on
    spectral stability, rhythm stability, and energy consistency.
    
    Args:
        buffer: AudioBuffer to analyze
        beat_grid: BeatGrid with beat information
        min_bars: Minimum number of bars for intro window
        max_bars: Maximum number of bars for intro window
        config: AnalysisConfig with configurable parameters (uses defaults if None)
    
    Returns:
        Tuple of (start_s, end_s) defining the intro window
    
    Raises:
        ValueError: If intro window cannot be determined
    """
    if config is None:
        config = AnalysisConfig()
    
    # Use config values if provided, otherwise use function parameters
    min_bars = max(min_bars, config.intro_min_bars)
    max_bars = min(max_bars, config.intro_max_bars)
    
    if len(beat_grid.downbeats_s) < min_bars:
        raise ValueError(
            f"Insufficient downbeats for intro detection: "
            f"found {len(beat_grid.downbeats_s)}, need at least {min_bars}"
        )
    
    # Start at the beginning (first downbeat or start of audio)
    start_s = 0.0
    
    # Get candidate bar counts to evaluate
    candidate_bars = [
        bars for bars in config.intro_candidate_bars
        if min_bars <= bars <= max_bars and bars <= len(beat_grid.downbeats_s)
    ]
    
    # If no candidates in range, use min/max
    if not candidate_bars:
        candidate_bars = [min_bars, max_bars]
    
    # Evaluate each candidate window
    window_scores = []
    
    for bars in candidate_bars:
        # Find the end of this candidate window
        if bars <= len(beat_grid.downbeats_s):
            end_s = beat_grid.downbeats_s[bars - 1]
        else:
            # Fall back to using beats if we don't have enough downbeats
            target_beats = bars * 4
            if target_beats <= len(beat_grid.beats_s):
                end_s = beat_grid.beats_s[target_beats - 1]
            else:
                end_s = beat_grid.beats_s[-1] if len(beat_grid.beats_s) > 0 else buffer.length_s()
        
        # Ensure we don't exceed buffer length
        end_s = min(end_s, buffer.length_s())
        
        if end_s <= start_s:
            continue
        
        # Extract candidate window
        candidate_buffer = buffer.trim(start_s, end_s)
        
        # Calculate stability metrics for this window
        # Spectral stability
        spectral_stability = calculate_spectral_stability(candidate_buffer, config)
        
        # Rhythm stability (need to create a beat grid for just this window)
        # For simplicity, use the full beat grid but filter beats within window
        window_beats = [b for b in beat_grid.beats_s if start_s <= b <= end_s]
        window_downbeats = [b for b in beat_grid.downbeats_s if start_s <= b <= end_s]
        
        if len(window_beats) >= 2:
            # Create a temporary beat grid for this window
            from mixinto.utils.types import BeatGrid
            window_beat_grid = BeatGrid(
                bpm=beat_grid.bpm,
                beats_s=window_beats,
                downbeats_s=window_downbeats,
                confidence=beat_grid.confidence,
            )
            rhythm_stability = calculate_rhythm_stability(candidate_buffer, window_beat_grid, config)
        else:
            rhythm_stability = 0.0
        
        # Energy consistency
        energy_mean, energy_std = calculate_energy_features(candidate_buffer, config)
        if energy_mean > 0:
            energy_consistency = 1.0 / (1.0 + (energy_std / energy_mean) * 2.0)
        else:
            energy_consistency = 0.0
        
        # Change point detection - check for significant changes in energy
        # Simple approach: check if energy variance exceeds threshold
        change_penalty = 0.0
        if energy_std / (energy_mean + 0.001) > config.change_point_threshold:
            change_penalty = 0.2  # Penalize windows with high energy variance
        
        # Combine stability scores with weights
        weights = config.intro_stability_weights
        combined_score = (
            weights[0] * spectral_stability +
            weights[1] * rhythm_stability +
            weights[2] * energy_consistency
        ) - change_penalty
        
        window_scores.append((bars, end_s, combined_score))
    
    if not window_scores:
        # Fallback to simple approach if no candidates scored
        target_bars = min(max_bars, len(beat_grid.downbeats_s))
        target_bars = max(min_bars, target_bars)
        if target_bars <= len(beat_grid.downbeats_s):
            end_s = beat_grid.downbeats_s[target_bars - 1]
        else:
            end_s = beat_grid.beats_s[min(target_bars * 4, len(beat_grid.beats_s) - 1)]
        end_s = min(end_s, buffer.length_s())
    else:
        # Select window with highest stability score
        window_scores.sort(key=lambda x: x[2], reverse=True)
        best_bars, end_s, best_score = window_scores[0]
    
    # Validate window
    if end_s <= start_s:
        raise ValueError(f"Invalid intro window: start={start_s}, end={end_s}")
    
    return (start_s, end_s)

"""Candidate segment generation across the track."""
from mixinto.utils.types import AudioBuffer, BeatGrid, SegmentCandidate, FeatureTimeline, AnalysisConfig


def generate_segment_candidates(
    buffer: AudioBuffer,
    beat_grid: BeatGrid,
    feature_timeline: FeatureTimeline,
    config: AnalysisConfig | None = None,
) -> list[SegmentCandidate]:
    """
    Generate candidate segments across the track.
    
    Creates segments of various bar lengths at different positions,
    respecting search region constraints.
    
    Args:
        buffer: AudioBuffer to analyze
        beat_grid: BeatGrid with beat information
        feature_timeline: FeatureTimeline with per-bar features
        config: AnalysisConfig with parameters (uses defaults if None)
    
    Returns:
        List of SegmentCandidate objects
    """
    if config is None:
        from mixinto.utils.types import AnalysisConfig
        config = AnalysisConfig()
    
    candidates = []
    bar_start_times_s = feature_timeline.bar_start_times_s
    bar_count = feature_timeline.bar_count
    
    # Determine search region
    search_region = config.segment_search_region
    max_bar = bar_count
    
    if search_region == "first_N_bars":
        max_bar = min(bar_count, config.segment_search_first_n_bars)
    elif search_region == "pre_vocal_only":
        # Find first bar with significant vocal presence
        vocal_threshold = 0.5
        for bar_idx in range(bar_count):
            if feature_timeline.vocal_presence[bar_idx] > vocal_threshold:
                max_bar = bar_idx
                break
    
    # Generate candidates for each target length
    for target_length_bars in config.segment_candidate_lengths:
        if target_length_bars > max_bar:
            continue  # Skip if segment is longer than available bars
        
        # Generate candidates with hop
        hop_bars = config.segment_hop_bars
        for start_bar in range(0, max_bar - target_length_bars + 1, hop_bars):
            end_bar = start_bar + target_length_bars
            
            # Get time boundaries
            start_s = bar_start_times_s[start_bar]
            end_s = bar_start_times_s[end_bar] if end_bar < len(bar_start_times_s) else buffer.length_s()
            
            # Ensure we don't exceed buffer length
            end_s = min(end_s, buffer.length_s())
            
            if end_s <= start_s:
                continue
            
            candidate = SegmentCandidate(
                start_bar=start_bar,
                end_bar=end_bar,
                start_s=start_s,
                end_s=end_s,
                bar_count=target_length_bars,
            )
            candidates.append(candidate)
    
    return candidates


def filter_candidates_by_vocals(
    candidates: list[SegmentCandidate],
    feature_timeline: FeatureTimeline,
    max_vocal_presence: float = 0.5,
) -> list[SegmentCandidate]:
    """
    Filter out candidates with too much vocal presence.
    
    Args:
        candidates: List of SegmentCandidate objects
        feature_timeline: FeatureTimeline with vocal presence data
        max_vocal_presence: Maximum allowed vocal presence (0-1)
    
    Returns:
        Filtered list of candidates
    """
    filtered = []
    
    for candidate in candidates:
        # Calculate average vocal presence in segment
        vocal_scores = feature_timeline.vocal_presence[candidate.start_bar:candidate.end_bar]
        if len(vocal_scores) == 0:
            continue
        
        avg_vocal = sum(vocal_scores) / len(vocal_scores)
        max_vocal = max(vocal_scores) if vocal_scores else 0.0
        
        # Use max (p90-like) to penalize spikes
        if max_vocal <= max_vocal_presence:
            filtered.append(candidate)
    
    return filtered

"""Segment scoring with robust statistics."""
import numpy as np

from mixinto.utils.types import (
    SegmentCandidate,
    SegmentScore,
    FeatureTimeline,
    AnalysisConfig,
)


def score_segment(
    candidate: SegmentCandidate,
    feature_timeline: FeatureTimeline,
    beat_grid,
    config: AnalysisConfig | None = None,
) -> SegmentScore:
    """
    Score a candidate segment using robust statistics.
    
    Uses mean + percentile-based aggregation to avoid "one bad bar"
    from ruining an otherwise good segment.
    
    Args:
        candidate: SegmentCandidate to score
        feature_timeline: FeatureTimeline with per-bar features
        beat_grid: BeatGrid for tempo confidence
        config: AnalysisConfig with parameters (uses defaults if None)
    
    Returns:
        SegmentScore with detailed scoring breakdown
    """
    if config is None:
        from mixinto.utils.types import AnalysisConfig
        config = AnalysisConfig()
    
    # Extract feature arrays for this segment
    start_idx = candidate.start_bar
    end_idx = candidate.end_bar
    
    # Get feature slices
    tempo_conf = feature_timeline.tempo_confidence[start_idx:end_idx]
    rhythm_stab = feature_timeline.rhythm_stability[start_idx:end_idx]
    spectral_stab = feature_timeline.spectral_stability[start_idx:end_idx]
    energy_cons = feature_timeline.energy_consistency[start_idx:end_idx]
    vocal_pres = feature_timeline.vocal_presence[start_idx:end_idx]
    loop_seam = feature_timeline.loop_seam_score[start_idx:end_idx]
    
    # Robust aggregation per feature
    # Use mean + p10 for "must be consistently good" features
    # Use p90 for "penalize spikes" features
    
    def robust_mean(values: list[float], p10_weight: float = 0.3) -> float:
        """Mean weighted with 10th percentile to avoid bad outliers."""
        if len(values) == 0:
            return 0.0
        arr = np.array(values)
        mean_val = np.mean(arr)
        p10_val = np.percentile(arr, 10)
        return float((1.0 - p10_weight) * mean_val + p10_weight * p10_val)
    
    def robust_max(values: list[float]) -> float:
        """90th percentile to catch spikes."""
        if len(values) == 0:
            return 0.0
        return float(np.percentile(values, 90))
    
    # Aggregate features
    loopability = robust_mean(loop_seam, p10_weight=0.3) if loop_seam else 0.0
    tempo_confidence = robust_mean(tempo_conf, p10_weight=0.2) if tempo_conf else beat_grid.confidence
    rhythm_stability = robust_mean(rhythm_stab, p10_weight=0.3) if rhythm_stab else 0.0
    spectral_stability = robust_mean(spectral_stab, p10_weight=0.3) if spectral_stab else 0.0
    energy_consistency = robust_mean(energy_cons, p10_weight=0.2) if energy_cons else 0.0
    
    # Vocal penalty: use p90 to penalize spikes
    vocal_max = robust_max(vocal_pres) if vocal_pres else 0.0
    vocal_penalty = 1.0 - vocal_max  # Invert: higher vocal = lower score
    
    # Calculate energy stats for component breakdown
    energy_mean = np.mean(energy_cons) if energy_cons else 0.0
    energy_std = np.std(energy_cons) if energy_cons else 0.0
    
    # Apply weights from config
    weights = config.segment_weights
    
    final_score = (
        weights.get("loopability", 0.25) * loopability +
        weights.get("tempo_confidence", 0.20) * tempo_confidence +
        weights.get("rhythm_stability", 0.20) * rhythm_stability +
        weights.get("spectral_stability", 0.20) * spectral_stability +
        weights.get("energy_consistency", 0.10) * energy_consistency +
        weights.get("vocal_penalty", 0.05) * vocal_penalty
    )
    
    # Clip to [0, 1]
    final_score = float(np.clip(final_score, 0.0, 1.0))
    
    # Generate flags
    flags = []
    if tempo_confidence < 0.5:
        flags.append("low_beat_confidence")
    if rhythm_stability < 0.5:
        flags.append("low_rhythm_stability")
    if spectral_stability < 0.5:
        flags.append("low_spectral_stability")
    if vocal_max > 0.5:
        flags.append("high_vocal_presence")
    if energy_consistency < 0.5:
        flags.append("low_energy_consistency")
    if loopability < config.loopability_min_score:
        flags.append("low_loopability")
    
    # Component breakdown for debugging
    component_breakdown = {
        "loopability": loopability,
        "tempo_confidence": tempo_confidence,
        "rhythm_stability": rhythm_stability,
        "spectral_stability": spectral_stability,
        "energy_consistency": energy_consistency,
        "vocal_penalty": vocal_penalty,
        "vocal_max": vocal_max,
        "energy_mean": energy_mean,
        "energy_std": energy_std,
    }
    
    return SegmentScore(
        segment=candidate,
        loopability=loopability,
        tempo_confidence=tempo_confidence,
        rhythm_stability=rhythm_stability,
        spectral_stability=spectral_stability,
        energy_consistency=energy_consistency,
        vocal_penalty=vocal_penalty,
        final_score=final_score,
        flags=flags,
        component_breakdown=component_breakdown,
    )


def select_top_segments(
    segment_scores: list[SegmentScore],
    top_k: int = 5,
    viable_threshold: float = 0.5,
) -> tuple[list[SegmentScore], int]:
    """
    Select top-K segments and count viable ones.
    
    Args:
        segment_scores: List of scored segments
        top_k: Number of top segments to return
        viable_threshold: Minimum score for a segment to be considered viable
    
    Returns:
        Tuple of (top_segments, viable_count)
    """
    # Sort by score (descending)
    sorted_scores = sorted(segment_scores, key=lambda s: s.final_score, reverse=True)
    
    # Count viable segments
    viable_count = sum(1 for s in sorted_scores if s.final_score >= viable_threshold)
    
    # Take top-K
    top_segments = sorted_scores[:top_k]
    
    return top_segments, viable_count


def calculate_track_extendability(
    top_segments: list[SegmentScore],
    viable_count: int,
    total_candidates: int,
) -> tuple[float, float]:
    """
    Calculate track-level extendability metrics.
    
    Args:
        top_segments: Top-K segment scores
        viable_count: Number of viable segments
        total_candidates: Total number of candidates evaluated
    
    Returns:
        Tuple of (track_extendability, confidence)
    """
    if len(top_segments) == 0:
        return 0.0, 0.0
    
    # Track extendability = score of best segment
    track_extendability = top_segments[0].final_score
    
    # Confidence based on:
    # 1. Margin between top-1 and top-2 (if available)
    # 2. Coverage (viable_count / total_candidates)
    # 3. Consistency of top segments
    
    margin = 0.0
    if len(top_segments) >= 2:
        margin = top_segments[0].final_score - top_segments[1].final_score
        margin = min(1.0, margin * 2.0)  # Scale margin
    
    coverage = viable_count / max(total_candidates, 1)
    
    # Consistency: how similar are top segments?
    if len(top_segments) >= 2:
        top_scores = [s.final_score for s in top_segments[:3]]
        consistency = 1.0 - np.std(top_scores) if len(top_scores) > 1 else 1.0
    else:
        consistency = 1.0
    
    # Combine factors
    confidence = (
        0.4 * margin +
        0.3 * coverage +
        0.3 * consistency
    )
    
    return float(track_extendability), float(np.clip(confidence, 0.0, 1.0))

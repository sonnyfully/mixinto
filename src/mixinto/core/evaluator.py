"""Evaluation and safety logic for extension candidates."""
import numpy as np

from mixinto.core.presets import get_preset
from mixinto.utils.types import BeatGrid, IntroProfile, ExtendabilityProfile, Preset


def evaluate_safety(
    profile: IntroProfile | ExtendabilityProfile,
    beat_grid: BeatGrid,
    preset: Preset | str,
) -> tuple[bool, str | None]:
    """
    Evaluate whether a track is safe to extend based on preset criteria.
    
    Supports both legacy IntroProfile and new ExtendabilityProfile.
    
    Args:
        profile: IntroProfile or ExtendabilityProfile with analysis results
        beat_grid: BeatGrid with beat information
        preset: Preset object or name
    
    Returns:
        Tuple of (is_safe, refusal_reason)
        - is_safe: True if safe to extend, False otherwise
        - refusal_reason: None if safe, otherwise reason for refusal
    """
    # Get preset if string provided
    if isinstance(preset, str):
        preset = get_preset(preset)
    
    # Check mix safety score (works for both profile types via properties)
    if profile.mix_safety_score < preset.min_mix_safety_score:
        return (
            False,
            f"Mix safety score too low: {profile.mix_safety_score:.2f} "
            f"(minimum: {preset.min_mix_safety_score:.2f})",
        )
    
    # Check vocal presence
    # Note: For ExtendabilityProfile, vocal_presence is actually vocal_penalty (inverted)
    # So we need to handle this differently
    if isinstance(profile, ExtendabilityProfile):
        # Use the actual vocal presence from the best segment
        vocal_presence = 1.0 - profile.best_segment.vocal_penalty
    else:
        vocal_presence = profile.vocal_presence
    
    if vocal_presence > preset.max_vocal_presence:
        return (
            False,
            f"Vocal presence too high: {vocal_presence:.2f} "
            f"(maximum: {preset.max_vocal_presence:.2f})",
        )
    
    # Check beat confidence
    if beat_grid.confidence < preset.min_beat_confidence:
        return (
            False,
            f"Beat confidence too low: {beat_grid.confidence:.2f} "
            f"(minimum: {preset.min_beat_confidence:.2f})",
        )
    
    # Check flags for critical issues
    critical_flags = [
        "low_beat_confidence",
        "low_rhythm_stability",
    ]
    
    for flag in critical_flags:
        if flag in profile.flags:
            return (
                False,
                f"Critical issue detected: {flag}",
            )
    
    # All checks passed
    return (True, None)


def calculate_seam_quality(
    original_end: float,
    extended_start: float,
    max_error_ms: float,
) -> float:
    """
    Calculate seam quality score based on alignment error.
    
    Args:
        original_end: End time of original segment
        extended_start: Start time of extended segment
        max_error_ms: Maximum acceptable error in milliseconds
    
    Returns:
        Quality score in [0, 1] where 1 is perfect alignment
    """
    error_s = abs(original_end - extended_start)
    error_ms = error_s * 1000.0
    
    if error_ms <= max_error_ms:
        # Perfect or acceptable alignment
        quality = 1.0 - (error_ms / max_error_ms) * 0.1  # Small penalty for any error
    else:
        # Poor alignment
        quality = max(0.0, 1.0 - (error_ms / max_error_ms))
    
    return float(np.clip(quality, 0.0, 1.0))

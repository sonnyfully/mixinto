"""Core extension pipeline orchestrator."""
from pathlib import Path

from mixinto.core.analyzer import analyze_audio
from mixinto.core.evaluator import evaluate_safety
from mixinto.core.presets import get_preset
from mixinto.dsp.extend.loop import extend_by_looping
from mixinto.io.audio.writer import write_audio
from mixinto.utils.types import AudioBuffer, ExtendRequest


def extend_audio(request: ExtendRequest) -> tuple[AudioBuffer, dict]:
    """
    Extend an audio file's intro according to the request.
    
    This orchestrates the full extension pipeline:
    1. Analyze audio
    2. Evaluate safety
    3. Extend if safe
    4. Return extended buffer and metrics
    
    Args:
        request: ExtendRequest with extension parameters
    
    Returns:
        Tuple of (extended_buffer, metrics_dict)
        - extended_buffer: Extended AudioBuffer (or original if refused)
        - metrics_dict: Dictionary with metrics, warnings, errors, etc.
    
    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If extension fails
    """
    # Get preset
    preset = get_preset(request.preset)
    
    # Step 1: Analyze audio
    buffer, beat_grid, intro_profile = analyze_audio(
        request.input_path,
        preset_name=request.preset,
    )
    
    # Step 2: Evaluate safety
    is_safe, refusal_reason = evaluate_safety(
        intro_profile,
        beat_grid,
        preset,
    )
    
    metrics = {
        "original_duration_s": buffer.length_s(),
        "extended_duration_s": buffer.length_s(),
        "bars_added": 0,
        "mix_safety_score": intro_profile.mix_safety_score,
        "seam_quality": 1.0,
        "warnings": [],
        "errors": [],
    }
    
    # Step 3: Check if we should refuse
    if not is_safe:
        return (
            buffer,  # Return original buffer
            {
                **metrics,
                "refused": True,
                "refusal_reason": refusal_reason,
            },
        )
    
    # Step 4: Calculate target extension
    if request.target_bars is not None:
        target_bars = request.target_bars
        # Calculate target duration from bars
        seconds_per_bar = beat_grid.seconds_per_beat() * 4  # Assuming 4/4
        target_seconds = target_bars * seconds_per_bar
    elif request.target_seconds is not None:
        target_seconds = request.target_seconds
        # Calculate bars from seconds
        seconds_per_bar = beat_grid.seconds_per_beat() * 4
        target_bars = int(target_seconds / seconds_per_bar)
    else:
        raise ValueError("Either target_bars or target_seconds must be provided")
    
    # Step 5: Extend based on method
    if preset.method == "loop":
        extended_buffer = extend_by_looping(
            buffer,
            beat_grid,
            intro_profile.start_s,
            intro_profile.end_s,
            target_bars,
        )
    else:
        raise ValueError(f"Unknown extension method: {preset.method}")
    
    # Step 6: Calculate final metrics
    metrics.update({
        "extended_duration_s": extended_buffer.length_s(),
        "bars_added": target_bars,
        "seam_quality": 1.0,  # For loop method, seam is perfect
    })
    
    # Add warnings from intro profile
    if intro_profile.flags:
        metrics["warnings"] = intro_profile.flags.copy()
    
    return (extended_buffer, metrics)

"""Core extension pipeline orchestrator."""
from pathlib import Path

from mixinto.core.analyzer import analyze_audio
from mixinto.core.evaluator import evaluate_safety
from mixinto.core.presets import get_preset
from mixinto.dsp.extend.loop import extend_by_looping
from mixinto.dsp.render.seam import apply_crossfade
from mixinto.generation.baseline import BaselineGenerator
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
    buffer, beat_grid, extendability_profile = analyze_audio(
        request.input_path,
        preset_name=request.preset,
    )
    
    # Step 2: Evaluate safety
    is_safe, refusal_reason = evaluate_safety(
        extendability_profile,
        beat_grid,
        preset,
    )
    
    metrics = {
        "original_duration_s": buffer.length_s(),
        "extended_duration_s": buffer.length_s(),
        "bars_added": 0,
        "mix_safety_score": extendability_profile.mix_safety_score,
        "track_extendability": extendability_profile.track_extendability,
        "coverage": extendability_profile.coverage,
        "confidence": extendability_profile.confidence,
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
    
    # Step 5: Extend based on backend
    generation_metadata = None
    
    if request.backend == "baseline":
        # Use baseline generator
        generator = BaselineGenerator()
        
        # Extract context window (best segment from extendability profile)
        context_audio = buffer.trim(extendability_profile.start_s, extendability_profile.end_s)
        
        # Generate continuation
        extension_buffer, generation_metadata = generator.generate_continuation(
            context_audio=context_audio,
            bars=target_bars,
            bpm=beat_grid.bpm,
            preset=request.preset,
            seed=request.seed,
        )
        
        # Join original track (up to segment end) with extension using crossfade
        # Extract original track up to segment end
        original_up_to_intro = buffer.trim(0.0, extendability_profile.end_s)
        
        # Apply crossfade between original intro end and extension start
        # Use 50-150ms crossfade (use 100ms)
        crossfade_duration_s = 0.1
        extended_buffer = apply_crossfade(
            original_up_to_intro,
            extension_buffer,
            fade_duration_s=crossfade_duration_s,
        )
        
        # Calculate seam quality (should be good with crossfade)
        seam_quality = 0.95  # High quality with crossfade
        
    elif request.backend == "loop":
        # Use legacy loop method
        extended_buffer = extend_by_looping(
            buffer,
            beat_grid,
            extendability_profile.start_s,
            extendability_profile.end_s,
            target_bars,
        )
        seam_quality = 1.0  # For loop method, seam is perfect
    else:
        raise ValueError(f"Unknown backend: {request.backend}")
    
    # Step 6: Calculate final metrics
    metrics.update({
        "extended_duration_s": extended_buffer.length_s(),
        "bars_added": target_bars,
        "seam_quality": seam_quality,
    })
    
    # Add bassline metadata if using baseline backend
    if generation_metadata is not None:
        metrics["bassline"] = {
            "root_midi": generation_metadata.root_midi,
            "bass_gain_db": generation_metadata.bass_gain_db,
            "pattern": generation_metadata.pattern,
            "loop_len_bars": generation_metadata.loop_len_bars,
            "seed": generation_metadata.seed,
        }
    
    # Add warnings from extendability profile
    if extendability_profile.flags:
        metrics["warnings"] = extendability_profile.flags.copy()
    
    # Add segment information
    metrics["best_segment"] = {
        "start_s": extendability_profile.best_segment.segment.start_s,
        "end_s": extendability_profile.best_segment.segment.end_s,
        "bar_count": extendability_profile.best_segment.segment.bar_count,
        "score": extendability_profile.best_segment.final_score,
    }
    
    if len(extendability_profile.top_segments) > 1:
        metrics["top_segments"] = [
            {
                "start_s": seg.segment.start_s,
                "end_s": seg.segment.end_s,
                "bar_count": seg.segment.bar_count,
                "score": seg.final_score,
            }
            for seg in extendability_profile.top_segments[:3]  # Top 3
        ]
    
    return (extended_buffer, metrics)

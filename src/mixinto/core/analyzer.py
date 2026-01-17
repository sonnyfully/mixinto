"""Core analysis pipeline orchestrator."""
from pathlib import Path

from mixinto.core.presets import get_preset
from mixinto.dsp.beats.grid import build_beat_grid
from mixinto.dsp.segments.candidates import generate_segment_candidates
from mixinto.dsp.segments.scoring import (
    calculate_track_extendability,
    score_segment,
    select_top_segments,
)
from mixinto.dsp.tempo.detector import detect_tempo
from mixinto.features.timeline import extract_feature_timeline
from mixinto.io.audio.loader import load_audio
from mixinto.utils.types import AudioBuffer, BeatGrid, ExtendabilityProfile


def analyze_audio(
    file_path: str | Path,
    preset_name: str = "dj_safe",
) -> tuple[AudioBuffer, BeatGrid, ExtendabilityProfile]:
    """
    Analyze an audio file and return buffer, beat grid, and extendability profile.
    
    This orchestrates the full segment-aware analysis pipeline:
    1. Load audio
    2. Detect tempo
    3. Build beat grid
    4. Extract per-bar feature timeline
    5. Generate candidate segments
    6. Score all candidates
    7. Select top-K segments
    8. Calculate track-level extendability
    
    Args:
        file_path: Path to audio file
        preset_name: Preset name
    
    Returns:
        Tuple of (AudioBuffer, BeatGrid, ExtendabilityProfile)
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If analysis fails
    """
    # Get preset and analysis config
    preset = get_preset(preset_name)
    config = preset.analysis_config
    
    # Step 1: Load audio
    buffer = load_audio(file_path)
    
    # Step 2: Detect tempo with config
    bpm, tempo_confidence = detect_tempo(buffer, config=config)
    
    # Step 3: Build beat grid
    beat_grid = build_beat_grid(buffer, bpm=bpm, confidence=tempo_confidence)
    
    # Step 4: Extract per-bar feature timeline (NEW)
    feature_timeline = extract_feature_timeline(buffer, beat_grid, config)
    
    # Step 5: Generate candidate segments (NEW)
    candidates = generate_segment_candidates(buffer, beat_grid, feature_timeline, config)
    
    if len(candidates) == 0:
        raise ValueError("No candidate segments generated")
    
    # Step 6: Score all candidates (NEW)
    segment_scores = []
    for candidate in candidates:
        score = score_segment(candidate, feature_timeline, beat_grid, config)
        segment_scores.append(score)
    
    # Step 7: Select top-K segments (NEW)
    top_segments, viable_count = select_top_segments(
        segment_scores,
        top_k=config.segment_top_k,
        viable_threshold=config.segment_viable_threshold,
    )
    
    if len(top_segments) == 0:
        raise ValueError("No viable segments found")
    
    # Step 8: Calculate track-level extendability (NEW)
    track_extendability, confidence = calculate_track_extendability(
        top_segments,
        viable_count,
        len(candidates),
    )
    
    # Create extendability profile
    extendability_profile = ExtendabilityProfile(
        track_extendability=track_extendability,
        coverage=viable_count,
        confidence=confidence,
        top_segments=top_segments,
        feature_timeline=feature_timeline,
        best_segment=top_segments[0],
    )
    
    return (buffer, beat_grid, extendability_profile)

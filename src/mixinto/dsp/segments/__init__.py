"""Segment detection module."""
from mixinto.dsp.segments.candidates import generate_segment_candidates
from mixinto.dsp.segments.intro import detect_intro_window
from mixinto.dsp.segments.scoring import (
    calculate_track_extendability,
    score_segment,
    select_top_segments,
)

__all__ = [
    "detect_intro_window",
    "generate_segment_candidates",
    "score_segment",
    "select_top_segments",
    "calculate_track_extendability",
]

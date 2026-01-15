"""Preset definitions and loading."""
from mixinto.utils.types import AnalysisConfig, Preset


# Built-in presets
PRESETS: dict[str, Preset] = {
    "dj_safe": Preset(
        name="dj_safe",
        min_mix_safety_score=0.5,
        max_vocal_presence=0.5,
        min_beat_confidence=0.5,
        method="loop",
        max_seam_error_ms=25.0,
        export_format="wav",
        normalize=True,
        analysis_config=AnalysisConfig(),  # Uses defaults
    ),
    "dj_safe_strict": Preset(
        name="dj_safe_strict",
        min_mix_safety_score=0.7,
        max_vocal_presence=0.3,
        min_beat_confidence=0.7,
        method="loop",
        max_seam_error_ms=25.0,
        export_format="wav",
        normalize=True,
        analysis_config=AnalysisConfig(
            tempo_confidence_threshold=0.5,
            intro_min_bars=12,
            intro_max_bars=20,
        ),
    ),
    "dj_safe_lenient": Preset(
        name="dj_safe_lenient",
        min_mix_safety_score=0.4,
        max_vocal_presence=0.6,
        min_beat_confidence=0.4,
        method="loop",
        max_seam_error_ms=25.0,
        export_format="wav",
        normalize=True,
        analysis_config=AnalysisConfig(
            tempo_confidence_threshold=0.2,
            intro_min_bars=4,
            intro_max_bars=16,
        ),
    ),
}


def get_preset(name: str) -> Preset:
    """
    Get a preset by name.
    
    Args:
        name: Preset name
    
    Returns:
        Preset object
    
    Raises:
        ValueError: If preset not found
    """
    if name not in PRESETS:
        raise ValueError(f"Unknown preset: {name}. Available: {list(PRESETS.keys())}")
    
    return PRESETS[name]

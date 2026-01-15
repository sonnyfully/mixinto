"""Core analysis pipeline orchestrator."""
from pathlib import Path

from mixinto.core.presets import get_preset
from mixinto.dsp.beats.grid import build_beat_grid
from mixinto.dsp.segments.intro import detect_intro_window
from mixinto.dsp.tempo.detector import detect_tempo
from mixinto.features.stability import (
    calculate_energy_features,
    calculate_rhythm_stability,
    calculate_spectral_stability,
)
from mixinto.features.vocals import detect_vocal_presence
from mixinto.io.audio.loader import load_audio
from mixinto.utils.types import AudioBuffer, BeatGrid, IntroProfile


def analyze_audio(
    file_path: str | Path,
    preset_name: str = "dj_safe",
) -> tuple[AudioBuffer, BeatGrid, IntroProfile]:
    """
    Analyze an audio file and return buffer, beat grid, and intro profile.
    
    This orchestrates the full analysis pipeline:
    1. Load audio
    2. Detect tempo
    3. Build beat grid
    4. Find intro window
    5. Extract features
    6. Calculate mix safety score
    
    Args:
        file_path: Path to audio file
        preset_name: Preset name
    
    Returns:
        Tuple of (AudioBuffer, BeatGrid, IntroProfile)
    
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
    
    # Step 4: Detect intro window with config
    intro_start_s, intro_end_s = detect_intro_window(
        buffer,
        beat_grid,
        min_bars=config.intro_min_bars,
        max_bars=config.intro_max_bars,
        config=config,
    )
    
    # Extract intro segment
    intro_buffer = buffer.trim(intro_start_s, intro_end_s)
    
    # Step 5: Extract features from intro with config
    # Energy features
    energy_mean, energy_std = calculate_energy_features(intro_buffer, config)
    
    # Stability features
    spectral_stability = calculate_spectral_stability(intro_buffer, config)
    rhythm_stability = calculate_rhythm_stability(intro_buffer, beat_grid, config)
    
    # Vocal presence (no config needed for now)
    vocal_presence = detect_vocal_presence(intro_buffer)
    
    # Step 6: Calculate mix safety score
    # Combine various factors into a single score
    # Higher is better for mixing
    mix_safety_score = (
        0.3 * tempo_confidence +  # Beat confidence is important
        0.25 * spectral_stability +  # Spectral stability
        0.25 * rhythm_stability +  # Rhythm stability
        0.1 * (1.0 - vocal_presence) +  # Lower vocals = better for mixing
        0.1 * (1.0 - min(1.0, energy_std / (energy_mean + 0.001)))  # Energy consistency
    )
    
    # Collect flags (warnings/issues)
    flags = []
    if tempo_confidence < 0.5:
        flags.append("low_beat_confidence")
    if spectral_stability < 0.5:
        flags.append("low_spectral_stability")
    if rhythm_stability < 0.5:
        flags.append("low_rhythm_stability")
    if vocal_presence > 0.5:
        flags.append("high_vocal_presence")
    if energy_std > energy_mean * 2.0:
        flags.append("high_energy_variance")
    
    # Create intro profile
    intro_profile = IntroProfile(
        start_s=intro_start_s,
        end_s=intro_end_s,
        energy_mean=energy_mean,
        energy_std=energy_std,
        spectral_stability=spectral_stability,
        rhythm_stability=rhythm_stability,
        vocal_presence=vocal_presence,
        mix_safety_score=mix_safety_score,
        flags=flags,
    )
    
    return (buffer, beat_grid, intro_profile)

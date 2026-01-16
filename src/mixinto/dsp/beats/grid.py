"""Beat grid and downbeat detection functionality."""
import numpy as np
import librosa

from mixinto.utils.types import AudioBuffer, BeatGrid, AnalysisConfig


def build_beat_grid(
    buffer: AudioBuffer,
    bpm: float | None = None,
    confidence: float | None = None,
    config: AnalysisConfig | None = None,
) -> BeatGrid:
    """
    Build a beat grid (beats and downbeats) for an audio buffer.
    
    Args:
        buffer: AudioBuffer to analyze
        bpm: Known BPM (if None, will be detected)
        confidence: Known confidence (if None, will be calculated)
        config: AnalysisConfig with configurable parameters (uses defaults if None)
    
    Returns:
        BeatGrid with beats and downbeats
    
    Raises:
        ValueError: If beat grid cannot be constructed
    """
    # Convert to mono
    mono_buffer = buffer.to_mono()
    samples = mono_buffer.samples.flatten()
    
    # Detect tempo if not provided
    if bpm is None:
        from mixinto.dsp.tempo.detector import detect_tempo
        bpm, confidence = detect_tempo(buffer, config=config)
    
    if confidence is None:
        confidence = 1.0  # Default if not provided
    
    # Detect beats using librosa
    # Use the known BPM to help with beat tracking
    tempo = float(bpm)
    
    # Get onset strength
    # Use default parameters (onset_strength doesn't need custom frame_length)
    onset_strength = librosa.onset.onset_strength(
        y=samples,
        sr=buffer.sample_rate,
    )
    
    # Track beats
    # Note: beat_track doesn't accept 'tempo' parameter in all versions
    # Instead, we'll use the onset envelope and let it detect beats
    # The tempo is used for validation/confidence, not as a direct parameter
    beats_frames, _ = librosa.beat.beat_track(
        onset_envelope=onset_strength,
        sr=buffer.sample_rate,
        units="time",
    )
    
    # Convert to time in seconds
    beats_s = beats_frames.tolist()
    
    # Detect downbeats (bar starts)
    # Simple heuristic: assume 4/4 time, so every 4th beat is a downbeat
    # We can refine this later with more sophisticated detection
    downbeats_s = []
    if len(beats_s) >= 4:
        # Start with the first beat as a downbeat
        downbeats_s.append(beats_s[0])
        
        # Find subsequent downbeats (every 4 beats)
        for i in range(4, len(beats_s), 4):
            downbeats_s.append(beats_s[i])
    
    # If we have very few beats, try a different approach
    if len(beats_s) < 4:
        # Use onset detection as fallback
        onsets = librosa.onset.onset_detect(
            y=samples,
            sr=buffer.sample_rate,
            units="time",
        )
        beats_s = onsets.tolist()
        
        # For downbeats, use every 4th onset
        if len(beats_s) >= 4:
            downbeats_s = beats_s[::4]
        else:
            downbeats_s = [beats_s[0]] if len(beats_s) > 0 else []
    
    # Validate we have beats
    if len(beats_s) == 0:
        raise ValueError("No beats detected in audio")
    
    return BeatGrid(
        bpm=tempo,
        beats_s=beats_s,
        downbeats_s=downbeats_s,
        confidence=float(confidence),
    )

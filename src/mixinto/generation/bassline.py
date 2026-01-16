"""Bassline synthesis for DJ-safe audio extension."""
import random
from typing import Optional

import librosa
import numpy as np
from scipy import signal

from mixinto.utils.types import AudioBuffer, AudioMetadata


def estimate_root_note(context: AudioBuffer, bpm: float) -> Optional[int]:
    """
    Estimate the root note (bass note) from context audio using a cheap heuristic.
    
    This function:
    1. Converts to mono
    2. Low-passes / focuses on 40-200 Hz band
    3. Computes a coarse pitch class estimate
    4. Returns a MIDI note number (36-52 range) or None if confidence is low
    
    Args:
        context: AudioBuffer containing context audio
        bpm: BPM of the track (for potential tempo-based filtering)
    
    Returns:
        MIDI note number (36-52 range) or None if confidence is low
    """
    # Convert to mono
    mono_buffer = context.to_mono()
    samples = mono_buffer.samples
    if samples.ndim > 1:
        samples = samples.squeeze()
    
    sr = mono_buffer.sample_rate
    
    # Low-pass filter to focus on 40-200 Hz band
    # Use a simple butterworth filter if scipy is available
    nyquist = sr / 2.0
    low_cutoff = 40.0 / nyquist
    high_cutoff = 200.0 / nyquist
    
    # Clamp to valid range
    low_cutoff = max(0.01, min(0.99, low_cutoff))
    high_cutoff = max(0.01, min(0.99, high_cutoff))
    
    if high_cutoff > low_cutoff:
        # Design bandpass filter
        b, a = signal.butter(4, [low_cutoff, high_cutoff], btype='band')
        filtered = signal.filtfilt(b, a, samples)
    else:
        # Fallback: just use low-pass
        b, a = signal.butter(4, high_cutoff, btype='low')
        filtered = signal.filtfilt(b, a, samples)
    
    # Try to use librosa's chroma_cqt if available for pitch estimation
    try:
        # Use a small hop length for better time resolution
        chroma = librosa.feature.chroma_cqt(
            y=filtered,
            sr=sr,
            hop_length=512,
            fmin=40.0,  # Start from 40 Hz (E1)
        )
        
        # Average chroma across time to get overall pitch class
        chroma_mean = np.mean(chroma, axis=1)
        
        # Find the dominant pitch class
        dominant_class = np.argmax(chroma_mean)
        
        # Map chroma class to MIDI note
        # Chroma classes: C, C#, D, D#, E, F, F#, G, G#, A, A#, B
        # We want to find the root in the bass range (36-52 MIDI = C2-E3)
        # Start from C2 (MIDI 36) and find the strongest class
        
        # Find the strongest chroma in the bass range
        # Map to MIDI: C=0, C#=1, D=2, D#=3, E=4, F=5, F#=6, G=7, G#=8, A=9, A#=10, B=11
        # For bass range, try C2 (36), D2 (38), E2 (40), F2 (41), G2 (43), A2 (45), B2 (47)
        # Then C3 (48), D3 (50), E3 (52)
        
        bass_notes = [36, 38, 40, 41, 43, 45, 47, 48, 50, 52]  # Common bass notes
        bass_chroma_classes = [0, 2, 4, 5, 7, 9, 11, 0, 2, 4]  # Corresponding chroma classes
        
        # Find which bass note matches the dominant chroma
        best_match = None
        best_score = 0.0
        
        for note, chroma_class in zip(bass_notes, bass_chroma_classes):
            if chroma_class == dominant_class:
                score = chroma_mean[dominant_class]
                if score > best_score:
                    best_score = score
                    best_match = note
        
        # If we found a match with reasonable confidence
        if best_match is not None and best_score > 0.1:
            return best_match
        
    except Exception:
        # Fallback: use FFT peak detection
        pass
    
    # Fallback method: FFT peak detection
    # Compute FFT of filtered signal
    fft_size = min(8192, len(filtered))
    fft = np.fft.rfft(filtered[:fft_size])
    freqs = np.fft.rfftfreq(fft_size, 1.0 / sr)
    magnitude = np.abs(fft)
    
    # Focus on 40-200 Hz range
    freq_mask = (freqs >= 40) & (freqs <= 200)
    if not np.any(freq_mask):
        return None
    
    freq_range = freqs[freq_mask]
    mag_range = magnitude[freq_mask]
    
    # Find dominant frequency
    peak_idx = np.argmax(mag_range)
    peak_freq = freq_range[peak_idx]
    peak_mag = mag_range[peak_idx]
    
    # Check if peak is strong enough
    if peak_mag < np.max(magnitude) * 0.1:
        return None
    
    # Convert frequency to MIDI note
    # MIDI note = 69 + 12 * log2(freq / 440)
    midi_note = 69 + 12 * np.log2(peak_freq / 440.0)
    midi_note = int(round(midi_note))
    
    # Clamp to bass range (36-52)
    midi_note = max(36, min(52, midi_note))
    
    return midi_note


def generate_bassline(
    sr: int,
    bpm: float,
    bars: int,
    seed: int,
    root_midi: Optional[int],
    preset: str,
) -> AudioBuffer:
    """
    Generate a mono bassline audio of exact duration bars * 4 beats at the given BPM.
    
    Uses a simple oscillator (sine + touch of 2nd harmonic) with ADSR envelope.
    Rhythm patterns vary by preset:
    - dj_safe: steady pattern (quarter or half notes), minimal variation
    - more_motion: deterministic variation every 4 bars (seeded)
    - no_vocals: same as dj_safe (conservative)
    
    Args:
        sr: Sample rate
        bpm: BPM of the track
        bars: Number of bars to generate
        seed: Random seed for determinism
        root_midi: Root MIDI note (36-52), or None to use default (G2 = 43)
        preset: Preset name (dj_safe, more_motion, no_vocals)
    
    Returns:
        AudioBuffer containing the generated bassline (mono)
    """
    # Set random seed for determinism
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    
    # Default to G2 (MIDI 43) if root not provided
    if root_midi is None:
        root_midi = 43
    
    # Clamp to valid range
    root_midi = max(36, min(52, root_midi))
    
    # Convert MIDI to frequency
    root_freq = 440.0 * (2.0 ** ((root_midi - 69) / 12.0))
    
    # Calculate duration
    seconds_per_beat = 60.0 / bpm
    seconds_per_bar = seconds_per_beat * 4  # 4/4 time
    duration_s = bars * seconds_per_bar
    num_samples = int(duration_s * sr)
    
    # Generate time array
    t = np.linspace(0, duration_s, num_samples, endpoint=False)
    
    # Initialize output
    output = np.zeros(num_samples)
    
    # Determine rhythm pattern based on preset
    beats_per_bar = 4
    total_beats = bars * beats_per_bar
    
    if preset == "dj_safe" or preset == "no_vocals":
        # Steady pattern: quarter notes or half notes
        # Use half notes for very steady feel
        note_duration_beats = 2.0  # Half notes
        pattern = [1.0] * int(total_beats / note_duration_beats)  # All notes on
    elif preset == "more_motion":
        # More variation: quarter notes with deterministic variation every 4 bars
        note_duration_beats = 1.0  # Quarter notes
        pattern_length = int(total_beats / note_duration_beats)
        pattern = []
        
        for i in range(pattern_length):
            bar_idx = i // beats_per_bar
            beat_in_bar = i % beats_per_bar
            
            # Every 4 bars, add some variation
            if bar_idx % 4 == 0 and beat_in_bar == 0:
                # On downbeat of every 4th bar, maybe add a slight accent
                pattern.append(1.0)
            elif beat_in_bar == 0:
                # Downbeats always on
                pattern.append(1.0)
            else:
                # Use seeded random to decide if note plays
                if rng.random() > 0.3:  # 70% chance
                    pattern.append(1.0)
                else:
                    pattern.append(0.0)
    else:
        # Default: steady quarter notes
        note_duration_beats = 1.0
        pattern = [1.0] * int(total_beats / note_duration_beats)
    
    # Generate notes
    note_duration_s = note_duration_beats * seconds_per_beat
    note_samples = int(note_duration_s * sr)
    
    pattern_idx = 0
    sample_idx = 0
    
    while sample_idx < num_samples:
        # Check if note should play
        if pattern_idx < len(pattern) and pattern[pattern_idx] > 0:
            # Generate note
            note_start = sample_idx
            note_end = min(sample_idx + note_samples, num_samples)
            note_length = note_end - note_start
            
            if note_length > 0:
                # ADSR envelope parameters
                attack_s = 0.01  # 10ms attack
                decay_s = 0.05   # 50ms decay
                sustain_level = 0.7  # 70% sustain
                release_s = 0.05  # 50ms release
                
                attack_samples = int(attack_s * sr)
                decay_samples = int(decay_s * sr)
                release_samples = int(release_s * sr)
                sustain_samples = note_length - attack_samples - decay_samples - release_samples
                
                # Clamp to valid ranges
                attack_samples = max(0, min(attack_samples, note_length // 4))
                decay_samples = max(0, min(decay_samples, note_length // 4))
                release_samples = max(0, min(release_samples, note_length // 4))
                sustain_samples = max(0, note_length - attack_samples - decay_samples - release_samples)
                
                # Generate envelope
                envelope = np.ones(note_length)
                
                # Attack
                if attack_samples > 0:
                    envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
                
                # Decay
                if decay_samples > 0:
                    decay_start = attack_samples
                    decay_end = decay_start + decay_samples
                    envelope[decay_start:decay_end] = np.linspace(
                        1, sustain_level, decay_samples
                    )
                
                # Sustain
                if sustain_samples > 0:
                    sustain_start = attack_samples + decay_samples
                    sustain_end = sustain_start + sustain_samples
                    envelope[sustain_start:sustain_end] = sustain_level
                
                # Release
                if release_samples > 0:
                    release_start = note_length - release_samples
                    envelope[release_start:] = np.linspace(
                        sustain_level, 0, release_samples
                    )
                
                # Generate oscillator (sine + touch of 2nd harmonic)
                note_t = t[note_start:note_end] - t[note_start]
                fundamental = np.sin(2 * np.pi * root_freq * note_t)
                harmonic2 = 0.15 * np.sin(2 * np.pi * root_freq * 2 * note_t)  # 15% 2nd harmonic
                oscillator = fundamental + harmonic2
                
                # Apply envelope
                note_audio = oscillator * envelope
                
                # Add to output
                output[note_start:note_end] += note_audio
        
        # Move to next note
        sample_idx += note_samples
        pattern_idx += 1
    
    # Apply micro fades at start and end to avoid clicks
    fade_samples = int(0.01 * sr)  # 10ms fade
    if fade_samples > 0 and fade_samples < len(output):
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        output[:fade_samples] *= fade_in
        output[-fade_samples:] *= fade_out
    
    # Light soft clip / limiter to prevent clipping
    # Soft clip: tanh for gentle saturation
    output = np.tanh(output * 0.8) * 0.95  # Scale down slightly
    
    # Ensure no clipping
    peak = np.max(np.abs(output))
    if peak > 0.99:
        output = output / peak * 0.99
    
    # Reshape to (samples, 1) for mono
    output = output.reshape(-1, 1)
    
    # Create metadata
    meta = AudioMetadata(
        source_path=None,
        sample_rate=sr,
        channels=1,
        duration_s=duration_s,
        format=None,
    )
    
    return AudioBuffer(
        samples=output,
        sample_rate=sr,
        meta=meta,
    )


def mix_bass_into_extension(
    extension: AudioBuffer,
    bass: AudioBuffer,
    bass_gain_db: float,
) -> AudioBuffer:
    """
    Mix bassline into extension audio.
    
    Matches lengths exactly, upmixes bass to stereo if extension is stereo,
    applies gain in linear scale from dB, and prevents clipping.
    
    Args:
        extension: Extension audio buffer
        bass: Bassline audio buffer (mono)
        bass_gain_db: Gain for bass in dB (negative values reduce volume)
    
    Returns:
        AudioBuffer with bass mixed in
    """
    # Ensure lengths match exactly
    extension_len = len(extension)
    bass_len = len(bass)
    
    if extension_len != bass_len:
        # Trim or pad bass to match extension
        if bass_len > extension_len:
            bass_samples = bass.samples[:extension_len]
        else:
            # Pad with zeros
            pad_len = extension_len - bass_len
            if bass.samples.ndim == 1:
                pad = np.zeros(pad_len)
                bass_samples = np.concatenate([bass.samples, pad])
            else:
                pad = np.zeros((pad_len, bass.samples.shape[1]))
                bass_samples = np.concatenate([bass.samples, pad], axis=0)
    else:
        bass_samples = bass.samples
    
    # Convert gain from dB to linear
    bass_gain_linear = 10.0 ** (bass_gain_db / 20.0)
    
    # Apply gain
    bass_samples = bass_samples * bass_gain_linear
    
    # Upmix bass to stereo if extension is stereo
    extension_samples = extension.samples
    if extension.meta.channels == 2 and bass.meta.channels == 1:
        # Upmix mono bass to stereo
        if bass_samples.ndim == 1:
            bass_samples = bass_samples.reshape(-1, 1)
        bass_samples = np.repeat(bass_samples, 2, axis=1)
    elif extension.meta.channels == 1 and bass.meta.channels == 1:
        # Both mono, ensure proper shape
        if bass_samples.ndim == 1:
            bass_samples = bass_samples.reshape(-1, 1)
        if extension_samples.ndim == 1:
            extension_samples = extension_samples.reshape(-1, 1)
    
    # Mix
    mixed_samples = extension_samples + bass_samples
    
    # Prevent clipping with soft limiter
    peak = np.max(np.abs(mixed_samples))
    if peak > 0.99:
        # Soft limit: compress above 0.99
        threshold = 0.99
        ratio = 4.0  # 4:1 compression ratio
        excess = peak - threshold
        if excess > 0:
            compressed_peak = threshold + excess / ratio
            mixed_samples = mixed_samples / peak * compressed_peak
    
    # Final hard limit to 0.99
    mixed_samples = np.clip(mixed_samples, -0.99, 0.99)
    
    # Create metadata
    meta = AudioMetadata(
        source_path=extension.meta.source_path,
        sample_rate=extension.sample_rate,
        channels=extension.meta.channels,
        duration_s=extension.meta.duration_s,
        format=extension.meta.format,
    )
    
    return AudioBuffer(
        samples=mixed_samples,
        sample_rate=extension.sample_rate,
        meta=meta,
    )

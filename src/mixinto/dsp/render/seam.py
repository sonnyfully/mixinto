"""Seam handling for clean audio joins."""
import numpy as np

from mixinto.utils.types import AudioBuffer


def apply_crossfade(
    buffer1: AudioBuffer,
    buffer2: AudioBuffer,
    fade_duration_s: float = 0.1,
) -> AudioBuffer:
    """
    Apply a crossfade between two audio buffers.
    
    Args:
        buffer1: First buffer (will fade out at the end)
        buffer2: Second buffer (will fade in at the start)
        fade_duration_s: Duration of crossfade in seconds
    
    Returns:
        Combined buffer with crossfade applied
    """
    # Ensure same sample rate and channels
    if buffer1.sample_rate != buffer2.sample_rate:
        raise ValueError("Buffers must have same sample rate")
    if buffer1.meta.channels != buffer2.meta.channels:
        raise ValueError("Buffers must have same number of channels")
    
    sample_rate = buffer1.sample_rate
    fade_samples = int(fade_duration_s * sample_rate)
    
    # Get samples
    samples1 = buffer1.samples
    samples2 = buffer2.samples
    
    # Ensure proper shape
    if samples1.ndim == 1:
        samples1 = samples1.reshape(-1, 1)
    if samples2.ndim == 1:
        samples2 = samples2.reshape(-1, 1)
    
    # Limit fade to buffer lengths
    fade_samples = min(fade_samples, len(samples1), len(samples2))
    
    if fade_samples == 0:
        # No crossfade, just concatenate
        combined = np.concatenate([samples1, samples2], axis=0)
    else:
        # Create fade curves
        fade_out = np.linspace(1.0, 0.0, fade_samples)
        fade_in = np.linspace(0.0, 1.0, fade_samples)
        
        # Reshape for broadcasting
        if samples1.ndim == 2:
            fade_out = fade_out.reshape(-1, 1)
            fade_in = fade_in.reshape(-1, 1)
        
        # Apply fades
        samples1_faded = samples1.copy()
        samples2_faded = samples2.copy()
        
        samples1_faded[-fade_samples:] *= fade_out
        samples2_faded[:fade_samples] *= fade_in
        
        # Combine: overlap the faded regions
        # Take non-faded part of buffer1, then crossfaded overlap, then non-faded part of buffer2
        if len(samples1) > fade_samples:
            part1 = samples1_faded[:-fade_samples]
        else:
            part1 = np.array([]).reshape(0, samples1.shape[1] if samples1.ndim == 2 else 1)
        
        # Overlap region (sum of faded parts)
        overlap = samples1_faded[-fade_samples:] + samples2_faded[:fade_samples]
        
        if len(samples2) > fade_samples:
            part2 = samples2_faded[fade_samples:]
        else:
            part2 = np.array([]).reshape(0, samples2.shape[1] if samples2.ndim == 2 else 1)
        
        # Concatenate
        combined = np.concatenate([part1, overlap, part2], axis=0)
    
    # Create combined buffer
    from mixinto.utils.types import AudioMetadata
    
    combined_meta = AudioMetadata(
        source_path=buffer1.meta.source_path,
        sample_rate=sample_rate,
        channels=buffer1.meta.channels,
        duration_s=len(combined) / sample_rate,
        format=buffer1.meta.format,
    )
    
    return AudioBuffer(
        samples=combined,
        sample_rate=sample_rate,
        meta=combined_meta,
    )


def join_buffers(
    buffers: list[AudioBuffer],
    crossfade_duration_s: float = 0.05,
) -> AudioBuffer:
    """
    Join multiple audio buffers with crossfades.
    
    Args:
        buffers: List of AudioBuffers to join
        crossfade_duration_s: Duration of crossfade between buffers
    
    Returns:
        Combined AudioBuffer
    """
    if len(buffers) == 0:
        raise ValueError("Cannot join empty buffer list")
    
    if len(buffers) == 1:
        return buffers[0]
    
    # Start with first buffer
    result = buffers[0]
    
    # Join each subsequent buffer with crossfade
    for next_buffer in buffers[1:]:
        result = apply_crossfade(result, next_buffer, crossfade_duration_s)
    
    return result

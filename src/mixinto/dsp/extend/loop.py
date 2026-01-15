"""Simple looping extension functionality."""
import numpy as np

from mixinto.utils.types import AudioBuffer, BeatGrid


def extend_by_looping(
    buffer: AudioBuffer,
    beat_grid: BeatGrid,
    intro_start_s: float,
    intro_end_s: float,
    target_bars: int,
) -> AudioBuffer:
    """
    Extend audio by looping the intro segment.
    
    Args:
        buffer: Original audio buffer
        beat_grid: Beat grid for bar alignment
        intro_start_s: Start of intro window in seconds
        intro_end_s: End of intro window in seconds
        target_bars: Number of bars to extend by
    
    Returns:
        Extended AudioBuffer
    """
    # Extract intro segment
    intro_buffer = buffer.trim(intro_start_s, intro_end_s)
    
    # Calculate how many bars the intro segment is
    intro_duration_s = intro_end_s - intro_start_s
    seconds_per_bar = beat_grid.seconds_per_beat() * 4  # Assuming 4/4 time
    intro_bars = intro_duration_s / seconds_per_bar
    
    # Calculate how many times to loop
    target_duration_s = target_bars * seconds_per_bar
    num_loops = int(np.ceil(target_duration_s / intro_duration_s))
    
    # Loop the intro segment
    intro_samples = intro_buffer.samples
    looped_samples_list = []
    
    for _ in range(num_loops):
        looped_samples_list.append(intro_samples)
    
    # Concatenate all loops
    extended_samples = np.concatenate(looped_samples_list, axis=0)
    
    # Trim to exact target duration
    target_samples = int(target_duration_s * buffer.sample_rate)
    if len(extended_samples) > target_samples:
        extended_samples = extended_samples[:target_samples]
    
    # Create extended buffer
    from mixinto.utils.types import AudioMetadata
    
    extended_meta = AudioMetadata(
        source_path=buffer.meta.source_path,
        sample_rate=buffer.sample_rate,
        channels=buffer.meta.channels,
        duration_s=target_duration_s,
        format=buffer.meta.format,
    )
    
    extended_buffer = AudioBuffer(
        samples=extended_samples,
        sample_rate=buffer.sample_rate,
        meta=extended_meta,
    )
    
    return extended_buffer

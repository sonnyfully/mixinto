"""Audio file writing functionality."""
from pathlib import Path

import numpy as np
import soundfile as sf

from mixinto.utils.types import AudioBuffer


def write_audio(buffer: AudioBuffer, output_path: str | Path, format: str = "wav") -> None:
    """
    Write an AudioBuffer to a file.
    
    Args:
        buffer: AudioBuffer to write
        output_path: Path where the file should be written
        format: Audio format (wav, flac, etc.)
    
    Raises:
        ValueError: If the buffer is invalid or write fails
    """
    path = Path(output_path)
    
    # Ensure output directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare samples for writing
    samples = buffer.samples
    
    # soundfile expects (samples, channels) format
    if samples.ndim == 1:
        # Mono audio - reshape to (samples, 1)
        samples = samples.reshape(-1, 1)
    elif samples.ndim == 2 and samples.shape[1] == 1:
        # Already in correct format
        pass
    elif samples.ndim == 2:
        # Multi-channel, already correct
        pass
    else:
        raise ValueError(f"Invalid audio buffer shape: {samples.shape}")
    
    # Normalize if needed (soundfile expects float32 in range [-1, 1])
    if samples.dtype != np.float32:
        # Convert to float32 and normalize if needed
        if samples.dtype in (np.int16, np.int32):
            # Integer samples - normalize to [-1, 1]
            max_val = np.iinfo(samples.dtype).max
            samples = samples.astype(np.float32) / max_val
        else:
            samples = samples.astype(np.float32)
    
    # Ensure samples are in valid range
    samples = np.clip(samples, -1.0, 1.0)
    
    try:
        # Determine subtype based on format
        subtype = None
        if format.lower() == "wav":
            subtype = "PCM_24"  # High quality default
        elif format.lower() == "flac":
            subtype = "PCM_24"
        
        # Write the file
        sf.write(
            str(path),
            samples,
            buffer.sample_rate,
            format=format,
            subtype=subtype,
        )
        
    except Exception as e:
        raise ValueError(f"Failed to write audio file {path}: {e}") from e

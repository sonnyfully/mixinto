"""Audio file loading functionality."""
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from mixinto.utils.types import AudioBuffer, AudioMetadata


def load_audio(file_path: str | Path, sample_rate: int | None = None) -> AudioBuffer:
    """
    Load an audio file and return an AudioBuffer.
    
    Args:
        file_path: Path to the audio file
        sample_rate: Target sample rate (None to use file's native rate)
    
    Returns:
        AudioBuffer containing the loaded audio data
    
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file cannot be read
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    
    # Load audio using librosa (handles resampling and mono conversion if needed)
    # librosa always returns mono, so we'll use soundfile for multi-channel support
    try:
        # Use soundfile to get raw samples and metadata
        data, native_sr = sf.read(str(path), always_2d=True)
        
        # Determine channels
        if data.ndim == 1:
            channels = 1
            samples = data
        else:
            channels = data.shape[1]
            # Convert to (samples, channels) format
            samples = data
        
        # Resample if needed
        target_sr = sample_rate if sample_rate is not None else native_sr
        if target_sr != native_sr:
            # For resampling, convert to mono first if multi-channel
            if channels > 1:
                mono_data = np.mean(samples, axis=1)
            else:
                mono_data = samples
            # Resample using librosa
            resampled = librosa.resample(
                mono_data, 
                orig_sr=native_sr, 
                target_sr=target_sr
            )
            samples = resampled.reshape(-1, 1) if channels > 1 else resampled
            channels = 1 if channels > 1 else channels
        else:
            # Ensure proper shape
            if channels == 1 and samples.ndim == 1:
                samples = samples.reshape(-1, 1)
            elif channels > 1 and samples.ndim == 1:
                samples = samples.reshape(-1, channels)
        
        # Get file format
        file_format = path.suffix.lower().lstrip('.')
        if not file_format:
            file_format = None
        
        # Calculate duration
        duration_s = len(samples) / target_sr
        
        # Create metadata
        metadata = AudioMetadata(
            source_path=str(path),
            sample_rate=target_sr,
            channels=channels,
            duration_s=duration_s,
            format=file_format,
        )
        
        # Ensure samples are in correct format (samples, channels)
        if samples.ndim == 1:
            samples = samples.reshape(-1, 1)
        
        return AudioBuffer(
            samples=samples,
            sample_rate=target_sr,
            meta=metadata,
        )
        
    except Exception as e:
        raise ValueError(f"Failed to load audio file {path}: {e}") from e

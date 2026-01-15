"""Test audio I/O functionality."""
import pytest
from pathlib import Path
import numpy as np

from mixinto.io.audio import load_audio, write_audio
from mixinto.utils.types import AudioBuffer


def test_load_audio(temp_audio_file, sample_audio_buffer):
    """Test loading audio file."""
    buffer = load_audio(temp_audio_file)
    
    assert buffer.sample_rate == sample_audio_buffer.sample_rate
    assert buffer.meta.channels == 1
    assert buffer.length_s() > 0
    assert len(buffer.samples) > 0


def test_load_audio_nonexistent():
    """Test loading non-existent file raises error."""
    with pytest.raises(FileNotFoundError):
        load_audio("nonexistent.wav")


def test_write_audio(sample_audio_buffer, tmp_path):
    """Test writing audio file."""
    output_path = tmp_path / "output.wav"
    write_audio(sample_audio_buffer, output_path)
    
    assert output_path.exists()
    
    # Verify we can load it back
    loaded = load_audio(output_path)
    assert loaded.sample_rate == sample_audio_buffer.sample_rate
    assert loaded.meta.channels == sample_audio_buffer.meta.channels


def test_write_audio_stereo(stereo_audio_buffer, tmp_path):
    """Test writing stereo audio file."""
    output_path = tmp_path / "stereo_output.wav"
    write_audio(stereo_audio_buffer, output_path)
    
    assert output_path.exists()
    
    # Verify we can load it back
    loaded = load_audio(output_path)
    assert loaded.sample_rate == stereo_audio_buffer.sample_rate
    assert loaded.meta.channels == 2


def test_audio_buffer_to_mono(stereo_audio_buffer):
    """Test converting stereo to mono."""
    mono = stereo_audio_buffer.to_mono()
    
    assert mono.meta.channels == 1
    assert mono.sample_rate == stereo_audio_buffer.sample_rate
    assert mono.samples.ndim == 2  # Should be (samples, 1)
    assert mono.samples.shape[1] == 1


def test_audio_buffer_trim(sample_audio_buffer):
    """Test trimming audio buffer."""
    trimmed = sample_audio_buffer.trim(1.0, 3.0)
    
    assert trimmed.length_s() == pytest.approx(2.0, abs=0.1)
    assert trimmed.sample_rate == sample_audio_buffer.sample_rate


def test_audio_buffer_length_s(sample_audio_buffer):
    """Test length calculation."""
    expected_length = sample_audio_buffer.meta.duration_s
    assert sample_audio_buffer.length_s() == pytest.approx(expected_length, abs=0.01)

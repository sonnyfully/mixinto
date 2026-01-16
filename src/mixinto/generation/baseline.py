"""Baseline generator that creates extension audio with optional bassline."""
import random
from dataclasses import dataclass

import numpy as np

from mixinto.generation.bassline import (
    estimate_root_note,
    generate_bassline,
    mix_bass_into_extension,
)
from mixinto.utils.types import AudioBuffer, AudioMetadata


@dataclass
class GenerationMetadata:
    """Metadata about the generation process."""
    seed: int
    loop_len_bars: int
    root_midi: int | None
    bass_gain_db: float
    pattern: str


class BaselineGenerator:
    """
    Baseline generator that creates extension audio by looping context and adding bassline.
    
    This implements the Stage 4 generator interface:
    generate_continuation(context_audio, bars, bpm, preset, seed) -> AudioBuffer
    """
    
    def generate_continuation(
        self,
        context_audio: AudioBuffer,
        bars: int,
        bpm: float,
        preset: str,
        seed: int,
    ) -> tuple[AudioBuffer, GenerationMetadata]:
        """
        Generate continuation audio by looping context and adding bassline.
        
        Args:
            context_audio: Context audio window (from analyze)
            bars: Number of bars to generate
            bpm: BPM of the track
            preset: Preset name (dj_safe, more_motion, no_vocals)
            seed: Random seed for determinism
        
        Returns:
            Tuple of (AudioBuffer, GenerationMetadata)
        """
        # Set random seed for determinism
        rng = random.Random(seed)
        
        # Calculate timing
        seconds_per_beat = 60.0 / bpm
        seconds_per_bar = seconds_per_beat * 4  # 4/4 time
        target_length_s = bars * seconds_per_bar
        
        # Step 1: Create extension by looping context audio
        # Choose loop length deterministically from seed (default 2 bars for dj_safe)
        if preset == "dj_safe" or preset == "no_vocals":
            default_loop_bars = 2
        else:
            # For more_motion, allow some variation
            default_loop_bars = rng.choice([1, 2, 4])
        
        loop_len_bars = default_loop_bars
        loop_length_s = loop_len_bars * seconds_per_bar
        
        # Extract loop segment from context (use last N bars of context)
        context_length_s = context_audio.length_s()
        
        # Find the last complete loop_len_bars in the context
        # Start from the end and work backwards
        loop_start_s = max(0, context_length_s - loop_length_s)
        loop_buffer = context_audio.trim(loop_start_s, context_length_s)
        
        # Calculate how many times to loop
        num_loops = int(np.ceil(target_length_s / loop_length_s))
        
        # Create looped extension
        loop_samples = loop_buffer.samples
        looped_samples_list = []
        
        for i in range(num_loops):
            looped_samples_list.append(loop_samples)
        
        # Concatenate all loops
        extended_samples = np.concatenate(looped_samples_list, axis=0)
        
        # Trim to exact target duration
        target_samples = int(target_length_s * context_audio.sample_rate)
        if len(extended_samples) > target_samples:
            extended_samples = extended_samples[:target_samples]
        
        # Apply micro crossfades at loop boundaries to reduce clicking
        # Crossfade duration: 50-150ms
        crossfade_duration_s = 0.1  # 100ms default
        crossfade_samples = int(crossfade_duration_s * context_audio.sample_rate)
        
        # Apply crossfades between loop repetitions
        if num_loops > 1 and crossfade_samples > 0:
            loop_samples_count = len(loop_samples)
            
            for i in range(1, num_loops):
                loop_start_idx = i * loop_samples_count
                loop_end_idx = min(loop_start_idx + loop_samples_count, len(extended_samples))
                
                # Crossfade between previous loop end and current loop start
                fade_start = max(0, loop_start_idx - crossfade_samples)
                fade_end = min(len(extended_samples), loop_start_idx + crossfade_samples)
                
                if fade_end > fade_start:
                    fade_length = fade_end - fade_start
                    fade_curve = np.linspace(0, 1, fade_length)
                    
                    # Reshape for broadcasting if stereo
                    if extended_samples.ndim == 2:
                        fade_curve = fade_curve.reshape(-1, 1)
                    
                    # Apply crossfade
                    extended_samples[fade_start:fade_end] *= fade_curve
        
        # Create extension buffer
        extension_meta = AudioMetadata(
            source_path=context_audio.meta.source_path,
            sample_rate=context_audio.sample_rate,
            channels=context_audio.meta.channels,
            duration_s=target_length_s,
            format=context_audio.meta.format,
        )
        
        extension_buffer = AudioBuffer(
            samples=extended_samples,
            sample_rate=context_audio.sample_rate,
            meta=extension_meta,
        )
        
        # Step 2: Generate bassline
        # Estimate root note from context
        root_midi = estimate_root_note(context_audio, bpm)
        
        # Determine bass gain based on preset
        if preset == "dj_safe" or preset == "no_vocals":
            bass_gain_db = -15.0  # Subtle: -15 dB
        elif preset == "more_motion":
            bass_gain_db = -12.0  # Slightly louder: -12 dB
        else:
            bass_gain_db = -15.0  # Default to subtle
        
        # Generate bassline
        bassline = generate_bassline(
            sr=context_audio.sample_rate,
            bpm=bpm,
            bars=bars,
            seed=seed,
            root_midi=root_midi,
            preset=preset,
        )
        
        # Step 3: Mix bass into extension
        extended_with_bass = mix_bass_into_extension(
            extension=extension_buffer,
            bass=bassline,
            bass_gain_db=bass_gain_db,
        )
        
        # Create metadata
        metadata = GenerationMetadata(
            seed=seed,
            loop_len_bars=loop_len_bars,
            root_midi=root_midi,
            bass_gain_db=bass_gain_db,
            pattern=preset,
        )
        
        return extended_with_bass, metadata

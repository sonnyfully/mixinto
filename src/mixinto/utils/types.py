from dataclasses import dataclass
from typing import Self
import numpy as np
from pydantic import BaseModel, Field, model_validator


@dataclass
class AudioMetadata:
    source_path: str | None
    sample_rate: int
    channels: int
    duration_s: float
    format: str | None

@dataclass
class AudioBuffer:
    samples: np.ndarray
    sample_rate: int
    meta: AudioMetadata

    def __len__(self) -> int:
        return self.samples.shape[0]

    def to_mono(self) -> 'AudioBuffer':
        if self.meta.channels == 1:
            return self
        # Average across channels (axis=1 for (samples, channels) format)
        mono_samples = self.samples.mean(axis=1)
        # Reshape to (samples, 1) to maintain 2D shape
        if mono_samples.ndim == 1:
            mono_samples = mono_samples.reshape(-1, 1)
        # Create new metadata with channels=1
        from dataclasses import replace
        mono_meta = replace(self.meta, channels=1)
        return AudioBuffer(
            samples=mono_samples,
            sample_rate=self.sample_rate,
            meta=mono_meta
        )

    def length_s(self) -> float:
        return len(self) / self.sample_rate

    def trim(self, start_s: float, end_s: float) -> 'AudioBuffer':
        start_idx = int(start_s * self.sample_rate)
        end_idx = int(end_s * self.sample_rate)
        return AudioBuffer(
            samples=self.samples[start_idx:end_idx],
            sample_rate=self.sample_rate,
            meta=self.meta
        )

@dataclass
class BeatGrid:
    bpm: float
    beats_s: list[float]
    downbeats_s: list[float]
    confidence: float

    def seconds_per_beat(self) -> float:
        return 60 / self.bpm
    
    def nearest_beat(self, time_s: float) -> float:
        return min(self.beats_s, key=lambda x: abs(x - time_s))

@dataclass
class IntroProfile:
    start_s: float
    end_s: float
    energy_mean: float
    energy_std: float
    spectral_stability: float
    rhythm_stability: float
    vocal_presence: float
    mix_safety_score: float
    flags: list[str]

class ExtendRequest(BaseModel):
    input_path: str
    preset: str = "dj_safe"
    output_path: str | None = None

    target_bars: int | None = Field(default=None, ge=0, description="Number of bars to extend the intro by.")
    target_seconds: float | None = Field(default=None, ge=0)

    dry_run: bool = False
    overwrite: bool = False
    force_bpm: float | None = Field(default=None, ge=0)
    backend: str = Field(default="baseline", description="Generation backend (baseline, loop)")
    seed: int = Field(default=0, description="Random seed for deterministic generation")

    @model_validator(mode="after")
    def check_target(self) -> Self:
        if self.dry_run:
            return self


        bars_set = self.target_bars is not None
        secs_set = self.target_seconds is not None
        if bars_set == secs_set:
            raise ValueError("Either target_bars or target_seconds must be provided, but not both.")
        
        return self

class AnalysisConfig(BaseModel):
    """Configuration for audio analysis parameters."""
    # Tempo detection
    tempo_min: float = Field(default=60.0, ge=30.0, le=300.0)
    tempo_max: float = Field(default=200.0, ge=30.0, le=300.0)
    tempo_confidence_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    onset_hop_length: int = Field(default=512, ge=64)
    onset_frame_length: int = Field(default=2048, ge=256)
    use_tempo_hints: bool = Field(default=True)
    
    # Intro detection
    intro_min_bars: int = Field(default=8, ge=4)
    intro_max_bars: int = Field(default=16, ge=8)
    intro_candidate_bars: list[int] = Field(default_factory=lambda: [4, 8, 12, 16, 20])
    intro_stability_weights: list[float] = Field(default_factory=lambda: [0.4, 0.4, 0.2])
    change_point_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    
    # Stability scoring
    spectral_features: list[str] = Field(default_factory=lambda: ["chroma", "mfcc"])
    stability_window_size_bars: float = Field(default=1.0, ge=0.25)
    time_weight_decay: float = Field(default=0.9, ge=0.0, le=1.0)
    rhythm_tempo_drift_threshold: float = Field(default=2.0, ge=0.0)
    energy_variance_threshold: float = Field(default=0.2, ge=0.0, le=1.0)
    
    @model_validator(mode="after")
    def validate_tempo_range(self) -> Self:
        if self.tempo_min >= self.tempo_max:
            raise ValueError("tempo_min must be less than tempo_max")
        return self
    
    @model_validator(mode="after")
    def validate_stability_weights(self) -> Self:
        if len(self.intro_stability_weights) != 3:
            raise ValueError("intro_stability_weights must have exactly 3 elements")
        if abs(sum(self.intro_stability_weights) - 1.0) > 0.01:
            raise ValueError("intro_stability_weights must sum to approximately 1.0")
        return self


class Preset(BaseModel):
    name: str

    min_mix_safety_score: float = Field(default=0.5, ge=0, le=1)
    max_vocal_presence: float = Field(default=0.5, ge=0, le=1)
    min_beat_confidence: float = Field(default=0.5, ge=0, le=1)

    method: str = "loop"

    max_seam_error_ms: float = Field(default=25.0, ge=0)

    export_format: str = "wav"
    normalize: bool = True
    
    analysis_config: AnalysisConfig = Field(default_factory=AnalysisConfig)




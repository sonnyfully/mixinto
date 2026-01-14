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
        return AudioBuffer(
            samples=self.samples.mean(axis=1),
            sample_rate=self.sample_rate,
            meta=self.meta
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

    @model_validator(mode="after")
    def check_target(self) -> Self:
        if self.dry_run:
            return self


        bars_set = self.target_bars is not None
        secs_set = self.target_seconds is not None
        if bars_set == secs_set:
            raise ValueError("Either target_bars or target_seconds must be provided, but not both.")
        
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




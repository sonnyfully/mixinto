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


@dataclass
class FeatureTimeline:
    """Per-bar/per-window feature timeline across the track."""
    bar_count: int
    bar_start_times_s: list[float]
    tempo_confidence: list[float]
    rhythm_stability: list[float]
    spectral_stability: list[float]
    energy_consistency: list[float]
    vocal_presence: list[float]
    loop_seam_score: list[float]  # NEW: loopability score per window


@dataclass
class SegmentCandidate:
    """A candidate segment for extension."""
    start_bar: int
    end_bar: int
    start_s: float
    end_s: float
    bar_count: int


@dataclass
class SegmentScore:
    """Scoring details for a segment candidate."""
    segment: SegmentCandidate
    loopability: float
    tempo_confidence: float
    rhythm_stability: float
    spectral_stability: float
    energy_consistency: float
    vocal_penalty: float
    final_score: float
    flags: list[str]
    component_breakdown: dict[str, float]  # Detailed breakdown for debugging


@dataclass
class ExtendabilityProfile:
    """Segment-aware extendability profile replacing IntroProfile."""
    # Track-level summary
    track_extendability: float  # max(segment_score) over all candidates
    coverage: int  # number of viable segments (score >= threshold)
    confidence: float  # confidence in the top segment
    
    # Top-K segments
    top_segments: list[SegmentScore]  # Sorted by score (best first)
    
    # Feature timeline (for visualization/debugging)
    feature_timeline: FeatureTimeline | None
    
    # Best segment (for backward compatibility)
    best_segment: SegmentScore
    
    # Legacy compatibility fields (derived from best_segment)
    @property
    def start_s(self) -> float:
        return self.best_segment.segment.start_s
    
    @property
    def end_s(self) -> float:
        return self.best_segment.segment.end_s
    
    @property
    def mix_safety_score(self) -> float:
        return self.best_segment.final_score
    
    @property
    def spectral_stability(self) -> float:
        return self.best_segment.spectral_stability
    
    @property
    def rhythm_stability(self) -> float:
        return self.best_segment.rhythm_stability
    
    @property
    def vocal_presence(self) -> float:
        # vocal_penalty is (1 - vocal_presence), so invert it back
        return 1.0 - self.best_segment.vocal_penalty
    
    @property
    def energy_mean(self) -> float:
        # Approximate from component breakdown if available
        return self.best_segment.component_breakdown.get("energy_mean", 0.0)
    
    @property
    def energy_std(self) -> float:
        # Approximate from component breakdown if available
        return self.best_segment.component_breakdown.get("energy_std", 0.0)
    
    @property
    def flags(self) -> list[str]:
        return self.best_segment.flags

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
    
    # Segment-aware analysis (NEW)
    segment_candidate_lengths: list[int] = Field(default_factory=lambda: [8, 16, 32], description="Bar lengths for candidate segments")
    segment_hop_bars: int = Field(default=1, ge=1, description="Hop size in bars for candidate generation")
    segment_scoring_window_bars: int = Field(default=4, ge=1, description="Window size in bars for rolling feature aggregation")
    segment_base_resolution_bars: int = Field(default=1, ge=1, description="Base resolution for per-bar feature extraction")
    segment_search_region: str = Field(default="anywhere", description="Search region: 'anywhere', 'first_N_bars', 'pre_vocal_only'")
    segment_search_first_n_bars: int = Field(default=64, ge=8, description="If search_region='first_N_bars', use this many bars")
    segment_viable_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum score for a segment to be considered viable")
    segment_top_k: int = Field(default=5, ge=1, description="Number of top segments to return")
    
    # Segment scoring weights (NEW - replaces old mix safety weights)
    segment_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "loopability": 0.25,
            "tempo_confidence": 0.20,
            "rhythm_stability": 0.20,
            "spectral_stability": 0.20,
            "energy_consistency": 0.10,
            "vocal_penalty": 0.05,
        },
        description="Weights for segment scoring components"
    )
    
    # Loopability scoring
    loopability_seam_window_ms: float = Field(default=50.0, ge=0.0, description="Window in ms for seam quality check")
    loopability_min_score: float = Field(default=0.3, ge=0.0, le=1.0, description="Minimum loopability score to consider")
    
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
    
    @model_validator(mode="after")
    def validate_segment_weights(self) -> Self:
        """Validate that segment weights sum to approximately 1.0."""
        total = sum(self.segment_weights.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"segment_weights must sum to approximately 1.0, got {total:.3f}")
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




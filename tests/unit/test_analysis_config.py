"""Test analysis configuration functionality."""
import pytest

from mixinto.utils.types import AnalysisConfig, Preset


def test_analysis_config_defaults():
    """Test that AnalysisConfig has sensible defaults."""
    config = AnalysisConfig()
    
    assert config.tempo_min == 60.0
    assert config.tempo_max == 200.0
    assert config.tempo_confidence_threshold == 0.3
    assert config.intro_min_bars == 8
    assert config.intro_max_bars == 16
    assert len(config.intro_candidate_bars) > 0
    assert len(config.intro_stability_weights) == 3
    assert abs(sum(config.intro_stability_weights) - 1.0) < 0.01


def test_analysis_config_validation():
    """Test that AnalysisConfig validates input correctly."""
    # Test tempo range validation
    with pytest.raises(ValueError, match="tempo_min must be less than tempo_max"):
        AnalysisConfig(tempo_min=200.0, tempo_max=100.0)
    
    # Test stability weights validation
    with pytest.raises(ValueError, match="intro_stability_weights"):
        AnalysisConfig(intro_stability_weights=[0.5, 0.5])


def test_preset_with_analysis_config():
    """Test that Preset includes AnalysisConfig."""
    preset = Preset(name="test")
    
    assert preset.analysis_config is not None
    assert isinstance(preset.analysis_config, AnalysisConfig)


def test_preset_custom_analysis_config():
    """Test creating preset with custom analysis config."""
    custom_config = AnalysisConfig(
        tempo_min=80.0,
        tempo_max=160.0,
        intro_min_bars=12,
    )
    
    preset = Preset(
        name="custom",
        analysis_config=custom_config,
    )
    
    assert preset.analysis_config.tempo_min == 80.0
    assert preset.analysis_config.tempo_max == 160.0
    assert preset.analysis_config.intro_min_bars == 12

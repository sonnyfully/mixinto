# Mix Safety Score Calculation

This document explains in detail how the mix safety score is calculated, including all components, weights, modules, and algorithms used. Use this to understand and tweak the scoring system.

## Overview

The mix safety score is a composite metric (0.0 to 1.0) that evaluates how suitable an audio segment is for DJ mixing. Higher scores indicate better mix safety. The score is calculated in `src/mixinto/core/analyzer.py` after analyzing the intro window of a track.

## Calculation Formula

The mix safety score is a weighted sum of five components:

```python
mix_safety_score = (
    0.30 * tempo_confidence +                                    # Component 1
    0.25 * spectral_stability +                                  # Component 2
    0.25 * rhythm_stability +                                    # Component 3
    0.10 * (1.0 - vocal_presence) +                             # Component 4
    0.10 * (1.0 - min(1.0, energy_std / (energy_mean + 0.001))) # Component 5
)
```

**Total weights sum to 1.0** ✓

## Component Breakdown

### 1. Tempo Confidence (Weight: 30%)

**Location:** `src/mixinto/dsp/tempo/detector.py` → `detect_tempo()`

**What it measures:** How reliably the BPM was detected and how consistent the beat tracking is.

**Calculation:**
1. **Onset Strength Analysis:**
   - Uses `librosa.onset.onset_strength()` to detect rhythmic events
   - Parameters: `hop_length` (default: 512), `n_fft` (default: 2048)
   - Configurable via `AnalysisConfig.onset_hop_length` and `AnalysisConfig.onset_frame_length`

2. **Tempo Estimation:**
   - Uses `librosa.feature.rhythm.tempo()` (or `librosa.beat.tempo()` for older librosa)
   - Estimates BPM from onset envelope

3. **Confidence Calculation** (three factors combined):
   ```python
   # Factor 1: Onset strength consistency (40% weight)
   cv = std_strength / (mean_strength + 0.001)
   consistency_score = 1.0 / (1.0 + cv * 2.0)
   
   # Factor 2: Overall onset strength (40% weight)
   strength_score = min(1.0, mean_strength / 0.5)
   
   # Factor 3: Tempo reasonableness (20% weight)
   tempo_score = 1.0
   if tempo < 80 or tempo > 180:
       tempo_score = 0.8  # Slight penalty for extreme tempos
   
   confidence = 0.4 * consistency_score + 0.4 * strength_score + 0.2 * tempo_score
   ```

4. **Octave Correction:**
   - If tempo is outside `config.tempo_min` to `config.tempo_max`, tries doubling/halving
   - Default range: 60-200 BPM (configurable)

**Range:** 0.0 to 1.0 (clipped)

**Why it matters:** Unreliable tempo detection means the extension won't align properly with the original track.

---

### 2. Spectral Stability (Weight: 25%)

**Location:** `src/mixinto/features/stability.py` → `calculate_spectral_stability()`

**What it measures:** How consistent the timbre/spectral content is over time. Stable timbre = easier to loop/extend.

**Calculation:**

1. **Feature Extraction** (configurable via `config.spectral_features`, default: `["chroma", "mfcc"]`):
   
   **a) Chroma (Pitch Class) Stability:**
   ```python
   chroma = librosa.feature.chroma_stft(
       y=samples,
       sr=sample_rate,
       n_fft=config.onset_frame_length,  # default: 2048
       hop_length=config.onset_hop_length  # default: 512
   )
   # Shape: (12, time_frames) - 12 pitch classes
   
   chroma_std = np.std(chroma, axis=1)  # Std dev across time for each pitch class
   mean_std = np.mean(chroma_std)
   chroma_stability = 1.0 / (1.0 + mean_std * 10.0)
   ```
   - Lower variance in chroma = higher stability
   - Formula: `1 / (1 + mean_std * 10)` normalizes to [0, 1]

   **b) MFCC (Mel-Frequency Cepstral Coefficients) Stability:**
   ```python
   mfcc = librosa.feature.mfcc(
       y=samples,
       sr=sample_rate,
       n_fft=config.onset_frame_length,
       hop_length=config.onset_hop_length,
       n_mfcc=13
   )
   mfcc_stable = mfcc[:6, :]  # Use first 6 MFCCs (more stable)
   mfcc_std = np.std(mfcc_stable, axis=1)
   mean_std = np.mean(mfcc_std)
   mfcc_stability = 1.0 / (1.0 + mean_std * 5.0)
   ```
   - Uses first 6 MFCCs (lower-order = more timbral, less transient)
   - Formula: `1 / (1 + mean_std * 5)` (different scaling than chroma)

   **c) Tonnetz (Optional):**
   ```python
   tonnetz = librosa.feature.tonnetz(y=samples, sr=sample_rate, hop_length=hop_length)
   tonnetz_std = np.std(tonnetz, axis=1)
   mean_std = np.mean(tonnetz_std)
   tonnetz_stability = 1.0 / (1.0 + mean_std * 8.0)
   ```

2. **Combination:**
   ```python
   stability = np.mean([chroma_stability, mfcc_stability, ...])  # Average of all enabled features
   ```

**Range:** 0.0 to 1.0 (clipped)

**Why it matters:** Unstable timbre (e.g., lots of instrument changes, effects) makes looping sound obvious.

**Configuration:**
- `config.spectral_features`: List of features to use (default: `["chroma", "mfcc"]`)
- `config.time_weight_decay`: Placeholder for future time-weighting (currently unused)

---

### 3. Rhythm Stability (Weight: 25%)

**Location:** `src/mixinto/features/stability.py` → `calculate_rhythm_stability()`

**What it measures:** How consistent the beat intervals are. Stable rhythm = predictable, mixable.

**Calculation:**

1. **Inter-Beat Interval Analysis:**
   ```python
   intervals = np.diff(beat_grid.beats_s)  # Time differences between consecutive beats
   expected_interval = 60.0 / beat_grid.bpm  # Expected interval in seconds
   ```

2. **Coefficient of Variation (CV):**
   ```python
   mean_interval = np.mean(intervals)
   std_interval = np.std(intervals)
   cv = std_interval / mean_interval  # Lower CV = more stable
   ```

3. **Tempo Drift Detection** (if ≥8 beats available):
   ```python
   # Split intervals into 4 segments
   segment_size = len(intervals) // 4
   segment_means = [np.mean(intervals[i:i+segment_size]) for i in range(0, len(intervals), segment_size)]
   
   # Calculate drift in BPM
   max_drift = max(segment_means) - min(segment_means)
   drift_bpm = (max_drift / expected_interval) * beat_grid.bpm
   
   # Penalize if drift exceeds threshold (default: 2.0 BPM)
   if drift_bpm > config.rhythm_tempo_drift_threshold:
       drift_penalty = min(1.0, drift_bpm / (config.rhythm_tempo_drift_threshold * 2))
       cv *= (1.0 + drift_penalty)  # Increase CV (reduce stability)
   ```

4. **Stability Score:**
   ```python
   stability = 1.0 / (1.0 + cv * 5.0)
   ```
   - Lower CV → higher stability
   - Formula normalizes CV to [0, 1] range

**Range:** 0.0 to 1.0 (clipped)

**Why it matters:** Inconsistent rhythm (e.g., live drums, tempo changes) makes extension sound off-beat.

**Configuration:**
- `config.rhythm_tempo_drift_threshold`: Maximum acceptable tempo drift in BPM (default: 2.0)

---

### 4. Vocal Presence (Weight: 10%)

**Location:** `src/mixinto/features/vocals.py` → `detect_vocal_presence()`

**What it measures:** How much vocal content is present. Lower vocals = better for mixing (less distracting).

**Calculation:**

1. **Spectral Feature Extraction:**
   ```python
   spectral_centroid = librosa.feature.spectral_centroid(y=samples, sr=sample_rate)[0]
   spectral_rolloff = librosa.feature.spectral_rolloff(y=samples, sr=sample_rate)[0]
   ```

2. **Normalization:**
   ```python
   # Centroid: typical range 1000-4000 Hz for music
   centroid_score = min(1.0, (mean_centroid - 1000) / 3000)
   centroid_score = max(0.0, centroid_score)
   
   # Rolloff: typical range 2000-8000 Hz
   rolloff_score = min(1.0, (mean_rolloff - 2000) / 6000)
   rolloff_score = max(0.0, rolloff_score)
   ```

3. **Combination:**
   ```python
   vocal_score = 0.7 * centroid_score + 0.3 * rolloff_score
   ```

4. **Inversion for Mix Safety:**
   ```python
   mix_safety_contribution = 0.10 * (1.0 - vocal_presence)
   ```
   - Higher vocal presence → lower mix safety contribution
   - Formula: `1.0 - vocal_presence` inverts the score

**Range:** 0.0 to 1.0 (clipped)

**Why it matters:** Vocals are distracting during mixing and make looping obvious.

**Note:** This is a heuristic-based approach. More sophisticated methods (e.g., source separation, vocal activity detection) could be added later.

---

### 5. Energy Consistency (Weight: 10%)

**Location:** `src/mixinto/features/stability.py` → `calculate_energy_features()`

**What it measures:** How consistent the energy (volume) is over time. Consistent energy = stable, mixable.

**Calculation:**

1. **RMS Energy Calculation:**
   ```python
   rms = librosa.feature.rms(
       y=samples,
       frame_length=config.onset_frame_length,  # default: 2048
       hop_length=config.onset_hop_length       # default: 512
   )[0]
   ```

2. **Statistics:**
   ```python
   energy_mean = np.mean(rms)
   energy_std = np.std(rms)
   ```

3. **Energy Ramp Detection** (if significant changes detected):
   ```python
   # Detect energy builds/ramps using rolling windows
   window_size = min(10, len(rms) // 4)
   energy_changes = []
   for i in range(window_size, len(rms)):
       prev_mean = np.mean(rms[i-window_size:i])
       curr_mean = np.mean(rms[i-window_size//2:i+window_size//2])
       if prev_mean > 0:
           change = (curr_mean - prev_mean) / prev_mean
           energy_changes.append(abs(change))
   
   # If max change exceeds threshold, increase variance estimate
   if max_change > config.energy_variance_threshold:  # default: 0.2
       std_energy *= (1.0 + max_change)
   ```

4. **Consistency Score:**
   ```python
   # Coefficient of variation (CV) normalized
   energy_cv = energy_std / (energy_mean + 0.001)
   consistency = 1.0 - min(1.0, energy_cv)
   mix_safety_contribution = 0.10 * consistency
   ```
   - Lower CV → higher consistency
   - Formula: `1.0 - min(1.0, energy_std / energy_mean)` normalizes to [0, 1]

**Range:** 0.0 to 1.0 (clipped)

**Why it matters:** Energy ramps/builds indicate the track is changing, making it harder to loop seamlessly.

**Configuration:**
- `config.energy_variance_threshold`: Threshold for detecting significant energy changes (default: 0.2)

---

## Intro Window Selection

Before calculating the mix safety score, the system must select which part of the track to analyze. This happens in `src/mixinto/dsp/segments/intro.py` → `detect_intro_window()`.

**Process:**

1. **Candidate Windows:**
   - Evaluates multiple bar-length candidates (default: `[4, 8, 12, 16, 20]` bars)
   - Constrained by `config.intro_min_bars` (default: 8) and `config.intro_max_bars` (default: 16)

2. **Scoring Each Candidate:**
   ```python
   # For each candidate window:
   spectral_stability = calculate_spectral_stability(candidate_buffer, config)
   rhythm_stability = calculate_rhythm_stability(candidate_buffer, window_beat_grid, config)
   energy_mean, energy_std = calculate_energy_features(candidate_buffer, config)
   energy_consistency = 1.0 / (1.0 + (energy_std / energy_mean) * 2.0)
   
   # Change point penalty
   if energy_std / (energy_mean + 0.001) > config.change_point_threshold:  # default: 0.3
       change_penalty = 0.2
   
   # Combined score
   weights = config.intro_stability_weights  # default: [0.4, 0.4, 0.2]
   combined_score = (
       weights[0] * spectral_stability +
       weights[1] * rhythm_stability +
       weights[2] * energy_consistency
   ) - change_penalty
   ```

3. **Selection:**
   - Chooses the window with the highest `combined_score`
   - This window is then used for the final mix safety score calculation

**Configuration:**
- `config.intro_min_bars`: Minimum bars for intro (default: 8)
- `config.intro_max_bars`: Maximum bars for intro (default: 16)
- `config.intro_candidate_bars`: List of bar counts to evaluate (default: `[4, 8, 12, 16, 20]`)
- `config.intro_stability_weights`: Weights for `[spectral, rhythm, energy]` (default: `[0.4, 0.4, 0.2]`)
- `config.change_point_threshold`: Energy variance threshold for change detection (default: 0.3)

---

## Flags and Warnings

After calculating the mix safety score, the system generates flags for potential issues:

```python
flags = []
if tempo_confidence < 0.5:
    flags.append("low_beat_confidence")
if spectral_stability < 0.5:
    flags.append("low_spectral_stability")
if rhythm_stability < 0.5:
    flags.append("low_rhythm_stability")
if vocal_presence > 0.5:
    flags.append("high_vocal_presence")
if energy_std > energy_mean * 2.0:
    flags.append("high_energy_variance")
```

These flags are included in the `IntroProfile` and can be used for debugging or user feedback.

---

## Configuration Parameters

All parameters are configurable via `AnalysisConfig` (defined in `src/mixinto/utils/types.py`):

### Tempo Detection
- `tempo_min`: Minimum BPM (default: 60.0)
- `tempo_max`: Maximum BPM (default: 200.0)
- `tempo_confidence_threshold`: Minimum confidence for valid detection (default: 0.3)
- `onset_hop_length`: Hop length for onset detection (default: 512)
- `onset_frame_length`: Frame length for onset detection (default: 2048)
- `use_tempo_hints`: Enable octave correction (default: True)

### Intro Detection
- `intro_min_bars`: Minimum bars for intro (default: 8)
- `intro_max_bars`: Maximum bars for intro (default: 16)
- `intro_candidate_bars`: Bar counts to evaluate (default: `[4, 8, 12, 16, 20]`)
- `intro_stability_weights`: Weights for `[spectral, rhythm, energy]` (default: `[0.4, 0.4, 0.2]`)
- `change_point_threshold`: Energy variance threshold (default: 0.3)

### Stability Scoring
- `spectral_features`: Features to use (default: `["chroma", "mfcc"]`)
- `stability_window_size_bars`: Window size for stability (default: 1.0)
- `time_weight_decay`: Time weighting decay (default: 0.9, currently unused)
- `rhythm_tempo_drift_threshold`: Max tempo drift in BPM (default: 2.0)
- `energy_variance_threshold`: Energy change threshold (default: 0.2)

---

## Current Weight Distribution

| Component | Weight | Rationale |
|-----------|--------|-----------|
| Tempo Confidence | 30% | Most critical - unreliable tempo = unusable |
| Spectral Stability | 25% | Important - unstable timbre = obvious loops |
| Rhythm Stability | 25% | Important - inconsistent rhythm = off-beat |
| Vocal Presence | 10% | Moderate - vocals are distracting but not fatal |
| Energy Consistency | 10% | Moderate - energy ramps are problematic but manageable |

**Total:** 100%

---

## Potential Issues and Tuning

### Issue: Score too low for good intros

**Possible causes:**
1. **Tempo confidence too strict:** The 30% weight might be penalizing tracks with subtle beats
2. **Spectral stability too sensitive:** The normalization factors (`* 10.0`, `* 5.0`) might be too aggressive
3. **Energy consistency penalizing builds:** Energy ramps are common in intros but are penalized
4. **Intro window selection:** The selected window might not be the best part

**Tuning suggestions:**
1. **Adjust weights:** Increase weight of components that are scoring well, decrease problematic ones
2. **Relax thresholds:** Lower `tempo_confidence_threshold`, `change_point_threshold`, etc.
3. **Normalization factors:** Adjust the `* 10.0`, `* 5.0` factors in stability calculations
4. **Energy ramp handling:** Make energy ramps less penalizing if they're gradual
5. **Intro window weights:** Adjust `intro_stability_weights` to favor different features

### Issue: Score too high for bad intros

**Possible causes:**
1. **Vocal detection too lenient:** The heuristic might miss subtle vocals
2. **Rhythm stability not catching subtle changes:** The CV calculation might miss gradual tempo drift
3. **Spectral stability missing timbre changes:** The features might not capture certain types of instability

**Tuning suggestions:**
1. **Stricter vocal detection:** Lower the normalization ranges or add more features
2. **More sensitive rhythm detection:** Reduce `rhythm_tempo_drift_threshold` or improve drift detection
3. **Additional spectral features:** Add `tonnetz` or other features to `spectral_features`

---

## Code Locations Summary

| Component | File | Function |
|-----------|------|----------|
| Main calculation | `src/mixinto/core/analyzer.py` | `analyze_audio()` (lines 80-89) |
| Tempo confidence | `src/mixinto/dsp/tempo/detector.py` | `detect_tempo()` |
| Spectral stability | `src/mixinto/features/stability.py` | `calculate_spectral_stability()` |
| Rhythm stability | `src/mixinto/features/stability.py` | `calculate_rhythm_stability()` |
| Vocal presence | `src/mixinto/features/vocals.py` | `detect_vocal_presence()` |
| Energy features | `src/mixinto/features/stability.py` | `calculate_energy_features()` |
| Intro window selection | `src/mixinto/dsp/segments/intro.py` | `detect_intro_window()` |
| Configuration | `src/mixinto/utils/types.py` | `AnalysisConfig` class |

---

## Next Steps for Tuning

1. **Add logging:** Log individual component scores to see which ones are causing low scores
2. **Test on known good/bad intros:** Create a test suite with manually labeled intros
3. **Adjust weights incrementally:** Change one weight at a time and observe impact
4. **Consider new features:** Add features like harmonic-percussive separation, onset density, etc.
5. **Preset-specific tuning:** Different presets might need different weights/thresholds

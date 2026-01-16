# mixinto

mixinto is a DJ-first tool that extends a track’s intro into a longer, mix-safe lead-in.

Instead of simply looping audio, mixinto aims to generate *new* intro bars via **constrained continuation**: it tries to preserve the track’s existing “intro state” (tempo, groove, timbre, energy) while adding time you can comfortably mix into.

This repo is at **first commit** stage. The initial focus is establishing a clean system design, narrow MVP scope, and a reliable pipeline that can say “no” when a track isn’t safe to extend.

---

## What it will do

Given a song file, mixinto will:
- Detect BPM + downbeats and build a bar grid
- Identify a stable intro window suitable for extension
- Generate multiple candidate intro continuations
- Score and rerank candidates for DJ mix safety (tempo-lock, low novelty, seam quality)
- Export an extended intro track (e.g., +16 / +32 / +64 bars)
- Emit a report describing confidence, metrics, and failure reasons

---

## Why this exists

DJs often need more time to mix into a track than the original intro provides. Extended mixes exist for some releases, but not most. Manually editing intros is time-consuming and looping can sound obvious and lifeless.

mixinto’s goal is to create extensions that are:
- **tempo-locked**
- **bar-aligned**
- **timbre-consistent**
- **low novelty / mix-safe**
- **seam-clean** (no obvious join artifacts)

---

## Approach (high level)

mixinto is a **constrained generation system**, not a “press button, get random music” demo.

Planned pipeline:
1. **Analyze**
   - BPM + downbeats (with confidence)
   - intro stability scoring
   - basic vocal/lead detection (for strict modes)
2. **Select context**
   - choose an intro window (e.g., 8–16 bars) that’s stable and representative
3. **Generate candidates**
   - use an audio continuation / inpainting backbone to generate K candidates
4. **Evaluate + rerank**
   - hard reject unsafe candidates
   - score the rest across mix-safety metrics
5. **Seam handling**
   - overlap/crossfade or seam inpainting for clean joins
6. **Export + report**
   - WAV output + JSON metrics/warnings/refusal reasons

The “impressive” part is the **control loop**: metrics, thresholds, reranking, and refusal modes that turn generation into something usable.

---

## MVP scope (initial)

The MVP will intentionally target a narrow subset:
- Beat-stable, 4/4 tracks
- Dance/electronic intros (house/techno/trance; others only if grid is stable)
- Local execution (no cloud requirement)
- No training on copyrighted commercial catalogs

If a track can’t be extended safely, the tool should refuse with clear reasons.

---

## Usage

### Analyze a track

Analyze a track to detect BPM, beat grid, and intro characteristics:

```bash
mixinto analyze input.wav
mixinto analyze input.wav --json --report analysis.json
```

### Extend a track

Extend a track's intro by adding bars with a DJ-safe bassline:

```bash
# Basic usage: extend by 32 bars using baseline generator
mixinto extend input.wav --bars 32 --out output.wav

# Specify preset and backend
mixinto extend input.wav --bars 16 --preset dj_safe --backend baseline --out output.wav

# Use deterministic seed for reproducible results
mixinto extend input.wav --bars 32 --seed 42 --out output.wav

# Use random seed for varied results
mixinto extend input.wav --bars 32 --seed random --out output.wav

# Generate JSON report
mixinto extend input.wav --bars 32 --report report.json --out output.wav
```

**Options:**
- `--bars, -b`: Number of bars to extend (required, or use `--seconds`)
- `--seconds, -s`: Number of seconds to extend (alternative to `--bars`)
- `--preset, -p`: Preset name (`dj_safe`, `dj_safe_strict`, `dj_safe_lenient`) - default: `dj_safe`
- `--backend`: Generation backend (`baseline`, `loop`) - default: `baseline`
- `--seed`: Random seed for deterministic generation (integer or `random` for random seed) - default: `0`
- `--out, -o`: Output file path (defaults to `output/{input_filename}_extended.wav`)
- `--report, -r`: Path for JSON report file
- `--overwrite`: Overwrite output file if it exists

**Baseline Generator:**
The baseline generator creates extension audio by:
1. Looping a bar-aligned slice from the intro context window
2. Adding a subtle, DJ-safe bassline that matches the track's root note
3. Applying crossfades to ensure seamless joins

The bassline is deterministic (controlled by seed), conservative in volume, and designed to sit under the original audio without adding distracting elements.
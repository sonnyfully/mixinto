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

## Planned interface

CLI (initial):

```bash
mixinto extend input.wav --bars 32 --preset dj_safe --out output.wav
mixinto analyze input.wav --json
mixinto batch ./tracks --bars 32 --preset dj_safe --out ./out
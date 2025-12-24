# Phonklean

Phonklean — a small, practical tool for restoring phonk tracks that have been overly saturated, clipped, or damaged by heavy "90s" effects.
<p align="center"><img width="256" height="256" alt="None" src="https://github.com/user-attachments/assets/d3d6aae2-b134-49d6-8b91-a06966023dca" /></p>  

Based on my `censorMyPy` project (which focused on censoring explicit lyrics), this project shifts focus to audio repair: recovering musical detail lost to distortion and clipping common in phonk and similar 2025-era genres.

What it does
- Reconstructs lost detail using the htdemucs transformer inpainting model.
- Reduces clipping and 90s-style saturation artifacts while preserving character.
- **Removes drum bleed and gating artifacts** from vocals using harmonic-percussive source separation (HPSS).
- Works on full mixes or separated stems.

How it works (overview)
1. Separate stems with htdemucs.
2. Detect clipped or distorted regions.
3. Use the transformer inpainting model to reconstruct missing waveform detail.
4. Blend and export a restored mix.

Quick start
- Place your audio in the project folder and run:

```bash
python phonklean.py input.wav
```

Notes
- Experimental — results vary by source material and severity of clipping.
- Best-effort approach: aims to recover musical detail without removing the genre's character.
- Inspired by and built on ideas from `censorMyPy`.

Smart Autotune (on vocals)
- **What:** Optional, vocals-only pitch-correction that runs independently of the other vocal restoration steps.
- **Example:**
```bash
python phonklean.py my_song.mp3 --autotune-vocals --autotune-key auto --autotune-strength 0.8
```

Usage & CLI options
- `input_file` : positional path to the input audio file (WAV/MP3). If omitted, defaults to `Song_name.mp3`.
- `--no-vocals`, `-nv` : skip all vocal-processing steps (useful when you only want instrument restoration).
- `--vf-mode`, `-vfm [0|1|2]` : selects `VoiceFixer` restoration mode when available (0 default, 1 additional pre-processing, 2 train mode).
- `--autotune-vocals` : enable vocals-only pitch correction (writes `vocals_autotuned.wav`).
- `--autotune-key [KEY|auto]` : root key for autotune; use `auto` to detect key from the vocals stem.
- `--autotune-scale [chromatic|major|minor]` : scale used for snapping pitches. If `chromatic`, detected scale may be used when key=`auto`.
- `--autotune-strength [0.0-1.0]` : control how strongly notes are corrected (0 = passthrough, 1 = full correction).
- `--bleed-removal` : enable HPSS-based drum bleed removal from vocals (removes over-gated drum artifacts).
- `--bleed-margin [0.5-4.0]` : HPSS separation margin in dB (higher = stricter separation; default 2.0).
- `--bleed-blend [0.0-1.0]` : percussive blend ratio (0.0 = remove all transients, 0.5 = 50%% blend; default 0.3).

Examples

```bash
# Basic restore (auto separation + restore)
python phonklean.py my_song.mp3

# Run only instrument restoration, skip vocals
python phonklean.py my_song.mp3 --no-vocals

# Enable smart autotune on detected key with moderate strength
python phonklean.py my_song.mp3 --autotune-vocals --autotune-key auto --autotune-strength 0.6

# Remove drum bleed from vocals (default blend 30%)
python phonklean.py my_song.mp3 --bleed-removal

# Aggressive drum bleed removal (10% percussive blend, tight separation)
python phonklean.py my_song.mp3 --bleed-removal --bleed-blend 0.1 --bleed-margin 3.0
```

License
- MIT — use and adapt freely; attribution appreciated.

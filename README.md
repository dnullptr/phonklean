# Phonklean

Phonklean — a small, practical tool for restoring phonk tracks that have been overly saturated, clipped, or damaged by heavy "90s" effects.
<p align="center"><img width="256" height="256" alt="None" src="https://github.com/user-attachments/assets/d3d6aae2-b134-49d6-8b91-a06966023dca" /></p>  

Based on my `censorMyPy` project (which focused on censoring explicit lyrics), this project shifts focus to audio repair: recovering musical detail lost to distortion and clipping common in phonk and similar 2025-era genres.

What it does
- Reconstructs lost detail using the htdemucs transformer inpainting model.
- Reduces clipping and 90s-style saturation artifacts while preserving character.
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

License
- MIT — use and adapt freely; attribution appreciated.

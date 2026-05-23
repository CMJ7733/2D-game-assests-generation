# PixelForge

**Text → 2D pixel-art game character with idle + walk animation, exported to your game engine in 2 minutes.**

![sprite sheet](assets/examples/knight.png)

## What it is

PixelForge generates **consistent, animatable** character sprite sheets from natural-language descriptions.
Unlike single-shot text-to-image tools, PixelForge produces a **12-frame sprite sheet** (4 idle + 8 walk-cycle)
with the same character across all poses, and exports directly into:

- Generic PNG + JSON (Phaser, custom engines)
- Godot 4 SpriteFrames (.tres) — drag and drop
- Unity sprite sheet + .meta

## Tech

Local Stable Diffusion 1.5 + ControlNet OpenPose + IP-Adapter (identity locking) + 12-pose pre-built skeleton library.
Replicate / fal.ai fallback if local inference is unavailable.

Tested on MacBook M-series 16GB.

## Quick start

See [docs/INSTALL.md](docs/INSTALL.md).

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
./scripts/download_models.sh
python -m pixelforge.app
```

Open http://127.0.0.1:7860.

## Architecture

10-module pipeline: prompt_engineer → reference_builder → pose_library → frame_generator → post_processor → sheet_composer → exporter.
See [docs/superpowers/specs/2026-05-23-pixelforge-design.md](docs/superpowers/specs/2026-05-23-pixelforge-design.md).

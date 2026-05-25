<div align="center">

# 🔥 PixelForge

**Text → 4-View Pixel Art Character Reference Sheet**

*Generate consistent front / left / right / back views of a pixel art character from a single text description — all running locally on your Mac.*

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Stable Diffusion 1.5](https://img.shields.io/badge/SD-1.5-green.svg)](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)
[![Gradio](https://img.shields.io/badge/UI-Gradio-orange.svg)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 🎮 What It Does

Give PixelForge a character description in plain English (or Chinese) and it generates a **4-view character reference sheet** — front, left, right, and back — in a consistent pixel art style.

[Image 1]

Each view uses view-specific trigger tokens fine-tuned into the base model, ensuring the character faces the correct direction while maintaining a unified aesthetic.

---

## 🧩 Character Sheet Layout

| Back | Left |
|:---:|:---:|
| [Image 2] | [Image 3] |
| Front | Right |
| [Image 4] | [Image 5] |

All four views are composed into a single downloadable character sheet PNG, ready for your game project.

---

## 🏗️ Architecture

```
User Prompt
    │
    ▼
┌──────────────────┐
│  prompt_engineer  │  View-specific triggers + anchors
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ reference_builder │  SD 1.5 text-to-image × 4 views
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  post_processor   │  Quantize → Upscale → Snap
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   app.py (Gradio) │  2×2 grid UI + download
└──────────────────┘
```

| Module | Role |
|--------|------|
| `prompt_engineer` | Injects view-specific trigger tokens (`PixelartFSS/LSS/RSS/BSS`) + perspective anchors |
| `reference_builder` | Loads SD 1.5 pixel art model, generates one image per view |
| `post_processor` | Color quantization → LANCZOS upscale → pixel snap for clean pixel art |
| `orchestrator` | Coordinates the full 4-view pipeline with progress callbacks |
| `app.py` | Gradio web UI with 2×2 image grid, stop button, and character sheet download |

---

## 🎨 Pixel Art Model

Powered by [**Onodofthenorth/SD_PixelArt_SpriteSheet_Generator**](https://huggingface.co/Onodofthenorth/SD_PixelArt_SpriteSheet_Generator) — a Stable Diffusion 1.5 fine-tune specifically for pixel art sprite generation. This model includes dedicated trigger tokens for different character viewpoints.

## 📐 Inspiration from LPC

The 4-view reference sheet format is inspired by the [**Liberated Pixel Cup (LPC)**](https://lpc.opengameart.org/) project and its sprite sheet conventions. LPC established the standard of providing character art from multiple directions (front, back, left, right) to enable 2D game animation. PixelForge adapts this concept — instead of hand-drawing each view, we generate them via diffusion with view-specific prompts.

---

## ⚡ Quick Start

### Prerequisites

- **macOS** with Apple Silicon (M1/M2/M3/M4) — MPS acceleration
- **Python 3.11**
- **~4 GB RAM** for model inference (float32 on MPS)
- **~4 GB disk** for cached model weights

### Install

```bash
# Clone
git clone https://github.com/<your-username>/2D-game.git
cd 2D-game

# Setup
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"

# Download models (~4 GB)
./scripts/download_models.sh
```

### Run

```bash
python -m pixelforge.app
```

Open **http://127.0.0.1:7860** in your browser, type a character description, and hit **Generate**.

---

## 🖥️ UI Preview

[Image 6]

- Dark theme with emerald accents
- 2×2 image grid with labeled views
- Real-time progress bar per view
- One-click character sheet PNG download
- Stop button to cancel generation mid-way

---

## 🔧 Configuration

All settings live in `src/pixelforge/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sd_model_id` | `Onodofthenorth/SD_PixelArt_SpriteSheet_Generator` | HuggingFace model ID |
| `image_size` | `512` | Generation resolution (per view) |
| `num_inference_steps` | `20` | SD denoising steps |
| `guidance_scale` | `7.5` | Classifier-free guidance strength |
| `seed` | `42` | Reproducible generation seed |
| `target_sprite_size` | `(256, 256)` | Post-processed output size per view |
| `palette_colors` | `32` | Color quantization level |
| `device` | `mps` | Compute device (`mps` / `cpu`) |
| `dtype` | `float32` | Model precision (float16 causes NaN on MPS with .bin models) |

---

## 📂 Project Structure

```
2D-game/
├── src/pixelforge/
│   ├── app.py              # Gradio web UI
│   ├── orchestrator.py     # Pipeline coordination
│   ├── reference_builder.py # SD 1.5 inference
│   ├── prompt_engineer.py  # View-specific prompt engineering
│   ├── post_processor.py   # Quantize + upscale + snap
│   └── config.py           # Central configuration
├── scripts/
│   └── download_models.sh  # Model download helper
├── tests/                  # Pytest test suite
├── output/                 # Generated character sheets
└── docs/                   # Design docs & specs
```

---

## 🧪 Testing

```bash
pytest tests/ -v
```

---

## ⚠️ Known Limitations

- **MPS float16 NaN** — The base model uses `.bin` format which produces NaN on Apple MPS with float16. float32 is required (~4 GB RAM).
- **View consistency** — Without IP-Adapter, each view is independently generated; character identity may vary slightly across views.
- **English prompts recommended** — The pixel art model was trained primarily on English captions; non-English input is passed through as-is.

---

## 📜 License

This project is licensed under the MIT License. The base pixel art model follows its own [license terms](https://huggingface.co/Onodofthenorth/SD_PixelArt_SpriteSheet_Generator).

---

<div align="center">

*Built with ❤️ for the pixel art game dev community*

</div>

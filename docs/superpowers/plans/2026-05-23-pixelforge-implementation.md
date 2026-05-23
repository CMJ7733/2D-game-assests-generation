# PixelForge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a text-to-pixel-art-character-sprite generator with idle+walk animations, exporting to Sprite Sheet PNG/JSON, Godot .tres, and Unity .meta, in 72 hours on a 16GB M-chip MacBook.

**Architecture:** Gradio UI orchestrates a 10-module pipeline (prompt → reference image → frame generation with IP-Adapter + ControlNet pose library → pixel post-processing → sprite sheet → multi-format export). Local SD1.5 via Diffusers/MPS is primary; Replicate API is fallback. A Quick Mode (single-image sprite sheet) acts as Plan B when consistency fails.

**Tech Stack:** Python 3.11, uv (package manager), diffusers + transformers + accelerate (PyTorch MPS), ControlNet OpenPose, IP-Adapter, rembg/RMBG-1.4, Pillow, scikit-image, Gradio 4, Jinja2 (templates), Replicate SDK (fallback), Loguru.

**Reference spec:** `docs/superpowers/specs/2026-05-23-pixelforge-design.md`

---

## Phase 0 — Project Setup (target: 0h–2h)

### Task 1: Initialize repository & Python environment

**Files:**
- Create: `pyproject.toml`
- Create: `.python-version`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `README.md`

- [ ] **Step 1: Initialize git**

Run from project root:
```bash
git init
git branch -m main
```

- [ ] **Step 2: Set Python version**

Create `.python-version`:
```
3.11
```

- [ ] **Step 3: Create pyproject.toml**

Create `pyproject.toml`:
```toml
[project]
name = "pixelforge"
version = "0.1.0"
description = "Text-to-pixel-art game character sprite generator"
requires-python = ">=3.11,<3.12"
dependencies = [
    "torch>=2.3.0",
    "diffusers>=0.27.0",
    "transformers>=4.40.0",
    "accelerate>=0.30.0",
    "safetensors>=0.4.0",
    "controlnet-aux>=0.0.7",
    "Pillow>=10.0.0",
    "rembg>=2.0.50",
    "numpy>=1.26.0,<2.0",
    "scikit-image>=0.22.0",
    "opencv-python-headless>=4.9.0",
    "replicate>=0.25.0",
    "httpx>=0.27.0",
    "openai>=1.30.0",
    "gradio>=4.30.0",
    "jinja2>=3.1.0",
    "pyyaml>=6.0",
    "pydantic>=2.7.0",
    "loguru>=0.7.0",
    "rich>=13.7.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-asyncio", "ruff>=0.4.0"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/pixelforge"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

- [ ] **Step 4: Create .gitignore**

Create `.gitignore`:
```
__pycache__/
*.pyc
.venv/
.env
models/
output/
.cache/
.DS_Store
.pytest_cache/
*.egg-info/
dist/
build/
```

- [ ] **Step 5: Create .env.example**

Create `.env.example`:
```
# HuggingFace mirror — defaults to China mirror
HF_ENDPOINT=https://hf-mirror.com

# Fallback API providers (optional, only used if local engine fails)
REPLICATE_API_TOKEN=
FAL_API_KEY=

# Prompt translation (optional)
OPENAI_API_KEY=
```

- [ ] **Step 6: Create venv and install**

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
```
Expected: `Successfully installed ...` ending without errors.

- [ ] **Step 7: Sanity-check MPS available**

```bash
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
```
Expected: `MPS: True`

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml .python-version .gitignore .env.example
git commit -m "chore: bootstrap project with Python 3.11 + dependencies"
```

---

### Task 2: Create source tree skeleton

**Files:**
- Create: `src/pixelforge/__init__.py`
- Create: `src/pixelforge/config.py`
- Create: empty stubs for all 10 modules
- Create: `tests/__init__.py`

- [ ] **Step 1: Create directories**

```bash
mkdir -p src/pixelforge assets/poses assets/templates assets/examples models output tests scripts
touch src/pixelforge/__init__.py tests/__init__.py
```

- [ ] **Step 2: Create config.py**

Create `src/pixelforge/config.py`:
```python
"""Central configuration. Imported before diffusers/transformers to set HF endpoint."""
import os

# MUST be set before any HuggingFace library imports.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from pathlib import Path
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).parent.parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"
POSES_DIR = ASSETS_DIR / "poses"
TEMPLATES_DIR = ASSETS_DIR / "templates"
EXAMPLES_DIR = ASSETS_DIR / "examples"


class GenerationConfig(BaseModel):
    sd_model_id: str = "runwayml/stable-diffusion-v1-5"
    controlnet_model_id: str = "lllyasviel/sd-controlnet-openpose"
    ip_adapter_repo: str = "h94/IP-Adapter"
    ip_adapter_subfolder: str = "models"
    ip_adapter_weight_name: str = "ip-adapter_sd15.bin"
    pixel_lora_repo: str | None = None  # Set after downloading a suitable pixel LoRA
    rmbg_model_id: str = "briaai/RMBG-1.4"

    image_size: int = 512
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    seed: int = 42

    ip_adapter_scale: float = 0.7
    controlnet_conditioning_scale: float = 0.9

    target_sprite_size: tuple[int, int] = (64, 64)
    palette_colors: int = 24

    device: str = "mps"
    dtype: str = "float16"


DEFAULT_CONFIG = GenerationConfig()


def ensure_dirs() -> None:
    for d in (ASSETS_DIR, MODELS_DIR, OUTPUT_DIR, POSES_DIR, TEMPLATES_DIR, EXAMPLES_DIR):
        d.mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 3: Create empty module stubs**

Create each of these files containing only a module docstring (one-line):
```bash
for m in prompt_engineer reference_builder pose_library frame_generator post_processor sheet_composer exporter engine_router quick_mode app; do
    echo "\"\"\"$m module.\"\"\"" > "src/pixelforge/${m}.py"
done
```

- [ ] **Step 4: Verify import**

```bash
python -c "from pixelforge import config; config.ensure_dirs(); print('OK')"
```
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add src tests assets/.gitkeep 2>/dev/null; git add src tests
git commit -m "chore: scaffold module skeleton + config"
```

---

### Task 3: Model downloader script

**Files:**
- Create: `scripts/download_models.sh`
- Create: `scripts/download_models.py`

- [ ] **Step 1: Write Python downloader**

Create `scripts/download_models.py`:
```python
"""Pre-download all required models to local cache via hf-mirror by default."""
import os
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from huggingface_hub import snapshot_download
from pathlib import Path

MODELS = [
    ("runwayml/stable-diffusion-v1-5", None),
    ("lllyasviel/sd-controlnet-openpose", None),
    ("h94/IP-Adapter", "models"),
    ("briaai/RMBG-1.4", None),
]

CACHE = Path(__file__).parent.parent / "models"
CACHE.mkdir(parents=True, exist_ok=True)


def main() -> int:
    if "--official" in sys.argv:
        os.environ["HF_ENDPOINT"] = "https://huggingface.co"
        print("Using OFFICIAL HuggingFace endpoint")
    else:
        print(f"Using endpoint: {os.environ['HF_ENDPOINT']}")

    for repo_id, subfolder in MODELS:
        print(f"\n→ Downloading {repo_id}" + (f"/{subfolder}" if subfolder else ""))
        kwargs = {"repo_id": repo_id, "cache_dir": str(CACHE)}
        if subfolder:
            kwargs["allow_patterns"] = [f"{subfolder}/*"]
        try:
            snapshot_download(**kwargs)
            print(f"  ✓ {repo_id}")
        except Exception as e:
            print(f"  ✗ {repo_id}: {e}")
            return 1
    print("\nAll models downloaded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Write shell wrapper**

Create `scripts/download_models.sh`:
```bash
#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
python scripts/download_models.py "$@"
```
```bash
chmod +x scripts/download_models.sh
```

- [ ] **Step 3: Run downloader in background**

```bash
./scripts/download_models.sh &
DOWNLOAD_PID=$!
echo "Models downloading in background (PID $DOWNLOAD_PID). Continue working."
```

- [ ] **Step 4: Commit**

```bash
git add scripts/
git commit -m "feat: model downloader script with hf-mirror default"
```

---

## Phase 1 — Day 1 Foundation: Quick Mode End-to-End (target: 2h–14h)

> Goal: Hit M1 — a Gradio UI that takes a prompt and outputs a (rough) sprite sheet using a single-image direct-generation path. This is the safety net.

### Task 4: prompt_engineer module (TDD)

**Files:**
- Modify: `src/pixelforge/prompt_engineer.py`
- Create: `tests/test_prompt_engineer.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_prompt_engineer.py`:
```python
from pixelforge.prompt_engineer import enhance_prompt, build_negative_prompt


def test_enhance_prompt_adds_pixel_art_anchors():
    out = enhance_prompt("knight with sword")
    assert "pixel art" in out.lower()
    assert "side view" in out.lower()
    assert "knight with sword" in out


def test_enhance_prompt_handles_chinese_input_passthrough():
    # When translator disabled, pass through with English wrapping
    out = enhance_prompt("骑士", translate=False)
    assert "骑士" in out
    assert "pixel art" in out.lower()


def test_negative_prompt_excludes_anti_pixel_terms():
    neg = build_negative_prompt()
    assert "blurry" in neg
    assert "3d" in neg
    assert "photorealistic" in neg
    assert "multiple characters" in neg


def test_enhance_prompt_supports_state_hint():
    out = enhance_prompt("knight", state="walking")
    assert "walking" in out.lower()
```

- [ ] **Step 2: Run test, verify it fails**

```bash
pytest tests/test_prompt_engineer.py -v
```
Expected: 4 FAILED — `enhance_prompt` not defined.

- [ ] **Step 3: Implement prompt_engineer**

Replace `src/pixelforge/prompt_engineer.py`:
```python
"""Prompt enhancement and Chinese→English translation."""
from __future__ import annotations
import os

_BASE_ANCHORS = [
    "pixel art",
    "game asset",
    "full body",
    "side view",
    "white background",
    "16-bit style",
    "crisp pixels",
]

_NEGATIVE_TERMS = [
    "blurry", "3d", "photorealistic", "multiple characters",
    "watermark", "text", "signature", "low quality", "deformed",
    "cropped", "out of frame", "extra limbs",
]


def _translate_zh_to_en(text: str) -> str:
    """Best-effort translation. Falls back to passthrough if no API key."""
    if not os.environ.get("OPENAI_API_KEY"):
        return text
    try:
        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Translate to concise English for a Stable Diffusion prompt. Output translation only, no explanation."},
                {"role": "user", "content": text},
            ],
            max_tokens=80,
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return text


def enhance_prompt(user_text: str, state: str | None = None, translate: bool = True) -> str:
    """Combine user text with pixel-art anchors and optional state hint."""
    base = user_text.strip()
    if translate and any(ord(ch) > 127 for ch in base):
        base = _translate_zh_to_en(base) or base
    parts = [base]
    if state:
        parts.append(state)
    parts.extend(_BASE_ANCHORS)
    return ", ".join(parts)


def build_negative_prompt() -> str:
    return ", ".join(_NEGATIVE_TERMS)
```

- [ ] **Step 4: Run tests, verify pass**

```bash
pytest tests/test_prompt_engineer.py -v
```
Expected: 4 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/pixelforge/prompt_engineer.py tests/test_prompt_engineer.py
git commit -m "feat(prompt_engineer): pixel-art prompt enhancement + zh→en translation"
```

---

### Task 5: engine_router module (TDD)

**Files:**
- Modify: `src/pixelforge/engine_router.py`
- Create: `tests/test_engine_router.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_engine_router.py`:
```python
from unittest.mock import MagicMock, patch
from pixelforge.engine_router import EngineRouter, EngineUnavailableError
import pytest


def test_router_prefers_local_when_available():
    r = EngineRouter(local_available=True, api_available=True)
    assert r.choose() == "local"


def test_router_falls_back_to_api_when_local_unavailable():
    r = EngineRouter(local_available=False, api_available=True)
    assert r.choose() == "api"


def test_router_raises_when_nothing_available():
    r = EngineRouter(local_available=False, api_available=False)
    with pytest.raises(EngineUnavailableError):
        r.choose()


def test_router_force_api_mode():
    r = EngineRouter(local_available=True, api_available=True, force="api")
    assert r.choose() == "api"
```

- [ ] **Step 2: Run, verify fail**

```bash
pytest tests/test_engine_router.py -v
```
Expected: 4 FAILED.

- [ ] **Step 3: Implement engine_router**

Replace `src/pixelforge/engine_router.py`:
```python
"""Route generation requests between local diffusers engine and remote API fallback."""
from __future__ import annotations
import os
from typing import Literal


class EngineUnavailableError(RuntimeError):
    pass


Engine = Literal["local", "api"]


class EngineRouter:
    def __init__(
        self,
        local_available: bool | None = None,
        api_available: bool | None = None,
        force: Engine | None = None,
    ):
        self.local_available = (
            local_available if local_available is not None else self._probe_local()
        )
        self.api_available = (
            api_available if api_available is not None else self._probe_api()
        )
        self.force = force

    @staticmethod
    def _probe_local() -> bool:
        try:
            import torch
            return torch.backends.mps.is_available() or torch.cuda.is_available()
        except ImportError:
            return False

    @staticmethod
    def _probe_api() -> bool:
        return bool(os.environ.get("REPLICATE_API_TOKEN") or os.environ.get("FAL_API_KEY"))

    def choose(self) -> Engine:
        if self.force:
            return self.force
        if self.local_available:
            return "local"
        if self.api_available:
            return "api"
        raise EngineUnavailableError(
            "Neither local (MPS/CUDA) nor API (Replicate/fal) backend available."
        )
```

- [ ] **Step 4: Run, verify pass**

```bash
pytest tests/test_engine_router.py -v
```
Expected: 4 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/pixelforge/engine_router.py tests/test_engine_router.py
git commit -m "feat(engine_router): local/API routing with auto-probe"
```

---

### Task 6: reference_builder module — single image generation

**Files:**
- Modify: `src/pixelforge/reference_builder.py`
- Create: `tests/test_reference_builder.py`

> Note: This task involves real model loading. Tests use mocks for unit testing; a `--integration` flag runs real inference if models are present.

- [ ] **Step 1: Write unit tests with mocks**

Create `tests/test_reference_builder.py`:
```python
from unittest.mock import MagicMock, patch
from PIL import Image
import pytest

from pixelforge.reference_builder import ReferenceBuilder


@patch("pixelforge.reference_builder.StableDiffusionPipeline")
def test_builder_initializes_pipeline_with_correct_model(mock_sd):
    rb = ReferenceBuilder()
    rb._ensure_loaded()
    mock_sd.from_pretrained.assert_called_once()
    call_args = mock_sd.from_pretrained.call_args
    assert "stable-diffusion-v1-5" in call_args[0][0]


@patch("pixelforge.reference_builder.StableDiffusionPipeline")
def test_generate_returns_pil_image(mock_sd):
    fake_image = Image.new("RGB", (512, 512), "red")
    mock_pipe = MagicMock()
    mock_pipe.return_value = MagicMock(images=[fake_image])
    mock_sd.from_pretrained.return_value = mock_pipe

    rb = ReferenceBuilder()
    img = rb.generate("knight character", negative_prompt="blurry")
    assert isinstance(img, Image.Image)
    assert img.size == (512, 512)
```

- [ ] **Step 2: Run, verify fail**

```bash
pytest tests/test_reference_builder.py -v
```
Expected: FAILED (ReferenceBuilder not defined).

- [ ] **Step 3: Implement reference_builder**

Replace `src/pixelforge/reference_builder.py`:
```python
"""Generate a single 'master' character reference image with SD1.5."""
from __future__ import annotations
from pixelforge.config import DEFAULT_CONFIG  # this also sets HF_ENDPOINT

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
from loguru import logger


class ReferenceBuilder:
    def __init__(self, config=None):
        self.cfg = config or DEFAULT_CONFIG
        self._pipe = None

    def _ensure_loaded(self) -> None:
        if self._pipe is not None:
            return
        logger.info(f"Loading SD1.5 from {self.cfg.sd_model_id}...")
        dtype = torch.float16 if self.cfg.dtype == "float16" else torch.float32
        self._pipe = StableDiffusionPipeline.from_pretrained(
            self.cfg.sd_model_id,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(self.cfg.device)
        self._pipe.enable_attention_slicing()
        try:
            self._pipe.enable_vae_slicing()
        except AttributeError:
            pass
        logger.info("SD1.5 loaded.")

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        seed: int | None = None,
    ) -> Image.Image:
        self._ensure_loaded()
        generator = torch.Generator(device=self.cfg.device).manual_seed(
            seed if seed is not None else self.cfg.seed
        )
        result = self._pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=self.cfg.num_inference_steps,
            guidance_scale=self.cfg.guidance_scale,
            width=self.cfg.image_size,
            height=self.cfg.image_size,
            generator=generator,
        )
        return result.images[0]
```

- [ ] **Step 4: Run unit tests, verify pass**

```bash
pytest tests/test_reference_builder.py -v
```
Expected: 2 PASSED.

- [ ] **Step 5: Integration smoke test (only if models downloaded)**

Run manually:
```bash
python -c "
from pixelforge.reference_builder import ReferenceBuilder
from pixelforge.prompt_engineer import enhance_prompt, build_negative_prompt

rb = ReferenceBuilder()
img = rb.generate(enhance_prompt('knight with red cape'), build_negative_prompt())
img.save('output/smoke_reference.png')
print('Saved output/smoke_reference.png')
"
```
Expected: File exists, contains a recognizable character image (10-15s on M chip).

- [ ] **Step 6: Commit**

```bash
git add src/pixelforge/reference_builder.py tests/test_reference_builder.py
git commit -m "feat(reference_builder): single-image SD1.5 generation"
```

---

### Task 7: quick_mode module — Plan B sprite sheet direct generation

**Files:**
- Modify: `src/pixelforge/quick_mode.py`
- Create: `tests/test_quick_mode.py`

- [ ] **Step 1: Write unit tests**

Create `tests/test_quick_mode.py`:
```python
from unittest.mock import MagicMock, patch
from PIL import Image
from pixelforge.quick_mode import QuickModeGenerator


def test_quick_mode_uses_sprite_sheet_prompt_anchors():
    qm = QuickModeGenerator()
    prompt = qm._build_sheet_prompt("knight")
    assert "sprite sheet" in prompt.lower()
    assert "horizontal" in prompt.lower() or "grid" in prompt.lower()
    assert "knight" in prompt


def test_slice_sheet_returns_expected_frame_count():
    sheet = Image.new("RGB", (512, 64), "white")
    qm = QuickModeGenerator()
    frames = qm._slice_sheet(sheet, columns=8)
    assert len(frames) == 8
    assert frames[0].size == (64, 64)


def test_slice_sheet_handles_2x4_grid():
    sheet = Image.new("RGB", (256, 128), "white")
    qm = QuickModeGenerator()
    frames = qm._slice_sheet(sheet, columns=4, rows=2)
    assert len(frames) == 8
    assert frames[0].size == (64, 64)
```

- [ ] **Step 2: Run, verify fail**

```bash
pytest tests/test_quick_mode.py -v
```
Expected: FAILED.

- [ ] **Step 3: Implement quick_mode**

Replace `src/pixelforge/quick_mode.py`:
```python
"""Plan B: single-image sprite sheet generation + slicing."""
from __future__ import annotations
from PIL import Image
from pixelforge.config import DEFAULT_CONFIG
from pixelforge.reference_builder import ReferenceBuilder


class QuickModeGenerator:
    def __init__(self, config=None):
        self.cfg = config or DEFAULT_CONFIG
        self._ref = ReferenceBuilder(self.cfg)

    def _build_sheet_prompt(self, user_prompt: str) -> str:
        return (
            f"{user_prompt}, sprite sheet, horizontal strip, 8 frames walking cycle, "
            "side view, pixel art, game asset, transparent background, white background, "
            "consistent character across frames"
        )

    def _slice_sheet(self, sheet: Image.Image, columns: int = 8, rows: int = 1) -> list[Image.Image]:
        w, h = sheet.size
        frame_w, frame_h = w // columns, h // rows
        frames = []
        for r in range(rows):
            for c in range(columns):
                box = (c * frame_w, r * frame_h, (c + 1) * frame_w, (r + 1) * frame_h)
                frames.append(sheet.crop(box))
        return frames

    def generate(self, user_prompt: str, columns: int = 8) -> list[Image.Image]:
        prompt = self._build_sheet_prompt(user_prompt)
        # Generate at columns*64 wide, 64 tall by abusing the pipe's width/height
        old_w = self.cfg.image_size
        # Stay within memory: cap to 512x512, slice into 8x1
        sheet = self._ref.generate(prompt, "")
        return self._slice_sheet(sheet, columns=columns, rows=1)
```

- [ ] **Step 4: Run, verify pass**

```bash
pytest tests/test_quick_mode.py -v
```
Expected: 3 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/pixelforge/quick_mode.py tests/test_quick_mode.py
git commit -m "feat(quick_mode): direct sprite-sheet generation + slicing"
```

---

### Task 8: post_processor module — pixelization pipeline (TDD)

**Files:**
- Modify: `src/pixelforge/post_processor.py`
- Create: `tests/test_post_processor.py`

- [ ] **Step 1: Write tests for individual stages**

Create `tests/test_post_processor.py`:
```python
import numpy as np
from PIL import Image
from pixelforge.post_processor import (
    PostProcessor,
    resize_to_target,
    quantize_palette,
)


def test_resize_to_target_preserves_size():
    img = Image.new("RGBA", (512, 512), (255, 0, 0, 255))
    out = resize_to_target(img, (64, 64))
    assert out.size == (64, 64)


def test_quantize_palette_reduces_colors():
    img = Image.new("RGB", (32, 32))
    pixels = img.load()
    for x in range(32):
        for y in range(32):
            pixels[x, y] = (x * 8, y * 8, 0)
    quantized = quantize_palette(img, n_colors=8)
    palette = quantized.convert("P", palette=Image.Palette.ADAPTIVE, colors=8)
    # After quantization, unique RGB tuples should be <= n_colors
    unique = set(quantized.getdata())
    assert len(unique) <= 8


def test_postprocessor_full_pipeline_outputs_target_size():
    img = Image.new("RGB", (512, 512), (100, 50, 200))
    pp = PostProcessor(target_size=(64, 64), palette_colors=16)
    out = pp.process(img)
    assert out.size == (64, 64)
    assert out.mode in ("RGBA", "P")
```

- [ ] **Step 2: Run, verify fail**

```bash
pytest tests/test_post_processor.py -v
```
Expected: FAILED.

- [ ] **Step 3: Implement post_processor**

Replace `src/pixelforge/post_processor.py`:
```python
"""Post-process raw SD frames into clean pixel-art sprite frames."""
from __future__ import annotations
import numpy as np
from PIL import Image
from loguru import logger


def remove_background(img: Image.Image) -> Image.Image:
    """Remove background, leaving alpha transparency. Falls back to threshold-based key."""
    try:
        from rembg import remove
        return remove(img.convert("RGBA"))
    except Exception as e:
        logger.warning(f"rembg failed ({e}), using threshold fallback")
        arr = np.array(img.convert("RGBA"))
        # Simple white-background chroma key
        mask = (arr[..., 0] > 240) & (arr[..., 1] > 240) & (arr[..., 2] > 240)
        arr[..., 3] = np.where(mask, 0, 255)
        return Image.fromarray(arr, "RGBA")


def trim_to_content(img: Image.Image, padding: int = 2) -> Image.Image:
    """Crop transparent borders, keeping `padding` px of margin."""
    arr = np.array(img.convert("RGBA"))
    alpha = arr[..., 3]
    if alpha.max() == 0:
        return img
    rows = np.any(alpha > 0, axis=1)
    cols = np.any(alpha > 0, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmin = max(0, rmin - padding)
    rmax = min(arr.shape[0], rmax + padding + 1)
    cmin = max(0, cmin - padding)
    cmax = min(arr.shape[1], cmax + padding + 1)
    return Image.fromarray(arr[rmin:rmax, cmin:cmax], "RGBA")


def resize_to_target(img: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    """Downsample preserving aspect ratio and pad to target size with transparency."""
    img = img.convert("RGBA")
    w, h = img.size
    tw, th = target_size
    scale = min(tw / w, th / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = img.resize((new_w, new_h), Image.NEAREST)
    canvas = Image.new("RGBA", target_size, (0, 0, 0, 0))
    canvas.paste(resized, ((tw - new_w) // 2, (th - new_h) // 2), resized)
    return canvas


def quantize_palette(img: Image.Image, n_colors: int = 24) -> Image.Image:
    """Reduce to a limited palette while preserving transparency."""
    rgba = img.convert("RGBA")
    alpha = rgba.split()[-1]
    rgb = rgba.convert("RGB")
    quantized = rgb.quantize(colors=n_colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    out = quantized.convert("RGB")
    out.putalpha(alpha)
    return out


class PostProcessor:
    def __init__(self, target_size: tuple[int, int] = (64, 64), palette_colors: int = 24):
        self.target_size = target_size
        self.palette_colors = palette_colors

    def process(self, raw_frame: Image.Image) -> Image.Image:
        stage1 = remove_background(raw_frame)
        stage2 = trim_to_content(stage1)
        stage3 = resize_to_target(stage2, self.target_size)
        stage4 = quantize_palette(stage3, self.palette_colors)
        return stage4
```

- [ ] **Step 4: Run, verify pass**

```bash
pytest tests/test_post_processor.py -v
```
Expected: 3 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/pixelforge/post_processor.py tests/test_post_processor.py
git commit -m "feat(post_processor): rembg + trim + resize + palette quantization"
```

---

### Task 9: sheet_composer module — sprite sheet assembly (TDD)

**Files:**
- Modify: `src/pixelforge/sheet_composer.py`
- Create: `tests/test_sheet_composer.py`

- [ ] **Step 1: Write tests**

Create `tests/test_sheet_composer.py`:
```python
from PIL import Image
from pixelforge.sheet_composer import SheetComposer, FrameMetadata


def test_compose_horizontal_strip():
    frames = [Image.new("RGBA", (64, 64), (i * 30, 0, 0, 255)) for i in range(4)]
    sc = SheetComposer()
    sheet, metadata = sc.compose(frames, frame_size=(64, 64))
    assert sheet.size == (256, 64)
    assert metadata["frame_size"] == [64, 64]
    assert metadata["columns"] == 4


def test_compose_includes_animation_definitions():
    frames = [Image.new("RGBA", (64, 64), (0, 0, 0, 0)) for _ in range(12)]
    sc = SheetComposer()
    sheet, meta = sc.compose(
        frames,
        frame_size=(64, 64),
        animations={
            "idle": {"frames": [0, 1, 2, 3], "fps": 6, "loop": True},
            "walk": {"frames": list(range(4, 12)), "fps": 12, "loop": True},
        },
    )
    assert "idle" in meta["animations"]
    assert "walk" in meta["animations"]
    assert meta["animations"]["walk"]["fps"] == 12
    assert meta["animations"]["idle"]["loop"] is True
```

- [ ] **Step 2: Run, verify fail**

```bash
pytest tests/test_sheet_composer.py -v
```

- [ ] **Step 3: Implement sheet_composer**

Replace `src/pixelforge/sheet_composer.py`:
```python
"""Compose individual sprite frames into a sprite sheet + animation metadata JSON."""
from __future__ import annotations
from dataclasses import dataclass
from PIL import Image
from typing import Any


@dataclass
class FrameMetadata:
    index: int
    state: str
    duration_ms: int


class SheetComposer:
    def compose(
        self,
        frames: list[Image.Image],
        frame_size: tuple[int, int] = (64, 64),
        animations: dict[str, dict[str, Any]] | None = None,
    ) -> tuple[Image.Image, dict[str, Any]]:
        n = len(frames)
        fw, fh = frame_size
        sheet = Image.new("RGBA", (fw * n, fh), (0, 0, 0, 0))
        for i, frame in enumerate(frames):
            if frame.size != (fw, fh):
                frame = frame.resize((fw, fh), Image.NEAREST)
            sheet.paste(frame, (i * fw, 0), frame.convert("RGBA"))

        metadata = {
            "frame_size": [fw, fh],
            "columns": n,
            "rows": 1,
            "frame_count": n,
            "animations": animations or {
                "all": {"frames": list(range(n)), "fps": 8, "loop": True}
            },
        }
        return sheet, metadata
```

- [ ] **Step 4: Run, verify pass**

```bash
pytest tests/test_sheet_composer.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/pixelforge/sheet_composer.py tests/test_sheet_composer.py
git commit -m "feat(sheet_composer): horizontal strip + animation metadata"
```

---

### Task 10: exporter — generic PNG+JSON export (TDD)

**Files:**
- Modify: `src/pixelforge/exporter.py`
- Create: `tests/test_exporter.py`

- [ ] **Step 1: Write tests**

Create `tests/test_exporter.py`:
```python
import json
from pathlib import Path
from PIL import Image
from pixelforge.exporter import Exporter


def test_export_generic_writes_png_and_json(tmp_path):
    sheet = Image.new("RGBA", (256, 64), (255, 0, 0, 255))
    meta = {
        "frame_size": [64, 64],
        "columns": 4,
        "rows": 1,
        "frame_count": 4,
        "animations": {"idle": {"frames": [0, 1, 2, 3], "fps": 6, "loop": True}},
    }
    ex = Exporter(output_dir=tmp_path)
    paths = ex.export_generic(sheet, meta, base_name="hero")
    assert (tmp_path / "hero.png").exists()
    assert (tmp_path / "hero.json").exists()
    loaded = json.loads((tmp_path / "hero.json").read_text())
    assert loaded["frame_count"] == 4
    assert paths["png"].name == "hero.png"
    assert paths["json"].name == "hero.json"
```

- [ ] **Step 2: Run, verify fail**

```bash
pytest tests/test_exporter.py -v
```

- [ ] **Step 3: Implement exporter (generic only for now)**

Replace `src/pixelforge/exporter.py`:
```python
"""Export sprite sheet + metadata in multiple game-engine-friendly formats."""
from __future__ import annotations
import json
from pathlib import Path
from PIL import Image
from typing import Any


class Exporter:
    def __init__(self, output_dir: Path | str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_generic(
        self,
        sheet: Image.Image,
        metadata: dict[str, Any],
        base_name: str = "character",
    ) -> dict[str, Path]:
        png_path = self.output_dir / f"{base_name}.png"
        json_path = self.output_dir / f"{base_name}.json"
        sheet.save(png_path)
        json_path.write_text(json.dumps(metadata, indent=2))
        return {"png": png_path, "json": json_path}
```

- [ ] **Step 4: Run, verify pass**

```bash
pytest tests/test_exporter.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/pixelforge/exporter.py tests/test_exporter.py
git commit -m "feat(exporter): generic PNG + JSON export"
```

---

### Task 11: Minimal Gradio app — Quick Mode end-to-end

**Files:**
- Modify: `src/pixelforge/app.py`

- [ ] **Step 1: Implement minimal app**

Replace `src/pixelforge/app.py`:
```python
"""Gradio entry point. Start with Quick Mode end-to-end."""
from __future__ import annotations
from pixelforge.config import DEFAULT_CONFIG, ensure_dirs, OUTPUT_DIR

import gradio as gr
from PIL import Image
from pathlib import Path

from pixelforge.prompt_engineer import enhance_prompt, build_negative_prompt
from pixelforge.quick_mode import QuickModeGenerator
from pixelforge.post_processor import PostProcessor
from pixelforge.sheet_composer import SheetComposer
from pixelforge.exporter import Exporter


def run_quick_mode(prompt: str) -> tuple[Image.Image, str]:
    ensure_dirs()
    enhanced = enhance_prompt(prompt)
    qm = QuickModeGenerator()
    raw_frames = qm.generate(enhanced, columns=8)

    pp = PostProcessor(target_size=DEFAULT_CONFIG.target_sprite_size,
                       palette_colors=DEFAULT_CONFIG.palette_colors)
    processed = [pp.process(f) for f in raw_frames]

    sc = SheetComposer()
    sheet, meta = sc.compose(
        processed,
        frame_size=DEFAULT_CONFIG.target_sprite_size,
        animations={
            "walk": {"frames": list(range(len(processed))), "fps": 12, "loop": True}
        },
    )

    ex = Exporter(OUTPUT_DIR)
    paths = ex.export_generic(sheet, meta, base_name="character_quick")
    return sheet, f"Saved to {paths['png']} and {paths['json']}"


def build_app() -> gr.Blocks:
    with gr.Blocks(title="PixelForge — Quick Mode") as app:
        gr.Markdown("# PixelForge\n*Quick Mode: text → pixel character sprite sheet*")
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Character description",
                                    placeholder="A knight with a red cape and blonde hair")
                btn = gr.Button("Generate", variant="primary")
                status = gr.Textbox(label="Status", interactive=False)
            with gr.Column():
                output = gr.Image(label="Sprite Sheet", type="pil")
        btn.click(run_quick_mode, inputs=prompt, outputs=[output, status])
    return app


if __name__ == "__main__":
    build_app().launch(server_name="127.0.0.1", server_port=7860)
```

- [ ] **Step 2: Run smoke test**

```bash
python -m pixelforge.app
```
Open `http://127.0.0.1:7860`, enter "knight with red cape", click Generate. Wait ~30-60s.
Expected: A sprite sheet image appears + status message with file paths.

- [ ] **Step 3: Commit**

```bash
git add src/pixelforge/app.py
git commit -m "feat(app): Quick Mode Gradio end-to-end"
```

- [ ] **Step 4: 🏁 Milestone M1 — record screen demo**

Record a 30-second clip of the Quick Mode flow working end-to-end. Save to `docs/demos/m1_quickmode.mp4` (mp4 not committed; reference in README).

---

## Phase 2 — Day 1 Foundation: Pose Library + ControlNet Frame Generator (target: 14h–24h)

> Goal: Hit M2 — generate per-frame characters using a pose-skeleton library, even without identity consistency yet.

### Task 12: Build pose skeleton library (manual asset prep)

**Files:**
- Create: `assets/poses/idle_0.png` … `idle_3.png` (4 images)
- Create: `assets/poses/walk_0.png` … `walk_7.png` (8 images)
- Create: `assets/poses/metadata.json`
- Create: `assets/poses/README.md`

> This is manual asset work; not pure code. The skeletons should be 512×512 black-background images with white OpenPose-style stick figures matching idle (subtle bob) and walk-cycle poses (side-view).

- [ ] **Step 1: Source initial pose skeletons**

Use `controlnet-aux` OpenposeDetector on reference photos/illustrations of:
- 4 idle poses (subtle weight shifts, breathing)
- 8 walk-cycle poses (1 full cycle, side view)

Or hand-paint in any image editor. Save each as 512×512 PNG with black background and white skeleton.

Helper script `scripts/prepare_poses.py`:
```python
"""Quick utility: open the controlnet-aux openpose detector on a folder of reference photos and dump the skeleton output to assets/poses/."""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import sys
from pathlib import Path
from controlnet_aux import OpenposeDetector
from PIL import Image

ROOT = Path(__file__).parent.parent
IN_DIR = ROOT / "scripts" / "pose_refs"
OUT_DIR = ROOT / "assets" / "poses"

def main():
    if not IN_DIR.exists():
        print(f"Place reference photos in {IN_DIR} first.")
        return 1
    detector = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
    for src in sorted(IN_DIR.glob("*.png")):
        img = Image.open(src).convert("RGB")
        pose = detector(img)
        out = OUT_DIR / src.name
        pose.save(out)
        print(f"  {src.name} → {out}")

if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Generate / hand-curate the 12 skeleton PNGs**

Place result in `assets/poses/idle_0.png` … `idle_3.png` and `walk_0.png` … `walk_7.png`.

Each file should be 512×512, black background, white stick-figure skeleton.

- [ ] **Step 3: Write metadata.json**

Create `assets/poses/metadata.json`:
```json
{
  "frame_size": 512,
  "states": {
    "idle": {
      "frames": [
        {"file": "idle_0.png", "duration_ms": 250},
        {"file": "idle_1.png", "duration_ms": 250},
        {"file": "idle_2.png", "duration_ms": 250},
        {"file": "idle_3.png", "duration_ms": 250}
      ],
      "loop": true,
      "fps": 6
    },
    "walk": {
      "frames": [
        {"file": "walk_0.png", "duration_ms": 83},
        {"file": "walk_1.png", "duration_ms": 83},
        {"file": "walk_2.png", "duration_ms": 83},
        {"file": "walk_3.png", "duration_ms": 83},
        {"file": "walk_4.png", "duration_ms": 83},
        {"file": "walk_5.png", "duration_ms": 83},
        {"file": "walk_6.png", "duration_ms": 83},
        {"file": "walk_7.png", "duration_ms": 83}
      ],
      "loop": true,
      "fps": 12
    }
  }
}
```

- [ ] **Step 4: Write README**

Create `assets/poses/README.md`:
```markdown
# Pose Skeleton Library

12 OpenPose-format skeleton PNGs at 512×512, used as ControlNet conditioning input
to enforce character pose across animation frames.

- `idle_0..3.png` — 4-frame idle cycle (subtle weight shift)
- `walk_0..7.png` — 8-frame side-view walk cycle

Source: hand-curated via `scripts/prepare_poses.py` applied to reference frames.
```

- [ ] **Step 5: Commit**

```bash
git add assets/poses/ scripts/prepare_poses.py
git commit -m "assets: pose skeleton library (idle×4 + walk×8 OpenPose)"
```

---

### Task 13: pose_library loader (TDD)

**Files:**
- Modify: `src/pixelforge/pose_library.py`
- Create: `tests/test_pose_library.py`

- [ ] **Step 1: Write tests**

Create `tests/test_pose_library.py`:
```python
from pixelforge.pose_library import PoseLibrary, AnimationState


def test_load_idle_returns_four_frames():
    lib = PoseLibrary()
    seq = lib.load("idle")
    assert len(seq.frames) == 4
    assert all(img.size == (512, 512) for img in seq.frames)


def test_load_walk_returns_eight_frames():
    lib = PoseLibrary()
    seq = lib.load("walk")
    assert len(seq.frames) == 8
    assert seq.fps == 12
    assert seq.loop is True


def test_load_combined_returns_idle_then_walk():
    lib = PoseLibrary()
    combined = lib.load_combined(["idle", "walk"])
    assert len(combined.frames) == 12
    # First 4 are idle, next 8 are walk — exposed via animations dict
    assert combined.animations["idle"]["frames"] == [0, 1, 2, 3]
    assert combined.animations["walk"]["frames"] == [4, 5, 6, 7, 8, 9, 10, 11]
```

- [ ] **Step 2: Run, verify fail**

```bash
pytest tests/test_pose_library.py -v
```

- [ ] **Step 3: Implement pose_library**

Replace `src/pixelforge/pose_library.py`:
```python
"""Load pre-built pose skeleton sequences for ControlNet conditioning."""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from PIL import Image
from pixelforge.config import POSES_DIR


@dataclass
class AnimationState:
    name: str
    frames: list[Image.Image]
    fps: int
    loop: bool


@dataclass
class CombinedSequence:
    frames: list[Image.Image]
    animations: dict[str, dict]
    fps_per_state: dict[str, int]


class PoseLibrary:
    def __init__(self, poses_dir: Path | None = None):
        self.dir = poses_dir or POSES_DIR
        self._meta = json.loads((self.dir / "metadata.json").read_text())

    def load(self, state: str) -> AnimationState:
        if state not in self._meta["states"]:
            raise KeyError(f"Unknown state: {state}")
        s = self._meta["states"][state]
        frames = [Image.open(self.dir / f["file"]).convert("RGB") for f in s["frames"]]
        return AnimationState(name=state, frames=frames, fps=s["fps"], loop=s["loop"])

    def load_combined(self, states: list[str]) -> CombinedSequence:
        all_frames: list[Image.Image] = []
        animations: dict[str, dict] = {}
        fps_per_state: dict[str, int] = {}
        idx = 0
        for state in states:
            anim = self.load(state)
            indices = list(range(idx, idx + len(anim.frames)))
            animations[state] = {"frames": indices, "fps": anim.fps, "loop": anim.loop}
            fps_per_state[state] = anim.fps
            all_frames.extend(anim.frames)
            idx += len(anim.frames)
        return CombinedSequence(
            frames=all_frames, animations=animations, fps_per_state=fps_per_state
        )
```

- [ ] **Step 4: Run, verify pass**

```bash
pytest tests/test_pose_library.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/pixelforge/pose_library.py tests/test_pose_library.py
git commit -m "feat(pose_library): metadata-driven OpenPose sequence loader"
```

---

### Task 14: frame_generator stage 1 — ControlNet only (TDD)

**Files:**
- Modify: `src/pixelforge/frame_generator.py`
- Create: `tests/test_frame_generator.py`

> This task adds ControlNet conditioning. IP-Adapter is added in Phase 3 (Task 18).

- [ ] **Step 1: Write unit tests (mocked)**

Create `tests/test_frame_generator.py`:
```python
from unittest.mock import MagicMock, patch
from PIL import Image
from pixelforge.frame_generator import FrameGenerator


@patch("pixelforge.frame_generator.StableDiffusionControlNetPipeline")
@patch("pixelforge.frame_generator.ControlNetModel")
def test_generate_per_pose_returns_one_image_per_pose(mock_cn, mock_pipe_cls):
    pose_a = Image.new("RGB", (512, 512), "black")
    pose_b = Image.new("RGB", (512, 512), "black")

    fake_output = MagicMock(images=[Image.new("RGB", (512, 512), "blue")])
    mock_pipe_instance = MagicMock(return_value=fake_output)
    mock_pipe_cls.from_pretrained.return_value = mock_pipe_instance

    fg = FrameGenerator()
    results = fg.generate_frames(
        prompt="knight",
        negative_prompt="blurry",
        pose_images=[pose_a, pose_b],
    )
    assert len(results) == 2
    assert all(isinstance(img, Image.Image) for img in results)
```

- [ ] **Step 2: Run, verify fail**

```bash
pytest tests/test_frame_generator.py -v
```

- [ ] **Step 3: Implement frame_generator (ControlNet only)**

Replace `src/pixelforge/frame_generator.py`:
```python
"""Generate animation frames conditioned on pose skeletons (and later IP-Adapter)."""
from __future__ import annotations
from pixelforge.config import DEFAULT_CONFIG

import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from PIL import Image
from loguru import logger


class FrameGenerator:
    def __init__(self, config=None):
        self.cfg = config or DEFAULT_CONFIG
        self._pipe = None
        self._reference_image: Image.Image | None = None  # set in phase 3

    def _ensure_loaded(self) -> None:
        if self._pipe is not None:
            return
        dtype = torch.float16 if self.cfg.dtype == "float16" else torch.float32
        logger.info("Loading ControlNet OpenPose model...")
        controlnet = ControlNetModel.from_pretrained(
            self.cfg.controlnet_model_id, torch_dtype=dtype
        )
        logger.info("Loading SD1.5 + ControlNet pipeline...")
        self._pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.cfg.sd_model_id,
            controlnet=controlnet,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(self.cfg.device)
        self._pipe.enable_attention_slicing()
        try:
            self._pipe.enable_vae_slicing()
        except AttributeError:
            pass
        logger.info("FrameGenerator pipeline loaded.")

    def set_reference(self, image: Image.Image) -> None:
        """Save reference image for later IP-Adapter use (Phase 3)."""
        self._reference_image = image

    def generate_frames(
        self,
        prompt: str,
        negative_prompt: str,
        pose_images: list[Image.Image],
        seed: int | None = None,
    ) -> list[Image.Image]:
        self._ensure_loaded()
        seed_val = seed if seed is not None else self.cfg.seed
        results: list[Image.Image] = []
        for i, pose in enumerate(pose_images):
            generator = torch.Generator(device=self.cfg.device).manual_seed(seed_val)
            output = self._pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=pose,
                num_inference_steps=self.cfg.num_inference_steps,
                guidance_scale=self.cfg.guidance_scale,
                controlnet_conditioning_scale=self.cfg.controlnet_conditioning_scale,
                generator=generator,
            )
            results.append(output.images[0])
            logger.info(f"Frame {i+1}/{len(pose_images)} generated.")
        return results
```

- [ ] **Step 4: Run unit tests**

```bash
pytest tests/test_frame_generator.py -v
```

- [ ] **Step 5: Integration smoke test**

Run manually:
```bash
python -c "
from pixelforge.prompt_engineer import enhance_prompt, build_negative_prompt
from pixelforge.pose_library import PoseLibrary
from pixelforge.frame_generator import FrameGenerator

lib = PoseLibrary()
fg = FrameGenerator()
poses = lib.load('walk').frames[:4]  # 4 frames only for speed
frames = fg.generate_frames(
    prompt=enhance_prompt('knight with red cape'),
    negative_prompt=build_negative_prompt(),
    pose_images=poses,
)
for i, f in enumerate(frames):
    f.save(f'output/m2_walk_{i}.png')
print('Saved 4 walk frames')
"
```
Expected: 4 PNGs in output/, showing characters in 4 different walk-cycle poses (consistency not yet enforced).

- [ ] **Step 6: Commit**

```bash
git add src/pixelforge/frame_generator.py tests/test_frame_generator.py
git commit -m "feat(frame_generator): ControlNet OpenPose per-frame generation"
```

- [ ] **Step 7: 🏁 Milestone M2 — record demo clip**

Record per-frame generation with visible pose match. Save as `docs/demos/m2_controlnet.mp4` (referenced in README, not committed if large).

---

## Phase 3 — Day 2 Consistency Sprint (target: 24h–48h)

> Goal: Hit M3 — all 12 frames look like the same character via IP-Adapter + tuned weights.

### Task 15: Integrate IP-Adapter for identity locking

**Files:**
- Modify: `src/pixelforge/frame_generator.py`
- Modify: `tests/test_frame_generator.py`

- [ ] **Step 1: Extend test to cover reference-conditioned mode**

Add to `tests/test_frame_generator.py`:
```python
@patch("pixelforge.frame_generator.StableDiffusionControlNetPipeline")
@patch("pixelforge.frame_generator.ControlNetModel")
def test_generate_with_reference_calls_ip_adapter(mock_cn, mock_pipe_cls):
    pose = Image.new("RGB", (512, 512), "black")
    ref = Image.new("RGB", (512, 512), "red")

    mock_pipe_instance = MagicMock(return_value=MagicMock(images=[Image.new("RGB", (512, 512), "green")]))
    mock_pipe_cls.from_pretrained.return_value = mock_pipe_instance

    fg = FrameGenerator()
    fg.set_reference(ref)
    fg.generate_frames(prompt="knight", negative_prompt="", pose_images=[pose])

    # Verify IP-Adapter was loaded
    assert mock_pipe_instance.load_ip_adapter.called
    assert mock_pipe_instance.set_ip_adapter_scale.called
    # Verify reference image was passed via kwargs
    call_kwargs = mock_pipe_instance.call_args.kwargs
    assert "ip_adapter_image" in call_kwargs
```

- [ ] **Step 2: Run, expect failure**

```bash
pytest tests/test_frame_generator.py::test_generate_with_reference_calls_ip_adapter -v
```

- [ ] **Step 3: Update frame_generator to support IP-Adapter**

Modify `src/pixelforge/frame_generator.py` — change `_ensure_loaded` and `generate_frames`:

```python
def _ensure_loaded(self) -> None:
    if self._pipe is not None:
        return
    dtype = torch.float16 if self.cfg.dtype == "float16" else torch.float32
    logger.info("Loading ControlNet OpenPose model...")
    controlnet = ControlNetModel.from_pretrained(
        self.cfg.controlnet_model_id, torch_dtype=dtype
    )
    logger.info("Loading SD1.5 + ControlNet pipeline...")
    self._pipe = StableDiffusionControlNetPipeline.from_pretrained(
        self.cfg.sd_model_id,
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(self.cfg.device)
    self._pipe.enable_attention_slicing()
    try:
        self._pipe.enable_vae_slicing()
    except AttributeError:
        pass
    # Load IP-Adapter
    logger.info("Loading IP-Adapter...")
    self._pipe.load_ip_adapter(
        self.cfg.ip_adapter_repo,
        subfolder=self.cfg.ip_adapter_subfolder,
        weight_name=self.cfg.ip_adapter_weight_name,
    )
    self._pipe.set_ip_adapter_scale(self.cfg.ip_adapter_scale)
    logger.info("FrameGenerator pipeline + IP-Adapter loaded.")

def generate_frames(
    self,
    prompt: str,
    negative_prompt: str,
    pose_images: list[Image.Image],
    seed: int | None = None,
) -> list[Image.Image]:
    self._ensure_loaded()
    seed_val = seed if seed is not None else self.cfg.seed
    results: list[Image.Image] = []
    for i, pose in enumerate(pose_images):
        generator = torch.Generator(device=self.cfg.device).manual_seed(seed_val)
        pipe_kwargs = dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=pose,
            num_inference_steps=self.cfg.num_inference_steps,
            guidance_scale=self.cfg.guidance_scale,
            controlnet_conditioning_scale=self.cfg.controlnet_conditioning_scale,
            generator=generator,
        )
        if self._reference_image is not None:
            pipe_kwargs["ip_adapter_image"] = self._reference_image
        output = self._pipe(**pipe_kwargs)
        results.append(output.images[0])
        logger.info(f"Frame {i+1}/{len(pose_images)} generated.")
    return results
```

- [ ] **Step 4: Run tests, verify pass**

```bash
pytest tests/test_frame_generator.py -v
```

- [ ] **Step 5: Integration smoke test — A/B**

Run manually, comparing IP-Adapter on vs off:
```bash
python scripts/ipa_ab_test.py
```

Create `scripts/ipa_ab_test.py`:
```python
"""Compare frame generation with and without IP-Adapter."""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from pathlib import Path
from pixelforge.prompt_engineer import enhance_prompt, build_negative_prompt
from pixelforge.reference_builder import ReferenceBuilder
from pixelforge.pose_library import PoseLibrary
from pixelforge.frame_generator import FrameGenerator

OUT = Path("output/ipa_ab")
OUT.mkdir(parents=True, exist_ok=True)

prompt = enhance_prompt("knight with red cape, blonde hair")
neg = build_negative_prompt()

# Reference image
rb = ReferenceBuilder()
ref = rb.generate(prompt, neg)
ref.save(OUT / "reference.png")

poses = PoseLibrary().load("walk").frames[:4]

# Without IP-Adapter
fg = FrameGenerator()
no_ipa = fg.generate_frames(prompt, neg, poses)
for i, f in enumerate(no_ipa):
    f.save(OUT / f"no_ipa_{i}.png")

# With IP-Adapter
fg2 = FrameGenerator()
fg2.set_reference(ref)
with_ipa = fg2.generate_frames(prompt, neg, poses)
for i, f in enumerate(with_ipa):
    f.save(OUT / f"with_ipa_{i}.png")

print("A/B output saved to output/ipa_ab/")
```

Run and inspect. Expected: `with_ipa_*.png` shows visibly more consistent character across the 4 frames.

- [ ] **Step 6: Commit**

```bash
git add src/pixelforge/frame_generator.py tests/test_frame_generator.py scripts/ipa_ab_test.py
git commit -m "feat(frame_generator): IP-Adapter identity locking via reference image"
```

---

### Task 16: Parameter tuning sprint — find a robust preset

**Files:**
- Modify: `src/pixelforge/config.py`
- Create: `scripts/sweep.py`

> This task is empirical, not TDD. Sweep parameter combinations on 3 test prompts and pick a single robust preset.

- [ ] **Step 1: Build sweep script**

Create `scripts/sweep.py`:
```python
"""Sweep IP-Adapter scale, ControlNet conditioning scale, and CFG to find a robust preset."""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from pathlib import Path
from itertools import product
from pixelforge.config import DEFAULT_CONFIG
from pixelforge.prompt_engineer import enhance_prompt, build_negative_prompt
from pixelforge.reference_builder import ReferenceBuilder
from pixelforge.pose_library import PoseLibrary
from pixelforge.frame_generator import FrameGenerator

OUT = Path("output/sweep")
OUT.mkdir(parents=True, exist_ok=True)

PROMPTS = [
    "knight with red cape and blonde hair",
    "blue-robed female mage with purple long hair",
    "rogue with green hood and dual daggers",
]

# Grid: ip_adapter_scale × controlnet_scale × guidance
GRID = list(product(
    [0.5, 0.7, 0.9],
    [0.7, 0.9, 1.0],
    [6.0, 7.5, 9.0],
))

rb = ReferenceBuilder()
poses = PoseLibrary().load("walk").frames[:2]  # 2 frames per combination for speed
neg = build_negative_prompt()

for p_idx, raw_prompt in enumerate(PROMPTS):
    enhanced = enhance_prompt(raw_prompt)
    ref = rb.generate(enhanced, neg)
    ref.save(OUT / f"ref_{p_idx}.png")
    for ipa, cn, cfg in GRID:
        DEFAULT_CONFIG.ip_adapter_scale = ipa
        DEFAULT_CONFIG.controlnet_conditioning_scale = cn
        DEFAULT_CONFIG.guidance_scale = cfg
        fg = FrameGenerator()
        fg.set_reference(ref)
        frames = fg.generate_frames(enhanced, neg, poses)
        for f_idx, f in enumerate(frames):
            f.save(OUT / f"p{p_idx}_ipa{ipa}_cn{cn}_cfg{cfg}_f{f_idx}.png")
        print(f"prompt={p_idx} ipa={ipa} cn={cn} cfg={cfg} done")
```

- [ ] **Step 2: Run sweep**

```bash
python scripts/sweep.py
```
Expected runtime: ~25 minutes (3 prompts × 27 combos × 2 frames × ~7s/frame on M chip).

- [ ] **Step 3: Inspect output and pick preset**

Open `output/sweep/` in Preview/Finder, visually compare. Note which combo gives best consistency + pose fidelity. Update `src/pixelforge/config.py` defaults to the chosen combo.

Example update if (0.7, 0.9, 7.5) wins:
```python
ip_adapter_scale: float = 0.7
controlnet_conditioning_scale: float = 0.9
guidance_scale: float = 7.5
```

- [ ] **Step 4: Commit**

```bash
git add src/pixelforge/config.py scripts/sweep.py
git commit -m "tune: parameter sweep results — selected IP-Adapter/CN/CFG preset"
```

---

### Task 17: Consistency check + auto-retry

**Files:**
- Create: `src/pixelforge/consistency.py`
- Create: `tests/test_consistency.py`
- Modify: `src/pixelforge/frame_generator.py`

- [ ] **Step 1: Write tests for similarity utility**

Create `tests/test_consistency.py`:
```python
from PIL import Image
import numpy as np
from pixelforge.consistency import image_similarity, retry_on_low_similarity


def test_similarity_high_for_identical_images():
    img = Image.new("RGB", (64, 64), (100, 50, 200))
    sim = image_similarity(img, img)
    assert sim > 0.99


def test_similarity_low_for_different_images():
    a = Image.new("RGB", (64, 64), (0, 0, 0))
    b = Image.new("RGB", (64, 64), (255, 255, 255))
    sim = image_similarity(a, b)
    assert sim < 0.5
```

- [ ] **Step 2: Run, verify fail**

```bash
pytest tests/test_consistency.py -v
```

- [ ] **Step 3: Implement consistency utility**

Create `src/pixelforge/consistency.py`:
```python
"""Lightweight image similarity for consistency checks. Uses average color histogram + perceptual hash fallback."""
from __future__ import annotations
import numpy as np
from PIL import Image


def image_similarity(a: Image.Image, b: Image.Image, size: int = 64) -> float:
    """Return 0..1 similarity score based on downscaled color histogram cosine similarity."""
    def feat(img: Image.Image) -> np.ndarray:
        small = img.convert("RGB").resize((size, size), Image.LANCZOS)
        arr = np.array(small).astype(np.float32).flatten()
        norm = np.linalg.norm(arr)
        return arr / norm if norm > 0 else arr

    fa, fb = feat(a), feat(b)
    return float(np.dot(fa, fb))


def retry_on_low_similarity(
    generate_fn,
    reference: Image.Image,
    threshold: float = 0.85,
    max_retries: int = 2,
):
    """Call generate_fn(); if result similarity to reference < threshold, retry up to max_retries times. Returns best result."""
    best_img = None
    best_sim = -1.0
    for attempt in range(max_retries + 1):
        img = generate_fn()
        sim = image_similarity(img, reference)
        if sim > best_sim:
            best_img, best_sim = img, sim
        if sim >= threshold:
            return img, sim
    return best_img, best_sim
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_consistency.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/pixelforge/consistency.py tests/test_consistency.py
git commit -m "feat(consistency): similarity check + retry helper"
```

---

### Task 18: End-to-end orchestrator with main path A

**Files:**
- Create: `src/pixelforge/orchestrator.py`
- Create: `tests/test_orchestrator.py`

- [ ] **Step 1: Write orchestration test (with mocked engines)**

Create `tests/test_orchestrator.py`:
```python
from unittest.mock import MagicMock, patch
from PIL import Image
from pathlib import Path
from pixelforge.orchestrator import generate_character


@patch("pixelforge.orchestrator.PoseLibrary")
@patch("pixelforge.orchestrator.FrameGenerator")
@patch("pixelforge.orchestrator.ReferenceBuilder")
def test_orchestrator_full_pipeline_returns_paths(mock_rb, mock_fg, mock_pl, tmp_path):
    # Mock reference
    fake_ref = Image.new("RGB", (512, 512), "red")
    mock_rb.return_value.generate.return_value = fake_ref

    # Mock 12 pose frames
    fake_poses = [Image.new("RGB", (512, 512), "black")] * 12
    fake_combined = MagicMock(
        frames=fake_poses,
        animations={
            "idle": {"frames": list(range(4)), "fps": 6, "loop": True},
            "walk": {"frames": list(range(4, 12)), "fps": 12, "loop": True},
        },
    )
    mock_pl.return_value.load_combined.return_value = fake_combined

    # Mock generated frames
    mock_fg.return_value.generate_frames.return_value = [
        Image.new("RGB", (512, 512), (i * 20, 0, 0)) for i in range(12)
    ]

    result = generate_character(
        user_prompt="knight",
        output_dir=tmp_path,
        use_quick_mode=False,
    )

    assert "png" in result["paths"]
    assert result["paths"]["png"].exists()
    assert result["paths"]["json"].exists()
    assert result["sheet"].size[0] > 0
```

- [ ] **Step 2: Run, verify fail**

```bash
pytest tests/test_orchestrator.py -v
```

- [ ] **Step 3: Implement orchestrator**

Create `src/pixelforge/orchestrator.py`:
```python
"""Top-level pipeline orchestration: prompt → reference → frames → sheet → export."""
from __future__ import annotations
from pathlib import Path
from PIL import Image
from typing import Any, Iterator
from loguru import logger

from pixelforge.config import DEFAULT_CONFIG, OUTPUT_DIR
from pixelforge.prompt_engineer import enhance_prompt, build_negative_prompt
from pixelforge.reference_builder import ReferenceBuilder
from pixelforge.pose_library import PoseLibrary
from pixelforge.frame_generator import FrameGenerator
from pixelforge.quick_mode import QuickModeGenerator
from pixelforge.post_processor import PostProcessor
from pixelforge.sheet_composer import SheetComposer
from pixelforge.exporter import Exporter


def generate_character(
    user_prompt: str,
    output_dir: Path | str = OUTPUT_DIR,
    base_name: str = "character",
    use_quick_mode: bool = False,
    states: list[str] | None = None,
    progress_callback=None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    states = states or ["idle", "walk"]
    intermediates: dict[str, Any] = {}

    def emit(msg: str, image: Image.Image | None = None):
        if progress_callback:
            progress_callback(msg, image)
        logger.info(msg)

    enhanced = enhance_prompt(user_prompt)
    neg = build_negative_prompt()
    intermediates["enhanced_prompt"] = enhanced

    if use_quick_mode:
        emit("Quick Mode: generating sprite sheet directly...")
        qm = QuickModeGenerator()
        raw_frames = qm.generate(enhanced, columns=8)
        animations = {
            "walk": {"frames": list(range(len(raw_frames))), "fps": 12, "loop": True}
        }
    else:
        emit("Generating reference image...")
        rb = ReferenceBuilder()
        ref = rb.generate(enhanced, neg)
        intermediates["reference"] = ref
        emit("Reference image generated.", ref)

        emit("Loading pose library...")
        lib = PoseLibrary()
        combined = lib.load_combined(states)
        intermediates["poses"] = combined.frames
        animations = combined.animations

        emit(f"Generating {len(combined.frames)} animation frames...")
        fg = FrameGenerator()
        fg.set_reference(ref)
        raw_frames = fg.generate_frames(enhanced, neg, combined.frames)
        intermediates["raw_frames"] = raw_frames

    emit(f"Post-processing {len(raw_frames)} frames...")
    pp = PostProcessor(
        target_size=DEFAULT_CONFIG.target_sprite_size,
        palette_colors=DEFAULT_CONFIG.palette_colors,
    )
    processed = [pp.process(f) for f in raw_frames]
    intermediates["processed_frames"] = processed

    emit("Composing sprite sheet...")
    sc = SheetComposer()
    sheet, meta = sc.compose(
        processed, frame_size=DEFAULT_CONFIG.target_sprite_size, animations=animations
    )
    emit("Exporting...")
    ex = Exporter(output_dir)
    paths = ex.export_generic(sheet, meta, base_name=base_name)

    return {
        "sheet": sheet,
        "metadata": meta,
        "paths": paths,
        "intermediates": intermediates,
    }
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_orchestrator.py -v
```

- [ ] **Step 5: Integration smoke**

Run manually:
```bash
python -c "
from pixelforge.orchestrator import generate_character
result = generate_character('knight with red cape')
print('Sheet path:', result['paths']['png'])
print('JSON path:', result['paths']['json'])
"
```
Expected: ~2-3 minutes, produces `output/character.png` (768×64 sprite sheet) and `output/character.json`.

- [ ] **Step 6: Commit**

```bash
git add src/pixelforge/orchestrator.py tests/test_orchestrator.py
git commit -m "feat(orchestrator): full pipeline orchestration with intermediates"
```

- [ ] **Step 7: 🏁 Milestone M3 — verify 12-frame consistency**

Visually inspect `output/character.png`. All 12 frames should look like the same character in different poses. If consistency is poor:
- Re-run Task 16 sweep with finer grid
- Try lowering IP-Adapter scale to 0.5
- Increase guidance to 8.5
- Last resort: fall back to Quick Mode

---

## Phase 4 — Day 3: Exports + UI Polish + Demo Prep (target: 48h–72h)

### Task 19: Godot 4 SpriteFrames .tres exporter

**Files:**
- Create: `assets/templates/godot_spriteframes.tres.j2`
- Modify: `src/pixelforge/exporter.py`
- Modify: `tests/test_exporter.py`

- [ ] **Step 1: Write test for .tres export**

Add to `tests/test_exporter.py`:
```python
def test_export_godot_writes_tres_file(tmp_path):
    sheet = Image.new("RGBA", (256, 64), (255, 0, 0, 255))
    meta = {
        "frame_size": [64, 64],
        "columns": 4,
        "rows": 1,
        "frame_count": 4,
        "animations": {"idle": {"frames": [0, 1, 2, 3], "fps": 6, "loop": True}},
    }
    ex = Exporter(output_dir=tmp_path)
    ex.export_generic(sheet, meta, base_name="hero")
    tres_path = ex.export_godot(meta, base_name="hero", png_relative="hero.png")
    assert tres_path.exists()
    content = tres_path.read_text()
    assert "SpriteFrames" in content
    assert "idle" in content
```

- [ ] **Step 2: Run, verify fail**

```bash
pytest tests/test_exporter.py::test_export_godot_writes_tres_file -v
```

- [ ] **Step 3: Create Jinja2 template for Godot 4 SpriteFrames**

Create `assets/templates/godot_spriteframes.tres.j2`:
```jinja
[gd_resource type="SpriteFrames" load_steps={{ 2 + frame_count }} format=3]

[ext_resource type="Texture2D" path="res://{{ png_relative }}" id="1_tex"]

{% for fr in range(frame_count) -%}
[sub_resource type="AtlasTexture" id="atlas_{{ fr }}"]
atlas = ExtResource("1_tex")
region = Rect2({{ (fr % columns) * frame_w }}, {{ (fr // columns) * frame_h }}, {{ frame_w }}, {{ frame_h }})

{% endfor %}
[resource]
animations = [
{%- for name, anim in animations.items() %}
    {
        "frames": [
            {%- for idx in anim.frames -%}
            {"duration": 1.0, "texture": SubResource("atlas_{{ idx }}")}{% if not loop.last %},{% endif %}
            {%- endfor -%}
        ],
        "loop": {{ anim.loop | lower }},
        "name": &"{{ name }}",
        "speed": {{ anim.fps }}.0
    }{% if not loop.last %},{% endif %}
{%- endfor %}
]
```

- [ ] **Step 4: Extend Exporter with export_godot**

Add to `src/pixelforge/exporter.py`:
```python
from jinja2 import Environment, FileSystemLoader
from pixelforge.config import TEMPLATES_DIR


class Exporter:
    # ... existing __init__ and export_generic ...

    def export_godot(
        self,
        metadata: dict,
        base_name: str = "character",
        png_relative: str | None = None,
    ) -> Path:
        env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)))
        template = env.get_template("godot_spriteframes.tres.j2")
        rendered = template.render(
            png_relative=png_relative or f"{base_name}.png",
            frame_count=metadata["frame_count"],
            columns=metadata["columns"],
            frame_w=metadata["frame_size"][0],
            frame_h=metadata["frame_size"][1],
            animations=metadata["animations"],
        )
        out = self.output_dir / f"{base_name}.tres"
        out.write_text(rendered)
        return out
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_exporter.py -v
```

- [ ] **Step 6: Manual Godot validation**

Open Godot 4.x → create new 2D scene → drag generated `.tres` into FileSystem → instantiate AnimatedSprite2D → assign SpriteFrames resource. Play scene; animation should loop visibly.

If format errors appear, check Godot version (`4.2` vs `4.3` differ slightly). Inspect the .tres in a text editor and adjust template.

- [ ] **Step 7: Commit**

```bash
git add assets/templates/godot_spriteframes.tres.j2 src/pixelforge/exporter.py tests/test_exporter.py
git commit -m "feat(exporter): Godot 4 SpriteFrames .tres export"
```

---

### Task 20: Unity sprite + .meta exporter

**Files:**
- Create: `assets/templates/unity_sprite.meta.j2`
- Modify: `src/pixelforge/exporter.py`
- Modify: `tests/test_exporter.py`

- [ ] **Step 1: Write test**

Add to `tests/test_exporter.py`:
```python
def test_export_unity_writes_meta_file(tmp_path):
    sheet = Image.new("RGBA", (256, 64), (255, 0, 0, 255))
    meta = {
        "frame_size": [64, 64],
        "columns": 4,
        "rows": 1,
        "frame_count": 4,
        "animations": {"idle": {"frames": [0, 1, 2, 3], "fps": 6, "loop": True}},
    }
    ex = Exporter(output_dir=tmp_path)
    ex.export_generic(sheet, meta, base_name="hero")
    meta_path = ex.export_unity(meta, base_name="hero")
    assert meta_path.exists()
    content = meta_path.read_text()
    assert "TextureImporter" in content
    assert "spriteSheet" in content
```

- [ ] **Step 2: Run, verify fail**

```bash
pytest tests/test_exporter.py::test_export_unity_writes_meta_file -v
```

- [ ] **Step 3: Create Unity .meta template**

Create `assets/templates/unity_sprite.meta.j2`:
```jinja
fileFormatVersion: 2
guid: {{ guid }}
TextureImporter:
  internalIDToNameTable:
{%- for fr in range(frame_count) %}
  - first:
      213: {{ 21300000 + fr }}
    second: {{ base_name }}_{{ fr }}
{%- endfor %}
  externalObjects: {}
  serializedVersion: 12
  mipmaps:
    mipMapMode: 0
    enableMipMap: 0
  bumpmap:
    convertToNormalMap: 0
  isReadable: 0
  streamingMipmaps: 0
  vTonemapper: 0
  alphaTestReferenceValue: 0.5
  mipMapFadeDistanceStart: 1
  mipMapFadeDistanceEnd: 3
  textureFormat: 1
  maxTextureSize: 2048
  textureSettings:
    serializedVersion: 2
    filterMode: 0
    aniso: 1
    mipBias: 0
    wrapU: 1
    wrapV: 1
    wrapW: 1
  nPOTScale: 0
  lightmap: 0
  compressionQuality: 50
  spriteMode: 2
  spriteExtrude: 1
  spriteMeshType: 1
  alignment: 0
  spritePivot: {x: 0.5, y: 0.5}
  spritePixelsToUnits: {{ frame_h }}
  spriteBorder: {x: 0, y: 0, z: 0, w: 0}
  spriteGenerateFallbackPhysicsShape: 1
  alphaUsage: 1
  alphaIsTransparency: 1
  spriteTessellationDetail: -1
  textureType: 8
  textureShape: 1
  singleChannelComponent: 0
  flipbookRows: 1
  flipbookColumns: 1
  maxTextureSizeSet: 0
  compressionQualitySet: 0
  textureFormatSet: 0
  ignorePngGamma: 0
  applyGammaDecoding: 0
  cookieLightType: 1
  platformSettings: []
  spriteSheet:
    serializedVersion: 2
    sprites:
{%- for fr in range(frame_count) %}
    - serializedVersion: 2
      name: {{ base_name }}_{{ fr }}
      rect:
        serializedVersion: 2
        x: {{ (fr % columns) * frame_w }}
        y: {{ sheet_h - (((fr // columns) + 1) * frame_h) }}
        width: {{ frame_w }}
        height: {{ frame_h }}
      alignment: 0
      pivot: {x: 0.5, y: 0.5}
      border: {x: 0, y: 0, z: 0, w: 0}
      outline: []
      physicsShape: []
      tessellationDetail: 0
      bones: []
      spriteID: {{ '5e97eb03825dee720800000000000000' }}
      vertices: []
      indices:
      edges: []
      weights: []
{%- endfor %}
    outline: []
    physicsShape: []
    bones: []
    spriteID: 5e97eb03825dee720800000000000000
    internalID: 0
    vertices: []
    indices:
    edges: []
    weights: []
    secondaryTextures: []
    nameFileIdTable: {}
  spritePackingTag:
  pSDRemoveMatte: 0
  pSDShowRemoveMatteOption: 0
  userData:
  assetBundleName:
  assetBundleVariant:
```

- [ ] **Step 4: Extend Exporter**

Add to `src/pixelforge/exporter.py`:
```python
import uuid


class Exporter:
    # ... existing methods ...

    def export_unity(
        self,
        metadata: dict,
        base_name: str = "character",
    ) -> Path:
        env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)))
        template = env.get_template("unity_sprite.meta.j2")
        # Unity needs the texture height to flip Y coordinates
        sheet_h = metadata["frame_size"][1] * metadata["rows"]
        rendered = template.render(
            guid=uuid.uuid4().hex,
            base_name=base_name,
            frame_count=metadata["frame_count"],
            columns=metadata["columns"],
            frame_w=metadata["frame_size"][0],
            frame_h=metadata["frame_size"][1],
            sheet_h=sheet_h,
        )
        out = self.output_dir / f"{base_name}.png.meta"
        out.write_text(rendered)
        return out
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_exporter.py -v
```

- [ ] **Step 6: Manual Unity validation (best-effort, can be cut)**

Open Unity 2022.3 LTS → drop generated `.png` + `.png.meta` into Assets folder. The sprite should auto-slice into N sub-sprites.

If broken: ship without `.png.meta` and add a README section "Unity import: set Sprite Mode = Multiple, Pixels Per Unit = 64, Filter Mode = Point, then click Slice."

- [ ] **Step 7: Commit**

```bash
git add assets/templates/unity_sprite.meta.j2 src/pixelforge/exporter.py tests/test_exporter.py
git commit -m "feat(exporter): Unity sprite .meta export (best effort)"
```

---

### Task 21: Wire all exporters into orchestrator

**Files:**
- Modify: `src/pixelforge/orchestrator.py`

- [ ] **Step 1: Add multi-format export to orchestrator**

Modify the end of `generate_character` in `src/pixelforge/orchestrator.py`:

Replace:
```python
emit("Exporting...")
ex = Exporter(output_dir)
paths = ex.export_generic(sheet, meta, base_name=base_name)
```

With:
```python
emit("Exporting...")
ex = Exporter(output_dir)
paths = ex.export_generic(sheet, meta, base_name=base_name)
try:
    paths["tres"] = ex.export_godot(meta, base_name=base_name, png_relative=f"{base_name}.png")
except Exception as e:
    logger.warning(f"Godot export failed: {e}")
try:
    paths["meta"] = ex.export_unity(meta, base_name=base_name)
except Exception as e:
    logger.warning(f"Unity export failed: {e}")
```

- [ ] **Step 2: Run end-to-end**

```bash
python -c "
from pixelforge.orchestrator import generate_character
r = generate_character('knight with red cape')
for k, v in r['paths'].items():
    print(f'{k}: {v}')
"
```
Expected: 4 paths printed (png, json, tres, meta).

- [ ] **Step 3: Commit**

```bash
git add src/pixelforge/orchestrator.py
git commit -m "feat(orchestrator): wire Godot + Unity exporters"
```

---

### Task 22: Polish Gradio UI — intermediate-product display + animation preview

**Files:**
- Modify: `src/pixelforge/app.py`

- [ ] **Step 1: Replace app with full version**

Replace `src/pixelforge/app.py`:
```python
"""PixelForge Gradio UI — full version with intermediate-product panels."""
from __future__ import annotations
from pixelforge.config import DEFAULT_CONFIG, ensure_dirs, OUTPUT_DIR

import gradio as gr
import zipfile
from pathlib import Path
from PIL import Image
from loguru import logger

from pixelforge.orchestrator import generate_character


def make_gif(frames: list[Image.Image], fps: int = 12) -> Path:
    """Write a preview animation GIF from a list of PIL images."""
    out = OUTPUT_DIR / "preview.gif"
    if not frames:
        return out
    duration = int(1000 / max(1, fps))
    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=duration,
        disposal=2,
        optimize=False,
    )
    return out


def make_zip(paths: dict, base_name: str) -> Path:
    zpath = OUTPUT_DIR / f"{base_name}.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        for p in paths.values():
            if isinstance(p, Path) and p.exists():
                zf.write(p, arcname=p.name)
    return zpath


def run_pipeline(prompt: str, quick_mode: bool, progress=gr.Progress()):
    ensure_dirs()
    progress(0.0, desc="Starting...")

    reference_state = {"img": None}
    pose_state = {"imgs": []}
    raw_state = {"imgs": []}
    proc_state = {"imgs": []}

    def cb(msg, img=None):
        logger.info(msg)
        if img is not None and "reference" in msg.lower():
            reference_state["img"] = img

    result = generate_character(
        user_prompt=prompt,
        use_quick_mode=quick_mode,
        progress_callback=cb,
    )

    inter = result["intermediates"]
    ref_img = inter.get("reference")
    poses = inter.get("poses", [])
    raw = inter.get("raw_frames", [])
    processed = inter.get("processed_frames", [])

    gif_path = make_gif(processed, fps=12) if processed else None
    zip_path = make_zip(result["paths"], "character")

    status = f"Done. Output: {result['paths']['png'].name}"

    return (
        result["sheet"],
        ref_img,
        poses[:4] if poses else None,
        raw[:4] if raw else None,
        processed[:4] if processed else None,
        str(gif_path) if gif_path else None,
        str(zip_path),
        status,
    )


def build_app() -> gr.Blocks:
    with gr.Blocks(title="PixelForge", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            "# PixelForge\n"
            "**Text → consistent pixel-art character sprite sheet with idle + walk animation.**\n\n"
            "Local Stable Diffusion 1.5 + ControlNet OpenPose + IP-Adapter. Exports to Godot / Unity / generic JSON."
        )
        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(
                    label="Character description",
                    placeholder="A knight with a red cape and blonde hair holding a sword",
                    lines=3,
                )
                quick = gr.Checkbox(label="Quick Mode (Plan B: direct sprite-sheet)", value=False)
                btn = gr.Button("Generate", variant="primary", size="lg")
                status = gr.Textbox(label="Status", interactive=False)
                zip_out = gr.File(label="Download all (ZIP)")
            with gr.Column(scale=2):
                sheet_out = gr.Image(label="Final Sprite Sheet", type="pil")
                preview = gr.Image(label="Animation Preview", type="filepath")

        with gr.Accordion("Pipeline Intermediates", open=False):
            with gr.Row():
                ref_out = gr.Image(label="1. Reference Character", type="pil")
                pose_out = gr.Gallery(label="2. Pose Skeletons (first 4)", columns=4, height=120)
                raw_out = gr.Gallery(label="3. Raw Frames (first 4)", columns=4, height=120)
                proc_out = gr.Gallery(label="4. Processed Frames (first 4)", columns=4, height=120)

        btn.click(
            run_pipeline,
            inputs=[prompt, quick],
            outputs=[sheet_out, ref_out, pose_out, raw_out, proc_out, preview, zip_out, status],
        )
    return app


if __name__ == "__main__":
    build_app().launch(server_name="127.0.0.1", server_port=7860)
```

- [ ] **Step 2: Run the app**

```bash
python -m pixelforge.app
```
Open browser, generate a character, verify all 4 intermediate panels populate.

- [ ] **Step 3: Commit**

```bash
git add src/pixelforge/app.py
git commit -m "feat(app): full UI with intermediate-product panels + GIF preview + ZIP export"
```

---

### Task 23: Pre-generate demo examples (safety net)

**Files:**
- Create: `scripts/generate_examples.py`
- Create: `assets/examples/*.png` + `*.json` + `*.tres` + `*.gif`

- [ ] **Step 1: Write example pre-generator**

Create `scripts/generate_examples.py`:
```python
"""Pre-generate 5 hand-picked demo examples for safety net during live demo."""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from pathlib import Path
from pixelforge.orchestrator import generate_character

EXAMPLES = [
    ("knight", "A knight with a red cape, blonde hair, holding a steel sword"),
    ("mage", "A female mage with a blue robe and long purple hair holding a wooden staff"),
    ("rogue", "A rogue with a green hood, leather armor, holding twin daggers"),
    ("archer", "A wood elf archer with a green tunic, brown hair, longbow"),
    ("warrior", "A barbarian warrior with bare chest, brown hair in a topknot, holding a great axe"),
]
OUT = Path("assets/examples")
OUT.mkdir(parents=True, exist_ok=True)

for slug, prompt in EXAMPLES:
    print(f"\n=== {slug}: {prompt}")
    result = generate_character(prompt, output_dir=OUT, base_name=slug)
    print(f"  → {result['paths']['png']}")
```

- [ ] **Step 2: Run it**

```bash
python scripts/generate_examples.py
```
Expected runtime: 5 × ~150s = ~12 min. Inspect outputs; if any look bad, edit the prompt and re-run that one.

- [ ] **Step 3: Commit examples**

```bash
git add assets/examples/ scripts/generate_examples.py
git commit -m "assets: pre-generated demo examples (knight/mage/rogue/archer/warrior)"
```

---

### Task 24: README + INSTALL + DEMO_SCRIPT

**Files:**
- Modify: `README.md`
- Create: `docs/INSTALL.md`
- Create: `docs/DEMO_SCRIPT.md`

- [ ] **Step 1: Write README**

Create `README.md`:
```markdown
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
```

- [ ] **Step 2: Write INSTALL.md**

Create `docs/INSTALL.md`:
```markdown
# Installation

## Prerequisites

- macOS with M-series chip (or any CUDA GPU)
- Python 3.11 (`pyenv install 3.11` or Homebrew)
- `uv` package manager: `brew install uv`
- ~10GB free disk space (models + cache)

## Steps

1. Clone:
   ```
   git clone <repo>
   cd pixelforge
   ```

2. Set up env:
   ```
   uv venv --python 3.11
   source .venv/bin/activate
   uv pip install -e ".[dev]"
   ```

3. Copy env file:
   ```
   cp .env.example .env
   # Optional: add REPLICATE_API_TOKEN or OPENAI_API_KEY
   ```

4. Download models (uses hf-mirror.com by default; ~10-15min on good connection):
   ```
   ./scripts/download_models.sh
   ```

   For official HuggingFace:
   ```
   ./scripts/download_models.sh --official
   ```

5. Run:
   ```
   python -m pixelforge.app
   ```

## Troubleshooting

- **`torch.backends.mps.is_available()` returns False**: macOS 12.3+ required; reinstall `torch>=2.3.0`.
- **Out of memory**: close Chrome / other heavy apps; set `dtype="float16"` in `config.py` (default).
- **Slow downloads**: try `./scripts/download_models.sh --official` if mirror is unstable.
- **rembg fails**: switches automatically to threshold fallback; output may have noisy edges.
```

- [ ] **Step 3: Write DEMO_SCRIPT.md**

Create `docs/DEMO_SCRIPT.md`:
```markdown
# Demo Script — 7 minutes

## Setup (before recording starts)

- [ ] `python -m pixelforge.app` running on `localhost:7860`
- [ ] Browser open on the Gradio page
- [ ] Godot 4.x project pre-opened with empty 2D scene
- [ ] Hardware: laptop unplugged-to-airplane-mode for "local-only" optics
- [ ] Backup: `assets/examples/` opened in Finder; preview GIFs ready
- [ ] Slides: 1-page intro + 1-page architecture diagram

## 0:00–0:30 — Open

> "Indie game devs' biggest bottleneck isn't code — it's art.
> An animated pixel character: ~$120, 1-3 days outsourced.
> Can AI make that minutes and near-zero cost?
> Meet **PixelForge**."

## 0:30–1:00 — One-liner

> "We're not another text-to-image tool. Text-to-image gives you a picture.
> Games need a **consistent** character that **animates** — and that imports cleanly into your engine. That's what we solve."

## 1:00–4:00 — Live Demo (core)

1. **Prompt** (30s): paste "A female mage with a blue robe and long purple hair holding a staff"
2. **Reference image** generates (15s) — point to it: ⭐ wow 1
3. **12-frame generation** (90s):
   - Talk through architecture while it runs
   - Pop open "Pipeline Intermediates" panel: ⭐ wow 2 (white box)
4. **Animation preview** loops (30s): ⭐ wow 3

## 4:00–5:30 — Integration

5. **Download ZIP**: show all 4 formats inside (.png, .json, .tres, .meta)
6. **Drag .tres into Godot** → AnimatedSprite2D → play scene
   - Mage walks on screen: ⭐ wow 4

## 5:30–6:30 — Tech + numbers

- Local M-chip 16GB inference (this laptop = proof)
- ControlNet pose library × IP-Adapter identity = consistency
- End-to-end: ~120s / character
- Cost: $0 local / $0.02 API / $120 outsourced

## 6:30–7:00 — Close

> "AI doesn't replace artists. It compresses the 0-to-1 from days to minutes.
> Every stage is upgradeable — swap SD1.5 for Cascade tomorrow.
> Open source, local, extensible."

## Anchors (if anything breaks)

- Anchor 1 — `assets/examples/` static showcase
- Anchor 2 — recorded video (`docs/demos/*.mp4`)
- Anchor 3 — Quick Mode toggle (Plan B always works)

## Q&A Prep

| Question | Answer |
|---|---|
| How is this different from MidJourney? | MJ is single image, no pose control, no engine export, no consistency across frames. We're a pipeline. |
| Why SD1.5 not SDXL? | SDXL too heavy for 16GB. SD1.5 + pixel LoRA gives equivalent style quality at 5× the speed. |
| Can it do more than walk/idle? | Pose library is data-driven; add attack/hurt = add 8 more skeleton PNGs + 1 JSON entry. |
| Quality vs. hand-drawn? | Comparable to mid-tier asset packs (Stardew Valley palette range). Not portfolio-grade yet. |
| Commercial use? | SD1.5 license permits. IP-Adapter is research-grade; recommend review for production. |
```

- [ ] **Step 4: Commit**

```bash
git add README.md docs/INSTALL.md docs/DEMO_SCRIPT.md
git commit -m "docs: README + INSTALL + DEMO_SCRIPT"
```

---

### Task 25: Benchmark + final smoke test

**Files:**
- Create: `scripts/benchmark.py`
- Modify: `tests/test_e2e_smoke.py`

- [ ] **Step 1: Write benchmark**

Create `scripts/benchmark.py`:
```python
"""Time each pipeline stage for a single character end-to-end."""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import time
from pathlib import Path
from pixelforge.orchestrator import generate_character

OUT = Path("output/bench")
OUT.mkdir(parents=True, exist_ok=True)

t0 = time.time()
result = generate_character("knight with red cape", output_dir=OUT, base_name="bench")
total = time.time() - t0
print(f"\n=== Benchmark Results ===")
print(f"Total end-to-end: {total:.1f}s")
print(f"Output: {result['paths']['png']}")
```

- [ ] **Step 2: Run benchmark**

```bash
python scripts/benchmark.py
```
Target: < 180s total on M-chip 16GB.
If > 180s: profile each stage (add `time.time()` between phases in orchestrator), reduce `num_inference_steps` from 20 → 15.

- [ ] **Step 3: Write end-to-end smoke test**

Create `tests/test_e2e_smoke.py`:
```python
"""End-to-end smoke test. Skipped in CI; run manually only."""
import pytest
from pathlib import Path


@pytest.mark.skipif(
    not Path("models").exists() or not list(Path("models").rglob("*.safetensors")),
    reason="Models not downloaded — skip e2e."
)
def test_quickmode_e2e(tmp_path):
    from pixelforge.orchestrator import generate_character
    result = generate_character(
        "test knight character",
        output_dir=tmp_path,
        base_name="smoke",
        use_quick_mode=True,
    )
    assert result["paths"]["png"].exists()
    assert result["paths"]["json"].exists()
```

- [ ] **Step 4: Run**

```bash
pytest tests/test_e2e_smoke.py -v
```

- [ ] **Step 5: Commit**

```bash
git add scripts/benchmark.py tests/test_e2e_smoke.py
git commit -m "test: end-to-end smoke + benchmark script"
```

---

### Task 26: Final cleanup + tag release

**Files:**
- Modify: `README.md` (add screenshots)
- Modify: `.gitignore` (ensure output/ ignored)

- [ ] **Step 1: Confirm everything runs from scratch**

```bash
rm -rf .venv output/*.png output/*.json output/*.tres
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
pytest -v
python -m pixelforge.app
```
Expected: all tests pass, UI launches.

- [ ] **Step 2: Add screenshot to README**

Take a screenshot of the Gradio app showing a complete generation (sprite sheet + intermediate panels). Save to `docs/screenshot.png`. Add to README:
```markdown
![screenshot](docs/screenshot.png)
```

- [ ] **Step 3: Rehearse demo 2-3 times**

Time each run. Watch for: > 10s pauses, error popups, off-camera prompts. Refine the DEMO_SCRIPT accordingly.

- [ ] **Step 4: 🏁 Milestone M5 — final commit and tag**

```bash
git add README.md docs/screenshot.png
git commit -m "docs: final README with screenshot"
git tag v0.1.0 -m "PixelForge v0.1.0 — hackathon submission"
```

---

## Appendix — Killswitch Order (if you run out of time)

In order of "I'll cut this first":

1. **Unity .meta** — replace with a README paragraph "How to import to Unity"
2. **Chinese prompt translation** — demo in English only
3. **Multi-direction support** — already MVP'd to side-view only
4. **Auto consistency check** — visual inspection during the sweep is enough
5. **Quick Mode UI toggle** — wire it to always-on if needed

**Never cut**: Local SD1.5 path / Sprite sheet + JSON export / Gradio UI / At least one engine integration (Godot).

---

## Self-Review

✅ **Spec coverage check** (against `docs/superpowers/specs/2026-05-23-pixelforge-design.md`):

| Spec section | Plan task |
|---|---|
| §3 Architecture | Tasks 1-2 (skeleton + config) |
| §4 Modules 1-10 | Tasks 4 (prompt), 5 (router), 6 (reference), 7 (quick), 8 (post), 9 (sheet), 10 (export), 13 (pose), 14-15 (frame), 18 (orchestrator), 22 (app) |
| §5 Data flow | Task 18 orchestrator implements full flow |
| §6 Tech stack | Task 1 pyproject.toml |
| §6.4 Repo structure | Task 2 |
| §7 Consistency approach | Tasks 14, 15, 16, 17 |
| §8 Error handling | Tasks 5 (router), 8 (post_processor fallback), 17 (consistency retry) |
| §9 Testing | All TDD tasks; Task 25 e2e smoke |
| §10 Day 1/2/3 schedule | Phase 1/2/3/4 alignment |
| §11 Demo script | Task 24 DEMO_SCRIPT.md |
| §12 Risk register | Killswitch appendix + INSTALL troubleshooting |
| §15 Prerequisites | Task 1 setup |

✅ **Placeholder scan**: no "TBD"/"TODO"/"add error handling" without specifics found.

✅ **Type/method consistency**: `generate_frames`, `set_reference`, `export_godot`, `export_unity`, `export_generic` consistent across tasks 14/15/18/19/20/21.

✅ **No spec gap**: every numbered module in §4 has a corresponding task. Risk mitigation (consistency check) covered in Task 17.

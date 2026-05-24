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
    sd_model_id: str = "Onodofthenorth/SD_PixelArt_SpriteSheet_Generator"
    controlnet_model_id: str = "lllyasviel/sd-controlnet-openpose"
    ip_adapter_repo: str = "h94/IP-Adapter"
    ip_adapter_subfolder: str = "models"
    ip_adapter_weight_name: str = "ip-adapter_sd15.bin"
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
    dtype: str = "float32"


DEFAULT_CONFIG = GenerationConfig()


def ensure_dirs() -> None:
    for d in (ASSETS_DIR, MODELS_DIR, OUTPUT_DIR, POSES_DIR, TEMPLATES_DIR, EXAMPLES_DIR):
        d.mkdir(parents=True, exist_ok=True)


def ensure_cached(repo_id: str, allow_patterns: list[str] | None = None) -> None:
    """Ensure a HF model repo is in local cache. Downloads via HF_ENDPOINT if missing."""
    from huggingface_hub import snapshot_download
    try:
        snapshot_download(repo_id, local_files_only=True, allow_patterns=allow_patterns)
    except Exception:
        endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
        print(f"[pixelforge] Downloading {repo_id} from {endpoint} ...")
        snapshot_download(repo_id, allow_patterns=allow_patterns)

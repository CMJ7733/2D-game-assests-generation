"""Central configuration. Imported before diffusers/transformers to set HF endpoint."""
import os
from typing import Literal

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
    rmbg_model_id: str = "briaai/RMBG-1.4"
    controlnet_model_id: str = "lllyasviel/sd-controlnet-openpose"
    ip_adapter_repo: str = "h94/IP-Adapter"
    ip_adapter_subfolder: str = "models"
    ip_adapter_weight_name: str = "ip-adapter_sd15.bin"

    image_size: int = 512
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    seed: int = 42

    target_sprite_size: tuple[int, int] = (256, 256)
    palette_colors: int = 32

    device: str = "mps"
    dtype: str = "float32"
    controlnet_conditioning_scale: float = 0.9
    ip_adapter_scale: float = 0.7
    consistency_threshold: float = 0.82
    consistency_max_retries: int = 1
    single_subject_min_confidence: float = 0.62
    single_subject_min_area_ratio: float = 0.18
    single_subject_split_enabled: bool = True
    max_extra_generations: int = 1

    profile: Literal["eco", "balanced", "quality"] = "balanced"
    allow_api_fallback: bool = False
    quick_mode_frames: int = 4


PROFILE_PRESETS: dict[str, dict] = {
    "eco": {
        "image_size": 320,
        "num_inference_steps": 12,
        "guidance_scale": 6.5,
        "target_sprite_size": (192, 192),
        "quick_mode_frames": 3,
        "consistency_threshold": 0.8,
        "consistency_max_retries": 0,
        "single_subject_min_confidence": 0.58,
        "single_subject_min_area_ratio": 0.18,
        "single_subject_split_enabled": True,
        "max_extra_generations": 0,
    },
    "balanced": {
        "image_size": 384,
        "num_inference_steps": 16,
        "guidance_scale": 7.0,
        "target_sprite_size": (224, 224),
        "quick_mode_frames": 4,
        "consistency_threshold": 0.9,
        "consistency_max_retries": 1,
        "single_subject_min_confidence": 0.62,
        "single_subject_min_area_ratio": 0.18,
        "single_subject_split_enabled": True,
        "max_extra_generations": 1,
    },
    "quality": {
        "image_size": 512,
        "num_inference_steps": 22,
        "guidance_scale": 7.5,
        "target_sprite_size": (256, 256),
        "quick_mode_frames": 6,
        "consistency_threshold": 0.95,
        "consistency_max_retries": 2,
        "single_subject_min_confidence": 0.66,
        "single_subject_min_area_ratio": 0.16,
        "single_subject_split_enabled": True,
        "max_extra_generations": 2,
    },
}


DEFAULT_CONFIG = GenerationConfig()


def config_for_profile(profile: str = "balanced", base: GenerationConfig | None = None) -> GenerationConfig:
    if profile not in PROFILE_PRESETS:
        raise ValueError(f"Unknown profile: {profile}")
    source = base or DEFAULT_CONFIG
    preset = PROFILE_PRESETS[profile]
    return source.model_copy(
        update={
            "profile": profile,
            "image_size": preset["image_size"],
            "num_inference_steps": preset["num_inference_steps"],
            "guidance_scale": preset["guidance_scale"],
            "target_sprite_size": preset["target_sprite_size"],
            "quick_mode_frames": preset["quick_mode_frames"],
            "consistency_threshold": preset["consistency_threshold"],
            "consistency_max_retries": preset["consistency_max_retries"],
            "single_subject_min_confidence": preset["single_subject_min_confidence"],
            "single_subject_min_area_ratio": preset["single_subject_min_area_ratio"],
            "single_subject_split_enabled": preset["single_subject_split_enabled"],
            "max_extra_generations": preset["max_extra_generations"],
        },
        deep=True,
    )


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

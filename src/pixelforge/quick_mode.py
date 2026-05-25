"""Quick Mode: reference → ControlNet+IP-Adapter walk cycle (4 frames, lighter than Normal Mode)."""
from __future__ import annotations
import gc
import os
from pixelforge.config import DEFAULT_CONFIG, config_for_profile
from pixelforge.prompt_engineer import build_negative_prompt
from pixelforge.reference_builder import ReferenceBuilder
from pixelforge.frame_generator import FrameGenerator
from pixelforge.pose_library import PoseLibrary
from PIL import Image
import torch


class QuickModeGenerator:
    def __init__(self, config=None):
        self.cfg = config or DEFAULT_CONFIG

    def generate(
        self,
        user_prompt: str,
        negative_prompt: str = "",
        profile: str | None = None,
        n_poses: int | None = None,
        allow_api_fallback: bool | None = None,
        progress_callback=None,
    ) -> list[Image.Image]:
        cfg = config_for_profile(profile or self.cfg.profile, base=self.cfg)
        if allow_api_fallback is not None:
            cfg = cfg.model_copy(update={"allow_api_fallback": allow_api_fallback})
        self.cfg = cfg
        pose_count = n_poses if n_poses is not None else cfg.quick_mode_frames

        neg = negative_prompt or build_negative_prompt()

        ref = self._generate_reference(user_prompt, neg, progress_callback, cfg)

        del self._ref_builder
        gc.collect()
        _safe_cleanup()

        frames = self._generate_walk_frames(
            user_prompt, neg, ref, pose_count, progress_callback, cfg
        )

        del self._frame_gen
        gc.collect()
        _safe_cleanup()

        return frames

    def _generate_reference(self, prompt: str, neg: str, progress_callback, cfg) -> Image.Image:
        self._ref_builder = ReferenceBuilder(cfg)
        ref = self._ref_builder.generate(prompt, neg, progress_callback=progress_callback)
        return ref

    def _generate_walk_frames(
        self,
        prompt: str,
        neg: str,
        reference: Image.Image,
        n_poses: int,
        progress_callback,
        cfg,
    ) -> list[Image.Image]:
        lib = PoseLibrary()
        walk_seq = lib.load("walk")
        poses = walk_seq.frames[:n_poses]

        self._frame_gen = FrameGenerator(cfg)
        self._frame_gen.set_reference(reference)
        return self._frame_gen.generate_frames(
            prompt, neg, poses, progress_callback=progress_callback,
        )


def _safe_cleanup() -> None:
    try:
        allow_mps_empty_cache = os.environ.get("PIXELFORGE_MPS_EMPTY_CACHE", "0") == "1"
        if (
            allow_mps_empty_cache
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            torch.mps.empty_cache()
    except Exception:
        pass

"""Quick Mode: reference → ControlNet+IP-Adapter walk cycle (4 frames, lighter than Normal Mode)."""
from __future__ import annotations
import gc
from pixelforge.config import DEFAULT_CONFIG
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
        n_poses: int = 4,
        progress_callback=None,
    ) -> list[Image.Image]:
        neg = negative_prompt or build_negative_prompt()

        ref = self._generate_reference(user_prompt, neg, progress_callback)

        del self._ref_builder
        gc.collect()
        torch.mps.empty_cache()

        frames = self._generate_walk_frames(user_prompt, neg, ref, n_poses, progress_callback)

        del self._frame_gen
        gc.collect()
        torch.mps.empty_cache()

        return frames

    def _generate_reference(self, prompt: str, neg: str, progress_callback) -> Image.Image:
        self._ref_builder = ReferenceBuilder(self.cfg)
        ref = self._ref_builder.generate(prompt, neg, progress_callback=progress_callback)
        return ref

    def _generate_walk_frames(
        self,
        prompt: str,
        neg: str,
        reference: Image.Image,
        n_poses: int,
        progress_callback,
    ) -> list[Image.Image]:
        lib = PoseLibrary()
        walk_seq = lib.load("walk")
        poses = walk_seq.frames[:n_poses]

        self._frame_gen = FrameGenerator(self.cfg)
        self._frame_gen.set_reference(reference)
        return self._frame_gen.generate_frames(
            prompt, neg, poses, progress_callback=progress_callback,
        )

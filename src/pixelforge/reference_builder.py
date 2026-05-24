"""Generate a single 'master' character reference image with SD1.5."""
from __future__ import annotations
from pixelforge.config import DEFAULT_CONFIG, ensure_cached

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
        ensure_cached(self.cfg.sd_model_id)
        dtype = torch.float16 if self.cfg.dtype == "float16" else torch.float32
        self._pipe = StableDiffusionPipeline.from_pretrained(
            self.cfg.sd_model_id,
            local_files_only=True,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(self.cfg.device)
        self._pipe.enable_attention_slicing()
        try:
            self._pipe.vae.enable_slicing()
        except AttributeError:
            pass
        logger.info("SD1.5 loaded.")

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        seed: int | None = None,
        progress_callback=None,
    ) -> Image.Image:
        self._ensure_loaded()
        generator = torch.Generator(device=self.cfg.device).manual_seed(
            seed if seed is not None else self.cfg.seed
        )
        total_steps = self.cfg.num_inference_steps

        def _on_step(pipe, step_index, timestep, callback_kwargs):
            step = step_index + 1
            if progress_callback:
                progress_callback(step / total_steps,
                                  f"Reference image — step {step}/{total_steps}")
            return callback_kwargs

        result = self._pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=total_steps,
            guidance_scale=self.cfg.guidance_scale,
            width=self.cfg.image_size,
            height=self.cfg.image_size,
            generator=generator,
            callback_on_step_end=_on_step,
        )
        return result.images[0]

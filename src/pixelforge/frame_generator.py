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
        # Load IP-Adapter
        logger.info("Loading IP-Adapter...")
        self._pipe.load_ip_adapter(
            self.cfg.ip_adapter_repo,
            subfolder=self.cfg.ip_adapter_subfolder,
            weight_name=self.cfg.ip_adapter_weight_name,
        )
        self._pipe.set_ip_adapter_scale(self.cfg.ip_adapter_scale)
        logger.info("FrameGenerator pipeline + IP-Adapter loaded.")

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

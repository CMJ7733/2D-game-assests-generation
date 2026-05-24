"""Generate animation frames conditioned on pose skeletons (and later IP-Adapter)."""
from __future__ import annotations
import threading
from pixelforge.config import DEFAULT_CONFIG, ensure_cached

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
        ensure_cached(self.cfg.controlnet_model_id)
        controlnet = ControlNetModel.from_pretrained(
            self.cfg.controlnet_model_id, local_files_only=True, torch_dtype=dtype
        )
        logger.info("Loading SD1.5 + ControlNet pipeline...")
        ensure_cached(self.cfg.sd_model_id)
        self._pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.cfg.sd_model_id,
            local_files_only=True,
            controlnet=controlnet,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(self.cfg.device)
        # Load IP-Adapter
        logger.info("Loading IP-Adapter...")
        ensure_cached(
            self.cfg.ip_adapter_repo,
            allow_patterns=[f"{self.cfg.ip_adapter_subfolder}/{self.cfg.ip_adapter_weight_name}"],
        )
        self._pipe.load_ip_adapter(
            self.cfg.ip_adapter_repo,
            subfolder=self.cfg.ip_adapter_subfolder,
            weight_name=self.cfg.ip_adapter_weight_name,
            local_files_only=True,
        )
        self._pipe.set_ip_adapter_scale(self.cfg.ip_adapter_scale)
        # NOTE: do NOT call enable_attention_slicing() — it is incompatible with
        # IP-Adapter in diffusers >= 0.37 (SlicedAttnProcessor can't handle tuple
        # encoder_hidden_states from IP-Adapter). VAE slicing is safe and sufficient.
        try:
            self._pipe.vae.enable_slicing()
        except AttributeError:
            pass
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
        progress_callback=None,
        stop_event: threading.Event | None = None,
    ) -> list[Image.Image]:
        self._ensure_loaded()
        seed_val = seed if seed is not None else self.cfg.seed
        n_frames = len(pose_images)
        total_steps = self.cfg.num_inference_steps
        results: list[Image.Image] = []
        for i, pose in enumerate(pose_images):
            if stop_event and stop_event.is_set():
                logger.info(f"Frame generation stopped by user at frame {i+1}/{n_frames}.")
                break
            generator = torch.Generator(device=self.cfg.device).manual_seed(seed_val)
            frame_start = i / n_frames
            frame_range = 1.0 / n_frames
            cur_i = i

            def _on_step(pipe, step_index, timestep, callback_kwargs, _i=cur_i, _fs=frame_start, _fr=frame_range):
                step = step_index + 1
                if progress_callback:
                    frac = _fs + (step / total_steps) * _fr
                    progress_callback(frac, f"Frame {_i+1}/{n_frames} — step {step}/{total_steps}")
                return callback_kwargs

            pipe_kwargs = dict(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=pose,
                num_inference_steps=total_steps,
                guidance_scale=self.cfg.guidance_scale,
                controlnet_conditioning_scale=self.cfg.controlnet_conditioning_scale,
                generator=generator,
                callback_on_step_end=_on_step,
            )
            if self._reference_image is not None:
                pipe_kwargs["ip_adapter_image"] = self._reference_image
            output = self._pipe(**pipe_kwargs)
            results.append(output.images[0])
            logger.info(f"Frame {i+1}/{n_frames} generated.")
        return results

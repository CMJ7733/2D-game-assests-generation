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

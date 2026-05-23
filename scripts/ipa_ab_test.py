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

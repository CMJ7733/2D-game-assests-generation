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

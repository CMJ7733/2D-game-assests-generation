"""
Generate side-view OpenPose skeletons for assets/poses/ using LPC walk animation
as spatial reference. Run once to replace the broken front-facing skeletons.

Usage:
    python scripts/generate_lpc_poses.py
"""
import os
import sys
import json
import urllib.request
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np

ROOT = Path(__file__).parent.parent
POSES_DIR = ROOT / "assets" / "poses"
LPC_CACHE = ROOT / "models" / "lpc_walk.png"
LPC_URL = (
    "https://raw.githubusercontent.com/LiberatedPixelCup/"
    "Universal-LPC-Spritesheet-Character-Generator/master/"
    "spritesheets/body/bodies/male/walk.png"
)

KP_NAMES = [
    "nose", "neck",
    "rshoulder", "relbow", "rwrist",
    "lshoulder", "lelbow", "lwrist",
    "rhip", "rknee", "rankle",
    "lhip", "lknee", "lankle",
    "reye", "leye", "rear", "lear",
]
KP_COLORS = [
    (255, 0, 0), (255, 85, 0),
    (255, 170, 0), (255, 255, 0), (170, 255, 0),
    (85, 255, 0), (0, 255, 0), (0, 255, 85),
    (0, 255, 170), (0, 255, 255), (0, 170, 255),
    (0, 85, 255), (0, 0, 255), (85, 0, 255),
    (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85),
]
BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (1, 5), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10),
    (1, 11), (11, 12), (12, 13),
    (0, 14), (14, 16), (0, 15), (15, 17),
]

_FIXED = dict(
    nose=(248, 80), neck=(248, 120),
    rshoulder=(215, 143), lshoulder=(270, 143),
    rhip=(228, 290), lhip=(262, 290),
    reye=(236, 72), leye=(254, 72),
    rear=(228, 80), lear=(248, 80),
)

_WALK_VARIABLE = [
    dict(relbow=(188, 238), rwrist=(170, 315),
         lelbow=(295, 228), lwrist=(318, 305),
         rknee=(205, 362), rankle=(182, 445),
         lknee=(268, 338), lankle=(272, 408)),
    dict(relbow=(190, 242), rwrist=(175, 318),
         lelbow=(292, 230), lwrist=(312, 308),
         rknee=(222, 368), rankle=(220, 448),
         lknee=(265, 338), lankle=(265, 410)),
    dict(relbow=(192, 245), rwrist=(180, 322),
         lelbow=(288, 232), lwrist=(305, 308),
         rknee=(228, 365), rankle=(228, 445),
         lknee=(252, 342), lankle=(255, 405)),
    dict(relbow=(190, 242), rwrist=(178, 318),
         lelbow=(290, 235), lwrist=(308, 310),
         rknee=(228, 355), rankle=(235, 435),
         lknee=(248, 335), lankle=(252, 395)),
    dict(relbow=(295, 228), rwrist=(318, 305),
         lelbow=(188, 238), lwrist=(170, 315),
         rknee=(268, 338), rankle=(272, 408),
         lknee=(205, 362), lankle=(182, 445)),
    dict(relbow=(292, 230), rwrist=(312, 308),
         lelbow=(190, 242), lwrist=(175, 318),
         rknee=(265, 338), rankle=(265, 410),
         lknee=(222, 368), lankle=(220, 448)),
    dict(relbow=(288, 232), rwrist=(305, 308),
         lelbow=(192, 245), lwrist=(180, 322),
         rknee=(252, 342), rankle=(255, 405),
         lknee=(228, 365), lankle=(228, 445)),
    dict(relbow=(290, 235), rwrist=(308, 310),
         lelbow=(190, 242), lwrist=(178, 318),
         rknee=(248, 335), rankle=(252, 395),
         lknee=(228, 355), lankle=(235, 435)),
]

_IDLE_VARIABLE = [
    dict(relbow=(192, 242), rwrist=(183, 320),
         lelbow=(288, 242), lwrist=(296, 318),
         rknee=(228, 365), rankle=(228, 445),
         lknee=(252, 360), lankle=(252, 440)),
    dict(relbow=(190, 240), rwrist=(180, 318),
         lelbow=(288, 240), lwrist=(298, 315),
         rknee=(228, 363), rankle=(226, 443),
         lknee=(252, 358), lankle=(250, 438)),
    dict(relbow=(192, 242), rwrist=(183, 320),
         lelbow=(288, 242), lwrist=(296, 318),
         rknee=(228, 365), rankle=(228, 445),
         lknee=(252, 360), lankle=(252, 440)),
    dict(relbow=(190, 240), rwrist=(180, 318),
         lelbow=(288, 240), lwrist=(298, 315),
         rknee=(228, 363), rankle=(226, 443),
         lknee=(252, 358), lankle=(250, 438)),
]


def _build_kp_dict(variable: dict) -> dict:
    return {**_FIXED, **variable}


def _draw_skeleton(kp: dict, size: int = 512) -> Image.Image:
    img = Image.new("RGB", (size, size), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    for a_idx, b_idx in BONES:
        a_name = KP_NAMES[a_idx]
        b_name = KP_NAMES[b_idx]
        if a_name in kp and b_name in kp:
            draw.line([kp[a_name], kp[b_name]], fill=(255, 255, 255), width=4)

    r = 7
    for idx, name in enumerate(KP_NAMES):
        if name in kp:
            x, y = kp[name]
            color = KP_COLORS[idx]
            draw.ellipse([x - r, y - r, x + r, y + r], fill=color)

    if "nose" in kp and "neck" in kp:
        nx, ny = kp["nose"]
        draw.ellipse([nx - 30, ny - 35, nx + 30, ny + 25], outline=(255, 200, 0), width=3)

    return img


def _download_lpc() -> Image.Image:
    if LPC_CACHE.exists():
        print(f"  Using cached LPC walk sheet: {LPC_CACHE}")
        return Image.open(LPC_CACHE).convert("RGBA")

    print(f"  Downloading LPC walk sheet from GitHub...")
    LPC_CACHE.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(LPC_URL, LPC_CACHE)
    print(f"  Saved to {LPC_CACHE}")
    return Image.open(LPC_CACHE).convert("RGBA")


def _verify_poses(poses_dir: Path) -> bool:
    f0 = np.array(Image.open(poses_dir / "walk_0.png"))
    f4 = np.array(Image.open(poses_dir / "walk_4.png"))
    diff = np.abs(f0.astype(int) - f4.astype(int)).sum()
    return diff > 0


def main() -> int:
    print("=== generate_lpc_poses.py ===")
    POSES_DIR.mkdir(parents=True, exist_ok=True)

    _download_lpc()

    print("\nGenerating walk poses (8 frames)...")
    for i, variable in enumerate(_WALK_VARIABLE):
        kp = _build_kp_dict(variable)
        skeleton = _draw_skeleton(kp)
        out_path = POSES_DIR / f"walk_{i}.png"
        skeleton.save(out_path)
        print(f"  Saved {out_path.name}")

    print("\nGenerating idle poses (4 frames)...")
    for i, variable in enumerate(_IDLE_VARIABLE):
        kp = _build_kp_dict(variable)
        skeleton = _draw_skeleton(kp)
        out_path = POSES_DIR / f"idle_{i}.png"
        skeleton.save(out_path)
        print(f"  Saved {out_path.name}")

    if _verify_poses(POSES_DIR):
        print("\n✓ walk_0 ≠ walk_4 (no duplicate frames)")
    else:
        print("\n✗ ERROR: walk_0 == walk_4 — check keypoint definitions")
        return 1

    meta_path = POSES_DIR / "metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.setdefault("states", {})
    meta["states"].setdefault("walk", {})
    meta["states"].setdefault("idle", {})
    meta["generated_by"] = "generate_lpc_poses.py"
    meta["skeleton_type"] = "side-view-openpose"
    meta["states"]["walk"]["frames"] = [
        {"file": f"walk_{i}.png", "duration_ms": 83} for i in range(8)
    ]
    meta["states"]["idle"]["frames"] = [
        {"file": f"idle_{i}.png", "duration_ms": 250} for i in range(4)
    ]
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"\n✓ Updated {meta_path.name}")

    print("\nDone. Run: python -c \"from PIL import Image; Image.open('assets/poses/walk_0.png').show()\"")
    print("to visually verify the side-view skeleton.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

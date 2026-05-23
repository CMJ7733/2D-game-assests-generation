"""Load pre-built pose skeleton sequences for ControlNet conditioning."""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from PIL import Image
from pixelforge.config import POSES_DIR


@dataclass
class AnimationState:
    name: str
    frames: list[Image.Image]
    fps: int
    loop: bool


@dataclass
class CombinedSequence:
    frames: list[Image.Image]
    animations: dict[str, dict]
    fps_per_state: dict[str, int]


class PoseLibrary:
    def __init__(self, poses_dir: Path | None = None):
        self.dir = poses_dir or POSES_DIR
        self._meta = json.loads((self.dir / "metadata.json").read_text())

    def load(self, state: str) -> AnimationState:
        if state not in self._meta["states"]:
            raise KeyError(f"Unknown state: {state}")
        s = self._meta["states"][state]
        frames = [Image.open(self.dir / f["file"]).convert("RGB") for f in s["frames"]]
        return AnimationState(name=state, frames=frames, fps=s["fps"], loop=s["loop"])

    def load_combined(self, states: list[str]) -> CombinedSequence:
        all_frames: list[Image.Image] = []
        animations: dict[str, dict] = {}
        fps_per_state: dict[str, int] = {}
        idx = 0
        for state in states:
            anim = self.load(state)
            indices = list(range(idx, idx + len(anim.frames)))
            animations[state] = {"frames": indices, "fps": anim.fps, "loop": anim.loop}
            fps_per_state[state] = anim.fps
            all_frames.extend(anim.frames)
            idx += len(anim.frames)
        return CombinedSequence(
            frames=all_frames, animations=animations, fps_per_state=fps_per_state
        )

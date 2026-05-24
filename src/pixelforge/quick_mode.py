"""Plan B: single-image sprite sheet generation + frame duplication."""
from __future__ import annotations
from PIL import Image
from pixelforge.config import DEFAULT_CONFIG
from pixelforge.prompt_engineer import build_negative_prompt
from pixelforge.reference_builder import ReferenceBuilder


class QuickModeGenerator:
    def __init__(self, config=None):
        self.cfg = config or DEFAULT_CONFIG
        self._ref = ReferenceBuilder(self.cfg)

    def generate(self, user_prompt: str, n_frames: int = 8) -> list[Image.Image]:
        neg = build_negative_prompt()
        img = self._ref.generate(user_prompt, neg)
        return [img.copy() for _ in range(n_frames)]

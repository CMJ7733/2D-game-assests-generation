"""Plan B: single-image sprite sheet generation + slicing."""
from __future__ import annotations
from PIL import Image
from pixelforge.config import DEFAULT_CONFIG
from pixelforge.reference_builder import ReferenceBuilder


class QuickModeGenerator:
    def __init__(self, config=None):
        self.cfg = config or DEFAULT_CONFIG
        self._ref = ReferenceBuilder(self.cfg)

    def _build_sheet_prompt(self, user_prompt: str) -> str:
        return (
            f"{user_prompt}, sprite sheet, horizontal strip, 8 frames walking cycle, "
            "side view, pixel art, game asset, transparent background, white background, "
            "consistent character across frames"
        )

    def _slice_sheet(self, sheet: Image.Image, columns: int = 8, rows: int = 1) -> list[Image.Image]:
        w, h = sheet.size
        frame_w, frame_h = w // columns, h // rows
        frames = []
        for r in range(rows):
            for c in range(columns):
                box = (c * frame_w, r * frame_h, (c + 1) * frame_w, (r + 1) * frame_h)
                frames.append(sheet.crop(box))
        return frames

    def generate(self, user_prompt: str, columns: int = 8) -> list[Image.Image]:
        prompt = self._build_sheet_prompt(user_prompt)
        # Stay within memory: cap to 512x512, slice into 8x1
        sheet = self._ref.generate(prompt, "")
        return self._slice_sheet(sheet, columns=columns, rows=1)

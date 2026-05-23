"""Compose individual sprite frames into a sprite sheet + animation metadata JSON."""
from __future__ import annotations
from dataclasses import dataclass
from PIL import Image
from typing import Any


@dataclass
class FrameMetadata:
    index: int
    state: str
    duration_ms: int


class SheetComposer:
    def compose(
        self,
        frames: list[Image.Image],
        frame_size: tuple[int, int] = (64, 64),
        animations: dict[str, dict[str, Any]] | None = None,
    ) -> tuple[Image.Image, dict[str, Any]]:
        n = len(frames)
        fw, fh = frame_size
        sheet = Image.new("RGBA", (fw * n, fh), (0, 0, 0, 0))
        for i, frame in enumerate(frames):
            if frame.size != (fw, fh):
                frame = frame.resize((fw, fh), Image.NEAREST)
            sheet.paste(frame, (i * fw, 0), frame.convert("RGBA"))

        metadata = {
            "frame_size": [fw, fh],
            "columns": n,
            "rows": 1,
            "frame_count": n,
            "animations": animations or {
                "all": {"frames": list(range(n)), "fps": 8, "loop": True}
            },
        }
        return sheet, metadata

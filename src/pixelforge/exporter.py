"""Export sprite sheet + metadata in multiple game-engine-friendly formats."""
from __future__ import annotations
import json
from pathlib import Path
from PIL import Image
from typing import Any


class Exporter:
    def __init__(self, output_dir: Path | str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_generic(
        self,
        sheet: Image.Image,
        metadata: dict[str, Any],
        base_name: str = "character",
    ) -> dict[str, Path]:
        png_path = self.output_dir / f"{base_name}.png"
        json_path = self.output_dir / f"{base_name}.json"
        sheet.save(png_path)
        json_path.write_text(json.dumps(metadata, indent=2))
        return {"png": png_path, "json": json_path}

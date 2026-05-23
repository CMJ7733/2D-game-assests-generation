"""Export sprite sheet + metadata in multiple game-engine-friendly formats."""
from __future__ import annotations
import json
import uuid
from pathlib import Path
from PIL import Image
from typing import Any
from jinja2 import Environment, FileSystemLoader
from pixelforge.config import TEMPLATES_DIR


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

    def export_godot(
        self,
        metadata: dict,
        base_name: str = "character",
        png_relative: str | None = None,
    ) -> Path:
        env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)))
        template = env.get_template("godot_spriteframes.tres.j2")
        rendered = template.render(
            png_relative=png_relative or f"{base_name}.png",
            frame_count=metadata["frame_count"],
            columns=metadata["columns"],
            frame_w=metadata["frame_size"][0],
            frame_h=metadata["frame_size"][1],
            animations=metadata["animations"],
        )
        out = self.output_dir / f"{base_name}.tres"
        out.write_text(rendered)
        return out

    def export_unity(
        self,
        metadata: dict,
        base_name: str = "character",
    ) -> Path:
        env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)))
        template = env.get_template("unity_sprite.meta.j2")
        sheet_h = metadata["frame_size"][1] * metadata["rows"]
        rendered = template.render(
            guid=uuid.uuid4().hex,
            base_name=base_name,
            frame_count=metadata["frame_count"],
            columns=metadata["columns"],
            frame_w=metadata["frame_size"][0],
            frame_h=metadata["frame_size"][1],
            sheet_h=sheet_h,
        )
        out = self.output_dir / f"{base_name}.png.meta"
        out.write_text(rendered)
        return out

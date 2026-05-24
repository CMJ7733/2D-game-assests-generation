"""Top-level pipeline orchestration: prompt → 4-view character sheet → export."""
from __future__ import annotations
import gc
import threading
from pathlib import Path
from PIL import Image
from typing import Any
import torch
from loguru import logger

from pixelforge.config import DEFAULT_CONFIG, OUTPUT_DIR
from pixelforge.prompt_engineer import enhance_prompt, build_negative_prompt
from pixelforge.reference_builder import ReferenceBuilder
from pixelforge.post_processor import PostProcessor

VIEWS = ["front", "left", "right", "back"]


def generate_character(
    user_prompt: str,
    output_dir: Path | str = OUTPUT_DIR,
    base_name: str = "character",
    progress_callback=None,
    stop_event: threading.Event | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    intermediates: dict[str, Any] = {}

    def _check_stop():
        if stop_event and stop_event.is_set():
            raise StopIteration("Stopped by user")

    def emit(fraction: float, msg: str):
        if progress_callback:
            progress_callback(fraction, msg)
        logger.info(msg)

    neg = build_negative_prompt()
    rb = ReferenceBuilder()
    pp = PostProcessor(
        target_size=DEFAULT_CONFIG.target_sprite_size,
        palette_colors=DEFAULT_CONFIG.palette_colors,
    )

    raw_views: dict[str, Image.Image] = {}
    processed_views: dict[str, Image.Image] = {}

    for i, view in enumerate(VIEWS):
        _check_stop()
        base_frac = i / len(VIEWS)
        next_frac = (i + 1) / len(VIEWS)

        emit(base_frac, f"Generating {view} view ({i+1}/{len(VIEWS)})...")
        prompt = enhance_prompt(user_prompt, view=view)

        def on_progress(frac: float, desc: str, _bf=base_frac, _nf=next_frac):
            emit(_bf + frac * (_nf - _bf) * 0.85, desc)

        img = rb.generate(prompt, neg, progress_callback=on_progress)
        raw_views[view] = img

        processed = pp.process(img)
        processed_views[view] = processed
        emit(next_frac * 0.9, f"{view.capitalize()} view done.")

    del rb
    gc.collect()
    torch.mps.empty_cache()

    intermediates["raw_views"] = raw_views
    intermediates["processed_views"] = processed_views

    sheet = _compose_sheet(processed_views)
    sheet_path = output_dir / f"{base_name}.png"
    sheet.save(sheet_path)

    emit(1.0, "Done!")
    return {
        "sheet": sheet,
        "raw_views": raw_views,
        "processed_views": processed_views,
        "paths": {"png": sheet_path},
        "intermediates": intermediates,
    }


def _compose_sheet(views: dict[str, Image.Image]) -> Image.Image:
    size = DEFAULT_CONFIG.target_sprite_size
    w, h = size
    canvas = Image.new("RGBA", (w * 2, h * 2), (40, 40, 40, 255))
    positions = {"back": (0, 0), "front": (0, h), "left": (w, 0), "right": (w, h)}
    for view, pos in positions.items():
        img = views.get(view)
        if img:
            img = img.resize(size, Image.LANCZOS) if img.size != size else img
            canvas.paste(img, pos)
    return canvas

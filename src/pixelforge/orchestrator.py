"""Top-level pipeline orchestration: prompt → reference → frames → sheet → export."""
from __future__ import annotations
from pathlib import Path
from PIL import Image
from typing import Any
from loguru import logger

from pixelforge.config import DEFAULT_CONFIG, OUTPUT_DIR
from pixelforge.prompt_engineer import enhance_prompt, build_negative_prompt
from pixelforge.reference_builder import ReferenceBuilder
from pixelforge.pose_library import PoseLibrary
from pixelforge.frame_generator import FrameGenerator
from pixelforge.quick_mode import QuickModeGenerator
from pixelforge.post_processor import PostProcessor
from pixelforge.sheet_composer import SheetComposer
from pixelforge.exporter import Exporter


def generate_character(
    user_prompt: str,
    output_dir: Path | str = OUTPUT_DIR,
    base_name: str = "character",
    use_quick_mode: bool = False,
    states: list[str] | None = None,
    progress_callback=None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    states = states or ["idle", "walk"]
    intermediates: dict[str, Any] = {}

    def emit(msg: str, image: Image.Image | None = None):
        if progress_callback:
            progress_callback(msg, image)
        logger.info(msg)

    enhanced = enhance_prompt(user_prompt)
    neg = build_negative_prompt()
    intermediates["enhanced_prompt"] = enhanced

    if use_quick_mode:
        emit("Quick Mode: generating sprite sheet directly...")
        qm = QuickModeGenerator()
        raw_frames = qm.generate(enhanced, columns=8)
        animations = {
            "walk": {"frames": list(range(len(raw_frames))), "fps": 12, "loop": True}
        }
    else:
        emit("Generating reference image...")
        rb = ReferenceBuilder()
        ref = rb.generate(enhanced, neg)
        intermediates["reference"] = ref
        emit("Reference image generated.", ref)

        emit("Loading pose library...")
        lib = PoseLibrary()
        combined = lib.load_combined(states)
        intermediates["poses"] = combined.frames
        animations = combined.animations

        emit(f"Generating {len(combined.frames)} animation frames...")
        fg = FrameGenerator()
        fg.set_reference(ref)
        raw_frames = fg.generate_frames(enhanced, neg, combined.frames)
        intermediates["raw_frames"] = raw_frames

    emit(f"Post-processing {len(raw_frames)} frames...")
    pp = PostProcessor(
        target_size=DEFAULT_CONFIG.target_sprite_size,
        palette_colors=DEFAULT_CONFIG.palette_colors,
    )
    processed = [pp.process(f) for f in raw_frames]
    intermediates["processed_frames"] = processed

    emit("Composing sprite sheet...")
    sc = SheetComposer()
    sheet, meta = sc.compose(
        processed, frame_size=DEFAULT_CONFIG.target_sprite_size, animations=animations
    )
    emit("Exporting...")
    ex = Exporter(output_dir)
    paths = ex.export_generic(sheet, meta, base_name=base_name)
    try:
        paths["tres"] = ex.export_godot(meta, base_name=base_name, png_relative=f"{base_name}.png")
    except Exception as e:
        logger.warning(f"Godot export failed: {e}")
    try:
        paths["meta"] = ex.export_unity(meta, base_name=base_name)
    except Exception as e:
        logger.warning(f"Unity export failed: {e}")

    return {
        "sheet": sheet,
        "metadata": meta,
        "paths": paths,
        "intermediates": intermediates,
    }

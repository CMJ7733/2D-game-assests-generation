"""Top-level pipeline orchestration: prompt → reference → frames → sheet → export."""
from __future__ import annotations
import gc
from pathlib import Path
from PIL import Image
from typing import Any
import torch
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

    def emit(fraction: float, msg: str):
        if progress_callback:
            progress_callback(fraction, msg)
        logger.info(msg)

    # Progress stage boundaries (normal mode)
    P_REF = 0.03   # reference generation starts
    P_REF_END = 0.20
    P_POSE = 0.22
    P_FRAMES_END = 0.85
    P_POST_END = 0.93
    P_SHEET_END = 0.97

    emit(0.02, "Enhancing prompt...")
    enhanced = enhance_prompt(user_prompt)
    neg = build_negative_prompt()
    intermediates["enhanced_prompt"] = enhanced

    if use_quick_mode:
        emit(0.05, "Quick Mode: generating sprite sheet directly...")
        qm = QuickModeGenerator()
        raw_frames = qm.generate(enhanced, columns=8)
        emit(0.85, f"Quick Mode: {len(raw_frames)} frames generated.")
        animations = {
            "walk": {"frames": list(range(len(raw_frames))), "fps": 12, "loop": True}
        }
    else:
        emit(P_REF, "Generating reference image...")
        rb = ReferenceBuilder()

        def on_ref_progress(frac: float, desc: str):
            emit(P_REF + frac * (P_REF_END - P_REF), desc)

        ref = rb.generate(enhanced, neg, progress_callback=on_ref_progress)
        intermediates["reference"] = ref
        del rb
        gc.collect()
        torch.mps.empty_cache()
        emit(P_REF_END, "Reference image generated.")

        emit(P_POSE, "Loading pose library...")
        lib = PoseLibrary()
        combined = lib.load_combined(states)
        intermediates["poses"] = combined.frames
        animations = combined.animations

        n_frames = len(combined.frames)
        emit(P_POSE, f"Generating {n_frames} animation frames...")
        fg = FrameGenerator()
        fg.set_reference(ref)

        def on_frame_progress(frac: float, desc: str):
            emit(P_POSE + frac * (P_FRAMES_END - P_POSE), desc)

        raw_frames = fg.generate_frames(
            enhanced, neg, combined.frames, progress_callback=on_frame_progress
        )
        intermediates["raw_frames"] = raw_frames
        del fg
        gc.collect()
        torch.mps.empty_cache()

    n_proc = len(raw_frames)
    pp = PostProcessor(
        target_size=DEFAULT_CONFIG.target_sprite_size,
        palette_colors=DEFAULT_CONFIG.palette_colors,
    )
    processed = []
    for j, f in enumerate(raw_frames):
        frac = j / max(1, n_proc)
        emit(P_FRAMES_END + frac * (P_POST_END - P_FRAMES_END),
             f"Post-processing frame {j+1}/{n_proc}...")
        processed.append(pp.process(f))
    intermediates["processed_frames"] = processed

    emit(P_SHEET_END, "Composing sprite sheet...")
    sc = SheetComposer()
    sheet, meta = sc.compose(
        processed, frame_size=DEFAULT_CONFIG.target_sprite_size, animations=animations
    )
    emit(P_SHEET_END, "Exporting...")
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

    emit(1.0, "Done!")
    return {
        "sheet": sheet,
        "metadata": meta,
        "paths": paths,
        "intermediates": intermediates,
    }

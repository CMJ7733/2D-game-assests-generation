"""Top-level pipeline orchestration: prompt → 4-view character sheet → export."""
from __future__ import annotations
import gc
import os
import threading
from pathlib import Path
from PIL import Image
from typing import Any
import torch
from loguru import logger

from pixelforge.config import DEFAULT_CONFIG, OUTPUT_DIR, config_for_profile
from pixelforge.consistency import image_similarity
from pixelforge.prompt_engineer import enhance_prompt, build_negative_prompt
from pixelforge.reference_builder import ReferenceBuilder
from pixelforge.post_processor import PostProcessor, SubjectGuard, SubjectGuardResult

VIEWS = ["front", "left", "right", "back"]


def generate_character(
    user_prompt: str,
    output_dir: Path | str = OUTPUT_DIR,
    base_name: str = "character",
    profile: str = "balanced",
    keep_raw_views: bool = False,
    allow_api_fallback: bool = False,
    progress_callback=None,
    stop_event: threading.Event | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = config_for_profile(profile)
    intermediates: dict[str, Any] = {}

    def _check_stop():
        if stop_event and stop_event.is_set():
            raise StopIteration("Stopped by user")

    def emit(fraction: float, msg: str):
        if progress_callback:
            progress_callback(fraction, msg)
        logger.info(msg)

    neg = build_negative_prompt()
    rb = ReferenceBuilder(cfg)
    pp = PostProcessor(
        target_size=cfg.target_sprite_size,
        palette_colors=cfg.palette_colors,
        subject_guard=SubjectGuard(
            min_confidence=cfg.single_subject_min_confidence,
            min_area_ratio=cfg.single_subject_min_area_ratio,
            split_enabled=cfg.single_subject_split_enabled,
        ),
    )

    raw_views: dict[str, Image.Image] = {}
    preview_views: dict[str, Image.Image] = {}
    processed_views: dict[str, Image.Image] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    reference_processed_identity: Image.Image | None = None
    extra_budget = max(0, int(cfg.max_extra_generations))
    degraded_views: list[str] = []

    try:
        for i, view in enumerate(VIEWS):
            _check_stop()
            base_frac = i / len(VIEWS)
            next_frac = (i + 1) / len(VIEWS)

            emit(base_frac, f"Generating {view} view ({i+1}/{len(VIEWS)})...")
            prompt = enhance_prompt(user_prompt, view=view)

            def on_progress(frac: float, desc: str, _bf=base_frac, _nf=next_frac):
                emit(_bf + frac * (_nf - _bf) * 0.85, desc)

            seed = cfg.seed + i * 1000
            candidate = rb.generate(prompt, neg, seed=seed, progress_callback=on_progress)
            best = _evaluate_candidate(
                candidate,
                pp,
                reference_processed_identity,
                min_confidence=cfg.single_subject_min_confidence,
            )

            fallback_triggered = False
            if not best["valid"]:
                while extra_budget > 0:
                    fallback_triggered = True
                    extra_budget -= 1
                    retry_seed = seed + (cfg.max_extra_generations - extra_budget) * 37
                    retry = rb.generate(
                        prompt, neg, seed=retry_seed, progress_callback=on_progress
                    )
                    retry_eval = _evaluate_candidate(
                        retry,
                        pp,
                        reference_processed_identity,
                        min_confidence=cfg.single_subject_min_confidence,
                    )
                    if retry_eval["score"] > best["score"]:
                        best = retry_eval
                    if retry_eval["valid"]:
                        best = retry_eval
                        break

            best_guard: SubjectGuardResult = best["guard"]
            best_guard.fallback_triggered = fallback_triggered

            diagnostics[view] = {
                "single_subject_confidence": best_guard.single_subject_confidence,
                "major_instance_count": best_guard.major_instance_count,
                "split_used": best_guard.split_used,
                "fallback_triggered": best_guard.fallback_triggered,
                "consistency_similarity": best["consistency_similarity"],
                "score": best["score"],
                "extra_budget_remaining": extra_budget,
            }

            if not best_guard.is_single_subject(cfg.single_subject_min_confidence):
                degraded_views.append(view)

            selected_raw: Image.Image = best["raw"]
            selected_processed: Image.Image = best["processed"]
            if keep_raw_views:
                raw_views[view] = selected_raw

            preview_views[view] = selected_raw.resize(cfg.target_sprite_size, Image.LANCZOS)
            processed_views[view] = selected_processed

            if view == "front" or reference_processed_identity is None:
                reference_processed_identity = selected_processed

            emit(
                min(0.99, next_frac * 0.88),
                f"{view.capitalize()} single-subject confidence: {best_guard.single_subject_confidence:.2f} "
                f"(instances={best_guard.major_instance_count}, fallback={best_guard.fallback_triggered})",
            )
            emit(next_frac * 0.9, f"{view.capitalize()} view done.")
    except RuntimeError as exc:
        if _is_memory_error(exc):
            _safe_device_cleanup()
            return {
                "sheet": None,
                "raw_views": raw_views if keep_raw_views else {},
                "preview_views": preview_views,
                "processed_views": processed_views,
                "diagnostics": diagnostics,
                "paths": {},
                "intermediates": intermediates,
                "message": _memory_error_message(profile, allow_api_fallback),
            }
        raise

    del rb
    _safe_device_cleanup()

    intermediates["raw_views"] = raw_views if keep_raw_views else {}
    intermediates["preview_views"] = preview_views
    intermediates["processed_views"] = processed_views
    intermediates["diagnostics"] = diagnostics

    sheet = _compose_sheet(processed_views, size=cfg.target_sprite_size)
    sheet_path = output_dir / f"{base_name}.png"
    sheet.save(sheet_path)

    final_message = (
        "Done!"
        if not degraded_views
        else _single_subject_warning_message(degraded_views, cfg.profile)
    )
    emit(1.0, final_message)
    return {
        "sheet": sheet,
        "raw_views": raw_views if keep_raw_views else {},
        "preview_views": preview_views,
        "processed_views": processed_views,
        "diagnostics": diagnostics,
        "paths": {"png": sheet_path},
        "intermediates": intermediates,
        "message": final_message,
    }


def _compose_sheet(views: dict[str, Image.Image], size: tuple[int, int] = DEFAULT_CONFIG.target_sprite_size) -> Image.Image:
    w, h = size
    canvas = Image.new("RGBA", (w * 2, h * 2), (40, 40, 40, 255))
    positions = {"back": (0, 0), "front": (0, h), "left": (w, 0), "right": (w, h)}
    for view, pos in positions.items():
        img = views.get(view)
        if img:
            img = img.resize(size, Image.LANCZOS) if img.size != size else img
            canvas.paste(img, pos)
    return canvas


def _safe_device_cleanup() -> None:
    gc.collect()

    try:
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        logger.debug(f"cuda cleanup skipped: {exc}")

    try:
        allow_mps_empty_cache = os.environ.get("PIXELFORGE_MPS_EMPTY_CACHE", "0") == "1"
        if (
            allow_mps_empty_cache
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            torch.mps.empty_cache()
    except Exception as exc:
        logger.debug(f"mps cleanup skipped: {exc}")


def _is_memory_error(exc: RuntimeError) -> bool:
    text = str(exc).lower()
    markers = ["out of memory", "mps backend out of memory", "cuda out of memory"]
    return any(marker in text for marker in markers)


def _memory_error_message(profile: str, allow_api_fallback: bool) -> str:
    parts = [
        "Error: generation failed due to out of memory.",
        f"Current profile: {profile}.",
        "Try switching to Eco profile, reducing inference steps, and closing other heavy apps.",
    ]
    if allow_api_fallback:
        parts.append("You can also retry with API fallback enabled.")
    else:
        parts.append("If needed, enable 'allow API fallback' and retry.")
    return " ".join(parts)


def _single_subject_warning_message(views: list[str], profile: str) -> str:
    joined = ", ".join(views)
    return (
        f"Done with warnings: single-subject guard did not fully pass for [{joined}] "
        f"in profile={profile}. Output kept best available candidates."
    )


def _evaluate_candidate(
    raw_img: Image.Image,
    processor: PostProcessor,
    reference_processed: Image.Image | None,
    min_confidence: float,
) -> dict[str, Any]:
    processed, guard = processor.process_with_diagnostics(raw_img)
    consistency_similarity = (
        image_similarity(processed, reference_processed) if reference_processed is not None else 1.0
    )
    # Weighted score: prioritize single-subject confidence, then cross-view consistency.
    score = float(guard.single_subject_confidence + 0.35 * consistency_similarity)
    valid = guard.is_single_subject(min_confidence)
    return {
        "raw": raw_img,
        "processed": processed,
        "guard": guard,
        "consistency_similarity": consistency_similarity,
        "score": score,
        "valid": valid,
    }

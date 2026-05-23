"""PixelForge Gradio UI — full version with intermediate-product panels."""
from __future__ import annotations
from pixelforge.config import DEFAULT_CONFIG, ensure_dirs, OUTPUT_DIR

import gradio as gr
import zipfile
from pathlib import Path
from PIL import Image
from loguru import logger

from pixelforge.orchestrator import generate_character


def make_gif(frames: list[Image.Image], fps: int = 12) -> Path:
    """Write a preview animation GIF from a list of PIL images."""
    out = OUTPUT_DIR / "preview.gif"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        return out
    duration = int(1000 / max(1, fps))
    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=duration,
        disposal=2,
        optimize=False,
    )
    return out


def make_zip(paths: dict, base_name: str) -> Path:
    zpath = OUTPUT_DIR / f"{base_name}.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        for p in paths.values():
            if isinstance(p, Path) and p.exists():
                zf.write(p, arcname=p.name)
    return zpath


def run_pipeline(prompt: str, quick_mode: bool, progress=gr.Progress()):
    ensure_dirs()
    progress(0.0, desc="Starting...")

    try:
        result = generate_character(
            user_prompt=prompt,
            use_quick_mode=quick_mode,
        )

        inter = result["intermediates"]
        ref_img = inter.get("reference")
        poses = inter.get("poses", [])
        raw = inter.get("raw_frames", [])
        processed = inter.get("processed_frames", [])

        gif_path = make_gif(processed, fps=12) if processed else None
        zip_path = make_zip(result["paths"], "character")

        status = f"Done. Output: {result['paths']['png'].name}"

        return (
            result["sheet"],
            ref_img,
            poses[:4] if poses else None,
            raw[:4] if raw else None,
            processed[:4] if processed else None,
            str(gif_path) if gif_path else None,
            str(zip_path),
            status,
        )
    except Exception as e:
        logger.exception("Pipeline failed")
        return (None, None, None, None, None, None, None, f"Error: {e}")


def build_app() -> gr.Blocks:
    with gr.Blocks(title="PixelForge", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            "# PixelForge\n"
            "**Text → consistent pixel-art character sprite sheet with idle + walk animation.**\n\n"
            "Local Stable Diffusion 1.5 + ControlNet OpenPose + IP-Adapter. Exports to Godot / Unity / generic JSON."
        )
        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(
                    label="Character description",
                    placeholder="A knight with a red cape and blonde hair holding a sword",
                    lines=3,
                )
                quick = gr.Checkbox(label="Quick Mode (Plan B: direct sprite-sheet)", value=False)
                btn = gr.Button("Generate", variant="primary", size="lg")
                status = gr.Textbox(label="Status", interactive=False)
                zip_out = gr.File(label="Download all (ZIP)")
            with gr.Column(scale=2):
                sheet_out = gr.Image(label="Final Sprite Sheet", type="pil")
                preview = gr.Image(label="Animation Preview", type="filepath")

        with gr.Accordion("Pipeline Intermediates", open=False):
            with gr.Row():
                ref_out = gr.Image(label="1. Reference Character", type="pil")
                pose_out = gr.Gallery(label="2. Pose Skeletons (first 4)", columns=4, height=120)
                raw_out = gr.Gallery(label="3. Raw Frames (first 4)", columns=4, height=120)
                proc_out = gr.Gallery(label="4. Processed Frames (first 4)", columns=4, height=120)

        btn.click(
            run_pipeline,
            inputs=[prompt, quick],
            outputs=[sheet_out, ref_out, pose_out, raw_out, proc_out, preview, zip_out, status],
        )
    return app


if __name__ == "__main__":
    build_app().launch(server_name="127.0.0.1", server_port=7860)

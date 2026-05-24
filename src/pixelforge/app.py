"""PixelForge Gradio UI — 4-view pixel art character reference sheet generator."""
from __future__ import annotations
from pixelforge.config import ensure_dirs, OUTPUT_DIR

import gradio as gr
import threading
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from loguru import logger

from pixelforge.orchestrator import generate_character

_stop_event = threading.Event()

VIEW_LABELS = {"back": "BACK", "left": "LEFT", "right": "RIGHT", "front": "FRONT"}
VIEW_POSITIONS = {"back": (0, 0), "left": (1, 0), "right": (1, 1), "front": (0, 1)}


def _label_image(img: Image.Image, label: str) -> Image.Image:
    img = img.convert("RGBA")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 16)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (img.width - tw) // 2
    y = img.height - th - 6
    draw.rectangle([x - 4, y - 2, x + tw + 4, y + th + 2], fill=(0, 0, 0, 180))
    draw.text((x, y), label, fill=(255, 255, 255, 230), font=font)
    return img


def _make_character_sheet(raw_views: dict, size: int = 512) -> Image.Image:
    gap = 4
    total_w = size * 2 + gap
    total_h = size * 2 + gap
    canvas = Image.new("RGBA", (total_w, total_h), (30, 30, 35, 255))

    for view, (col, row) in VIEW_POSITIONS.items():
        img = raw_views.get(view)
        if img:
            img = img.resize((size, size), Image.LANCZOS) if img.size != (size, size) else img
            labeled = _label_image(img, VIEW_LABELS[view])
            x = col * (size + gap)
            y = row * (size + gap)
            canvas.paste(labeled, (x, y))

    return canvas


def run_pipeline(prompt: str, progress=gr.Progress()):
    ensure_dirs()
    _stop_event.clear()

    def on_progress(fraction: float, desc: str):
        progress(fraction, desc=desc)

    try:
        result = generate_character(
            user_prompt=prompt,
            progress_callback=on_progress,
            stop_event=_stop_event,
        )

        raw = result["raw_views"]
        processed = result["processed_views"]

        sheet = _make_character_sheet(raw, size=512)
        sheet_path = OUTPUT_DIR / "character_sheet.png"
        sheet.save(sheet_path)

        return (
            raw.get("front"),
            raw.get("left"),
            raw.get("right"),
            raw.get("back"),
            str(sheet_path),
            f"Done! {len(raw)} views generated.",
        )
    except StopIteration:
        return (None, None, None, None, None, "Stopped by user.")
    except Exception as e:
        logger.exception("Pipeline failed")
        return (None, None, None, None, None, f"Error: {e}")


def stop_pipeline():
    _stop_event.set()
    return "Stopping..."


CSS = """
#main-title { text-align: center; margin-bottom: 0.2em; }
#subtitle { text-align: center; color: #888; font-size: 0.9em; margin-bottom: 1em; }
.view-img { border: 2px solid #333; border-radius: 8px; }
.view-label { text-align: center; font-weight: bold; color: #e0e0e0; font-size: 0.85em; margin-top: 0.3em; }
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(title="PixelForge", css=CSS, theme=gr.themes.Default(primary_hue="emerald")) as app:
        gr.HTML('<h1 id="main-title">PixelForge</h1>')
        gr.HTML('<p id="subtitle">Text → 4-View Pixel Art Character Reference Sheet</p>')

        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                prompt = gr.Textbox(
                    label="Character description",
                    placeholder="A knight with a red cape and blonde hair holding a sword",
                    lines=3,
                )
                with gr.Row():
                    btn = gr.Button("Generate", variant="primary", size="lg")
                    stop_btn = gr.Button("Stop", variant="stop", size="lg")
                status = gr.Textbox(label="Status", interactive=False)
                sheet_file = gr.File(label="Download Character Sheet")

        with gr.Row(equal_height=True):
            with gr.Column():
                gr.HTML('<p class="view-label">← Left Side</p>')
                img_left = gr.Image(label="", elem_classes=["view-img"], show_download_button=True)
            with gr.Column():
                gr.HTML('<p class="view-label">Front →</p>')
                img_front = gr.Image(label="", elem_classes=["view-img"], show_download_button=True)

        with gr.Row(equal_height=True):
            with gr.Column():
                gr.HTML('<p class="view-label">Right Side →</p>')
                img_right = gr.Image(label="", elem_classes=["view-img"], show_download_button=True)
            with gr.Column():
                gr.HTML('<p class="view-label">← Back</p>')
                img_back = gr.Image(label="", elem_classes=["view-img"], show_download_button=True)

        gen_event = btn.click(
            run_pipeline,
            inputs=[prompt],
            outputs=[img_front, img_left, img_right, img_back, sheet_file, status],
        )
        stop_btn.click(
            stop_pipeline,
            inputs=[],
            outputs=[status],
            cancels=[gen_event],
        )
    return app


if __name__ == "__main__":
    build_app().launch(server_name="127.0.0.1", server_port=7860)

"""PixelForge Gradio UI — 4-view pixel art character reference sheet generator."""
from __future__ import annotations
from pixelforge.config import ensure_dirs

import gradio as gr
import threading
from loguru import logger

from pixelforge.orchestrator import generate_character

_stop_event = threading.Event()


def run_pipeline(
    prompt: str,
    profile: str,
    allow_api_fallback: bool,
    progress=gr.Progress(),
):
    ensure_dirs()
    _stop_event.clear()

    def on_progress(fraction: float, desc: str):
        progress(fraction, desc=desc)

    try:
        result = generate_character(
            user_prompt=prompt,
            profile=profile.lower(),
            keep_raw_views=False,
            allow_api_fallback=allow_api_fallback,
            progress_callback=on_progress,
            stop_event=_stop_event,
        )

        if not result["paths"]:
            return (None, None, None, None, None, result["message"])

        processed = result["processed_views"]
        sheet_path = result["paths"]["png"]

        progress(1.0, desc="Done!")

        return (
            processed.get("front"),
            processed.get("left"),
            processed.get("right"),
            processed.get("back"),
            str(sheet_path),
            result["message"],
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
    with gr.Blocks(title="PixelForge") as app:
        gr.HTML('<h1 id="main-title">PixelForge</h1>')
        gr.HTML('<p id="subtitle">Text → 4-View Pixel Art Character Reference</p>')

        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                prompt = gr.Textbox(
                    label="Character description",
                    placeholder="A knight with a red cape and blonde hair holding a sword",
                    lines=3,
                )
                profile = gr.Dropdown(
                    choices=["Eco", "Balanced", "Quality"],
                    value="Balanced",
                    label="Performance profile",
                    info="Eco uses less memory and is faster. Quality uses more memory.",
                )
                allow_api_fallback = gr.Checkbox(
                    value=False,
                    label="Allow API fallback",
                    info="Only used when local generation fails and API keys are configured.",
                )
                with gr.Row():
                    btn = gr.Button("Generate", variant="primary", size="lg")
                    stop_btn = gr.Button("Stop", variant="stop", size="lg")
                status = gr.Textbox(label="Status", interactive=False)
                sheet_file = gr.File(label="Download Character Sheet")

        with gr.Row(equal_height=True):
            with gr.Column():
                gr.HTML('<p class="view-label">← Left Side</p>')
                img_left = gr.Image(label="", elem_classes=["view-img"])
            with gr.Column():
                gr.HTML('<p class="view-label">Front →</p>')
                img_front = gr.Image(label="", elem_classes=["view-img"])

        with gr.Row(equal_height=True):
            with gr.Column():
                gr.HTML('<p class="view-label">Right Side →</p>')
                img_right = gr.Image(label="", elem_classes=["view-img"])
            with gr.Column():
                gr.HTML('<p class="view-label">← Back</p>')
                img_back = gr.Image(label="", elem_classes=["view-img"])

        gen_event = btn.click(
            run_pipeline,
            inputs=[prompt, profile, allow_api_fallback],
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
    build_app().launch(server_name="127.0.0.1", server_port=7860, css=CSS, theme=gr.themes.Default(primary_hue="emerald"))

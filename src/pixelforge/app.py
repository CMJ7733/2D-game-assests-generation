"""Gradio entry point. Start with Quick Mode end-to-end."""
from __future__ import annotations
from pixelforge.config import DEFAULT_CONFIG, ensure_dirs, OUTPUT_DIR

import gradio as gr
from PIL import Image
from pathlib import Path

from pixelforge.prompt_engineer import enhance_prompt, build_negative_prompt
from pixelforge.quick_mode import QuickModeGenerator
from pixelforge.post_processor import PostProcessor
from pixelforge.sheet_composer import SheetComposer
from pixelforge.exporter import Exporter


def run_quick_mode(prompt: str) -> tuple[Image.Image, str]:
    ensure_dirs()
    enhanced = enhance_prompt(prompt)
    qm = QuickModeGenerator()
    raw_frames = qm.generate(enhanced, columns=8)

    pp = PostProcessor(target_size=DEFAULT_CONFIG.target_sprite_size,
                       palette_colors=DEFAULT_CONFIG.palette_colors)
    processed = [pp.process(f) for f in raw_frames]

    sc = SheetComposer()
    sheet, meta = sc.compose(
        processed,
        frame_size=DEFAULT_CONFIG.target_sprite_size,
        animations={
            "walk": {"frames": list(range(len(processed))), "fps": 12, "loop": True}
        },
    )

    ex = Exporter(OUTPUT_DIR)
    paths = ex.export_generic(sheet, meta, base_name="character_quick")
    return sheet, f"Saved to {paths['png']} and {paths['json']}"


def build_app() -> gr.Blocks:
    with gr.Blocks(title="PixelForge — Quick Mode") as app:
        gr.Markdown("# PixelForge\n*Quick Mode: text → pixel character sprite sheet*")
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Character description",
                                    placeholder="A knight with a red cape and blonde hair")
                btn = gr.Button("Generate", variant="primary")
                status = gr.Textbox(label="Status", interactive=False)
            with gr.Column():
                output = gr.Image(label="Sprite Sheet", type="pil")
        btn.click(run_quick_mode, inputs=prompt, outputs=[output, status])
    return app


if __name__ == "__main__":
    build_app().launch(server_name="127.0.0.1", server_port=7860)

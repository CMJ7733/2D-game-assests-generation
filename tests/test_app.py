from pathlib import Path
from unittest.mock import patch

from PIL import Image

from pixelforge.app import run_pipeline


@patch("pixelforge.app.generate_character")
def test_run_pipeline_uses_processed_views_for_ui(mock_generate):
    front_preview = Image.new("RGB", (32, 32), "red")
    front_processed = Image.new("RGB", (32, 32), "blue")

    mock_generate.return_value = {
        "paths": {"png": Path("output/character.png")},
        "preview_views": {
            "front": front_preview,
            "left": front_preview,
            "right": front_preview,
            "back": front_preview,
        },
        "processed_views": {
            "front": front_processed,
            "left": front_processed,
            "right": front_processed,
            "back": front_processed,
        },
        "message": "Done!",
    }

    out = run_pipeline("knight", "Balanced", False, progress=lambda *a, **k: None)
    img_front = out[0]
    assert img_front is front_processed

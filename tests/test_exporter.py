import json
from pathlib import Path
from PIL import Image
from pixelforge.exporter import Exporter


def test_export_generic_writes_png_and_json(tmp_path):
    sheet = Image.new("RGBA", (256, 64), (255, 0, 0, 255))
    meta = {
        "frame_size": [64, 64],
        "columns": 4,
        "rows": 1,
        "frame_count": 4,
        "animations": {"idle": {"frames": [0, 1, 2, 3], "fps": 6, "loop": True}},
    }
    ex = Exporter(output_dir=tmp_path)
    paths = ex.export_generic(sheet, meta, base_name="hero")
    assert (tmp_path / "hero.png").exists()
    assert (tmp_path / "hero.json").exists()
    loaded = json.loads((tmp_path / "hero.json").read_text())
    assert loaded["frame_count"] == 4
    assert paths["png"].name == "hero.png"
    assert paths["json"].name == "hero.json"

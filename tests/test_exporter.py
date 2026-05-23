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


def test_export_godot_writes_tres_file(tmp_path):
    sheet = Image.new("RGBA", (256, 64), (255, 0, 0, 255))
    meta = {
        "frame_size": [64, 64],
        "columns": 4,
        "rows": 1,
        "frame_count": 4,
        "animations": {"idle": {"frames": [0, 1, 2, 3], "fps": 6, "loop": True}},
    }
    ex = Exporter(output_dir=tmp_path)
    ex.export_generic(sheet, meta, base_name="hero")
    tres_path = ex.export_godot(meta, base_name="hero", png_relative="hero.png")
    assert tres_path.exists()
    content = tres_path.read_text()
    assert "SpriteFrames" in content
    assert "idle" in content


def test_export_unity_writes_meta_file(tmp_path):
    sheet = Image.new("RGBA", (256, 64), (255, 0, 0, 255))
    meta = {
        "frame_size": [64, 64],
        "columns": 4,
        "rows": 1,
        "frame_count": 4,
        "animations": {"idle": {"frames": [0, 1, 2, 3], "fps": 6, "loop": True}},
    }
    ex = Exporter(output_dir=tmp_path)
    ex.export_generic(sheet, meta, base_name="hero")
    meta_path = ex.export_unity(meta, base_name="hero")
    assert meta_path.exists()
    content = meta_path.read_text()
    assert "TextureImporter" in content
    assert "spriteSheet" in content

from PIL import Image
from pixelforge.sheet_composer import SheetComposer, FrameMetadata


def test_compose_horizontal_strip():
    frames = [Image.new("RGBA", (64, 64), (i * 30, 0, 0, 255)) for i in range(4)]
    sc = SheetComposer()
    sheet, metadata = sc.compose(frames, frame_size=(64, 64))
    assert sheet.size == (256, 64)
    assert metadata["frame_size"] == [64, 64]
    assert metadata["columns"] == 4


def test_compose_includes_animation_definitions():
    frames = [Image.new("RGBA", (64, 64), (0, 0, 0, 0)) for _ in range(12)]
    sc = SheetComposer()
    sheet, meta = sc.compose(
        frames,
        frame_size=(64, 64),
        animations={
            "idle": {"frames": [0, 1, 2, 3], "fps": 6, "loop": True},
            "walk": {"frames": list(range(4, 12)), "fps": 12, "loop": True},
        },
    )
    assert "idle" in meta["animations"]
    assert "walk" in meta["animations"]
    assert meta["animations"]["walk"]["fps"] == 12
    assert meta["animations"]["idle"]["loop"] is True

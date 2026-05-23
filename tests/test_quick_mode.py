from unittest.mock import MagicMock, patch
from PIL import Image
from pixelforge.quick_mode import QuickModeGenerator


def test_quick_mode_uses_sprite_sheet_prompt_anchors():
    qm = QuickModeGenerator()
    prompt = qm._build_sheet_prompt("knight")
    assert "sprite sheet" in prompt.lower()
    assert "horizontal" in prompt.lower() or "grid" in prompt.lower()
    assert "knight" in prompt


def test_slice_sheet_returns_expected_frame_count():
    sheet = Image.new("RGB", (512, 64), "white")
    qm = QuickModeGenerator()
    frames = qm._slice_sheet(sheet, columns=8)
    assert len(frames) == 8
    assert frames[0].size == (64, 64)


def test_slice_sheet_handles_2x4_grid():
    sheet = Image.new("RGB", (256, 128), "white")
    qm = QuickModeGenerator()
    frames = qm._slice_sheet(sheet, columns=4, rows=2)
    assert len(frames) == 8
    assert frames[0].size == (64, 64)

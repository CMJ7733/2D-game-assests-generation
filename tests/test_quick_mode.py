from unittest.mock import patch
from PIL import Image
from pixelforge.quick_mode import QuickModeGenerator
from pixelforge.prompt_engineer import build_negative_prompt


@patch("pixelforge.quick_mode.ReferenceBuilder")
def test_quick_mode_generates_n_frames(MockBuilder):
    fake_img = Image.new("RGB", (512, 512), "red")
    mock_instance = MockBuilder.return_value
    mock_instance.generate.return_value = fake_img

    qm = QuickModeGenerator()
    frames = qm.generate("knight", n_frames=8)

    assert len(frames) == 8
    assert all(f.size == (512, 512) for f in frames)
    mock_instance.generate.assert_called_once()
    call_args = mock_instance.generate.call_args
    assert "knight" in call_args[0][0]
    assert call_args[0][1] == build_negative_prompt()


@patch("pixelforge.quick_mode.ReferenceBuilder")
def test_quick_mode_passes_negative_prompt(MockBuilder):
    fake_img = Image.new("RGB", (512, 512), "red")
    mock_instance = MockBuilder.return_value
    mock_instance.generate.return_value = fake_img

    qm = QuickModeGenerator()
    qm.generate("knight")

    neg = mock_instance.generate.call_args[0][1]
    assert "front view" in neg
    assert "3d" in neg

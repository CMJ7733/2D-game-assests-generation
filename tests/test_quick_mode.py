from unittest.mock import patch, MagicMock
from PIL import Image
from pixelforge.quick_mode import QuickModeGenerator
from pixelforge.prompt_engineer import build_negative_prompt


@patch("pixelforge.quick_mode.FrameGenerator")
@patch("pixelforge.quick_mode.ReferenceBuilder")
@patch("pixelforge.quick_mode.PoseLibrary")
def test_quick_mode_generates_walk_frames(MockLib, MockBuilder, MockFrameGen):
    fake_ref = Image.new("RGB", (512, 512), "red")
    mock_rb = MockBuilder.return_value
    mock_rb.generate.return_value = fake_ref

    fake_poses = [Image.new("RGB", (512, 512), "blue") for _ in range(4)]
    mock_lib = MockLib.return_value
    mock_walk = MagicMock()
    mock_walk.frames = fake_poses
    mock_lib.load.return_value = mock_walk

    fake_frames = [Image.new("RGB", (512, 512), "green") for _ in range(4)]
    mock_fg = MockFrameGen.return_value
    mock_fg.generate_frames.return_value = fake_frames

    qm = QuickModeGenerator()
    frames = qm.generate("knight", n_poses=4)

    assert len(frames) == 4
    mock_rb.generate.assert_called_once()
    mock_fg.set_reference.assert_called_once_with(fake_ref)
    mock_fg.generate_frames.assert_called_once()
    assert mock_fg.generate_frames.call_args[0][0] == "knight"


@patch("pixelforge.quick_mode.FrameGenerator")
@patch("pixelforge.quick_mode.ReferenceBuilder")
@patch("pixelforge.quick_mode.PoseLibrary")
def test_quick_mode_uses_negative_prompt(MockLib, MockBuilder, MockFrameGen):
    mock_rb = MockBuilder.return_value
    mock_rb.generate.return_value = Image.new("RGB", (512, 512))

    mock_lib = MockLib.return_value
    mock_walk = MagicMock()
    mock_walk.frames = [Image.new("RGB", (512, 512))]
    mock_lib.load.return_value = mock_walk

    mock_fg = MockFrameGen.return_value
    mock_fg.generate_frames.return_value = [Image.new("RGB", (512, 512))]

    qm = QuickModeGenerator()
    qm.generate("knight")

    neg = mock_rb.generate.call_args[0][1]
    assert "front view" in neg
    assert "3d" in neg

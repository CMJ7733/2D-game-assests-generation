from unittest.mock import patch
from PIL import Image

from pixelforge.quick_mode import QuickModeGenerator


@patch("pixelforge.quick_mode.FrameGenerator")
@patch("pixelforge.quick_mode.PoseLibrary")
@patch("pixelforge.quick_mode.ReferenceBuilder")
def test_quick_mode_uses_profile_default_frame_count(
    mock_ref_builder, mock_pose_lib, mock_frame_gen
):
    mock_ref_builder.return_value.generate.return_value = Image.new("RGB", (512, 512), "red")
    mock_pose_lib.return_value.load.return_value.frames = [
        Image.new("RGB", (512, 512), "white") for _ in range(8)
    ]
    mock_frame_gen.return_value.generate_frames.return_value = [
        Image.new("RGB", (512, 512), "blue")
    ]

    qm = QuickModeGenerator()
    qm.generate("knight", profile="balanced")

    passed_poses = mock_frame_gen.return_value.generate_frames.call_args[0][2]
    assert len(passed_poses) == qm.cfg.quick_mode_frames

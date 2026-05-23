from unittest.mock import MagicMock, patch
from PIL import Image
from pathlib import Path
from pixelforge.orchestrator import generate_character


@patch("pixelforge.orchestrator.PoseLibrary")
@patch("pixelforge.orchestrator.FrameGenerator")
@patch("pixelforge.orchestrator.ReferenceBuilder")
def test_orchestrator_full_pipeline_returns_paths(mock_rb, mock_fg, mock_pl, tmp_path):
    # Mock reference
    fake_ref = Image.new("RGB", (512, 512), "red")
    mock_rb.return_value.generate.return_value = fake_ref

    # Mock 12 pose frames
    fake_poses = [Image.new("RGB", (512, 512), "black")] * 12
    fake_combined = MagicMock(
        frames=fake_poses,
        animations={
            "idle": {"frames": list(range(4)), "fps": 6, "loop": True},
            "walk": {"frames": list(range(4, 12)), "fps": 12, "loop": True},
        },
    )
    mock_pl.return_value.load_combined.return_value = fake_combined

    # Mock generated frames
    mock_fg.return_value.generate_frames.return_value = [
        Image.new("RGB", (512, 512), (i * 20, 0, 0)) for i in range(12)
    ]

    result = generate_character(
        user_prompt="knight",
        output_dir=tmp_path,
        use_quick_mode=False,
    )

    assert "png" in result["paths"]
    assert result["paths"]["png"].exists()
    assert result["paths"]["json"].exists()
    assert result["sheet"].size[0] > 0

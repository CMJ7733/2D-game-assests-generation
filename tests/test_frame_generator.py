from unittest.mock import MagicMock, patch
from PIL import Image
from pixelforge.frame_generator import FrameGenerator


@patch("pixelforge.frame_generator.StableDiffusionControlNetPipeline")
@patch("pixelforge.frame_generator.ControlNetModel")
def test_generate_per_pose_returns_one_image_per_pose(mock_cn, mock_pipe_cls):
    pose_a = Image.new("RGB", (512, 512), "black")
    pose_b = Image.new("RGB", (512, 512), "black")

    fake_output = MagicMock(images=[Image.new("RGB", (512, 512), "blue")])
    mock_pipe_instance = MagicMock(return_value=fake_output)
    mock_pipe_cls.from_pretrained.return_value.to.return_value = mock_pipe_instance

    fg = FrameGenerator()
    results = fg.generate_frames(
        prompt="knight",
        negative_prompt="blurry",
        pose_images=[pose_a, pose_b],
    )
    assert len(results) == 2
    assert all(isinstance(img, Image.Image) for img in results)

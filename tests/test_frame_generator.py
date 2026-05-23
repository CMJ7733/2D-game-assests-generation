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


@patch("pixelforge.frame_generator.StableDiffusionControlNetPipeline")
@patch("pixelforge.frame_generator.ControlNetModel")
def test_generate_with_reference_calls_ip_adapter(mock_cn, mock_pipe_cls):
    pose = Image.new("RGB", (512, 512), "black")
    ref = Image.new("RGB", (512, 512), "red")

    mock_pipe_instance = MagicMock(return_value=MagicMock(images=[Image.new("RGB", (512, 512), "green")]))
    mock_pipe_cls.from_pretrained.return_value.to.return_value = mock_pipe_instance

    fg = FrameGenerator()
    fg.set_reference(ref)
    fg.generate_frames(prompt="knight", negative_prompt="", pose_images=[pose])

    # Verify IP-Adapter was loaded
    assert mock_pipe_instance.load_ip_adapter.called
    assert mock_pipe_instance.set_ip_adapter_scale.called
    # Verify reference image was passed via kwargs
    call_kwargs = mock_pipe_instance.call_args.kwargs
    assert "ip_adapter_image" in call_kwargs

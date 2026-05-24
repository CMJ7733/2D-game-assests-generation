from unittest.mock import MagicMock, patch
from PIL import Image
import pytest

from pixelforge.reference_builder import ReferenceBuilder


@patch("pixelforge.reference_builder.ensure_cached")
@patch("pixelforge.reference_builder.StableDiffusionPipeline")
def test_builder_initializes_pipeline_with_correct_model(mock_sd, mock_cache):
    rb = ReferenceBuilder()
    rb._ensure_loaded()
    mock_sd.from_pretrained.assert_called_once()
    call_args = mock_sd.from_pretrained.call_args
    assert "SD_PixelArt_SpriteSheet_Generator" in call_args[0][0]


@patch("pixelforge.reference_builder.ensure_cached")
@patch("pixelforge.reference_builder.StableDiffusionPipeline")
def test_generate_returns_pil_image(mock_sd, mock_cache):
    fake_image = Image.new("RGB", (512, 512), "red")
    mock_pipe_instance = MagicMock()
    mock_pipe_instance.return_value = MagicMock(images=[fake_image])
    mock_sd.from_pretrained.return_value.to.return_value = mock_pipe_instance

    rb = ReferenceBuilder()
    img = rb.generate("knight character", negative_prompt="blurry")
    assert isinstance(img, Image.Image)
    assert img.size == (512, 512)

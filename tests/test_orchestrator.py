from unittest.mock import patch
from PIL import Image
from pixelforge.orchestrator import generate_character


@patch("pixelforge.orchestrator.PostProcessor")
@patch("pixelforge.orchestrator.ReferenceBuilder")
def test_orchestrator_generates_4_views(mock_rb, mock_pp, tmp_path):
    fake_img = Image.new("RGB", (512, 512), "red")
    mock_rb.return_value.generate.return_value = fake_img

    mock_pp_instance = mock_pp.return_value
    mock_pp_instance.process.return_value = Image.new("RGBA", (256, 256), "blue")

    result = generate_character(
        user_prompt="knight",
        output_dir=tmp_path,
    )

    assert "png" in result["paths"]
    assert result["paths"]["png"].exists()
    assert len(result["raw_views"]) == 4
    assert "front" in result["raw_views"]
    assert "back" in result["raw_views"]
    assert mock_rb.return_value.generate.call_count == 4

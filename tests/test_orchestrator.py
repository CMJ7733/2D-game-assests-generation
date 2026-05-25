from unittest.mock import patch
from PIL import Image
from pixelforge.orchestrator import generate_character
from pixelforge.post_processor import SubjectGuardResult


@patch("pixelforge.orchestrator.PostProcessor")
@patch("pixelforge.orchestrator.ReferenceBuilder")
def test_orchestrator_generates_4_views(mock_rb, mock_pp, tmp_path):
    fake_img = Image.new("RGB", (512, 512), "red")
    mock_rb.return_value.generate.return_value = fake_img

    mock_pp_instance = mock_pp.return_value
    mock_pp_instance.process_with_diagnostics.return_value = (
        Image.new("RGBA", (256, 256), "blue"),
        SubjectGuardResult(
            image=Image.new("RGBA", (256, 256), "blue"),
            single_subject_confidence=0.95,
            major_instance_count=1,
            split_used=False,
            fallback_triggered=False,
        ),
    )

    result = generate_character(
        user_prompt="knight",
        output_dir=tmp_path,
    )

    assert "png" in result["paths"]
    assert result["paths"]["png"].exists()
    assert result["raw_views"] == {}
    assert len(result["preview_views"]) == 4
    assert "front" in result["preview_views"]
    assert "back" in result["preview_views"]
    assert "diagnostics" in result
    assert "front" in result["diagnostics"]
    assert "message" in result
    assert "Done!" in result["message"]
    assert mock_rb.return_value.generate.call_count == 4


@patch("pixelforge.orchestrator.PostProcessor")
@patch("pixelforge.orchestrator.ReferenceBuilder")
def test_orchestrator_generates_without_extra_fallback_when_subject_is_valid(
    mock_rb, mock_pp, tmp_path
):
    fake_img = Image.new("RGB", (512, 512), "red")
    mock_rb.return_value.generate.return_value = fake_img
    mock_pp.return_value.process_with_diagnostics.return_value = (
        Image.new("RGBA", (256, 256), "blue"),
        SubjectGuardResult(
            image=Image.new("RGBA", (256, 256), "blue"),
            single_subject_confidence=0.95,
            major_instance_count=1,
            split_used=False,
            fallback_triggered=False,
        ),
    )

    result = generate_character(user_prompt="knight", output_dir=tmp_path)
    assert result["paths"]["png"].exists()
    assert mock_rb.return_value.generate.call_count == 4


@patch("pixelforge.orchestrator.PostProcessor")
@patch("pixelforge.orchestrator.ReferenceBuilder")
def test_orchestrator_can_keep_raw_views(mock_rb, mock_pp, tmp_path):
    fake_img = Image.new("RGB", (512, 512), "red")
    mock_rb.return_value.generate.return_value = fake_img

    mock_pp_instance = mock_pp.return_value
    mock_pp_instance.process_with_diagnostics.return_value = (
        Image.new("RGBA", (256, 256), "blue"),
        SubjectGuardResult(
            image=Image.new("RGBA", (256, 256), "blue"),
            single_subject_confidence=0.95,
            major_instance_count=1,
            split_used=False,
            fallback_triggered=False,
        ),
    )

    result = generate_character(
        user_prompt="knight",
        output_dir=tmp_path,
        keep_raw_views=True,
    )

    assert len(result["raw_views"]) == 4
    assert len(result["preview_views"]) == 4


@patch("pixelforge.orchestrator.torch.mps.empty_cache")
@patch("pixelforge.orchestrator.torch.backends.mps.is_available")
@patch("pixelforge.orchestrator.PostProcessor")
@patch("pixelforge.orchestrator.ReferenceBuilder")
def test_orchestrator_skips_mps_cache_cleanup_when_mps_unavailable(
    mock_rb, mock_pp, mock_mps_available, mock_empty_cache, tmp_path
):
    fake_img = Image.new("RGB", (512, 512), "red")
    mock_rb.return_value.generate.return_value = fake_img
    mock_pp.return_value.process_with_diagnostics.return_value = (
        Image.new("RGBA", (256, 256), "blue"),
        SubjectGuardResult(
            image=Image.new("RGBA", (256, 256), "blue"),
            single_subject_confidence=0.95,
            major_instance_count=1,
            split_used=False,
            fallback_triggered=False,
        ),
    )

    mock_mps_available.return_value = False
    result = generate_character("knight", output_dir=tmp_path)

    assert result["paths"]["png"].exists()
    mock_empty_cache.assert_not_called()


@patch("pixelforge.orchestrator.ReferenceBuilder")
def test_orchestrator_returns_structured_message_on_memory_error(mock_rb, tmp_path):
    mock_rb.return_value.generate.side_effect = RuntimeError("MPS backend out of memory")

    result = generate_character(
        user_prompt="knight",
        output_dir=tmp_path,
        profile="balanced",
        allow_api_fallback=False,
    )

    assert result["paths"] == {}
    assert result["message"].startswith("Error:")
    assert "out of memory" in result["message"].lower()
    assert "Eco" in result["message"]
    assert "allow API fallback" in result["message"]


@patch("pixelforge.orchestrator.PostProcessor")
@patch("pixelforge.orchestrator.ReferenceBuilder")
def test_orchestrator_triggers_single_subject_fallback_and_succeeds(
    mock_rb, mock_pp, tmp_path
):
    fake_img = Image.new("RGB", (512, 512), "red")
    mock_rb.return_value.generate.return_value = fake_img

    good = SubjectGuardResult(
        image=Image.new("RGBA", (256, 256), "blue"),
        single_subject_confidence=0.92,
        major_instance_count=1,
        split_used=False,
        fallback_triggered=False,
    )
    bad = SubjectGuardResult(
        image=Image.new("RGBA", (256, 256), "blue"),
        single_subject_confidence=0.40,
        major_instance_count=2,
        split_used=True,
        fallback_triggered=False,
    )

    # front good, left bad then retry good, right good, back good => one extra generation
    mock_pp.return_value.process_with_diagnostics.side_effect = [
        (good.image, good),
        (bad.image, bad),
        (good.image, good),
        (good.image, good),
        (good.image, good),
    ]

    result = generate_character("knight", output_dir=tmp_path, profile="balanced")
    assert result["paths"]["png"].exists()
    assert result["diagnostics"]["left"]["fallback_triggered"] is True
    assert mock_rb.return_value.generate.call_count == 5


@patch("pixelforge.orchestrator.PostProcessor")
@patch("pixelforge.orchestrator.ReferenceBuilder")
def test_orchestrator_keeps_output_when_budget_exhausted_and_subject_invalid(
    mock_rb, mock_pp, tmp_path
):
    fake_img = Image.new("RGB", (512, 512), "red")
    mock_rb.return_value.generate.return_value = fake_img

    good = SubjectGuardResult(
        image=Image.new("RGBA", (256, 256), "blue"),
        single_subject_confidence=0.92,
        major_instance_count=1,
        split_used=False,
        fallback_triggered=False,
    )
    bad = SubjectGuardResult(
        image=Image.new("RGBA", (256, 256), "blue"),
        single_subject_confidence=0.35,
        major_instance_count=2,
        split_used=True,
        fallback_triggered=False,
    )
    # front good, left bad, left retry bad => continue with warning in balanced budget=1
    mock_pp.return_value.process_with_diagnostics.side_effect = [
        (good.image, good),
        (bad.image, bad),
        (bad.image, bad),
        (good.image, good),
        (good.image, good),
    ]

    result = generate_character("knight", output_dir=tmp_path, profile="balanced")
    assert result["paths"]["png"].exists()
    assert "warnings" in result["message"].lower()
    assert result["diagnostics"]["left"]["major_instance_count"] == 2

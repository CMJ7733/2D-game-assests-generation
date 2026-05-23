"""End-to-end smoke test. Skipped in CI; run manually only."""
import pytest
from pathlib import Path


@pytest.mark.skipif(
    not Path("models").exists() or not list(Path("models").rglob("*.safetensors")),
    reason="Models not downloaded — skip e2e."
)
def test_quickmode_e2e(tmp_path):
    from pixelforge.orchestrator import generate_character
    result = generate_character(
        "test knight character",
        output_dir=tmp_path,
        base_name="smoke",
        use_quick_mode=True,
    )
    assert result["paths"]["png"].exists()
    assert result["paths"]["json"].exists()

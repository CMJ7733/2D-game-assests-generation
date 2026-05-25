"""End-to-end smoke test. Skipped in CI; run manually only."""
import pytest
import os
from pathlib import Path


@pytest.mark.skipif(
    (
        os.environ.get("RUN_E2E_SMOKE") != "1"
        or not Path("models").exists()
        or not list(Path("models").rglob("model_index.json"))
    ),
    reason="Set RUN_E2E_SMOKE=1 and download models to run e2e.",
)
def test_quickmode_e2e(tmp_path):
    from pixelforge.orchestrator import generate_character
    result = generate_character(
        "test knight character",
        output_dir=tmp_path,
        base_name="smoke",
        profile="eco",
        keep_raw_views=False,
    )
    assert result["paths"]["png"].exists()
    assert "message" in result

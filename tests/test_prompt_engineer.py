from pixelforge.prompt_engineer import enhance_prompt, build_negative_prompt


def test_enhance_prompt_adds_pixel_art_anchors():
    out = enhance_prompt("knight with sword")
    assert "pixel art" in out.lower()
    assert "side view" in out.lower()
    assert "knight with sword" in out


def test_enhance_prompt_handles_chinese_input_passthrough():
    # When translator disabled, pass through with English wrapping
    out = enhance_prompt("骑士", translate=False)
    assert "骑士" in out
    assert "pixel art" in out.lower()


def test_negative_prompt_excludes_anti_pixel_terms():
    neg = build_negative_prompt()
    assert "blurry" in neg
    assert "3d" in neg
    assert "photorealistic" in neg
    assert "multiple characters" in neg


def test_enhance_prompt_supports_state_hint():
    out = enhance_prompt("knight", state="walking")
    assert "walking" in out.lower()

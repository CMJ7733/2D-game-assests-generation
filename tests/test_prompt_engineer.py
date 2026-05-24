from pixelforge.prompt_engineer import enhance_prompt, build_negative_prompt


def test_enhance_prompt_adds_pixel_art_anchors():
    out = enhance_prompt("knight with sword")
    assert "PixelartLSS" in out
    assert "side-view" in out
    assert "facing left" in out.lower()
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


def test_enhance_prompt_includes_pixelartlss_trigger():
    out = enhance_prompt("knight with sword", translate=False)
    assert "PixelartLSS" in out


def test_enhance_prompt_includes_facing_left():
    out = enhance_prompt("knight", translate=False)
    assert "facing left" in out.lower()


def test_negative_prompt_blocks_front_view():
    neg = build_negative_prompt()
    assert "front view" in neg
    assert "facing forward" in neg

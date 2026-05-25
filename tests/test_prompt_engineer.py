from pixelforge.prompt_engineer import enhance_prompt, build_negative_prompt


def test_enhance_prompt_adds_view_anchors():
    out = enhance_prompt("knight with sword", view="left", translate=False)
    assert "PixelartLSS" in out
    assert "side-view" in out
    assert "facing left" in out.lower()
    assert "knight with sword" in out
    assert "single character" in out.lower()


def test_enhance_prompt_front_view_uses_fss_trigger():
    out = enhance_prompt("knight", view="front", translate=False)
    assert "PixelartFSS" in out
    assert "front view" in out.lower()


def test_enhance_prompt_back_view_uses_bss_trigger():
    out = enhance_prompt("knight", view="back", translate=False)
    assert "PixelartBSS" in out
    assert "back view" in out.lower()


def test_enhance_prompt_handles_chinese_input_passthrough():
    out = enhance_prompt("骑士", translate=False)
    assert "骑士" in out
    assert "PixelartLSS" in out


def test_negative_prompt_excludes_anti_pixel_terms():
    neg = build_negative_prompt()
    assert "blurry" in neg
    assert "3d" in neg
    assert "photorealistic" in neg
    assert "multiple characters" in neg
    assert "sprite sheet" in neg

import numpy as np
from PIL import Image
from pixelforge.post_processor import (
    PostProcessor,
    resize_to_target,
    quantize_palette,
)


def test_resize_to_target_preserves_size():
    img = Image.new("RGBA", (512, 512), (255, 0, 0, 255))
    out = resize_to_target(img, (64, 64))
    assert out.size == (64, 64)


def test_quantize_palette_reduces_colors():
    img = Image.new("RGB", (32, 32))
    pixels = img.load()
    for x in range(32):
        for y in range(32):
            pixels[x, y] = (x * 8, y * 8, 0)
    quantized = quantize_palette(img, n_colors=8)
    # After quantization, unique RGB tuples should be <= n_colors
    unique = set(quantized.convert("RGB").getdata())
    assert len(unique) <= 8


def test_postprocessor_full_pipeline_outputs_target_size():
    img = Image.new("RGB", (512, 512), (100, 50, 200))
    pp = PostProcessor(target_size=(64, 64), palette_colors=16)
    out = pp.process(img)
    assert out.size == (64, 64)
    assert out.mode == "RGBA"


def test_postprocessor_quantizes_before_resize():
    """After process(), a colored input should yield a small image where
    every pixel belongs to one of the quantized palette colors (no blended intermediates)."""
    from PIL import ImageDraw

    img = Image.new("RGB", (512, 512))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 255, 511], fill=(200, 50, 50))
    draw.rectangle([256, 0, 511, 511], fill=(50, 50, 200))

    pp = PostProcessor(target_size=(64, 64), palette_colors=8)
    out = pp.process(img)

    assert out.size == (64, 64)
    assert out.mode in ("RGBA", "RGB")

    arr = np.array(out.convert("RGB"))
    unique_colors = set(map(tuple, arr.reshape(-1, 3).tolist()))
    assert len(unique_colors) <= 8, f"Expected ≤8 colors, got {len(unique_colors)}"

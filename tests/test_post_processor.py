import numpy as np
import cv2
from PIL import Image
from PIL import ImageDraw
from pixelforge.post_processor import (
    PostProcessor,
    SubjectGuard,
    SubjectGuardResult,
    resize_to_target,
    quantize_palette,
    keep_primary_subject,
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


def test_keep_primary_subject_keeps_single_component_near_center():
    img = Image.new("RGBA", (120, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    # left / center / right characters; center one is slightly larger.
    draw.rectangle([6, 18, 24, 58], fill=(255, 50, 50, 255))
    draw.rectangle([48, 12, 74, 60], fill=(255, 50, 50, 255))
    draw.rectangle([92, 18, 110, 58], fill=(255, 50, 50, 255))

    out = keep_primary_subject(img)
    alpha = np.array(out)[..., 3]

    ys, xs = np.where(alpha > 0)
    assert len(xs) > 0
    cx = xs.mean()
    # remaining subject should be around center column.
    assert 45 <= cx <= 78


def test_subject_guard_splits_touching_subjects_with_watershed():
    img = Image.new("RGBA", (200, 120), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    # Two circles connected by a thin bridge => one connected component before split.
    draw.ellipse([20, 30, 80, 90], fill=(255, 80, 80, 255))
    draw.ellipse([120, 30, 180, 90], fill=(255, 80, 80, 255))
    draw.rectangle([80, 58, 120, 62], fill=(255, 80, 80, 255))

    guard = SubjectGuard(min_confidence=0.55, min_area_ratio=0.2, split_enabled=True)
    result = guard.enforce(img)

    assert isinstance(result, SubjectGuardResult)
    assert result.split_used is True
    assert result.major_instance_count >= 2

    alpha = np.array(result.image.convert("RGBA"))[..., 3]
    n_labels, _, _, _ = cv2.connectedComponentsWithStats((alpha > 8).astype(np.uint8), connectivity=8)
    assert n_labels - 1 == 1


def test_subject_guard_keeps_single_subject_without_split():
    img = Image.new("RGBA", (120, 120), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rectangle([35, 20, 85, 110], fill=(200, 200, 80, 255))

    guard = SubjectGuard(min_confidence=0.55, min_area_ratio=0.2, split_enabled=True)
    result = guard.enforce(img)

    assert result.split_used is False
    assert result.major_instance_count == 1
    assert result.single_subject_confidence >= 0.55

"""Post-process raw SD frames into clean pixel-art sprite frames."""
from __future__ import annotations
import numpy as np
from PIL import Image
from loguru import logger


def remove_background(img: Image.Image) -> Image.Image:
    """Remove background, leaving alpha transparency. Falls back to threshold-based key."""
    try:
        from rembg import remove
        return remove(img.convert("RGBA"))
    except Exception as e:
        logger.warning(f"rembg failed ({e}), using threshold fallback")
        arr = np.array(img.convert("RGBA"))
        # Simple white-background chroma key
        mask = (arr[..., 0] > 240) & (arr[..., 1] > 240) & (arr[..., 2] > 240)
        arr[..., 3] = np.where(mask, 0, 255)
        return Image.fromarray(arr, "RGBA")


def trim_to_content(img: Image.Image, padding: int = 2) -> Image.Image:
    """Crop transparent borders, keeping `padding` px of margin."""
    arr = np.array(img.convert("RGBA"))
    alpha = arr[..., 3]
    if alpha.max() == 0:
        return img
    rows = np.any(alpha > 0, axis=1)
    cols = np.any(alpha > 0, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmin = max(0, rmin - padding)
    rmax = min(arr.shape[0], rmax + padding + 1)
    cmin = max(0, cmin - padding)
    cmax = min(arr.shape[1], cmax + padding + 1)
    return Image.fromarray(arr[rmin:rmax, cmin:cmax], "RGBA")


def resize_to_target(img: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    """Downsample preserving aspect ratio and pad to target size with transparency."""
    img = img.convert("RGBA")
    w, h = img.size
    tw, th = target_size
    scale = min(tw / w, th / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = img.resize((new_w, new_h), Image.NEAREST)
    canvas = Image.new("RGBA", target_size, (0, 0, 0, 0))
    canvas.paste(resized, ((tw - new_w) // 2, (th - new_h) // 2), resized)
    return canvas


def quantize_palette(img: Image.Image, n_colors: int = 24) -> Image.Image:
    """Reduce to a limited palette while preserving transparency."""
    rgba = img.convert("RGBA")
    alpha = rgba.split()[-1]
    rgb = rgba.convert("RGB")
    quantized = rgb.quantize(colors=n_colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    out = quantized.convert("RGBA")
    out.putalpha(alpha)
    return out


class PostProcessor:
    def __init__(self, target_size: tuple[int, int] = (64, 64), palette_colors: int = 24):
        self.target_size = target_size
        self.palette_colors = palette_colors

    def process(self, raw_frame: Image.Image) -> Image.Image:
        stage1 = remove_background(raw_frame)
        stage2 = trim_to_content(stage1)
        stage3 = resize_to_target(stage2, self.target_size)
        stage4 = quantize_palette(stage3, self.palette_colors)
        return stage4

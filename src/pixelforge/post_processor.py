"""Post-process raw SD frames into clean pixel-art sprite frames."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
import cv2
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


@dataclass
class SubjectGuardResult:
    image: Image.Image
    single_subject_confidence: float
    major_instance_count: int
    split_used: bool
    fallback_triggered: bool = False

    def is_single_subject(self, min_confidence: float) -> bool:
        return self.major_instance_count == 1 and self.single_subject_confidence >= min_confidence

    def to_dict(self) -> dict:
        out = asdict(self)
        out["image"] = None
        return out


class SubjectGuard:
    """Algorithmic guardrail: split touching subjects and keep only the best subject."""

    def __init__(
        self,
        min_confidence: float = 0.62,
        min_area_ratio: float = 0.18,
        split_enabled: bool = True,
        min_alpha: int = 8,
        min_area_px: int = 32,
    ):
        self.min_confidence = min_confidence
        self.min_area_ratio = min_area_ratio
        self.split_enabled = split_enabled
        self.min_alpha = min_alpha
        self.min_area_px = min_area_px

    def enforce(self, img: Image.Image) -> SubjectGuardResult:
        rgba = np.array(img.convert("RGBA"))
        alpha = rgba[..., 3]
        mask = (alpha > self.min_alpha).astype(np.uint8)
        if mask.sum() == 0:
            return SubjectGuardResult(
                image=Image.fromarray(rgba, "RGBA"),
                single_subject_confidence=0.0,
                major_instance_count=0,
                split_used=False,
            )

        labels, stats, centroids = self._label_components(mask)
        major_labels = self._major_labels(stats)

        split_used = False
        if self.split_enabled and len(major_labels) <= 1:
            maybe_split = self._watershed_split(mask, rgba[..., :3])
            if maybe_split is not None:
                split_used = True
                labels, stats, centroids = self._label_components(maybe_split)
                major_labels = self._major_labels(stats)

        if not major_labels and stats.shape[0] > 1:
            major_labels = [1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))]

        if not major_labels:
            return SubjectGuardResult(
                image=Image.fromarray(rgba, "RGBA"),
                single_subject_confidence=0.0,
                major_instance_count=0,
                split_used=split_used,
            )

        best_label, best_score = self._best_label(major_labels, stats, centroids, mask.shape)
        out = rgba.copy()
        out[..., 3] = np.where(labels == best_label, alpha, 0).astype(np.uint8)
        confidence = float(max(0.0, min(1.0, best_score)))

        return SubjectGuardResult(
            image=Image.fromarray(out, "RGBA"),
            single_subject_confidence=confidence,
            major_instance_count=len(major_labels),
            split_used=split_used,
        )

    def _label_components(self, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n_labels <= 1:
            stats = np.zeros((1, 5), dtype=np.int32)
            centroids = np.zeros((1, 2), dtype=np.float32)
        return labels, stats, centroids

    def _major_labels(self, stats: np.ndarray) -> list[int]:
        if stats.shape[0] <= 1:
            return []
        max_area = int(stats[1:, cv2.CC_STAT_AREA].max())
        area_floor = max(self.min_area_px, int(max_area * self.min_area_ratio))
        labels: list[int] = []
        for label in range(1, stats.shape[0]):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area >= area_floor:
                labels.append(label)
        return labels

    def _best_label(
        self,
        labels: list[int],
        stats: np.ndarray,
        centroids: np.ndarray,
        shape: tuple[int, int],
    ) -> tuple[int, float]:
        h, w = shape
        center_x, center_y = w / 2.0, h / 2.0
        max_area = max(float(stats[label, cv2.CC_STAT_AREA]) for label in labels)
        best_label = labels[0]
        best_score = -1e9

        for label in labels:
            area = float(stats[label, cv2.CC_STAT_AREA])
            cx, cy = centroids[label]
            dx = (cx - center_x) / max(w, 1.0)
            dy = (cy - center_y) / max(h, 1.0)
            dist2 = float(dx * dx + dy * dy)
            bw = max(1.0, float(stats[label, cv2.CC_STAT_WIDTH]))
            bh = max(1.0, float(stats[label, cv2.CC_STAT_HEIGHT]))
            aspect = bw / bh
            aspect_penalty = abs(np.log(aspect))
            score = (area / max_area) - 0.22 * dist2 - 0.08 * aspect_penalty
            if score > best_score:
                best_score = score
                best_label = label

        return best_label, best_score

    def _watershed_split(self, mask: np.ndarray, rgb: np.ndarray) -> np.ndarray | None:
        fg = (mask * 255).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)
        sure_bg = cv2.dilate(opening, kernel, iterations=2)

        dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        if float(dist.max()) <= 0.0:
            return None
        _, sure_fg = cv2.threshold(dist, 0.35 * float(dist.max()), 255, 0)
        sure_fg = np.uint8(sure_fg)
        if int(sure_fg.sum()) == 0:
            return None

        unknown = cv2.subtract(sure_bg, sure_fg)
        n_markers, markers = cv2.connectedComponents(sure_fg)
        if n_markers <= 2:
            return None
        markers = markers + 1
        markers[unknown == 255] = 0

        ws = cv2.watershed(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), markers.astype(np.int32))
        split = np.zeros_like(mask, dtype=np.uint8)
        split[ws > 1] = 1
        return split


def keep_primary_subject(
    img: Image.Image,
    min_alpha: int = 8,
    min_area_px: int = 32,
) -> Image.Image:
    """Compatibility wrapper using SubjectGuard with conservative defaults."""
    guard = SubjectGuard(
        min_confidence=0.0,
        min_area_ratio=0.15,
        split_enabled=True,
        min_alpha=min_alpha,
        min_area_px=min_area_px,
    )
    return guard.enforce(img).image


def resize_to_target(img: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    """Downsample preserving aspect ratio using LANCZOS and pad to target size with transparency."""
    img = img.convert("RGBA")
    w, h = img.size
    tw, th = target_size
    scale = min(tw / w, th / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = img.resize((new_w, new_h), Image.LANCZOS)
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
    def __init__(
        self,
        target_size: tuple[int, int] = (64, 64),
        palette_colors: int = 24,
        subject_guard: SubjectGuard | None = None,
    ):
        self.target_size = target_size
        self.palette_colors = palette_colors
        self.subject_guard = subject_guard or SubjectGuard()

    def process(self, raw_frame: Image.Image) -> Image.Image:
        processed, _ = self.process_with_diagnostics(raw_frame)
        return processed

    def process_with_diagnostics(self, raw_frame: Image.Image) -> tuple[Image.Image, SubjectGuardResult]:
        stage1 = remove_background(raw_frame)
        guard_result = self.subject_guard.enforce(stage1)
        stage2 = guard_result.image
        stage3 = trim_to_content(stage2)
        stage4 = quantize_palette(stage3, self.palette_colors)
        stage5 = resize_to_target(stage4, self.target_size)
        stage6 = quantize_palette(stage5, self.palette_colors)
        guard_result.image = stage6
        return stage6, guard_result

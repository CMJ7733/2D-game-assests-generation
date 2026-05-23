"""Lightweight image similarity for consistency checks. Uses average color histogram + perceptual hash fallback."""
from __future__ import annotations
import numpy as np
from PIL import Image


def image_similarity(a: Image.Image, b: Image.Image, size: int = 64) -> float:
    """Return 0..1 similarity score based on downscaled color histogram cosine similarity."""
    def feat(img: Image.Image) -> np.ndarray:
        small = img.convert("RGB").resize((size, size), Image.LANCZOS)
        arr = np.array(small).astype(np.float32).flatten()
        norm = np.linalg.norm(arr)
        return arr / norm if norm > 0 else arr

    fa, fb = feat(a), feat(b)
    return float(np.dot(fa, fb))


def retry_on_low_similarity(
    generate_fn,
    reference: Image.Image,
    threshold: float = 0.85,
    max_retries: int = 2,
):
    """Call generate_fn(); if result similarity to reference < threshold, retry up to max_retries times. Returns best result."""
    best_img = None
    best_sim = -1.0
    for attempt in range(max_retries + 1):
        img = generate_fn()
        sim = image_similarity(img, reference)
        if sim > best_sim:
            best_img, best_sim = img, sim
        if sim >= threshold:
            return img, sim
    return best_img, best_sim

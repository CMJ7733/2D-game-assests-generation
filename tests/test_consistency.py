from PIL import Image
import numpy as np
from pixelforge.consistency import image_similarity, retry_on_low_similarity


def test_similarity_high_for_identical_images():
    img = Image.new("RGB", (64, 64), (100, 50, 200))
    sim = image_similarity(img, img)
    assert sim > 0.99


def test_similarity_low_for_different_images():
    a = Image.new("RGB", (64, 64), (0, 0, 0))
    b = Image.new("RGB", (64, 64), (255, 255, 255))
    sim = image_similarity(a, b)
    assert sim < 0.5

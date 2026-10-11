"""
src/eval/corruptions.py - DERMA-STRESS image corruptions (shared by the experiment and the UI).
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


def blur(im: Image.Image, radius: float = 4) -> Image.Image:
    return im.filter(ImageFilter.GaussianBlur(radius))


def dark(im: Image.Image, factor: float = 0.35) -> Image.Image:
    return ImageEnhance.Brightness(im).enhance(factor)


def noise(im: Image.Image, sigma: float = 0.12, seed: int = 0) -> Image.Image:
    a = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    a = np.clip(a + np.random.default_rng(seed).normal(0, sigma, a.shape), 0, 1)
    return Image.fromarray((a * 255).astype(np.uint8))


def jpeg(im: Image.Image, quality: int = 8) -> Image.Image:
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=quality)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def occlusion(im: Image.Image, area: float = 0.35) -> Image.Image:
    im = im.copy()
    w, h = im.size
    s = area ** 0.5
    ImageDraw.Draw(im).rectangle([w * (1 - s) / 2, h * (1 - s) / 2,
                                  w * (1 + s) / 2, h * (1 + s) / 2], fill=(0, 0, 0))
    return im


IMAGE_CORRUPTIONS = {
    "blur": blur, "dark": dark, "noise": noise, "jpeg": jpeg, "occlusion": occlusion,
}

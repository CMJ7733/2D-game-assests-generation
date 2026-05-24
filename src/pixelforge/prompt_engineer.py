"""Prompt enhancement and Chinese→English translation."""
from __future__ import annotations
import os

_VIEW_TRIGGERS = {
    "front": "PixelartFSS",
    "left": "PixelartLSS",
    "right": "PixelartRSS",
    "back": "PixelartBSS",
}

_VIEW_ANCHORS = {
    "front": ["front view", "facing camera"],
    "left": ["side-view", "facing left"],
    "right": ["side-view", "facing right"],
    "back": ["back view", "facing away"],
}

_COMMON_ANCHORS = [
    "2D game character",
    "full body",
    "white background",
    "clean pixel art",
]

_NEGATIVE_TERMS = [
    "blurry", "3d", "photorealistic", "multiple characters",
    "watermark", "text", "signature", "low quality", "deformed",
    "cropped", "out of frame", "extra limbs", "isometric",
]


def _translate_zh_to_en(text: str) -> str:
    if not os.environ.get("OPENAI_API_KEY"):
        return text
    try:
        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Translate to concise English for a Stable Diffusion prompt. Output translation only, no explanation."},
                {"role": "user", "content": text},
            ],
            max_tokens=80,
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return text


def enhance_prompt(user_text: str, view: str = "left", translate: bool = True) -> str:
    base = user_text.strip()
    if translate and any(ord(ch) > 127 for ch in base):
        base = _translate_zh_to_en(base) or base
    trigger = _VIEW_TRIGGERS.get(view, "PixelartLSS")
    view_anchors = _VIEW_ANCHORS.get(view, [])
    parts = [base, trigger] + view_anchors + _COMMON_ANCHORS
    return ", ".join(parts)


def build_negative_prompt() -> str:
    return ", ".join(_NEGATIVE_TERMS)

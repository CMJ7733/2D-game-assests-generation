"""Prompt enhancement and Chinese→English translation."""
from __future__ import annotations
import os

_BASE_ANCHORS = [
    "PixelartLSS",
    "side-view",
    "facing left",
    "2D game character",
    "full body",
    "white background",
    "clean pixel art",
]

_NEGATIVE_TERMS = [
    "blurry", "3d", "photorealistic", "multiple characters",
    "watermark", "text", "signature", "low quality", "deformed",
    "cropped", "out of frame", "extra limbs",
    "front view", "facing forward", "facing camera",
    "3/4 view", "isometric",
]


def _translate_zh_to_en(text: str) -> str:
    """Best-effort translation. Falls back to passthrough if no API key."""
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


def enhance_prompt(user_text: str, state: str | None = None, translate: bool = True) -> str:
    """Combine user text with pixel-art anchors and optional state hint."""
    base = user_text.strip()
    if translate and any(ord(ch) > 127 for ch in base):
        base = _translate_zh_to_en(base) or base
    parts = [base]
    if state:
        parts.append(state)
    parts.extend(_BASE_ANCHORS)
    return ", ".join(parts)


def build_negative_prompt() -> str:
    return ", ".join(_NEGATIVE_TERMS)

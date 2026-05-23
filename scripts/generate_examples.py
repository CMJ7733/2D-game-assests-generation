"""Pre-generate 5 hand-picked demo examples for safety net during live demo."""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from pathlib import Path
from pixelforge.orchestrator import generate_character

EXAMPLES = [
    ("knight", "A knight with a red cape, blonde hair, holding a steel sword"),
    ("mage", "A female mage with a blue robe and long purple hair holding a wooden staff"),
    ("rogue", "A rogue with a green hood, leather armor, holding twin daggers"),
    ("archer", "A wood elf archer with a green tunic, brown hair, longbow"),
    ("warrior", "A barbarian warrior with bare chest, brown hair in a topknot, holding a great axe"),
]
OUT = Path("assets/examples")
OUT.mkdir(parents=True, exist_ok=True)

for slug, prompt in EXAMPLES:
    print(f"\n=== {slug}: {prompt}")
    result = generate_character(prompt, output_dir=OUT, base_name=slug)
    print(f"  → {result['paths']['png']}")

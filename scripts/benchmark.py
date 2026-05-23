"""Time each pipeline stage for a single character end-to-end."""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import time
from pathlib import Path
from pixelforge.orchestrator import generate_character

OUT = Path("output/bench")
OUT.mkdir(parents=True, exist_ok=True)

t0 = time.time()
result = generate_character("knight with red cape", output_dir=OUT, base_name="bench")
total = time.time() - t0
print(f"\n=== Benchmark Results ===")
print(f"Total end-to-end: {total:.1f}s")
print(f"Output: {result['paths']['png']}")

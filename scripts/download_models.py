"""Pre-download all required models to HuggingFace default cache via hf-mirror."""
import os
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from huggingface_hub import snapshot_download

MODELS = [
    ("runwayml/stable-diffusion-v1-5", None),
    ("lllyasviel/sd-controlnet-openpose", None),
    ("h94/IP-Adapter", "models"),
    ("briaai/RMBG-1.4", None),
]


def main() -> int:
    if "--official" in sys.argv:
        os.environ["HF_ENDPOINT"] = "https://huggingface.co"
        print("Using OFFICIAL HuggingFace endpoint")
    else:
        print(f"Using endpoint: {os.environ['HF_ENDPOINT']}")

    for repo_id, subfolder in MODELS:
        print(f"\n→ Downloading {repo_id}" + (f"/{subfolder}" if subfolder else ""))
        kwargs = {"repo_id": repo_id}
        if subfolder:
            kwargs["allow_patterns"] = [f"{subfolder}/*"]
        try:
            snapshot_download(**kwargs)
            print(f"  ✓ {repo_id}")
        except Exception as e:
            print(f"  ✗ {repo_id}: {e}")
            return 1
    print("\nAll models downloaded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

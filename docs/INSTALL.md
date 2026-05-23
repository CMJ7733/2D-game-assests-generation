# Installation

## Prerequisites

- macOS with M-series chip (or any CUDA GPU)
- Python 3.11 (`pyenv install 3.11` or Homebrew)
- `uv` package manager: `brew install uv`
- ~10GB free disk space (models + cache)

## Steps

1. Clone:
   ```
   git clone <repo>
   cd pixelforge
   ```

2. Set up env:
   ```
   uv venv --python 3.11
   source .venv/bin/activate
   uv pip install -e ".[dev]"
   ```

3. Copy env file:
   ```
   cp .env.example .env
   # Optional: add REPLICATE_API_TOKEN or OPENAI_API_KEY
   ```

4. Download models (uses hf-mirror.com by default; ~10-15min on good connection):
   ```
   ./scripts/download_models.sh
   ```

   For official HuggingFace:
   ```
   ./scripts/download_models.sh --official
   ```

5. Run:
   ```
   python -m pixelforge.app
   ```

## Troubleshooting

- **`torch.backends.mps.is_available()` returns False**: macOS 12.3+ required; reinstall `torch>=2.3.0`.
- **Out of memory**: close Chrome / other heavy apps; set `dtype="float16"` in `config.py` (default).
- **Slow downloads**: try `./scripts/download_models.sh --official` if mirror is unstable.
- **rembg fails**: switches automatically to threshold fallback; output may have noisy edges.

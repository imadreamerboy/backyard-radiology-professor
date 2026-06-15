# Deploy the Modal GPU backend

The official Build Small Space must live under
`build-small-hackathon/backyard-radiology-professor`. Hugging Face rejected paid
L4 hardware for that org without billing credits, and ZeroGPU is not available
for Docker Spaces. The production path is therefore:

1. Run the real backend on Modal GPU.
2. Configure the official Hugging Face Space as a Gradio proxy with
   `RAD_TRAINER_REMOTE_BACKEND_URL`.

Modal deployment:

```bash
uv sync --extra deploy
$env:HF_TOKEN = (wsl.exe -d Ubuntu -- bash -lc 'cat ~/.cache/huggingface/token').Trim()
uv run modal setup
uv run modal deploy deploy/modal_app.py
```

The Modal app uses:

- `Dockerfile.modal`, equivalent to the local CUDA runtime without BuildKit-only
  cache mounts.
- GPU fallback `L40S`, then `L4`.
- A Modal Volume mounted at `/data` for GGUF, X-Raydar, and Hugging Face caches.
- `HF_TOKEN` from the local environment during deploy, or a Modal secret named
  `backyard-radiology-professor-hf`.

After deployment, copy the Modal web URL and set it on the official Space:

```bash
uv run python scripts/configure_hf_space.py --backend-url https://YOUR-MODAL-URL.modal.run
```

Then validate the official Space URL:

```bash
uv run python scripts/validate_golden_cases.py --app-url https://build-small-hackathon-backyard-radiology-professor.hf.space --timeout-seconds 1800
uv run python scripts/benchmark_runtime.py --app-url https://build-small-hackathon-backyard-radiology-professor.hf.space --runs 2 --timeout-seconds 1800 --gpu-baseline-mb 0
```

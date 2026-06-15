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
- Modal proxy authentication. Requests without the private key and secret are
  rejected before a GPU container starts.
- `min_containers=0` and a 60-second scale-down window. The app remains deployed
  but has no GPU compute cost after its last request finishes and the warm
  window expires.

Create a proxy-auth token in the Modal dashboard as documented in
[Webhook proxy auth](https://modal.com/docs/guide/webhook-proxy-auth). Store its
key and secret locally; do not commit them.

```bash
$env:RAD_TRAINER_MODAL_KEY = "wk-..."
$env:RAD_TRAINER_MODAL_SECRET = "ws-..."
```

Deploy Modal, copy its web URL, then configure the official Space. The script
stores the credentials as Hugging Face Space secrets:

```bash
uv run modal deploy deploy/modal_app.py
uv run python scripts/configure_hf_space.py `
  --backend-url https://YOUR-MODAL-URL.modal.run
```

Then validate the official Space URL:

```bash
uv run python scripts/validate_golden_cases.py --app-url https://build-small-hackathon-backyard-radiology-professor.hf.space --timeout-seconds 1800
uv run python scripts/benchmark_runtime.py --app-url https://build-small-hackathon-backyard-radiology-professor.hf.space --runs 2 --timeout-seconds 1800 --gpu-baseline-mb 0
```

The Space serves its bundled case catalog and status locally, so opening the
page does not wake Modal. Opening or analyzing a study starts the backend.

Check the current deployment without waking it:

```bash
uv run modal app list
uv run modal container list
```

Leave the Modal app deployed for ordinary idling. `modal app stop` permanently
stops the deployment and requires another `modal deploy`; it is not the normal
cost-control mechanism.

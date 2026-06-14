# Deploy the Space

1. Create a Docker Space with one Nvidia L4.
2. Accept the MedGemma license for the account behind the token.
3. Add `HF_TOKEN` as a Space secret.
4. Push this repository unchanged.

Startup downloads the exact GGUF revisions in `scripts/prepare_runtime.py`, prepares X-Raydar, starts the pinned CUDA llama.cpp router on port 8080, verifies both model presets, and serves Gradio on port 7860.

Required router aliases:

- `medgemma-professor`
- `medgemma-localizer`

Deployment acceptance:

```bash
uv run python scripts/validate_golden_cases.py --app-url https://SPACE.hf.space
uv run python scripts/benchmark_runtime.py --app-url https://SPACE.hf.space
```

Commit the resulting reports as `artifacts/validation/space-golden-cases.json`
and `artifacts/validation/space-runtime-benchmark.json`. The local equivalents
are generated from the identical container before deployment.

For a workstation GPU that also drives a desktop, record idle board usage before
starting the container and pass it as `--gpu-baseline-mb`. The benchmark reports
both raw board peak and application-attributed peak. Use `0` on a dedicated Space
GPU.

The Space profile uses an 8192-token professor context and full GPU offload.
Docker Compose selects `runtime/models.local-wsl.ini`, a separate 6144-token,
full-offload profile for a desktop RTX 4090. Move the
Space to L40S if the L4 exceeds 22 GB peak VRAM, stays below 5 generated tokens/s,
or has warm first-token latency above 20 seconds after the documented 6K fallback
has also been measured.

# Run with WSL 2

Use Docker Desktop with WSL integration enabled for Ubuntu. Keep the repository,
model files, and Docker build context inside the WSL ext4 filesystem. Loading a
27B GGUF through `/mnt/c` is substantially slower.

```bash
git clone <repository-url> ~/small-models-hackathon
cd ~/small-models-hackathon
cp .env.example .env
```

Set `HF_TOKEN` in `.env` after accepting the MedGemma license, then start the
full backend:

```bash
docker compose up --build
```

Docker Compose selects `runtime/models.local-wsl.ini`: full GPU offload with a
6144-token professor context. Open
[http://127.0.0.1:7860](http://127.0.0.1:7860) from Windows.

Useful checks:

```bash
docker compose logs -f radiology-trainer
curl http://127.0.0.1:7860/api/status
uv run python scripts/validate_golden_cases.py
uv run python scripts/benchmark_runtime.py --gpu-baseline-mb 0
```

Pass the GPU's measured idle board usage to `--gpu-baseline-mb` when it also
drives the desktop. The benchmark records both raw and application-attributed
peak memory.

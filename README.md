---
title: Backyard Radiology Professor
emoji: 🩻
colorFrom: gray
colorTo: teal
sdk: docker
app_port: 7860
tags:
  - build-small-hackathon
  - backyard-ai
  - gradio
  - llama-cpp
  - medgemma
  - radiology
  - dicom
models:
  - unsloth/medgemma-27b-it-GGUF
  - unsloth/medgemma-1.5-4b-it-GGUF
---

# Backyard Radiology Professor

An educational chest-radiograph workstation for deliberate practice. A trainee commits a blind interpretation before seeing independently attributed X-Raydar evidence, MedGemma localization, and feedback from a multimodal MedGemma professor.

The app is built for the Hugging Face Build Small hackathon Backyard AI track. It is a teaching demo, not clinical software.

## Runtime

- `unsloth/medgemma-27b-it-GGUF` Q4_K_M + F16 projector: professor review and multi-turn chat.
- `unsloth/medgemma-1.5-4b-it-GGUF` Q4_K_M + F16 projector: observations and bounding boxes.
- X-Raydar: independent three-model chest radiograph classifier.
- One pinned CUDA llama.cpp router with `--models-max 1`.
- Python dependencies and commands are managed with `uv`.

Only one MedGemma model is resident at a time. X-Raydar moves back to CPU after inference, the 4B localizer unloads, and the 27B professor then loads for chat.

## Run

```bash
cp .env.example .env
docker compose up --build
```

Set `HF_TOKEN` after accepting the MedGemma license. Open [http://localhost:7860](http://localhost:7860).

On Windows, run the repository and model cache from the WSL ext4 filesystem
rather than `/mnt/c`; see [docs/run_wsl.md](docs/run_wsl.md).

## Verify

```bash
uv run pytest
uv run ruff check src tests scripts app.py
uv run python scripts/generate_dicom_fixtures.py
uv run python scripts/validate_golden_cases.py
uv run python scripts/benchmark_runtime.py
```

Verified local and deployed results are stored under `artifacts/validation/`.
The public demo is complete only after the same real-backend suite passes against the deployed Space. L4 is the target; if the measured profile exceeds 22 GB, reduce professor context to 6K and partially offload layers. If warm performance remains below 5 tokens/s or first token exceeds 20 seconds, use L40S without changing model quality.

## Supported studies

- PNG/JPEG and other common image formats.
- Single CR/DX DICOM files.
- Multi-file or multi-frame chest CR/DX studies.
- ZIP archives with bounded, traversal-safe extraction.
- Uncompressed, JPEG, JPEG-LS, and JPEG2000 pixel data through pydicom/pylibjpeg.

CT, MR, non-chest studies, corrupt archives, and oversized uploads are rejected.

This software is for educational practice only.

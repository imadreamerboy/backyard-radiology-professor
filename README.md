---
title: Backyard Radiology Professor
emoji: 👀
colorFrom: gray
colorTo: green
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
  - track:backyard
  - sponsor:openai
  - sponsor:modal
  - achievement:offgrid
  - achievement:offbrand
  - achievement:llama
  - achievement:fieldnotes
models:
  - unsloth/medgemma-27b-it-GGUF
  - unsloth/medgemma-1.5-4b-it-GGUF
---

# Backyard Radiology Professor

An educational chest-radiograph workstation for deliberate practice. A trainee commits a blind interpretation before seeing independently attributed X-Raydar evidence, MedGemma localization, and feedback from a multimodal MedGemma professor.

The app is built for the Hugging Face Build Small hackathon Backyard AI track. It is a teaching demo, not clinical software.

## Why I built it

My brother recently started working as a radiologist. He has the medical training, but becoming fast and systematic still takes repeated exposure to real cases and useful feedback. I built this workstation to give him, and other early-career readers, another way to practise outside the clinical workflow.

The core idea is simple: make your own read first, then compare it with evidence from several models. The app deliberately keeps the blind interpretation, classifier output, localized regions, and professor feedback separate. That makes it easier to notice what you missed without pretending that one model has the final answer.

The three included cases are public educational examples. Uploaded studies should be public or de-identified chest CR/DX radiographs.

## How a session works

1. Choose a bundled practice case or open a chest radiograph or DICOM study.
2. Inspect the image with familiar viewer controls and write findings plus an impression in normal language.
3. Commit the blind read. It is locked before model evidence appears.
4. X-Raydar produces independent classifier evidence and MedGemma 1.5 4B proposes observations and regions.
5. MedGemma 27B compares the image, metadata, blind read, and model evidence, then continues as a radiology professor in chat.
6. Export the session or ask for explanations, differentials, and quizzes.

## Runtime

- `unsloth/medgemma-27b-it-GGUF` Q4_K_M + F16 projector: professor review and multi-turn chat.
- `unsloth/medgemma-1.5-4b-it-GGUF` Q4_K_M + F16 projector: observations and bounding boxes.
- X-Raydar: independent three-model chest radiograph classifier.
- One pinned CUDA llama.cpp router with both MedGemma aliases loaded.
- Python dependencies and commands are managed with `uv`.

The public backend targets an L40S profile so the 4B localizer and 27B professor stay resident. Session state is persisted on the backend volume so chat can resume after the GPU container idles down and wakes again.

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
The public demo is complete only after the same real-backend suite passes against
the deployed Space. If the official Build Small org cannot allocate paid GPU
hardware, run the real backend on Modal and use the official Hugging Face Space
as the public Gradio proxy. See
[docs/deploy_modal_backend.md](docs/deploy_modal_backend.md).

For the recorded submission workflow, see [docs/record_demo_video.md](docs/record_demo_video.md).

## Supported studies

- PNG/JPEG and other common image formats.
- Single CR/DX DICOM files.
- Multi-file or multi-frame chest CR/DX studies.
- ZIP archives with bounded, traversal-safe extraction.
- Uncompressed, JPEG, JPEG-LS, and JPEG2000 pixel data through pydicom/pylibjpeg.

CT, MR, non-chest studies, corrupt archives, and oversized uploads are rejected.

This software is for educational practice only.

---
title: Backyard Radiology Trainer
sdk: gradio
sdk_version: 6.17.3
app_file: app.py
short_description: Chest-X-ray-first educational radiology practice app
---

# Backyard Radiology Trainer

Chest-X-ray-first Gradio app for educational radiology practice. The user writes a blind read first, then the app reveals model evidence, a Nemotron-backed tutor response, and a short quiz.

This is not a clinical tool. It is designed for local educational use and a public demo-safe Space.

## MVP Scope

- Chest radiograph workflow first.
- Extensible anatomy registry for later plain-film pipelines.
- Demo mode runs without gated models or GPU.
- Local mode can call Hugging Face or OpenAI-compatible endpoints for Nemotron.
- Evidence layer is structured so X-Raydar, MedSigLIP, CXR Foundation, MedGemma, and SAM-style segmentation can be added behind stable interfaces.

## Run

```powershell
uv sync
uv run python app.py
```

Open the local URL Gradio prints.

## Model Modes

Default mode:

```powershell
$env:RAD_TRAINER_MODEL_MODE="demo"
$env:RAD_TRAINER_TUTOR_PROVIDER="demo"
uv run python app.py
```

Nemotron via Hugging Face:

```powershell
$env:HF_TOKEN="hf_..."
$env:RAD_TRAINER_TUTOR_PROVIDER="hf"
$env:RAD_TRAINER_NEMOTRON_MODEL="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
uv run python app.py
```

Nemotron via a local OpenAI-compatible server:

```powershell
$env:RAD_TRAINER_TUTOR_PROVIDER="openai"
$env:RAD_TRAINER_OPENAI_BASE_URL="http://localhost:8000/v1"
$env:RAD_TRAINER_OPENAI_API_KEY="local"
uv run python app.py
```

## Architecture

```text
Upload image
  -> preprocessing and quality notes
  -> anatomy router
  -> chest evidence pipeline
     -> classifier findings
     -> localization boxes
     -> optional segmentation overlays
  -> Nemotron tutor layer
  -> educational feedback and quiz
```

The app currently ships a deterministic demo evidence model so UI and flow can be tested anywhere. Real model adapters should produce the same `EvidenceBundle` shape.

## Next Model Integrations

1. X-Raydar for frontal chest finding probabilities.
2. MedSigLIP or CXR Foundation for embeddings, retrieval, and broad routing.
3. MedGemma 1.5 4B as a medical VLM for image-conditioned explanations.
4. SAM 3.1 as an interactive region-refinement overlay, not as a diagnostic source.

## Safety Defaults

- No uploads are persisted by default.
- The UI labels every output as educational.
- Tutor prompts forbid diagnosis claims and force uncertainty notes.
- The app asks for a blind read before revealing AI evidence.

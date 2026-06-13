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
- Local mode can optionally add MedGemma image-conditioned notes.
- Local classifier services can plug in through one HTTP evidence endpoint.
- Evidence layer is structured so X-Raydar, MedSigLIP, CXR Foundation, MedGemma, and SAM-style segmentation can be added behind stable interfaces.
- Synthetic example cases are included for public demo judging without patient data.

## Run

```powershell
uv sync
uv run python app.py
```

Open the local URL Gradio prints.

Regenerate synthetic demo cases:

```powershell
uv run python scripts/generate_demo_cases.py
```

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
$env:RAD_TRAINER_HF_PROVIDER="nvidia"
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

Optional MedGemma local VLM notes:

```powershell
uv sync --extra models
$env:RAD_TRAINER_MODEL_MODE="local"
$env:RAD_TRAINER_ENABLE_MEDICAL_VLM="true"
$env:RAD_TRAINER_MEDICAL_VLM="google/medgemma-1.5-4b-it"
uv run python app.py
```

External chest evidence endpoint:

```powershell
$env:RAD_TRAINER_MODEL_MODE="local"
$env:RAD_TRAINER_CHEST_EVIDENCE_URL="http://localhost:9000/analyze"
uv run python app.py
```

Prepare and run the real X-Raydar backend:

```powershell
uv sync --extra xraydar --extra dev
uv run python scripts/download_xraydar_backend.py
uv run python scripts/run_xraydar_service.py
```

Validate the app pipeline against X-Raydar's included demo DICOM labels:

```powershell
$env:RAD_TRAINER_CHEST_EVIDENCE_URL="http://127.0.0.1:9000/analyze"
uv run python scripts/validate_xraydar_demo.py
```

Smoke test the running Gradio app against the same backend:

```powershell
$env:RAD_TRAINER_MODEL_MODE="local"
$env:RAD_TRAINER_CHEST_EVIDENCE_URL="http://127.0.0.1:9000/analyze"
$env:RAD_TRAINER_CHEST_EVIDENCE_TIMEOUT_SECONDS="300"
uv run python app.py
uv run python scripts/smoke_gradio_xraydar.py --app-url http://127.0.0.1:7860
```

Expected endpoint response:

```json
{
  "anatomy": "chest",
  "findings": [
    {"label": "pleural effusion", "score": 0.83, "source": "xraydar"}
  ],
  "regions": [
    {"label": "pleural effusion", "x1": 0.08, "y1": 0.64, "x2": 0.41, "y2": 0.95}
  ],
  "model_notes": ["X-Raydar local service"]
}
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

## Submission Materials

- [Model integration plan](docs/model_integration.md)
- [Submission pack](docs/submission_pack.md)
- [Field notes draft](docs/field_notes.md)
- [Space deploy runbook](docs/deploy_space.md)
- [X-Raydar backend validation](docs/validation_xraydar_demo.md)

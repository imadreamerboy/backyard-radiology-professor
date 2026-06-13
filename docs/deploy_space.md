# Deploy to Hugging Face Spaces

This repo is ready for a Gradio Space. The environment here is not logged in to Hugging Face, so deployment requires a write token from the account that will own the Space.

## Option A: Website

1. Create a new Space at `https://huggingface.co/new-space`.
2. Choose SDK: `Gradio`.
3. Use this repo's `README.md` frontmatter and `app.py`.
4. Push the repository files to the Space repo.
5. Add secrets if using hosted Nemotron:
   - `HF_TOKEN`
   - `RAD_TRAINER_TUTOR_PROVIDER=hf`
   - `RAD_TRAINER_HF_PROVIDER=nvidia`
   - `RAD_TRAINER_NEMOTRON_MODEL=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`

## Option B: Git Remote

```powershell
uv run hf auth login
uv run hf repo create <space-name> --type space --space-sdk gradio
git remote add space https://huggingface.co/spaces/<user-or-org>/<space-name>
git push space feat/radiology-trainer-mvp:main
```

## Public Demo Defaults

For a safe public Space, either leave model env vars unset or point to a separately hosted X-Raydar-compatible evidence endpoint. With env vars unset, the app will use:

- synthetic demo cases
- deterministic demo evidence
- demo tutor text
- no patient-image persistence

For a real backend demo Space, set:

- `RAD_TRAINER_MODEL_MODE=local`
- `RAD_TRAINER_CHEST_EVIDENCE_URL=https://<your-evidence-service>/analyze`
- `RAD_TRAINER_CHEST_EVIDENCE_TIMEOUT_SECONDS=180`

Do not bundle X-Raydar weights into the public Space unless its non-commercial/research terms and Space hardware constraints are acceptable.

## Local Nemotron Demo

For a stronger recording, run locally with Nemotron:

```powershell
$env:RAD_TRAINER_TUTOR_PROVIDER="openai"
$env:RAD_TRAINER_OPENAI_BASE_URL="http://localhost:8000/v1"
$env:RAD_TRAINER_OPENAI_API_KEY="local"
$env:RAD_TRAINER_NEMOTRON_MODEL="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
uv run python app.py
```

Then record the same flow:

1. Pick `Right Lower Opacity`.
2. Click `Analyze`.
3. Show scorecard, tutor feedback, quiz, and session note.

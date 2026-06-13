# Submission Pack

## One-line Pitch

Backyard Radiology Trainer helps a new radiologist practice chest X-ray reads by forcing a blind read first, then revealing structured evidence, targeted feedback, and a copy-ready learning note.

## Backyard AI Fit

- **Specific person:** built for the author's brother, a new radiologist who wants more deliberate practice.
- **Specific problem:** he needs repetitions, feedback, and a safer way to ask questions while learning.
- **Small-model fit:** the app uses small-model orchestration instead of one giant opaque medical model.
- **Gradio polish:** the workflow is a reading workstation: upload, blind read, evidence, scorecard, tutor, quiz, session note.

## Model Story

Public demo mode is intentionally safe and deterministic:

- synthetic example images only
- no patient data
- no clinical claims
- no model pretending to diagnose

Local mode turns on the real stack:

- **Nemotron 3 Nano** as the tutor/orchestrator for critique, uncertainty, and quiz generation.
- **External chest evidence endpoint** for X-Raydar, MedSigLIP, CXR Foundation, or later anatomy-specific tools.
- **MedGemma 1.5 4B** as an optional image-conditioned medical VLM note source.

## Demo Video Script

1. Start on a synthetic case.
2. Write a blind read with one intentional omission.
3. Click Analyze.
4. Show the overlay and structured evidence.
5. Show the blind-read scorecard and missed finding.
6. Show tutor feedback and quiz.
7. Open the copy-ready session note.
8. Briefly show local-mode config for Nemotron and future X-Raydar/MedGemma tools.

## Judging Checklist

- Real person/problem: mention the brother and why this workflow fits daily practice.
- Actual use: record a quick before/after read session and include one quote or observation.
- Small-model honesty: emphasize tool routing and explicit evidence instead of a giant diagnosis model.
- Gradio polish: show the built-in examples, overlay, scorecard, and session note.
- NVIDIA angle: configure `RAD_TRAINER_TUTOR_PROVIDER=hf` or `openai` with a Nemotron model for the tutor layer.

## Remaining Before Submission

- Deploy the Space in demo mode.
- Add Space secrets for `HF_TOKEN` if using hosted Nemotron.
- Record a 60-90 second demo video.
- Ask the brother to run 2-3 cases and capture feedback.
- Publish field notes from `docs/field_notes.md`.


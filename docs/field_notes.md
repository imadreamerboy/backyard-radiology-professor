# Field Notes: Building a Chest X-ray Practice Coach for My Brother

My brother recently started working as a radiologist. He does not need another generic chatbot. He needs repetitions: look at an image, commit to a read, compare that read against evidence, and learn what to inspect next.

That shaped the app. Backyard Radiology Trainer does not start with an answer. It starts with a blind read. Only after the trainee writes down an impression does the app reveal evidence, critique the read, ask follow-up questions, and produce a session note.

## Why Small Models Fit

The medical setting makes a single all-knowing model the wrong product shape. The useful workflow is smaller and more explicit:

- a chest evidence layer for findings and regions
- a medical VLM hook for image-conditioned observations
- Nemotron as the tutor/orchestrator
- deterministic scoring so the public demo still teaches something without patient data

This is also safer. The app is educational, not clinical. It says what evidence it used, what it missed, and what the trainee should verify manually.

## What Changed During the Build

The first prototype was just upload plus explanation. That was not enough. The useful version became a practice loop:

1. upload a case
2. write a blind read
3. reveal evidence
4. get a scorecard
5. receive tutor feedback
6. answer a quiz
7. save the session note

The scorecard turned out to matter. It gives the learner a measurable target without pretending to be a clinical benchmark.

## What Works Now

- Chest-first Gradio app.
- Synthetic demo cases for public judging.
- Image and optional DICOM ingestion.
- Structured evidence table.
- Evidence overlay.
- Blind-read scorecard.
- Nemotron-compatible tutor layer.
- Optional MedGemma 1.5 VLM notes.
- External evidence endpoint for X-Raydar or other classifiers.

## What Comes Next

The next step is replacing demo evidence with a local chest model service, likely X-Raydar first, then MedSigLIP or CXR Foundation for routing and retrieval. After that, the same evidence contract can support other anatomy-specific plain-film pipelines.

The product goal stays the same: better practice for one real person, not a clinical decision system.


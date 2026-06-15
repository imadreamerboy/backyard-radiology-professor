# Record the demo video

Record a deterministic browser run first, then add a short voiceover in OBS
Studio or your video editor. The automated recording proves the real public
workflow; the narrated edit communicates the story clearly.

## Prepare

Deploy Modal, resume the Hugging Face Space, and verify:

```bash
uv run python scripts/validate_golden_cases.py \
  --app-url https://build-small-hackathon-backyard-radiology-professor.hf.space \
  --timeout-seconds 1800
```

Install the recording dependencies:

```bash
uv sync --extra video
uv run playwright install chromium
```

## Record the browser workflow

```bash
uv run python scripts/record_demo.py --headed
```

The script records a 1440x900 walkthrough under `artifacts/video/`. It opens the
tutorial, selects the scoliosis case, commits a blind read, shows evidence, and
asks the professor a grounded question. It waits for the real backend rather
than substituting demo output. If `ffmpeg` is installed, it also creates an MP4.

## Suggested two-minute narration

1. "I built Backyard Radiology Professor for my brother, who recently started working as a radiologist and wanted more structured practice."
2. "The learner starts with a blind read, before any model evidence is visible."
3. "X-Raydar provides independent classifier evidence. MedGemma 1.5 4B adds observations and clickable regions."
4. "The 27B MedGemma professor sees the image, metadata, blind read, and attributed evidence. It can explain misses, build a differential, or generate a quiz."
5. "The public Hugging Face Space is lightweight. The authenticated Modal GPU backend starts on demand and scales back to zero when idle."
6. "This is an educational demo, not clinical software."

For the final submission:

1. Run the automated capture once and keep the unedited WebM as proof.
2. Record the narration in OBS Studio while replaying the same workflow, or add
   narration over the generated MP4.
3. Show the model identity, evidence provenance, clickable regions, and one
   grounded professor response.
4. Keep the full workstation visible. Speed up only the cold-start wait and
   label that edit.
5. End on the educational-use statement and repository or Space URL.

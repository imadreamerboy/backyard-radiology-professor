# Future backlog

These ideas are intentionally outside the hackathon demo scope.

## Agentic study router

- Detect anatomy and modality before choosing a model path.
- Route chest radiographs to X-Raydar and future body-part-specific classifiers.
- Ask the MedGemma localizer for targeted regions after the classifier proposes a
  finding.
- Let the professor request additional model passes only when the request is
  visible and logged.

## Medical reference grounding

- Add a curated teaching-source retriever for radiology anatomy, signs, and
  differentials.
- Keep retrieval citations separate from image observations.
- Prefer offline or locally cached references for the local-first workflow.

## Longitudinal teaching mode

- Track repeated misses by concept and view.
- Generate spaced quiz cards from completed sessions.
- Keep all learner history local unless an explicit sync option is added.

## Additional models

- Evaluate body-part-specific classifiers before enabling non-chest studies.
- Add segmentation models only when their output can be controlled, attributed,
  hidden, and exported like current region overlays.

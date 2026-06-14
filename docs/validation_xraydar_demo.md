# Real-backend validation

The golden suite uses the three public X-Raydar demonstration cases:

- Normal chest radiograph.
- Thoracic scoliosis.
- Cardiomegaly with widened mediastinum and unfolded aorta.

For each case the validator requires:

- The reference labels match the bundled public labels.
- X-Raydar produces the expected high-ranking evidence.
- MedGemma 1.5 returns accepted observations and bounded regions.
- MedGemma 27B returns structured feedback and a quiz.
- Every model run reports success and its real model identity.

Run:

```bash
uv run python scripts/validate_golden_cases.py
```

Derived DX fixtures can be generated from the same public pixels:

```bash
uv run python scripts/generate_dicom_fixtures.py
```

These files are explicitly marked as derived software-test fixtures, not original acquisition DICOMs.

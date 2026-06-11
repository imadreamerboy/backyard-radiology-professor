# Model Integration Plan

The app is built around stable internal contracts, not one monolithic model call.

## Current MVP

- `DemoChestEvidenceModel`: deterministic, demo-safe evidence for UI and flow testing.
- `NemotronTutorModel`: active tutor adapter for Hugging Face or OpenAI-compatible endpoints.
- `EvidenceBundle`: shared contract for classifiers, localizers, retrieval tools, VLM notes, and segmentation outputs.

## Intended Local Stack

1. **X-Raydar**
   - Role: frontal chest finding probabilities.
   - Model: `dnamodel/xraydar-cv`.
   - Notes: non-commercial/research terms; requires the upstream `gmontana/xraydar-cv` code path and HF weights.

2. **MedSigLIP or CXR Foundation**
   - Role: anatomy routing, zero-shot labels, retrieval, out-of-distribution checks.
   - Notes: use before calling chest-specific tools so later body-part pipelines can be added cleanly.

3. **MedGemma 1.5 4B**
   - Role: image-conditioned medical explanation and bounding-box/anatomical localization.
   - Model: `google/medgemma-1.5-4b-it`.
   - Notes: gated/terms-governed; should be a medical VLM tool, not the only perceptual source.

4. **Nemotron 3 Nano**
   - Role: main tutor/orchestrator, blind-read critique, uncertainty handling, quiz generation.
   - Default model: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`.
   - Notes: run via vLLM/SGLang/OpenAI-compatible endpoint for local compute, or HF provider if available.

5. **SAM 3.1**
   - Role: interactive region refinement after classifier/VLM evidence proposes a target.
   - Notes: not a diagnostic model; use it for overlays and user-driven segmentation.

## Extension Pattern

Add a new anatomy by implementing an evidence adapter that returns `EvidenceBundle`, then register it behind the anatomy router. The UI and tutor layer should not need anatomy-specific changes.


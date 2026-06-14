# Field notes

The workstation is designed for a new radiologist who wants repeated, image-grounded practice rather than a generic medical chatbot.

Important product decisions:

- The blind read remains locked before evidence appears.
- Model evidence is independently attributed instead of blended into one answer.
- MedGemma 1.5 4B owns localization because its published chest anatomy bounding-box result is stronger than the earlier 27B model.
- MedGemma 27B owns teaching and conversation.
- The 4B model is not called autonomously during chat; this keeps behavior traceable and avoids repeated model swaps.
- CR/DX chest studies are supported. Other modalities are rejected instead of implying unsupported AI capability.

Measured runtime results belong in generated benchmark JSON. Do not copy stale VRAM or latency numbers into this document.

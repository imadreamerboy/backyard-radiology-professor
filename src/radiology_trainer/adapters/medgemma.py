from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from radiology_trainer.domain import EvidenceBundle


@dataclass
class MedGemmaVisionTool:
    model_id: str
    name: str = "medgemma-vision"
    _pipe: Any = field(default=None, init=False, repr=False)

    def describe(self, image: Image.Image, evidence: EvidenceBundle) -> str:
        pipe = self._load_pipeline()
        prompt = _build_prompt(evidence)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        output = pipe(text=messages, max_new_tokens=500, do_sample=False)
        generated = output[0]["generated_text"]
        if isinstance(generated, list):
            return generated[-1].get("content", "").strip()
        return str(generated).strip()

    def _load_pipeline(self):
        if self._pipe is not None:
            return self._pipe

        import torch
        from transformers import pipeline

        has_cuda = torch.cuda.is_available()
        self._pipe = pipeline(
            "image-text-to-text",
            model=self.model_id,
            torch_dtype=torch.bfloat16 if has_cuda else torch.float32,
            device=0 if has_cuda else -1,
        )
        return self._pipe


def _build_prompt(evidence: EvidenceBundle) -> str:
    top_findings = ", ".join(
        f"{finding.label}={finding.score:.2f}" for finding in evidence.top_findings(5)
    )
    return (
        "Educational chest X-ray support only. "
        "Briefly describe image-conditioned observations that a trainee should verify manually. "
        "Do not claim a clinical diagnosis. "
        f"Structured classifier evidence: {top_findings or 'none'}."
    )


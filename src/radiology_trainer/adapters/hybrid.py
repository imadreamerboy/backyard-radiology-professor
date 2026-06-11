from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from radiology_trainer.adapters.base import EvidenceModel
from radiology_trainer.domain import EvidenceBundle


@dataclass
class HybridChestEvidenceModel:
    base_model: EvidenceModel
    vision_tools: list[object]
    name: str = "hybrid-chest-evidence"

    def analyze(self, image: Image.Image) -> EvidenceBundle:
        evidence = self.base_model.analyze(image)
        for tool in self.vision_tools:
            try:
                note = tool.describe(image, evidence)
            except Exception as exc:
                evidence.model_notes.append(f"{tool.name} unavailable: {exc}")
                continue
            if note:
                evidence.model_notes.append(f"{tool.name}: {note}")
        return evidence


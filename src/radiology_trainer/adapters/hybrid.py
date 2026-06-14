from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from radiology_trainer.adapters.base import EvidenceModel
from radiology_trainer.adapters.medgemma import MedGemmaVisionTool
from radiology_trainer.domain import EvidenceBundle


@dataclass
class HybridChestEvidenceModel:
    base_model: EvidenceModel
    vision_tool: MedGemmaVisionTool
    name: str = "xraydar-medgemma-evidence"

    def analyze(self, image: Image.Image) -> EvidenceBundle:
        evidence = self.base_model.analyze(image)
        observations, regions, uncertainty, run = self.vision_tool.analyze(image, evidence)
        evidence.observations.extend(observations)
        evidence.regions.extend(regions)
        evidence.model_runs.append(run)
        evidence.agreement_notes.extend(uncertainty)
        if run.status == "error":
            evidence.model_notes.append(f"MedGemma output rejected: {run.detail}")
        return evidence

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from radiology_trainer.adapters.base import EvidenceModel, TutorModel
from radiology_trainer.adapters.demo import DemoChestEvidenceModel, DemoTutorModel
from radiology_trainer.adapters.hybrid import HybridChestEvidenceModel
from radiology_trainer.adapters.medgemma import MedGemmaVisionTool
from radiology_trainer.adapters.nemotron import NemotronTutorModel
from radiology_trainer.config import AppConfig
from radiology_trainer.domain import Anatomy, EvidenceBundle, StudentRead, TutorResponse


@dataclass
class RadiologyTrainerPipeline:
    evidence_model: EvidenceModel
    tutor_model: TutorModel

    def analyze(self, image: Image.Image, student_read: StudentRead) -> tuple[EvidenceBundle, TutorResponse]:
        evidence = self.evidence_model.analyze(image)
        tutor = self.tutor_model.coach(evidence, student_read)
        return evidence, tutor


class AnatomyRegistry:
    def __init__(self) -> None:
        self._models: dict[Anatomy, EvidenceModel] = {}

    def register(self, anatomy: Anatomy, model: EvidenceModel) -> None:
        self._models[anatomy] = model

    def get(self, anatomy: Anatomy) -> EvidenceModel:
        return self._models.get(anatomy) or self._models[Anatomy.CHEST]


def build_pipeline(config: AppConfig | None = None) -> RadiologyTrainerPipeline:
    cfg = config or AppConfig.from_env()
    evidence_model = _build_evidence_model(cfg)
    tutor_model = _build_tutor_model(cfg)
    return RadiologyTrainerPipeline(evidence_model=evidence_model, tutor_model=tutor_model)


def _build_evidence_model(config: AppConfig) -> EvidenceModel:
    base_model = DemoChestEvidenceModel()
    vision_tools = []
    if config.enable_medical_vlm:
        vision_tools.append(MedGemmaVisionTool(model_id=config.medical_vlm))

    if vision_tools:
        return HybridChestEvidenceModel(base_model=base_model, vision_tools=vision_tools)

    if config.model_mode != "demo":
        # Keep this explicit until real adapters are integrated. Silent fake inference is worse.
        return DemoChestEvidenceModel(name=f"demo-until-{config.chest_classifier}-adapter-is-wired")
    return base_model


def _build_tutor_model(config: AppConfig) -> TutorModel:
    if config.tutor_provider in {"hf", "openai"}:
        return NemotronTutorModel(config=config)
    return DemoTutorModel()

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from radiology_trainer.adapters.base import EvidenceModel, TutorModel
from radiology_trainer.adapters.demo import DemoChestEvidenceModel, DemoTutorModel
from radiology_trainer.adapters.hybrid import HybridChestEvidenceModel
from radiology_trainer.adapters.llama_client import LlamaCppClient
from radiology_trainer.adapters.medgemma import MedGemmaVisionTool
from radiology_trainer.adapters.professor import MedGemmaProfessorModel
from radiology_trainer.adapters.xraydar import XRaydarEvidenceModel
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
    if config.model_mode == "demo":
        return DemoChestEvidenceModel()

    base_model: EvidenceModel = XRaydarEvidenceModel(
        backend_dir=Path(config.xraydar_backend_dir),
        device_name=config.xraydar_device,
        model_revision=config.xraydar_revision,
    )
    if not config.enable_medical_vlm:
        return base_model

    client = LlamaCppClient(
        base_url=config.llama_base_url,
        api_key=config.llama_api_key,
        timeout_seconds=config.model_timeout_seconds,
    )
    return HybridChestEvidenceModel(
        base_model=base_model,
        vision_tool=MedGemmaVisionTool(
            client=client,
            model_id=config.localizer_model,
            model_revision=config.localizer_revision,
        ),
    )


def _build_tutor_model(config: AppConfig) -> TutorModel:
    if config.tutor_provider == "llama":
        client = LlamaCppClient(
            base_url=config.llama_base_url,
            api_key=config.llama_api_key,
            timeout_seconds=config.model_timeout_seconds,
        )
        return MedGemmaProfessorModel(config=config, client=client)
    return DemoTutorModel()

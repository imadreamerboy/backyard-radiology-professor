from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    model_mode: str = "demo"
    tutor_provider: str = "demo"
    nemotron_model: str = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    hf_provider: str | None = "nvidia"
    hf_token: str | None = None
    openai_base_url: str | None = None
    openai_api_key: str | None = None
    chest_classifier: str = "demo"
    medical_vlm: str = "google/medgemma-1.5-4b-it"
    enable_medical_vlm: bool = False
    enable_segmentation: bool = False

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            model_mode=os.getenv("RAD_TRAINER_MODEL_MODE", "demo").strip().lower(),
            tutor_provider=os.getenv("RAD_TRAINER_TUTOR_PROVIDER", "demo").strip().lower(),
            nemotron_model=os.getenv(
                "RAD_TRAINER_NEMOTRON_MODEL",
                "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
            ).strip(),
            hf_provider=os.getenv("RAD_TRAINER_HF_PROVIDER", "nvidia").strip() or None,
            hf_token=os.getenv("HF_TOKEN") or None,
            openai_base_url=os.getenv("RAD_TRAINER_OPENAI_BASE_URL") or None,
            openai_api_key=os.getenv("RAD_TRAINER_OPENAI_API_KEY") or None,
            chest_classifier=os.getenv("RAD_TRAINER_CHEST_CLASSIFIER", "demo").strip().lower(),
            medical_vlm=os.getenv(
                "RAD_TRAINER_MEDICAL_VLM",
                "google/medgemma-1.5-4b-it",
            ).strip(),
            enable_medical_vlm=os.getenv("RAD_TRAINER_ENABLE_MEDICAL_VLM", "false")
            .strip()
            .lower()
            in {"1", "true", "yes", "on"},
            enable_segmentation=os.getenv("RAD_TRAINER_ENABLE_SEGMENTATION", "false")
            .strip()
            .lower()
            in {"1", "true", "yes", "on"},
        )

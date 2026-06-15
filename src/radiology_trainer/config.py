from __future__ import annotations

import os
from dataclasses import dataclass

from radiology_trainer.runtime_manifest import (
    LOCALIZER_REVISION,
    PROFESSOR_REVISION,
    XRAYDAR_REVISION,
)


@dataclass(frozen=True)
class AppConfig:
    model_mode: str = "demo"
    tutor_provider: str = "demo"
    llama_base_url: str = "http://127.0.0.1:8080/v1"
    llama_api_key: str = "local"
    remote_backend_url: str = ""
    professor_model: str = "medgemma-professor"
    professor_revision: str = PROFESSOR_REVISION
    localizer_model: str = "medgemma-localizer"
    localizer_revision: str = LOCALIZER_REVISION
    model_timeout_seconds: float = 900.0
    chest_classifier: str = "demo"
    xraydar_backend_dir: str = "outputs/xraydar-cv"
    xraydar_revision: str = XRAYDAR_REVISION
    xraydar_device: str = "cuda"
    enable_medical_vlm: bool = False
    session_ttl_minutes: int = 120
    max_upload_mb: int = 200
    max_study_files: int = 32
    max_study_uncompressed_mb: int = 500
    chat_history_messages: int = 12

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            model_mode=os.getenv("RAD_TRAINER_MODEL_MODE", "demo").strip().lower(),
            tutor_provider=os.getenv("RAD_TRAINER_TUTOR_PROVIDER", "demo").strip().lower(),
            llama_base_url=os.getenv(
                "RAD_TRAINER_LLAMA_BASE_URL", "http://127.0.0.1:8080/v1"
            ).strip().rstrip("/"),
            llama_api_key=os.getenv("RAD_TRAINER_LLAMA_API_KEY", "local").strip(),
            remote_backend_url=os.getenv("RAD_TRAINER_REMOTE_BACKEND_URL", "")
            .strip()
            .rstrip("/"),
            professor_model=os.getenv(
                "RAD_TRAINER_PROFESSOR_MODEL",
                "medgemma-professor",
            ).strip(),
            professor_revision=os.getenv(
                "RAD_TRAINER_PROFESSOR_REVISION",
                PROFESSOR_REVISION,
            ).strip(),
            localizer_model=os.getenv(
                "RAD_TRAINER_LOCALIZER_MODEL",
                "medgemma-localizer",
            ).strip(),
            localizer_revision=os.getenv(
                "RAD_TRAINER_LOCALIZER_REVISION",
                LOCALIZER_REVISION,
            ).strip(),
            model_timeout_seconds=float(
                os.getenv("RAD_TRAINER_MODEL_TIMEOUT_SECONDS", "900")
            ),
            chest_classifier=os.getenv("RAD_TRAINER_CHEST_CLASSIFIER", "demo").strip().lower(),
            xraydar_backend_dir=os.getenv(
                "RAD_TRAINER_XRAYDAR_BACKEND_DIR",
                "outputs/xraydar-cv",
            ).strip(),
            xraydar_revision=os.getenv(
                "RAD_TRAINER_XRAYDAR_REVISION",
                XRAYDAR_REVISION,
            ).strip(),
            xraydar_device=os.getenv("RAD_TRAINER_XRAYDAR_DEVICE", "cuda").strip(),
            enable_medical_vlm=os.getenv("RAD_TRAINER_ENABLE_MEDICAL_VLM", "false")
            .strip()
            .lower()
            in {"1", "true", "yes", "on"},
            session_ttl_minutes=int(os.getenv("RAD_TRAINER_SESSION_TTL_MINUTES", "120")),
            max_upload_mb=int(os.getenv("RAD_TRAINER_MAX_UPLOAD_MB", "200")),
            max_study_files=int(os.getenv("RAD_TRAINER_MAX_STUDY_FILES", "32")),
            max_study_uncompressed_mb=int(
                os.getenv("RAD_TRAINER_MAX_STUDY_UNCOMPRESSED_MB", "500")
            ),
            chat_history_messages=int(
                os.getenv("RAD_TRAINER_CHAT_HISTORY_MESSAGES", "12")
            ),
        )

from __future__ import annotations

from typing import Protocol

from PIL import Image

from radiology_trainer.domain import EvidenceBundle, StudentRead, TutorResponse


class EvidenceModel(Protocol):
    name: str

    def analyze(self, image: Image.Image) -> EvidenceBundle:
        """Return structured model evidence for one image."""


class TutorModel(Protocol):
    name: str

    def coach(self, evidence: EvidenceBundle, student_read: StudentRead) -> TutorResponse:
        """Return educational feedback grounded in structured evidence."""


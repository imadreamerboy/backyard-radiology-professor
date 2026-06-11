from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from PIL import Image

from radiology_trainer.domain import (
    Anatomy,
    EvidenceBundle,
    FindingScore,
    RegionBox,
    StudentRead,
    TutorResponse,
)
from radiology_trainer.preprocessing import assess_quality, prepare_image, route_anatomy


CHEST_FINDINGS = [
    "cardiomegaly",
    "pleural effusion",
    "atelectasis",
    "pneumothorax",
    "consolidation",
    "pulmonary edema",
    "rib fracture",
]


@dataclass
class DemoChestEvidenceModel:
    name: str = "demo-chest-evidence"

    def analyze(self, image: Image.Image) -> EvidenceBundle:
        prepared = prepare_image(image)
        anatomy = route_anatomy(prepared)
        quality = assess_quality(prepared)
        scores = _score_image(prepared)

        findings = [
            FindingScore(
                label=label,
                score=score,
                source=self.name,
                explanation="Demo score from image statistics; replace with X-Raydar/local model.",
            )
            for label, score in scores
        ]
        regions = _demo_regions(findings) if anatomy == Anatomy.CHEST else []

        notes = [
            "Demo evidence is deterministic and UI-focused, not model inference.",
            "Local mode should replace this adapter with X-Raydar, MedSigLIP, and MedGemma tools.",
        ]
        if anatomy != Anatomy.CHEST:
            notes.append("Router did not strongly identify this as a chest radiograph.")

        return EvidenceBundle(
            anatomy=anatomy,
            quality=quality,
            findings=findings,
            regions=regions,
            agreement_notes=[
                "No real second-opinion model is active in demo mode.",
                "The app contract supports classifier, VLM, and segmentation agreement later.",
            ],
            model_notes=notes,
        )


@dataclass
class DemoTutorModel:
    name: str = "demo-nemotron-tutor"

    def coach(self, evidence: EvidenceBundle, student_read: StudentRead) -> TutorResponse:
        top = evidence.top_findings(3)
        top_labels = ", ".join(f"{item.label} ({item.score:.2f})" for item in top)
        student_text = student_read.observation.strip()

        feedback = []
        if student_text:
            feedback.append("Compare your read against the top evidence signals instead of accepting them.")
            feedback.append("Check whether your read mentions view, technical quality, and key negatives.")
        else:
            feedback.append("Write a blind read before revealing model evidence for better practice.")

        if top:
            feedback.append(f"Highest demo signals: {top_labels}.")

        return TutorResponse(
            summary=(
                "Educational demo response. In local mode this slot is intended for Nemotron "
                "as the tutoring and reasoning layer."
            ),
            feedback=feedback,
            suggested_checks=[
                "Confirm projection and image quality before findings.",
                "Trace tubes/lines and pleural margins before calling lung findings.",
                "Use a search pattern: airway, bones, cardiac silhouette, diaphragm, everything else.",
            ],
            uncertainty=[
                "Demo mode is not reading the radiograph.",
                "Any finding should be verified manually and with supervised teaching material.",
            ],
            quiz=[
                "What view is this image, and how does that affect heart-size assessment?",
                "Which regions would you inspect before excluding pneumothorax?",
                "What key negative finding should be stated explicitly?",
            ],
            provider=self.name,
        )


def _score_image(image: Image.Image) -> list[tuple[str, float]]:
    gray = prepare_image(image).convert("L").resize((256, 256))
    arr = np.asarray(gray).astype(np.float32) / 255.0
    digest = hashlib.sha256(arr.tobytes()).digest()
    brightness = float(np.mean(arr))
    contrast = float(np.std(arr))
    vertical_gradient = float(np.mean(np.abs(np.diff(arr, axis=0))))
    horizontal_gradient = float(np.mean(np.abs(np.diff(arr, axis=1))))

    features = [
        brightness,
        contrast,
        vertical_gradient,
        horizontal_gradient,
        abs(brightness - 0.48),
        min(contrast * 3.0, 1.0),
        min((vertical_gradient + horizontal_gradient) * 8.0, 1.0),
    ]

    scores: list[tuple[str, float]] = []
    for idx, label in enumerate(CHEST_FINDINGS):
        noise = digest[idx] / 255.0
        raw = 0.18 + (features[idx] * 0.45) + (noise * 0.22)
        scores.append((label, round(max(0.02, min(raw, 0.94)), 3)))
    return sorted(scores, key=lambda item: item[1], reverse=True)


def _demo_regions(findings: list[FindingScore]) -> list[RegionBox]:
    templates = {
        "cardiomegaly": (0.35, 0.46, 0.66, 0.78),
        "pleural effusion": (0.08, 0.64, 0.40, 0.94),
        "atelectasis": (0.18, 0.58, 0.45, 0.78),
        "pneumothorax": (0.62, 0.15, 0.91, 0.55),
        "consolidation": (0.18, 0.28, 0.50, 0.62),
        "pulmonary edema": (0.26, 0.25, 0.74, 0.63),
        "rib fracture": (0.08, 0.22, 0.32, 0.58),
    }
    regions: list[RegionBox] = []
    for finding in findings[:3]:
        x1, y1, x2, y2 = templates[finding.label]
        regions.append(
            RegionBox(
                label=finding.label,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                score=finding.score,
                source="demo-region-prior",
            )
        )
    return regions


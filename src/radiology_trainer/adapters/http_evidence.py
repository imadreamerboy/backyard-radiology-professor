from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import requests
from PIL import Image

from radiology_trainer.domain import Anatomy, EvidenceBundle, FindingScore, RegionBox
from radiology_trainer.preprocessing import assess_quality, prepare_image


@dataclass
class HTTPChestEvidenceModel:
    url: str
    timeout_seconds: float = 60.0
    name: str = "http-chest-evidence"

    def analyze(self, image: Image.Image) -> EvidenceBundle:
        prepared = prepare_image(image).convert("RGB")
        buffer = BytesIO()
        prepared.save(buffer, format="JPEG", quality=95)
        buffer.seek(0)

        response = requests.post(
            self.url,
            files={"image": ("xray.jpg", buffer, "image/jpeg")},
            data={"anatomy": "chest"},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()

        findings = [
            FindingScore(
                label=str(item["label"]),
                score=float(item["score"]),
                source=str(item.get("source", self.name)),
                explanation=str(item.get("explanation", "External chest evidence endpoint.")),
            )
            for item in payload.get("findings", [])
        ]
        regions = [
            RegionBox(
                label=str(item["label"]),
                x1=float(item["x1"]),
                y1=float(item["y1"]),
                x2=float(item["x2"]),
                y2=float(item["y2"]),
                score=float(item.get("score", 1.0)),
                source=str(item.get("source", self.name)),
            )
            for item in payload.get("regions", [])
        ]

        if not findings:
            raise ValueError("External evidence endpoint returned no findings.")

        return EvidenceBundle(
            anatomy=_parse_anatomy(payload.get("anatomy")),
            quality=assess_quality(prepared),
            findings=findings,
            regions=regions,
            agreement_notes=[str(note) for note in payload.get("agreement_notes", [])],
            model_notes=[str(note) for note in payload.get("model_notes", [])]
            or [f"Evidence supplied by {self.url}."],
        )


def _parse_anatomy(value) -> Anatomy:
    try:
        return Anatomy(str(value or "chest"))
    except ValueError:
        return Anatomy.CHEST


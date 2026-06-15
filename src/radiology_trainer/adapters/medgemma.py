from __future__ import annotations

import time
from dataclasses import dataclass

from PIL import Image
from pydantic import BaseModel, Field

from radiology_trainer.adapters.llama_client import LlamaCppClient
from radiology_trainer.domain import (
    EvidenceBundle,
    ModelRun,
    RegionBox,
    VisionObservation,
)
from radiology_trainer.runtime_manifest import LLAMA_CPP_BUILD, LOCALIZER_REPO
from radiology_trainer.structured_output import parse_json_model, parse_json_value


class _Observation(BaseModel):
    label: str
    description: str


class _Prediction(BaseModel):
    label: str
    rationale: str


class _RawBox(BaseModel):
    box_2d: list[float] = Field(min_length=4, max_length=4)
    label: str


class _AssessmentPayload(BaseModel):
    prediction: _Prediction
    observations: list[_Observation] = Field(default_factory=list, max_length=6)
    uncertainty: list[str] = Field(default_factory=list, max_length=4)


_ASSESSMENT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "observations": {
            "type": "array",
            "maxItems": 6,
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "minLength": 1, "maxLength": 60},
                    "description": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 180,
                    },
                },
                "required": ["label", "description"],
                "additionalProperties": False,
            },
        },
        "prediction": {
            "type": "object",
            "properties": {
                "label": {"type": "string", "minLength": 1, "maxLength": 80},
                "rationale": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 180,
                },
            },
            "required": ["label", "rationale"],
            "additionalProperties": False,
        },
        "uncertainty": {
            "type": "array",
            "maxItems": 4,
            "items": {"type": "string", "minLength": 1, "maxLength": 180},
        },
    },
    "required": ["prediction", "observations", "uncertainty"],
    "additionalProperties": False,
}

_BOX_JSON_SCHEMA = {
    "type": "array",
    "minItems": 1,
    "maxItems": 1,
    "items": {
        "type": "object",
        "properties": {
            "box_2d": {
                "type": "array",
                "minItems": 4,
                "maxItems": 4,
                "items": {"type": "number", "minimum": 0, "maximum": 1000},
            },
            "label": {"type": "string", "minLength": 1, "maxLength": 60},
        },
        "required": ["box_2d", "label"],
        "additionalProperties": False,
    },
}

_LOCALIZATION_TARGETS = {
    "scoliosis": "thoracic spine showing scoliosis",
    "cardiomegaly": "cardiac silhouette",
    "mediastinum widened": "widened upper mediastinum",
    "tortuosity aorta": "visible aortic contour",
}


@dataclass
class MedGemmaVisionTool:
    client: LlamaCppClient
    model_id: str
    model_revision: str
    name: str = "medgemma-localizer"

    def analyze(
        self,
        image: Image.Image,
        evidence: EvidenceBundle,
    ) -> tuple[list[VisionObservation], list[RegionBox], list[str], ModelRun]:
        started = time.perf_counter()
        targets = [
            finding.label
            for finding in evidence.top_findings(4)
            if finding.score >= 0.4
        ]
        padded_image = _pad_to_square(image)
        try:
            text = self.client.chat(
                model=self.model_id,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": _build_assessment_prompt()},
                            {
                                "type": "image_url",
                                "image_url": {"url": self.client.image_url(padded_image)},
                            },
                        ],
                    }
                ],
                max_tokens=450,
                temperature=0.0,
                json_schema=_ASSESSMENT_JSON_SCHEMA,
            )
            payload = parse_json_model(text, _AssessmentPayload)
            observations = []
            observations.append(
                VisionObservation(
                    label=f"Prediction: {payload.prediction.label}",
                    description=payload.prediction.rationale,
                    source=self.name,
                )
            )
            observations.extend([
                VisionObservation(
                    label=item.label,
                    description=item.description,
                    source=self.name,
                )
                for item in payload.observations
            ])
            regions = []
            box_uncertainty: list[str] = []
            for target in targets[:3]:
                try:
                    regions.append(
                        _localize_target(
                            self.client,
                            self.model_id,
                            padded_image,
                            image.size,
                            target,
                        )
                    )
                except Exception as exc:
                    box_uncertainty.append(f"No valid MedGemma box for {target}: {exc}")
        except Exception as exc:
            return [], [], [], _model_run(
                started,
                self.model_id,
                self.model_revision,
                "error",
                str(exc),
            )

        return (
            observations,
            regions,
            [*payload.uncertainty, *box_uncertainty],
            _model_run(started, self.model_id, self.model_revision, "ok"),
        )


def _build_assessment_prompt() -> str:
    return (
        "Educational chest X-ray review only. Independently inspect the image. Provide one "
        "best educational prediction in prediction.label with a short visual rationale. "
        "Use 'No acute abnormality' when no focal finding is visible. "
        "Then describe up to six visible observations that a trainee should verify. Keep "
        "every description under 20 words. Uncertainty entries must be short limitations, "
        "not anatomy labels. Do not infer diagnoses from metadata or prior model output. "
        "Return only a JSON object containing prediction, observations, and uncertainty."
    )


def _localize_target(
    client: LlamaCppClient,
    model_id: str,
    padded_image: Image.Image,
    original_size: tuple[int, int],
    target: str,
) -> RegionBox:
    object_name = _LOCALIZATION_TARGETS.get(target, target)
    prompt = (
        "Instructions: The query requires one bounding box. Coordinates are "
        "[y0, x0, y1, x1], where the first point is top-left and the second is "
        "bottom-right. Normalize coordinates to [0, 1000]. Return one parseable JSON "
        "list containing one object with box_2d and label keys. The box must tightly "
        f"localize only the requested anatomy or finding. Query: Where is the {object_name}?"
    )
    text = client.chat(
        model=model_id,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": client.image_url(padded_image)}},
                ],
            }
        ],
        max_tokens=160,
        temperature=0.0,
        json_schema=_BOX_JSON_SCHEMA,
    )
    values = parse_json_value(text)
    if not isinstance(values, list) or len(values) != 1:
        raise ValueError(f"MedGemma returned an invalid box list for {target}.")
    return _convert_box(_RawBox.model_validate(values[0]), original_size)


def _convert_box(item: _RawBox, original_size: tuple[int, int]) -> RegionBox:
    y0, x0, y1, x1 = item.box_2d
    width, height = original_size
    side = max(width, height)
    pad_x = (side - width) / 2
    pad_y = (side - height) / 2
    x_min = ((float(x0) / 1000 * side) - pad_x) / width
    x_max = ((float(x1) / 1000 * side) - pad_x) / width
    y_min = ((float(y0) / 1000 * side) - pad_y) / height
    y_max = ((float(y1) / 1000 * side) - pad_y) / height
    x_min, y_min, x_max, y_max = [
        max(0.0, min(value, 1.0)) for value in (x_min, y_min, x_max, y_max)
    ]
    if x_max <= x_min or y_max <= y_min:
        raise ValueError(f"Invalid MedGemma box for {item.label}: {item.box_2d}")
    return RegionBox(
        label=item.label,
        x1=x_min,
        y1=y_min,
        x2=x_max,
        y2=y_max,
        source="medgemma-localizer",
    )


def _pad_to_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    side = max(width, height)
    padded = Image.new("RGB", (side, side))
    padded.paste(image.convert("RGB"), ((side - width) // 2, (side - height) // 2))
    return padded


def _model_run(
    started: float,
    model_id: str,
    model_revision: str,
    status: str,
    detail: str = "",
) -> ModelRun:
    return ModelRun(
        role="vision",
        model_id=model_id,
        model_source=LOCALIZER_REPO,
        model_revision=model_revision,
        runtime_revision=LLAMA_CPP_BUILD,
        status=status,
        latency_ms=int((time.perf_counter() - started) * 1000),
        detail=detail,
    )

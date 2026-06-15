from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class Anatomy(StrEnum):
    CHEST = "chest"
    OTHER = "other"


class ImageQuality(BaseModel):
    width: int
    height: int
    grayscale: bool
    contrast: float
    brightness: float
    notes: list[str] = Field(default_factory=list)


class FindingScore(BaseModel):
    label: str
    score: float = Field(ge=0.0, le=1.0)
    source: str
    explanation: str


class VisionObservation(BaseModel):
    label: str
    description: str
    source: str


class RegionBox(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    image_id: str | None = None
    label: str
    x1: float = Field(ge=0.0, le=1.0)
    y1: float = Field(ge=0.0, le=1.0)
    x2: float = Field(ge=0.0, le=1.0)
    y2: float = Field(ge=0.0, le=1.0)
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    source: str


class ModelRun(BaseModel):
    role: str
    model_id: str
    model_source: str = ""
    model_revision: str = ""
    runtime: str = "llama.cpp"
    runtime_revision: str = ""
    status: Literal["ok", "error"]
    latency_ms: int = Field(ge=0)
    time_to_first_token_ms: int | None = Field(default=None, ge=0)
    tokens_per_second: float | None = Field(default=None, ge=0)
    detail: str = ""


class EvidenceBundle(BaseModel):
    anatomy: Anatomy
    quality: ImageQuality
    findings: list[FindingScore]
    observations: list[VisionObservation] = Field(default_factory=list)
    regions: list[RegionBox] = Field(default_factory=list)
    agreement_notes: list[str] = Field(default_factory=list)
    model_notes: list[str] = Field(default_factory=list)
    model_runs: list[ModelRun] = Field(default_factory=list)

    def top_findings(self, limit: int = 5) -> list[FindingScore]:
        return sorted(self.findings, key=lambda item: item.score, reverse=True)[:limit]


class StudentRead(BaseModel):
    observation: str
    question: str = ""


class DicomMetadata(BaseModel):
    modality: str = ""
    sop_class_uid: str = ""
    patient_name: str = ""
    patient_id: str = ""
    patient_age: str = ""
    patient_sex: str = ""
    study_date: str = ""
    accession_number: str = ""
    institution_name: str = ""
    body_part_examined: str = ""
    view_position: str = ""
    laterality: str = ""
    rows: int = 0
    columns: int = 0
    bits_stored: int = 0
    photometric_interpretation: str = ""
    window_center: float | None = None
    window_width: float | None = None
    pixel_spacing_mm: tuple[float, float] | None = None
    frame_number: int | None = None


class WindowPreset(BaseModel):
    id: str
    label: str
    center: float
    width: float = Field(gt=0)


class StudyImage(BaseModel):
    id: str
    label: str
    projection: str
    instance_number: int = 0
    width: int
    height: int
    image_url: str
    metadata: DicomMetadata
    window_presets: list[WindowPreset] = Field(default_factory=list)


class Study(BaseModel):
    id: str
    title: str
    source: Literal["demo", "upload"]
    case_id: str | None = None
    images: list[StudyImage]
    primary_image_id: str
    reference: ReferenceAnswer | None = None


class TutorResponse(BaseModel):
    student_read_assessment: str
    model_evidence: list[str]
    professor_assessment: str
    reading_approach: list[str]
    uncertainty: list[str]
    quiz: list[str]
    provider: str
    model_run: ModelRun | None = None


class ReadScorecard(BaseModel):
    total_score: int = Field(ge=0, le=100)
    coverage_score: int = Field(ge=0, le=100)
    technique_score: int = Field(ge=0, le=100)
    uncertainty_score: int = Field(ge=0, le=100)
    matched_findings: list[str] = Field(default_factory=list)
    missed_findings: list[str] = Field(default_factory=list)
    technique_hits: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
    practice_focus: str


class ReferenceAnswer(BaseModel):
    labels: list[str]
    urgency: str
    teaching_point: str
    source: str


class DemoCase(BaseModel):
    id: str
    title: str
    difficulty: Literal["foundation", "intermediate", "advanced"]
    image_path: str
    reference: ReferenceAnswer


class AnalysisResult(BaseModel):
    session_id: str | None = None
    case_id: str | None = None
    primary_image_id: str | None = None
    image_data_url: str
    evidence: EvidenceBundle
    scorecard: ReadScorecard
    tutor: TutorResponse
    reference: ReferenceAnswer | None = None
    session_note: str


class ChatMessage(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    model_run: ModelRun | None = None
    evidence_sources: list[str] = Field(default_factory=list)
    region_ids: list[str] = Field(default_factory=list)


class AnalysisSession(BaseModel):
    id: str
    status: Literal["ready", "analyzing", "complete", "error"]
    study: Study
    blind_read: str = ""
    result: AnalysisResult | None = None
    messages: list[ChatMessage] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


OverlayMode = Literal["none", "regions"]

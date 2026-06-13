from __future__ import annotations

from enum import StrEnum
from typing import Literal

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


class RegionBox(BaseModel):
    label: str
    x1: float = Field(ge=0.0, le=1.0)
    y1: float = Field(ge=0.0, le=1.0)
    x2: float = Field(ge=0.0, le=1.0)
    y2: float = Field(ge=0.0, le=1.0)
    score: float = Field(ge=0.0, le=1.0)
    source: str


class EvidenceBundle(BaseModel):
    anatomy: Anatomy
    quality: ImageQuality
    findings: list[FindingScore]
    regions: list[RegionBox] = Field(default_factory=list)
    agreement_notes: list[str] = Field(default_factory=list)
    model_notes: list[str] = Field(default_factory=list)

    def top_findings(self, limit: int = 5) -> list[FindingScore]:
        return sorted(self.findings, key=lambda item: item.score, reverse=True)[:limit]


class StudentRead(BaseModel):
    observation: str
    question: str = ""


class TutorResponse(BaseModel):
    summary: str
    feedback: list[str]
    suggested_checks: list[str]
    uncertainty: list[str]
    quiz: list[str]
    provider: str


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


OverlayMode = Literal["none", "regions"]

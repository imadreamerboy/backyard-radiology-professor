from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Iterator

from PIL import Image
from pydantic import BaseModel, Field

from radiology_trainer.adapters.llama_client import LlamaCppClient
from radiology_trainer.config import AppConfig
from radiology_trainer.domain import (
    ChatMessage,
    EvidenceBundle,
    ModelRun,
    ReferenceAnswer,
    StudentRead,
    TutorResponse,
)
from radiology_trainer.runtime_manifest import LLAMA_CPP_BUILD, PROFESSOR_REPO
from radiology_trainer.structured_output import parse_json_model


PROFESSOR_SYSTEM_PROMPT = """You are Professor MedGemma, an experienced thoracic
radiologist and precise, constructive educator. This is educational practice only.

Ground every answer in the supplied radiographs, the trainee's blind interpretation,
study metadata, independently attributed X-Raydar signals, MedGemma localization, and
the public reference labels when supplied.

For case review, reason in this order:
1. What the student said: accurately summarize strengths, omissions, and unsupported claims.
2. What the models suggest: attribute X-Raydar and MedGemma separately; scores are signals,
   not diagnoses.
3. Professor assessment: inspect the image yourself, reconcile conflicts, and state what the
   finding most likely is or could be. If a public reference is supplied, treat it as the
   educational answer and explain how the image supports it.
4. How to read it: give a short reusable search method for recognizing or excluding the finding.

For follow-up questions, answer the question first, then use the same four headings when they
add value. Separate observation, interpretation, differential, and uncertainty. Never invent
findings or claim certainty unsupported by the image. Do not give patient-specific treatment
or management instructions."""


INITIAL_REVIEW_PROMPT = """Return one JSON object with exactly these keys:
student_read_assessment, model_evidence, professor_assessment, reading_approach,
uncertainty, quiz.
model_evidence, reading_approach, uncertainty, and quiz must be arrays of short strings.
Use the public reference as the educational answer when present. Otherwise independently
inspect the image and reconcile it with the attributed model evidence. The reading approach
must be practical and reusable. Include one to three targeted quiz questions. Return JSON only."""

REPAIR_REVIEW_PROMPT = """Convert the previous professor review into one JSON object
with exactly these keys: student_read_assessment, model_evidence,
professor_assessment, reading_approach, uncertainty, quiz.

model_evidence, reading_approach, uncertainty, and quiz must be arrays of short
strings. If the previous answer omitted a field, infer it from the supplied case
context. Return JSON only."""

CHAT_REPLY_PROMPT = """Answer in concise Markdown. Use these headings when useful:
### What you said
### What the models suggest
### Professor assessment
### How to read it
### Uncertainty

Prefer short bullets under each heading. If the trainee asks for a quiz, include
### Quiz with one to three questions. Do not output JSON."""


class _TutorPayload(BaseModel):
    student_read_assessment: str
    model_evidence: list[str] = Field(min_length=1, max_length=6)
    professor_assessment: str
    reading_approach: list[str] = Field(min_length=1, max_length=6)
    uncertainty: list[str] = Field(min_length=1, max_length=4)
    quiz: list[str] = Field(min_length=1, max_length=3)


@dataclass
class MedGemmaProfessorModel:
    config: AppConfig
    client: LlamaCppClient
    name: str = "medgemma-professor"

    def coach(
        self,
        evidence: EvidenceBundle,
        student_read: StudentRead,
        *,
        images: list[Image.Image] | None = None,
        study_context: dict[str, Any] | None = None,
        reference: ReferenceAnswer | None = None,
    ) -> TutorResponse:
        started = time.perf_counter()
        try:
            content: list[dict[str, Any]] = [
                {"type": "text", "text": INITIAL_REVIEW_PROMPT},
                {
                    "type": "text",
                    "text": _context_json(
                        evidence,
                        student_read,
                        reference,
                        study_context=study_context,
                    ),
                },
            ]
            for image in (images or [])[:2]:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": self.client.image_url(image)},
                    }
                )
            messages = [
                {"role": "system", "content": PROFESSOR_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ]
            payload = _structured_review(
                self.client,
                model=self.config.professor_model,
                messages=messages,
            )
        except Exception as exc:
            return TutorResponse(
                student_read_assessment="The committed read could not be reviewed.",
                model_evidence=[f"Model error: {exc}"],
                professor_assessment="MedGemma professor output is unavailable.",
                reading_approach=["Review the independently attributed evidence manually."],
                uncertainty=["No generated professor guidance was accepted for this run."],
                quiz=[],
                provider=self.name,
                model_run=_run(
                    started,
                    self.config.professor_model,
                    self.config.professor_revision,
                    "error",
                    str(exc),
                ),
            )

        return TutorResponse(
            **payload.model_dump(),
            provider=self.name,
            model_run=_run(
                started,
                self.config.professor_model,
                self.config.professor_revision,
                "ok",
            ),
        )

    def stream_reply(
        self,
        *,
        evidence: EvidenceBundle,
        blind_read: str,
        question: str,
        history: list[ChatMessage],
        images: list[Image.Image],
        study_context: dict[str, Any] | None,
        reference: ReferenceAnswer | None,
    ) -> Iterator[tuple[str, dict[str, Any]]]:
        context = StudentRead(observation=blind_read, question=question)
        first_content: list[dict[str, Any]] = [
            {"type": "text", "text": CHAT_REPLY_PROMPT},
            {
                "type": "text",
                "text": _context_json(
                    evidence,
                    context,
                    reference,
                    study_context=study_context,
                ),
            }
        ]
        for image in images[:2]:
            first_content.append(
                {"type": "image_url", "image_url": {"url": self.client.image_url(image)}}
            )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": PROFESSOR_SYSTEM_PROMPT},
            {"role": "user", "content": first_content},
        ]
        for message in history[-self.config.chat_history_messages :]:
            messages.append({"role": message.role, "content": message.content})
        messages.append({"role": "user", "content": question})
        yield from self.client.stream_chat(
            model=self.config.professor_model,
            messages=messages,
            max_tokens=1200,
            temperature=0.2,
        )


def _context_json(
    evidence: EvidenceBundle,
    student_read: StudentRead,
    reference: ReferenceAnswer | None,
    *,
    study_context: dict[str, Any] | None = None,
) -> str:
    return json.dumps(
        {
            "student_blind_read": student_read.observation,
            "student_question": student_read.question,
            "study": study_context,
            "image_quality": evidence.quality.model_dump(),
            "xraydar_findings": [
                item.model_dump() for item in evidence.top_findings(8)
            ],
            "medgemma_observations": [
                item.model_dump() for item in evidence.observations
            ],
            "suggested_regions": [item.model_dump() for item in evidence.regions],
            "agreement_and_limitations": evidence.agreement_notes,
            "reference": reference.model_dump() if reference else None,
        },
        ensure_ascii=True,
    )


def _structured_review(
    client: LlamaCppClient,
    *,
    model: str,
    messages: list[dict[str, Any]],
) -> _TutorPayload:
    text = client.chat(
        model=model,
        messages=messages,
        max_tokens=900,
        temperature=0.1,
        json_schema=_TutorPayload.model_json_schema(),
    )
    try:
        return parse_json_model(text, _TutorPayload)
    except Exception:
        repair_messages = [
            *messages,
            {"role": "assistant", "content": text},
            {"role": "user", "content": REPAIR_REVIEW_PROMPT},
        ]
        repaired = client.chat(
            model=model,
            messages=repair_messages,
            max_tokens=700,
            temperature=0.0,
            json_schema=_TutorPayload.model_json_schema(),
        )
        return parse_json_model(repaired, _TutorPayload)


def _run(
    started: float,
    model_id: str,
    model_revision: str,
    status: str,
    detail: str = "",
) -> ModelRun:
    return ModelRun(
        role="professor",
        model_id=model_id,
        model_source=PROFESSOR_REPO,
        model_revision=model_revision,
        runtime_revision=LLAMA_CPP_BUILD,
        status=status,
        latency_ms=int((time.perf_counter() - started) * 1000),
        detail=detail,
    )

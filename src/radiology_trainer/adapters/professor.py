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
radiologist and demanding but constructive educator. This is educational practice only.

Ground every answer in the supplied radiographs, the trainee's blind interpretation,
the independently attributed X-Raydar signals, and MedGemma localization evidence.
Separate direct observations from interpretation, differential diagnosis, uncertainty,
and teaching points. Never invent image findings or claim certainty unsupported by the
evidence. Do not give patient-specific treatment or clinical management instructions.
Answer the trainee's actual question directly, then explain the reasoning concisely."""


INITIAL_REVIEW_PROMPT = """Return one JSON object with exactly these keys:
summary, feedback, suggested_checks, uncertainty, quiz.
feedback, suggested_checks, uncertainty, and quiz must be arrays of short strings.
Act as a senior radiology professor reviewing a committed blind read. The quiz should
contain one to three questions targeted to this case. Return JSON only."""


class _TutorPayload(BaseModel):
    summary: str
    feedback: list[str] = Field(min_length=1, max_length=6)
    suggested_checks: list[str] = Field(min_length=1, max_length=6)
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
        reference: ReferenceAnswer | None = None,
    ) -> TutorResponse:
        started = time.perf_counter()
        try:
            content: list[dict[str, Any]] = [
                {"type": "text", "text": INITIAL_REVIEW_PROMPT},
                {"type": "text", "text": _context_json(evidence, student_read, reference)},
            ]
            for image in (images or [])[:2]:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": self.client.image_url(image)},
                    }
                )
            text = self.client.chat(
                model=self.config.professor_model,
                messages=[
                    {"role": "system", "content": PROFESSOR_SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                max_tokens=900,
                temperature=0.1,
                json_schema=_TutorPayload.model_json_schema(),
            )
            payload = parse_json_model(text, _TutorPayload)
        except Exception as exc:
            return TutorResponse(
                summary="MedGemma professor output is unavailable.",
                feedback=[f"Model error: {exc}"],
                suggested_checks=["Review the independently attributed evidence manually."],
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
        reference: ReferenceAnswer | None,
    ) -> Iterator[tuple[str, dict[str, Any]]]:
        context = StudentRead(observation=blind_read, question=question)
        first_content: list[dict[str, Any]] = [
            {"type": "text", "text": _context_json(evidence, context, reference)}
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
) -> str:
    return json.dumps(
        {
            "student_blind_read": student_read.observation,
            "student_question": student_read.question,
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

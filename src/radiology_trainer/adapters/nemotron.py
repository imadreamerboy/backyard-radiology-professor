from __future__ import annotations

from dataclasses import dataclass

import requests
from huggingface_hub import InferenceClient

from radiology_trainer.config import AppConfig
from radiology_trainer.domain import EvidenceBundle, StudentRead, TutorResponse


SYSTEM_PROMPT = """You are an educational radiology tutor for chest X-ray practice.
Use the supplied structured evidence and the student's blind read.
Do not claim clinical diagnosis.
Do not invent findings not supported by the evidence.
Be concise, uncertainty-aware, and focused on teaching.
Return plain text with these sections:
Summary
Feedback
Suggested checks
Uncertainty
Quiz
"""


@dataclass
class NemotronTutorModel:
    config: AppConfig
    name: str = "nemotron-tutor"

    def coach(self, evidence: EvidenceBundle, student_read: StudentRead) -> TutorResponse:
        prompt = _build_prompt(evidence, student_read)

        try:
            if self.config.tutor_provider == "hf":
                text = self._call_huggingface(prompt)
            elif self.config.tutor_provider == "openai":
                text = self._call_openai_compatible(prompt)
            else:
                raise ValueError(f"Unsupported tutor provider: {self.config.tutor_provider}")
        except Exception as exc:
            return TutorResponse(
                summary="Nemotron call failed; showing grounded fallback feedback.",
                feedback=[f"Provider error: {exc}"],
                suggested_checks=[
                    "Verify HF_TOKEN or local endpoint configuration.",
                    "Review structured evidence manually before relying on tutor text.",
                ],
                uncertainty=[
                    "No live Nemotron response was used for this output.",
                ],
                quiz=[
                    "Which evidence item has the highest score?",
                    "Which region should be checked manually first?",
                    "What could make this model evidence misleading?",
                ],
                provider=f"{self.name}:fallback",
            )

        return _parse_text_response(text, provider=f"{self.name}:{self.config.tutor_provider}")

    def _call_huggingface(self, prompt: str) -> str:
        client = InferenceClient(
            model=self.config.nemotron_model,
            provider=self.config.hf_provider,
            token=self.config.hf_token,
        )
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=700,
            temperature=0.2,
        )
        return response.choices[0].message.content or ""

    def _call_openai_compatible(self, prompt: str) -> str:
        if not self.config.openai_base_url:
            raise ValueError("RAD_TRAINER_OPENAI_BASE_URL is required for openai provider")
        url = self.config.openai_base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.config.openai_api_key:
            headers["Authorization"] = f"Bearer {self.config.openai_api_key}"
        payload = {
            "model": self.config.nemotron_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 700,
        }
        response = requests.post(url, headers=headers, json=payload, timeout=90)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]


def _build_prompt(evidence: EvidenceBundle, student_read: StudentRead) -> str:
    top = [
        {
            "label": finding.label,
            "score": finding.score,
            "source": finding.source,
            "explanation": finding.explanation,
        }
        for finding in evidence.top_findings(5)
    ]
    regions = [
        {
            "label": region.label,
            "box": [region.x1, region.y1, region.x2, region.y2],
            "score": region.score,
            "source": region.source,
        }
        for region in evidence.regions[:5]
    ]
    return (
        "Student blind read:\n"
        f"{student_read.observation or '[empty]'}\n\n"
        "Student question:\n"
        f"{student_read.question or '[none]'}\n\n"
        "Structured evidence:\n"
        f"anatomy={evidence.anatomy}\n"
        f"quality={evidence.quality.model_dump()}\n"
        f"top_findings={top}\n"
        f"regions={regions}\n"
        f"agreement_notes={evidence.agreement_notes}\n"
        f"model_notes={evidence.model_notes}\n"
    )


def _parse_text_response(text: str, provider: str) -> TutorResponse:
    cleaned = text.strip() or "No text returned from tutor provider."
    lines = [line.strip("- ").strip() for line in cleaned.splitlines() if line.strip()]
    feedback = lines[:4] if lines else [cleaned]
    quiz = [line for line in lines if "?" in line][-3:]
    if not quiz:
        quiz = [
            "What image feature most supports the top finding?",
            "What alternative explanation should be considered?",
            "What should be checked manually before accepting this interpretation?",
        ]
    return TutorResponse(
        summary=cleaned.splitlines()[0][:500],
        feedback=feedback,
        suggested_checks=[
            "Check whether the tutor response stayed grounded in the evidence.",
            "Compare against the original image before accepting any finding.",
        ],
        uncertainty=[
            "Tutor text is educational and may be wrong.",
            "Model outputs are not validated for clinical decisions.",
        ],
        quiz=quiz[:3],
        provider=provider,
    )

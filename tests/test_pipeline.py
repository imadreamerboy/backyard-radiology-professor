from __future__ import annotations

from PIL import Image

from radiology_trainer.config import AppConfig
from radiology_trainer.domain import Anatomy, StudentRead
from radiology_trainer.pipeline import build_pipeline


def test_demo_pipeline_returns_structured_evidence() -> None:
    image = Image.new("L", (768, 768), color=120)
    pipeline = build_pipeline(AppConfig())

    evidence, tutor = pipeline.analyze(
        image=image,
        student_read=StudentRead(observation="PA chest radiograph. No focal opacity."),
    )

    assert evidence.anatomy == Anatomy.CHEST
    assert evidence.findings
    assert evidence.top_findings(3)[0].score >= evidence.top_findings(3)[-1].score
    assert evidence.regions
    assert tutor.quiz
    assert "educational" in tutor.summary.lower()


from __future__ import annotations

from PIL import Image

from radiology_trainer.config import AppConfig
from radiology_trainer.domain import Anatomy, StudentRead
from radiology_trainer.examples import example_cases
from radiology_trainer.image_io import load_xray_image
from radiology_trainer.learning import build_scorecard
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


def test_load_xray_image_from_png(tmp_path) -> None:
    path = tmp_path / "case.png"
    Image.new("L", (640, 640), color=100).save(path)

    image = load_xray_image(path)

    assert image.mode == "RGB"
    assert image.size == (640, 640)


def test_pipeline_uses_hybrid_evidence_when_medical_vlm_enabled() -> None:
    config = AppConfig(enable_medical_vlm=True)
    pipeline = build_pipeline(config)

    assert pipeline.evidence_model.name == "hybrid-chest-evidence"


def test_pipeline_uses_http_evidence_when_url_is_configured() -> None:
    config = AppConfig(chest_evidence_url="http://localhost:9000/analyze")
    pipeline = build_pipeline(config)

    assert pipeline.evidence_model.name == "http-chest-evidence"


def test_scorecard_rewards_structured_blind_read() -> None:
    image = Image.new("L", (768, 768), color=120)
    pipeline = build_pipeline(AppConfig())
    initial_evidence, _ = pipeline.analyze(
        image=image,
        student_read=StudentRead(observation=""),
    )
    top_label = initial_evidence.top_findings(1)[0].label
    student_read = StudentRead(
        observation=(
            f"PA chest radiograph. Image quality adequate. Possible {top_label}. "
            "No pneumothorax."
        )
    )
    evidence, _ = pipeline.analyze(image=image, student_read=student_read)

    scorecard = build_scorecard(evidence, student_read)

    assert scorecard.total_score > 50
    assert "projection/view" in scorecard.technique_hits
    assert scorecard.next_steps


def test_example_cases_exist_and_load() -> None:
    for path, _, _ in example_cases():
        image = load_xray_image(path)
        assert image.size == (900, 900)

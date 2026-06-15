from __future__ import annotations

from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from radiology_trainer.cases import demo_cases
from radiology_trainer.config import AppConfig
from radiology_trainer.adapters.professor import _context_json
from radiology_trainer.domain import Anatomy, StudentRead
from radiology_trainer.image_io import load_xray_image
from radiology_trainer.learning import build_scorecard
from radiology_trainer.pipeline import build_pipeline
from radiology_trainer.reporting import format_session_note
from radiology_trainer.ui import create_server


def test_demo_pipeline_returns_structured_evidence() -> None:
    image = Image.new("L", (768, 768), color=120)
    pipeline = build_pipeline(AppConfig())
    evidence, tutor = pipeline.analyze(
        image=image,
        student_read=StudentRead(observation="PA chest radiograph. No focal opacity."),
    )
    assert evidence.anatomy == Anatomy.CHEST
    assert evidence.findings
    assert evidence.regions
    assert tutor.quiz


def test_load_xray_image_from_png(tmp_path) -> None:
    path = tmp_path / "case.png"
    Image.new("L", (640, 640), color=100).save(path)
    image = load_xray_image(path)
    assert image.mode == "RGB"
    assert image.size == (640, 640)


def test_local_pipeline_uses_xraydar_and_medgemma() -> None:
    config = AppConfig(model_mode="local", tutor_provider="llama", enable_medical_vlm=True)
    pipeline = build_pipeline(config)
    assert pipeline.evidence_model.name == "xraydar-medgemma-evidence"
    assert pipeline.tutor_model.name == "medgemma-professor"


def test_scorecard_rewards_structured_blind_read() -> None:
    image = Image.new("L", (768, 768), color=120)
    pipeline = build_pipeline(AppConfig())
    evidence, _ = pipeline.analyze(image=image, student_read=StudentRead(observation=""))
    top_label = evidence.top_findings(1)[0].label
    student_read = StudentRead(
        observation=f"PA chest radiograph. Image quality adequate. Possible {top_label}. No pneumothorax."
    )
    scorecard = build_scorecard(evidence, student_read)
    assert scorecard.total_score > 50
    assert "projection/view" in scorecard.technique_hits


def test_real_case_catalog_has_balanced_three_case_set() -> None:
    cases = demo_cases()
    assert [case.id for case in cases] == ["normal", "scoliosis", "cardiomediastinal"]
    assert cases[0].reference.labels == []
    assert "scoliosis" in cases[1].reference.labels
    assert len(cases[2].reference.labels) == 3


def test_session_note_contains_training_artifact() -> None:
    image = Image.new("L", (768, 768), color=120)
    student_read = StudentRead(observation="PA chest. No pneumothorax.", question="What next?")
    evidence, tutor = build_pipeline(AppConfig()).analyze(image=image, student_read=student_read)
    scorecard = build_scorecard(evidence, student_read)
    note = format_session_note(evidence, scorecard, student_read, tutor)
    assert "Session note" in note
    assert "Educational practice only" in note


def test_server_requires_blind_read_and_returns_real_contract() -> None:
    client = TestClient(create_server(AppConfig()))
    status = client.get("/api/status").json()
    assert status["runtime_revision"]
    assert len(status["model_revisions"]) == 2
    assert status["xraydar"]["weights_revision"]

    buffer = BytesIO()
    Image.new("L", (320, 320), color=110).save(buffer, format="PNG")
    response = client.post(
        "/api/analyze",
        data={"observation": "PA chest. No focal opacity.", "question": ""},
        files={"image": ("case.png", buffer.getvalue(), "image/png")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["image_data_url"].startswith("data:image/png;base64,")
    assert payload["evidence"]["findings"]
    assert payload["scorecard"]["total_score"] >= 0

    rejected = client.post(
        "/api/analyze",
        data={"observation": "", "question": ""},
        files={"image": ("case.png", buffer.getvalue(), "image/png")},
    )
    assert rejected.status_code == 422


def test_session_api_streams_analysis_and_chat() -> None:
    client = TestClient(create_server(AppConfig()))
    buffer = BytesIO()
    Image.new("L", (320, 320), color=110).save(buffer, format="PNG")
    created = client.post(
        "/api/sessions",
        files={"files": ("case.png", buffer.getvalue(), "image/png")},
    )
    assert created.status_code == 200
    session = created.json()
    assert session["study"]["images"]

    image = session["study"]["images"][0]
    rendered = client.get(image["image_url"])
    assert rendered.status_code == 200
    assert rendered.headers["content-type"] == "image/png"

    analysis = client.post(
        f"/api/sessions/{session['id']}/analyze",
        json={"observation": "PA chest radiograph. No focal opacity."},
    )
    assert analysis.status_code == 200
    assert "event: complete" in analysis.text

    chat = client.post(
        f"/api/sessions/{session['id']}/chat",
        json={"message": "What should I inspect next?"},
    )
    assert chat.status_code == 200
    assert "event: delta" in chat.text
    assert "event: complete" in chat.text


def test_professor_context_includes_study_metadata() -> None:
    image = Image.new("L", (320, 320), color=110)
    evidence, _ = build_pipeline(AppConfig()).analyze(
        image=image,
        student_read=StudentRead(observation="PA chest. No focal opacity."),
    )
    context = _context_json(
        evidence,
        StudentRead(observation="PA chest. No focal opacity."),
        None,
        study_context={
            "title": "Derived PA chest",
            "primary_image_id": "primary",
            "images": [
                {
                    "id": "primary",
                    "projection": "PA",
                    "metadata": {"modality": "DX", "pixel_spacing_mm": [0.14, 0.14]},
                }
            ],
        },
    )
    assert '"study"' in context
    assert '"modality": "DX"' in context
    assert '"xraydar_findings"' in context

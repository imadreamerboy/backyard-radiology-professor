from __future__ import annotations

import gradio as gr

from radiology_trainer.config import AppConfig
from radiology_trainer.domain import StudentRead
from radiology_trainer.examples import example_case_label, example_cases
from radiology_trainer.image_io import load_xray_image
from radiology_trainer.learning import build_scorecard, format_scorecard
from radiology_trainer.overlays import draw_regions
from radiology_trainer.pipeline import build_pipeline
from radiology_trainer.reporting import format_session_note


APP_CSS = """
:root {
  --rt-accent: #3b82f6;
  --rt-ink: #0f172a;
  --rt-muted: #64748b;
}
.gradio-container {
  max-width: 1440px !important;
}
#app-title h1 {
  font-size: 32px;
  letter-spacing: 0;
  margin-bottom: 4px;
}
#app-title p {
  color: var(--rt-muted);
  margin-top: 0;
}
.status-pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  border: 1px solid #d8dee8;
  border-radius: 999px;
  padding: 4px 10px;
  color: #334155;
  background: #f8fafc;
  font-size: 13px;
}
.gradio-container button.primary,
.gradio-container button.primary:hover,
.gradio-container .primary > button {
  background: var(--rt-accent) !important;
  border-color: var(--rt-accent) !important;
  color: #ffffff !important;
}
.gradio-container button.primary:hover,
.gradio-container .primary > button:hover {
  background: #2563eb !important;
}
"""


def create_theme() -> gr.Theme:
    return gr.themes.Soft(
        primary_hue="blue",
        neutral_hue="slate",
        radius_size="sm",
    )


def create_app(config: AppConfig | None = None) -> gr.Blocks:
    cfg = config or AppConfig.from_env()
    cases = example_cases()

    with gr.Blocks(title="Backyard Radiology Trainer") as demo:
        gr.Markdown(
            """
# Backyard Radiology Trainer
Educational chest X-ray practice: blind read first, evidence second, tutor last.
""",
            elem_id="app-title",
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                file_input = gr.File(
                    label="X-ray image or DICOM",
                    file_types=["image", ".dcm", ".dicom"],
                    type="filepath",
                )
                overlay_output = gr.Image(label="Preview and evidence overlay", type="pil", height=560)

            with gr.Column(scale=4):
                blind_read = gr.Textbox(
                    label="Your blind read",
                    placeholder="Example: PA chest radiograph. Cardiomediastinal silhouette...",
                    lines=8,
                )
                question = gr.Textbox(
                    label="Question for the tutor",
                    placeholder="What should I double-check before calling this normal?",
                    lines=3,
                )
                run_button = gr.Button("Analyze", variant="primary")
                gr.HTML(
                    f"""
<span class="status-pill">mode: {cfg.model_mode}</span>
<span class="status-pill">tutor: {cfg.tutor_provider}</span>
<span class="status-pill">hf provider: {cfg.hf_provider or "auto"}</span>
<span class="status-pill">evidence: {"external" if cfg.chest_evidence_url else "demo"}</span>
<span class="status-pill">medical vlm: {"on" if cfg.enable_medical_vlm else "off"}</span>
<span class="status-pill">nemotron: {cfg.nemotron_model}</span>
"""
                )
                gr.Markdown("**Synthetic demo cases**")
                with gr.Row():
                    for idx, case in enumerate(cases):
                        gr.Button(example_case_label(case[0]), size="sm").click(
                            fn=lambda case_index=idx: _load_example_case(case_index),
                            inputs=None,
                            outputs=[file_input, blind_read, question],
                        )

        with gr.Row():
            evidence_table = gr.Dataframe(
                headers=["Finding", "Score", "Source", "Note"],
                label="Structured evidence",
                interactive=False,
                wrap=True,
            )
            quality_output = gr.JSON(label="Image quality and routing")

        with gr.Row():
            scorecard_output = gr.Markdown(label="Blind-read scorecard")
            tutor_summary = gr.Markdown(label="Tutor summary")
            tutor_quiz = gr.Markdown(label="Quiz")

        with gr.Accordion("Copy-ready session note", open=False):
            session_note = gr.Markdown()

        model_notes = gr.Markdown(label="Model notes")

        run_button.click(
            fn=_run_analysis,
            inputs=[file_input, blind_read, question],
            outputs=[
                overlay_output,
                evidence_table,
                quality_output,
                scorecard_output,
                tutor_summary,
                tutor_quiz,
                session_note,
                model_notes,
            ],
        )

    return demo


def _load_example_case(case_index: int) -> tuple[str, str, str]:
    file_path, blind_read, question = example_cases()[case_index]
    return file_path, blind_read, question


def _run_analysis(file_path: str | None, blind_read: str, question: str):
    if not file_path:
        raise gr.Error("Upload an X-ray image or DICOM first.")

    try:
        image = load_xray_image(file_path)
    except Exception as exc:
        raise gr.Error(str(exc)) from exc

    pipeline = build_pipeline(AppConfig.from_env())
    evidence, tutor = pipeline.analyze(
        image=image,
        student_read=StudentRead(observation=blind_read or "", question=question or ""),
    )
    scorecard = build_scorecard(
        evidence=evidence,
        student_read=StudentRead(observation=blind_read or "", question=question or ""),
    )

    overlay = draw_regions(image, evidence.regions) if evidence.regions else image
    rows = [
        [finding.label, round(finding.score, 3), finding.source, finding.explanation]
        for finding in evidence.top_findings(7)
    ]
    quality = {
        "anatomy": evidence.anatomy.value,
        "quality": evidence.quality.model_dump(),
        "agreement_notes": evidence.agreement_notes,
    }
    scorecard_md = format_scorecard(scorecard)
    tutor_md = _format_tutor(tutor)
    quiz_md = "\n".join(f"{idx + 1}. {item}" for idx, item in enumerate(tutor.quiz))
    session_note_md = format_session_note(
        evidence=evidence,
        scorecard=scorecard,
        student_read=StudentRead(observation=blind_read or "", question=question or ""),
        tutor=tutor,
    )
    notes_md = "\n".join(f"- {note}" for note in evidence.model_notes)

    return overlay, rows, quality, scorecard_md, tutor_md, quiz_md, session_note_md, notes_md


def _format_tutor(tutor) -> str:
    feedback = "\n".join(f"- {item}" for item in tutor.feedback)
    checks = "\n".join(f"- {item}" for item in tutor.suggested_checks)
    uncertainty = "\n".join(f"- {item}" for item in tutor.uncertainty)
    return f"""
**Provider:** `{tutor.provider}`

**Summary**

{tutor.summary}

**Feedback**

{feedback}

**Suggested checks**

{checks}

**Uncertainty**

{uncertainty}
"""

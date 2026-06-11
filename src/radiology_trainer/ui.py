from __future__ import annotations

import gradio as gr
from PIL import Image

from radiology_trainer.config import AppConfig
from radiology_trainer.domain import StudentRead
from radiology_trainer.overlays import draw_regions
from radiology_trainer.pipeline import build_pipeline


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
"""


def create_app(config: AppConfig | None = None) -> gr.Blocks:
    cfg = config or AppConfig.from_env()

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
                image_input = gr.Image(
                    label="X-ray",
                    type="pil",
                    height=560,
                    image_mode="RGB",
                )
                overlay_output = gr.Image(label="Evidence overlay", type="pil", height=560)

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
<span class="status-pill">nemotron: {cfg.nemotron_model}</span>
"""
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
            tutor_summary = gr.Markdown(label="Tutor summary")
            tutor_quiz = gr.Markdown(label="Quiz")

        model_notes = gr.Markdown(label="Model notes")

        run_button.click(
            fn=_run_analysis,
            inputs=[image_input, blind_read, question],
            outputs=[
                overlay_output,
                evidence_table,
                quality_output,
                tutor_summary,
                tutor_quiz,
                model_notes,
            ],
        )

    return demo


def _run_analysis(image: Image.Image | None, blind_read: str, question: str):
    if image is None:
        raise gr.Error("Upload an X-ray image first.")

    pipeline = build_pipeline(AppConfig.from_env())
    evidence, tutor = pipeline.analyze(
        image=image,
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
    tutor_md = _format_tutor(tutor)
    quiz_md = "\n".join(f"{idx + 1}. {item}" for idx, item in enumerate(tutor.quiz))
    notes_md = "\n".join(f"- {note}" for note in evidence.model_notes)

    return overlay, rows, quality, tutor_md, quiz_md, notes_md


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

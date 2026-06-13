from __future__ import annotations

from radiology_trainer.domain import EvidenceBundle, ReadScorecard, StudentRead, TutorResponse


def format_session_note(
    evidence: EvidenceBundle,
    scorecard: ReadScorecard,
    student_read: StudentRead,
    tutor: TutorResponse,
) -> str:
    top_findings = "\n".join(
        f"- {finding.label}: {finding.score:.2f} ({finding.source})"
        for finding in evidence.top_findings(5)
    )
    next_steps = "\n".join(f"- {step}" for step in scorecard.next_steps)
    quiz = "\n".join(f"{idx + 1}. {item}" for idx, item in enumerate(tutor.quiz))

    return f"""
## Session note

**Student blind read**

{student_read.observation or "No blind read provided."}

**Question**

{student_read.question or "No question provided."}

**Score**

{scorecard.total_score}/100 - {scorecard.practice_focus}

**Top evidence signals**

{top_findings or "- No evidence signals available."}

**Next practice steps**

{next_steps}

**Quiz**

{quiz}

**Safety note**

Educational practice only. Verify all findings manually and do not use this output for clinical decisions.
"""

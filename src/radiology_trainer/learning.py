from __future__ import annotations

import re

from radiology_trainer.domain import EvidenceBundle, ReadScorecard, StudentRead


FINDING_SYNONYMS = {
    "cardiomegaly": ["cardiomegaly", "enlarged heart", "heart size", "cardiac silhouette"],
    "pleural effusion": ["pleural effusion", "effusion", "blunting", "costophrenic"],
    "atelectasis": ["atelectasis", "volume loss", "linear opacity", "basal opacity"],
    "pneumothorax": ["pneumothorax", "pleural line", "collapsed lung"],
    "consolidation": ["consolidation", "airspace opacity", "focal opacity", "infection"],
    "pulmonary edema": ["pulmonary edema", "edema", "interstitial", "vascular congestion"],
    "rib fracture": ["rib fracture", "fracture", "rib"],
}

VIEW_TERMS = ["pa", "ap", "lateral", "frontal", "portable", "projection", "view"]
QUALITY_TERMS = ["quality", "rotation", "inspiration", "exposure", "penetration", "positioning"]
NEGATIVE_TERMS = ["no ", "without", "not ", "absent", "negative for"]
UNCERTAINTY_TERMS = ["possible", "consider", "cannot exclude", "mild", "likely", "uncertain"]


def build_scorecard(evidence: EvidenceBundle, student_read: StudentRead) -> ReadScorecard:
    text = _normalize(student_read.observation)
    if not text:
        return ReadScorecard(
            total_score=0,
            coverage_score=0,
            technique_score=0,
            uncertainty_score=0,
            practice_focus="Start with a blind read before revealing model evidence.",
            next_steps=[
                "State projection and image quality first.",
                "Use a fixed search pattern.",
                "Name key negatives before asking the tutor.",
            ],
        )

    relevant = [finding for finding in evidence.top_findings(5) if finding.score >= 0.35]
    matched = [finding.label for finding in relevant if _finding_is_mentioned(finding.label, text)]
    missed = [finding.label for finding in relevant if finding.label not in matched]

    coverage_score = int(round((len(matched) / max(len(relevant), 1)) * 100))

    technique_hits = []
    if _contains_any(text, VIEW_TERMS):
        technique_hits.append("projection/view")
    if _contains_any(text, QUALITY_TERMS):
        technique_hits.append("image quality")
    if _contains_any(text, NEGATIVE_TERMS):
        technique_hits.append("key negatives")
    technique_score = int(round((len(technique_hits) / 3) * 100))

    uncertainty_score = 100 if _contains_any(text, UNCERTAINTY_TERMS) else 35
    total = int(round(coverage_score * 0.55 + technique_score * 0.30 + uncertainty_score * 0.15))

    next_steps = _next_steps(missed, technique_hits)
    practice_focus = _practice_focus(missed, technique_hits)

    return ReadScorecard(
        total_score=total,
        coverage_score=coverage_score,
        technique_score=technique_score,
        uncertainty_score=uncertainty_score,
        matched_findings=matched,
        missed_findings=missed,
        technique_hits=technique_hits,
        next_steps=next_steps,
        practice_focus=practice_focus,
    )


def _finding_is_mentioned(label: str, text: str) -> bool:
    terms = FINDING_SYNONYMS.get(label, [label])
    return _contains_any(text, terms)


def _contains_any(text: str, terms: list[str]) -> bool:
    return any(term in text for term in terms)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _next_steps(missed: list[str], technique_hits: list[str]) -> list[str]:
    steps = []
    if "projection/view" not in technique_hits:
        steps.append("State the projection before estimating heart size or subtle lung findings.")
    if "image quality" not in technique_hits:
        steps.append("Add a short quality check: rotation, inspiration, exposure, and positioning.")
    if "key negatives" not in technique_hits:
        steps.append("Name the key negatives that would change management in a real read.")
    for label in missed[:3]:
        steps.append(f"Re-inspect the image for evidence related to {label}.")
    return steps[:5] or ["Tighten the impression into one or two prioritized findings."]


def _practice_focus(missed: list[str], technique_hits: list[str]) -> str:
    if missed:
        return f"Finding coverage: focus next on {missed[0]} and its mimics."
    if len(technique_hits) < 3:
        return "Read structure: make projection, quality, and key negatives automatic."
    return "Good structure. Next focus on sharper differentials and uncertainty wording."


def format_scorecard(scorecard: ReadScorecard) -> str:
    matched = ", ".join(scorecard.matched_findings) or "none"
    missed = ", ".join(scorecard.missed_findings) or "none"
    technique = ", ".join(scorecard.technique_hits) or "none"
    next_steps = "\n".join(f"- {step}" for step in scorecard.next_steps)
    return f"""
## Blind-read scorecard

**Overall:** {scorecard.total_score}/100

| Area | Score |
|---|---:|
| Finding coverage | {scorecard.coverage_score}/100 |
| Read technique | {scorecard.technique_score}/100 |
| Uncertainty wording | {scorecard.uncertainty_score}/100 |

**Matched evidence:** {matched}

**Needs re-check:** {missed}

**Technique covered:** {technique}

**Practice focus:** {scorecard.practice_focus}

**Next steps**

{next_steps}
"""

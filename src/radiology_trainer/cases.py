from __future__ import annotations

from pathlib import Path

from radiology_trainer.domain import DemoCase, ReferenceAnswer


_CASE_SPECS = (
    {
        "id": "normal",
        "title": "Normal chest radiograph",
        "difficulty": "foundation",
        "filename": "9f827b27749080ca0d3d8d2917a4bae5a5cd031c",
        "labels": [],
        "urgency": "normal",
        "teaching_point": "Use a fixed search pattern and state the important negatives.",
    },
    {
        "id": "scoliosis",
        "title": "Thoracic scoliosis",
        "difficulty": "intermediate",
        "filename": "04f72062c19d9cd7a55519708aa2cc58b5e52b52",
        "labels": ["scoliosis"],
        "urgency": "non-urgent",
        "teaching_point": "Trace the thoracic spine instead of limiting the read to lungs and heart.",
    },
    {
        "id": "cardiomediastinal",
        "title": "Cardiomediastinal findings",
        "difficulty": "advanced",
        "filename": "c3fb036149b3fa4f271dc3a9400690699c633c52",
        "labels": ["cardiomegaly", "mediastinum widened", "unfolded aorta"],
        "urgency": "critical",
        "teaching_point": "Separate cardiac size, mediastinal width, and aortic contour in the impression.",
    },
)


def demo_cases(backend_dir: str | Path = "outputs/xraydar-cv") -> list[DemoCase]:
    data_dir = Path(backend_dir) / "demo_data"
    return [
        DemoCase(
            id=spec["id"],
            title=spec["title"],
            difficulty=spec["difficulty"],
            image_path=str(data_dir / spec["filename"]),
            reference=ReferenceAnswer(
                labels=spec["labels"],
                urgency=spec["urgency"],
                teaching_point=spec["teaching_point"],
                source="X-Raydar public demo_data labels",
            ),
        )
        for spec in _CASE_SPECS
    ]


def get_demo_case(case_id: str, backend_dir: str | Path = "outputs/xraydar-cv") -> DemoCase:
    for case in demo_cases(backend_dir):
        if case.id == case_id:
            return case
    raise KeyError(case_id)

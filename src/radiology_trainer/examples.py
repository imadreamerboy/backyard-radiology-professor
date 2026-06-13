from __future__ import annotations

from pathlib import Path


def example_cases() -> list[list[str]]:
    root = Path(__file__).resolve().parents[2]
    assets = root / "assets" / "examples"
    return [
        [
            str(assets / "normal_search_pattern.png"),
            "PA chest radiograph. Image quality adequate. No focal opacity, no pleural effusion, no pneumothorax.",
            "Grade my search pattern and tell me what key negatives are missing.",
        ],
        [
            str(assets / "right_lower_opacity.png"),
            "PA chest radiograph. Possible right lower zone focal opacity. No large pneumothorax.",
            "What alternatives should I consider for this opacity?",
        ],
        [
            str(assets / "effusion_pattern.png"),
            "Frontal chest radiograph. Possible small pleural effusion with basal blunting.",
            "How should I describe the pleural bases and uncertainty?",
        ],
        [
            str(assets / "pneumothorax_check.png"),
            "AP chest radiograph. I would re-check the right apical pleural margin for pneumothorax.",
            "What visual signs help exclude pneumothorax?",
        ],
    ]


def example_case_label(path: str) -> str:
    return Path(path).stem.replace("_", " ").title()

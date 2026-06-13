from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path

from radiology_trainer.config import AppConfig
from radiology_trainer.domain import StudentRead
from radiology_trainer.image_io import load_xray_image
from radiology_trainer.pipeline import build_pipeline
from radiology_trainer.xraydar_labels import normalize_xraydar_label


def main() -> None:
    args = _parse_args()
    rows = _load_demo_rows(args.backend_dir)
    pipeline = build_pipeline(
        AppConfig(
            model_mode="local",
            chest_evidence_url=args.evidence_url,
            chest_evidence_timeout_seconds=args.timeout_seconds,
        )
    )
    results = []
    for row in rows:
        image_path = args.backend_dir / "demo_data" / Path(row["fnDcm"]).name
        labels = _parse_labels(row["list_of_labels"])
        evidence, _ = pipeline.analyze(
            image=load_xray_image(image_path),
            student_read=StudentRead(observation="Validation run."),
        )
        top_findings = evidence.top_findings(args.top_k)
        top = [
            {
                "label": normalize_xraydar_label(item.label),
                "score": round(item.score, 4),
            }
            for item in top_findings
        ]
        top_labels = [item["label"] for item in top]
        label_hit = bool(set(labels) & set(top_labels))
        high_confidence_findings = [
            normalize_xraydar_label(item.label)
            for item in evidence.findings
            if item.score >= args.positive_threshold
        ]
        normal_pass = not labels and not high_confidence_findings
        passed = label_hit if labels else normal_pass
        results.append(
            {
                "file": image_path.name,
                "expected": labels,
                "top_predictions": top,
                "high_confidence_findings": high_confidence_findings,
                "label_hit": label_hit,
                "normal_pass": normal_pass if not labels else None,
                "passed": passed,
            }
        )

    labeled = [item for item in results if item["expected"]]
    normal = [item for item in results if not item["expected"]]
    summary = {
        "dataset": "xraydar-cv demo_data",
        "evidence_url": args.evidence_url,
        "top_k": args.top_k,
        "positive_threshold": args.positive_threshold,
        "cases": len(results),
        "passed": sum(1 for item in results if item["passed"]),
        "labeled_cases": len(labeled),
        "labeled_top_k_hits": sum(1 for item in labeled if item["label_hit"]),
        "normal_cases": len(normal),
        "normal_threshold_passes": sum(1 for item in normal if item["normal_pass"]),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate app pipeline against X-Raydar demo cases.")
    parser.add_argument("--backend-dir", type=Path, default=Path("outputs/xraydar-cv"))
    parser.add_argument("--evidence-url", default="http://127.0.0.1:9000/analyze")
    parser.add_argument("--top-k", type=int, default=7)
    parser.add_argument("--positive-threshold", type=float, default=0.5)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--output", type=Path, default=Path("outputs/xraydar_demo_validation.json"))
    return parser.parse_args()


def _load_demo_rows(backend_dir: Path) -> list[dict[str, str]]:
    csv_path = backend_dir / "demo_data" / "demo_data.csv"
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _parse_labels(value: str) -> list[str]:
    labels = ast.literal_eval(value)
    return [normalize_xraydar_label(label) for label in labels]


if __name__ == "__main__":
    main()

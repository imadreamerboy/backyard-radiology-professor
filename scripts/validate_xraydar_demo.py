from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path

from radiology_trainer.adapters.xraydar import XRaydarEvidenceModel
from radiology_trainer.image_io import load_xray_image
from radiology_trainer.xraydar_labels import normalize_xraydar_label


def main() -> None:
    args = _parse_args()
    rows = _load_demo_rows(args.backend_dir)
    model = XRaydarEvidenceModel(args.backend_dir, args.device)
    results = []
    for row in rows:
        image_path = args.backend_dir / "demo_data" / Path(row["fnDcm"]).name
        labels = _parse_labels(row["list_of_labels"])
        evidence = model.analyze(load_xray_image(image_path))
        top = [
            {"label": normalize_xraydar_label(item.label), "score": round(item.score, 4)}
            for item in evidence.top_findings(args.top_k)
        ]
        top_labels = [item["label"] for item in top]
        high_confidence = [
            normalize_xraydar_label(item.label)
            for item in evidence.findings
            if item.score >= args.positive_threshold
        ]
        label_hit = bool(set(labels) & set(top_labels))
        normal_pass = not labels and not high_confidence
        results.append(
            {
                "file": image_path.name,
                "expected": labels,
                "top_predictions": top,
                "high_confidence_findings": high_confidence,
                "passed": label_hit if labels else normal_pass,
            }
        )

    summary = {
        "dataset": "xraydar-cv demo_data",
        "device": args.device,
        "cases": len(results),
        "passed": sum(item["passed"] for item in results),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if summary["passed"] != summary["cases"]:
        raise SystemExit(1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate direct X-Raydar inference.")
    parser.add_argument("--backend-dir", type=Path, default=Path("outputs/xraydar-cv"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top-k", type=int, default=7)
    parser.add_argument("--positive-threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=Path("outputs/xraydar_demo_validation.json"))
    return parser.parse_args()


def _load_demo_rows(backend_dir: Path) -> list[dict[str, str]]:
    with (backend_dir / "demo_data" / "demo_data.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        return list(csv.DictReader(handle))


def _parse_labels(value: str) -> list[str]:
    return [normalize_xraydar_label(label) for label in ast.literal_eval(value)]


if __name__ == "__main__":
    main()

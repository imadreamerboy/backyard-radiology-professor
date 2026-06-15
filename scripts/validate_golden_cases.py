from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import requests


CASES = (
    {
        "id": "normal",
        "blind_read": (
            "PA chest radiograph. Cardiomediastinal silhouette and lungs appear "
            "within normal limits. No focal airspace opacity or pleural effusion."
        ),
        "expected_reference": set(),
        "required_classifier": set(),
    },
    {
        "id": "scoliosis",
        "blind_read": (
            "PA chest radiograph. No focal pulmonary opacity. There may be mild "
            "thoracic spinal curvature."
        ),
        "expected_reference": {"scoliosis"},
        "required_classifier": {"scoliosis"},
    },
    {
        "id": "cardiomediastinal",
        "blind_read": (
            "PA chest radiograph. Enlarged cardiac silhouette with a widened "
            "mediastinal contour and unfolded aorta."
        ),
        "expected_reference": {
            "cardiomegaly",
            "mediastinum widened",
            "unfolded aorta",
        },
        "required_classifier": {
            "cardiomegaly",
            "mediastinum widened",
            "tortuosity aorta",
        },
    },
)


def main() -> None:
    args = _parse_args()
    status = requests.get(f"{args.app_url}/api/status?wake=true", timeout=args.timeout_seconds).json()
    reports = [_run_case(args, case) for case in CASES]
    passed = status.get("runtime_status") == "ready" and all(
        report["passed"] for report in reports
    )
    output = {
        "passed": passed,
        "app_url": args.app_url,
        "runtime_status": status,
        "cases": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    if not passed:
        raise SystemExit(1)


def _run_case(args: argparse.Namespace, case: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    response = requests.post(
        f"{args.app_url}/api/analyze",
        data={
            "case_id": case["id"],
            "observation": case["blind_read"],
            "question": "Compare my blind read with the evidence and test one missed concept.",
        },
        timeout=args.timeout_seconds,
    )
    response.raise_for_status()
    result = response.json()
    findings = {
        item["label"].lower(): float(item["score"])
        for item in result["evidence"]["findings"]
    }
    reference = set(result["reference"]["labels"])
    runs = [
        *result["evidence"]["model_runs"],
        result["tutor"].get("model_run"),
    ]
    runs = [run for run in runs if run]
    regions = result["evidence"]["regions"]
    checks = {
        "reference": reference == case["expected_reference"],
        "classifier": _classifier_matches(case["id"], findings, case),
        "vision_observations": bool(result["evidence"]["observations"]),
        "bounded_regions": all(_valid_box(box) for box in regions)
        and (case["id"] == "normal" or bool(regions)),
        "tutor_json": bool(result["tutor"]["summary"])
        and bool(result["tutor"]["feedback"])
        and bool(result["tutor"]["suggested_checks"]),
        "quiz": bool(result["tutor"]["quiz"]),
        "model_runs": len(runs) == 3
        and all(run.get("status") == "ok" for run in runs),
    }
    return {
        "id": case["id"],
        "passed": all(checks.values()),
        "checks": checks,
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
        "classifier_scores": dict(
            sorted(findings.items(), key=lambda item: item[1], reverse=True)[:8]
        ),
        "reference_labels": sorted(reference),
        "regions": regions,
        "model_runs": runs,
        "quiz": result["tutor"]["quiz"],
    }


def _classifier_matches(
    case_id: str,
    findings: dict[str, float],
    case: dict[str, Any],
) -> bool:
    if case_id == "normal":
        return not any(score >= 0.5 for score in findings.values())
    return all(findings.get(label, 0.0) >= 0.35 for label in case["required_classifier"])


def _valid_box(box: dict[str, Any]) -> bool:
    return (
        0 <= box["x1"] < box["x2"] <= 1
        and 0 <= box["y1"] < box["y2"] <= 1
        and bool(box["label"])
        and box["source"] == "medgemma-localizer"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate all three real-model golden cases."
    )
    parser.add_argument("--app-url", default="http://127.0.0.1:7860")
    parser.add_argument("--timeout-seconds", type=float, default=900)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/golden_case_validation.json"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()

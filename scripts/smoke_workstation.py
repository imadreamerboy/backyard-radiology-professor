from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests


def main() -> None:
    args = _parse_args()
    status = requests.get(f"{args.app_url}/api/status", timeout=30).json()
    with args.image.open("rb") as handle:
        response = requests.post(
            f"{args.app_url}/api/analyze",
            data={
                "observation": args.blind_read,
                "question": "What should I inspect next?",
            },
            files={"image": (args.image.name, handle, "application/dicom")},
            timeout=args.timeout_seconds,
        )
    response.raise_for_status()
    result = response.json()
    model_runs = [
        *result["evidence"]["model_runs"],
        result["tutor"].get("model_run"),
    ]
    model_runs = [run for run in model_runs if run]
    top_labels = [item["label"].lower() for item in result["evidence"]["findings"][:7]]
    passed = (
        status["runtime_status"] == "ready"
        and args.expected_label.lower() in top_labels
        and result["evidence"]["observations"]
        and result["evidence"]["regions"]
        and result["tutor"]["quiz"]
        and all(run["status"] == "ok" for run in model_runs)
    )
    summary = {
        "passed": passed,
        "status": status,
        "top_labels": top_labels,
        "regions": result["evidence"]["regions"],
        "model_runs": model_runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not passed:
        raise SystemExit(1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test the complete workstation backend.")
    parser.add_argument("--app-url", default="http://127.0.0.1:7860")
    parser.add_argument(
        "--image",
        type=Path,
        default=Path("outputs/xraydar-cv/demo_data/04f72062c19d9cd7a55519708aa2cc58b5e52b52"),
    )
    parser.add_argument("--expected-label", default="scoliosis")
    parser.add_argument(
        "--blind-read",
        default="PA chest radiograph. Image quality adequate. No focal pulmonary opacity.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=600)
    parser.add_argument("--output", type=Path, default=Path("outputs/workstation_smoke.json"))
    return parser.parse_args()


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from gradio_client import Client, handle_file

from radiology_trainer.xraydar_labels import normalize_xraydar_label


def main() -> None:
    args = _parse_args()
    result = Client(args.app_url).predict(
        handle_file(str(args.image)),
        "App API smoke test blind read.",
        "What should I double-check on this case?",
        api_name="/_run_analysis",
    )
    rows = _structured_rows(result[1])
    expected_label = normalize_xraydar_label(args.expected_label)
    top_labels = [normalize_xraydar_label(row[0]) for row in rows[: args.top_k]]
    top_source = str(rows[0][2]) if rows else ""
    top_score = float(rows[0][1]) if rows else 0.0
    tutor_summary = str(result[4])
    model_notes = str(result[7])
    live_tutor_ok = not args.require_live_tutor or (
        args.expected_tutor_provider in tutor_summary
        and "fallback" not in tutor_summary.lower()
        and "demo-nemotron-tutor" not in tutor_summary.lower()
    )
    passed = (
        expected_label in top_labels
        and top_source == "xraydar-cv"
        and top_score >= args.min_top_score
        and "X-Raydar CV backend" in model_notes
        and live_tutor_ok
    )
    summary = {
        "app_url": args.app_url,
        "image": str(args.image),
        "expected_label": expected_label,
        "top_k": args.top_k,
        "min_top_score": args.min_top_score,
        "top_findings": rows[: args.top_k],
        "require_live_tutor": args.require_live_tutor,
        "expected_tutor_provider": args.expected_tutor_provider,
        "live_tutor_ok": live_tutor_ok,
        "tutor_summary_excerpt": tutor_summary[:1000],
        "model_notes": model_notes,
        "passed": passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not passed:
        raise SystemExit(1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test a running Gradio app with X-Raydar.")
    parser.add_argument("--app-url", default="http://127.0.0.1:7860")
    parser.add_argument("--image", type=Path, default=Path("outputs/xraydar_scoliosis.dcm"))
    parser.add_argument("--expected-label", default="scoliosis")
    parser.add_argument("--top-k", type=int, default=7)
    parser.add_argument("--min-top-score", type=float, default=0.9)
    parser.add_argument("--require-live-tutor", action="store_true")
    parser.add_argument("--expected-tutor-provider", default="nemotron-tutor:openai")
    parser.add_argument("--output", type=Path, default=Path("outputs/gradio_xraydar_smoke.json"))
    return parser.parse_args()


def _structured_rows(value: Any) -> list[list[Any]]:
    if isinstance(value, dict):
        return value.get("data", [])
    raise TypeError(f"Unexpected structured evidence value: {type(value).__name__}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from radiology_trainer.adapters.llama_client import LlamaCppClient
from radiology_trainer.image_io import load_xray_image
from radiology_trainer.structured_output import _extract_json


def main() -> None:
    args = _parse_args()
    client = LlamaCppClient(args.base_url, args.api_key, args.timeout_seconds)
    models = _wait_for_models(client, args.wait_seconds)
    model_ids = {str(item.get("id", "")) for item in models}
    missing = {"medgemma-professor", "medgemma-localizer"} - model_ids
    if missing:
        raise RuntimeError(f"Router is missing models: {sorted(missing)}")

    professor_started = time.perf_counter()
    professor = client.chat(
        model="medgemma-professor",
        messages=[
            {"role": "system", "content": "Return JSON only."},
            {"role": "user", "content": 'Return {"status":"ready"}.'},
        ],
        max_tokens=64,
        json_schema={
            "type": "object",
            "properties": {"status": {"type": "string", "const": "ready"}},
            "required": ["status"],
            "additionalProperties": False,
        },
    )
    json.loads(_extract_json(professor))
    professor_ms = int((time.perf_counter() - professor_started) * 1000)

    image = load_xray_image(args.image)
    medgemma_started = time.perf_counter()
    medgemma = client.chat(
        model="medgemma-localizer",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": 'Return JSON only: {"observed":"yes"}.'},
                    {"type": "image_url", "image_url": {"url": client.image_url(image)}},
                ],
            }
        ],
        max_tokens=80,
        json_schema={
            "type": "object",
            "properties": {"observed": {"type": "string", "const": "yes"}},
            "required": ["observed"],
            "additionalProperties": False,
        },
    )
    json.loads(_extract_json(medgemma))
    medgemma_ms = int((time.perf_counter() - medgemma_started) * 1000)
    print(
        json.dumps(
            {
                "passed": True,
                "models": sorted(model_ids),
                "professor_latency_ms": professor_ms,
                "localizer_latency_ms": medgemma_ms,
            },
            indent=2,
        )
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test both llama.cpp router models.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--api-key", default="local")
    parser.add_argument(
        "--image",
        type=Path,
        default=Path("outputs/xraydar-cv/demo_data/04f72062c19d9cd7a55519708aa2cc58b5e52b52"),
    )
    parser.add_argument("--wait-seconds", type=float, default=600)
    parser.add_argument("--timeout-seconds", type=float, default=180)
    return parser.parse_args()


def _wait_for_models(client: LlamaCppClient, wait_seconds: float) -> list[dict]:
    deadline = time.monotonic() + wait_seconds
    last_error = ""
    required = {"medgemma-professor", "medgemma-localizer"}
    while time.monotonic() < deadline:
        try:
            models = client.list_models()
            available = {str(item.get("id", "")) for item in models}
            if required <= available:
                return models
        except Exception as exc:
            last_error = str(exc)
        time.sleep(3)
    raise TimeoutError(f"llama.cpp router did not become ready: {last_error}")


if __name__ == "__main__":
    main()

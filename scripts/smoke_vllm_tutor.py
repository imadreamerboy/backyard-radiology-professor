from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

import requests


DEFAULT_MODEL = "nemotron3-nano-4B-FP8"


def main() -> None:
    args = _parse_args()
    model_ids = _wait_for_models(args.base_url, args.wait_seconds, args.timeout_seconds)
    response_text = _chat(args)
    expected_ok = "VLLM_OK" in response_text
    summary = {
        "base_url": args.base_url,
        "model": args.model,
        "available_models": model_ids,
        "response": response_text,
        "passed": expected_ok,
    }
    print(json.dumps(summary, indent=2))
    if not expected_ok:
        raise SystemExit(1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test a vLLM OpenAI-compatible tutor.")
    parser.add_argument("--base-url", default=os.getenv("RAD_TRAINER_OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--model", default=os.getenv("RAD_TRAINER_NEMOTRON_MODEL", DEFAULT_MODEL))
    parser.add_argument("--api-key", default=os.getenv("RAD_TRAINER_OPENAI_API_KEY", "local"))
    parser.add_argument("--wait-seconds", type=float, default=600.0)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    return parser.parse_args()


def _wait_for_models(base_url: str, wait_seconds: float, timeout_seconds: float) -> list[str]:
    deadline = time.monotonic() + wait_seconds
    last_error = ""
    while time.monotonic() < deadline:
        try:
            response = requests.get(f"{base_url.rstrip('/')}/models", timeout=timeout_seconds)
            response.raise_for_status()
            data = response.json()
            return [item["id"] for item in data.get("data", [])]
        except Exception as exc:
            last_error = str(exc)
            time.sleep(5)
    raise TimeoutError(f"vLLM did not become ready within {wait_seconds}s: {last_error}")


def _chat(args: argparse.Namespace) -> str:
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": [
            {
                "role": "system",
                "content": "Do not reason. Reply exactly with: VLLM_OK local tutor ready.",
            },
            {"role": "user", "content": "Run a readiness check."},
        ],
        "temperature": 0,
        "max_tokens": 32,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}",
    }
    response = requests.post(
        f"{args.base_url.rstrip('/')}/chat/completions",
        headers=headers,
        json=payload,
        timeout=args.timeout_seconds,
    )
    response.raise_for_status()
    message = response.json()["choices"][0]["message"]
    content = message.get("content") or message.get("reasoning") or ""
    return str(content).strip()


if __name__ == "__main__":
    main()

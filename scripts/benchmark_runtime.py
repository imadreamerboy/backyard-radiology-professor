from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import requests


def main() -> None:
    args = _parse_args()
    sampler = _GpuSampler(args.app_url)
    sampler.start()
    try:
        status = requests.get(
            f"{args.app_url}/api/status?wake=true",
            timeout=args.timeout_seconds,
        ).json()
        runs = [_run(args, index) for index in range(args.runs)]
    finally:
        sampler.stop()
    output = {
        "passed": all(run["passed"] for run in runs),
        "app_url": args.app_url,
        "runtime_status": status,
        "host_baseline_gpu_memory_mb": args.gpu_baseline_mb,
        "peak_gpu_memory_mb": sampler.peak_memory_mb,
        "peak_application_gpu_memory_mb": sampler.application_peak_memory_mb(
            args.gpu_baseline_mb
        ),
        "local_peak_gpu_memory_mb": sampler.local_peak_memory_mb,
        "remote_peak_gpu_memory_mb": sampler.remote_peak_memory_mb,
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    if not output["passed"]:
        raise SystemExit(1)


def _run(args: argparse.Namespace, index: int) -> dict[str, Any]:
    created = requests.post(
        f"{args.app_url}/api/sessions",
        data={"case_id": args.case_id},
        timeout=60,
    )
    created.raise_for_status()
    session = created.json()
    image = requests.get(
        f"{args.app_url}{session['study']['images'][0]['image_url']}",
        timeout=60,
    ).content
    started = time.perf_counter()
    analysis_events = _stream(
        f"{args.app_url}/api/sessions/{session['id']}/analyze",
        {"observation": args.blind_read},
        args.timeout_seconds,
    )
    analysis_ms = int((time.perf_counter() - started) * 1000)
    started = time.perf_counter()
    chat_events = _stream(
        f"{args.app_url}/api/sessions/{session['id']}/chat",
        {"message": "Explain the most important teaching point in this case."},
        args.timeout_seconds,
    )
    chat_ms = int((time.perf_counter() - started) * 1000)
    completed = next(
        (event for event in reversed(analysis_events) if event.get("type") == "complete"),
        None,
    )
    chat_complete = next(
        (event for event in reversed(chat_events) if event.get("type") == "complete"),
        None,
    )
    requests.delete(f"{args.app_url}/api/sessions/{session['id']}", timeout=30)
    return {
        "index": index,
        "temperature": "cold" if index == 0 else "warm",
        "passed": completed is not None and chat_complete is not None,
        "analysis_latency_ms": analysis_ms,
        "chat_latency_ms": chat_ms,
        "image_sha256": hashlib.sha256(image).hexdigest(),
        "model_runs": (
            [
                *completed["session"]["result"]["evidence"]["model_runs"],
                completed["session"]["result"]["tutor"]["model_run"],
                chat_complete["message"]["model_run"],
            ]
            if completed and chat_complete
            else []
        ),
    }


def _stream(url: str, payload: dict[str, Any], timeout: float) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with requests.post(url, json=payload, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        for raw in response.iter_lines(decode_unicode=True):
            if raw and raw.startswith("data:"):
                events.append(json.loads(raw[5:].strip()))
    errors = [event["message"] for event in events if event.get("type") == "error"]
    if errors:
        raise RuntimeError("; ".join(errors))
    return events


class _GpuSampler:
    def __init__(self, app_url: str) -> None:
        self.app_url = app_url.rstrip("/")
        self.local_peak_memory_mb = 0
        self.remote_peak_memory_mb = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    @property
    def peak_memory_mb(self) -> int:
        return max(self.local_peak_memory_mb, self.remote_peak_memory_mb)

    def application_peak_memory_mb(self, baseline_mb: int) -> int:
        if self.remote_peak_memory_mb:
            return self.remote_peak_memory_mb
        return max(0, self.local_peak_memory_mb - baseline_mb)

    def _sample(self) -> None:
        while not self._stop.is_set():
            self._sample_local_gpu()
            self._sample_remote_gpu()
            self._stop.wait(1.0)

    def _sample_local_gpu(self) -> None:
        try:
            value = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=3,
            ).stdout.splitlines()[0]
            self.local_peak_memory_mb = max(self.local_peak_memory_mb, int(value.strip()))
        except Exception:
            pass

    def _sample_remote_gpu(self) -> None:
        try:
            status = requests.get(f"{self.app_url}/api/status?wake=true", timeout=10).json()
            gpu = status.get("gpu") or {}
            used = gpu.get("memory_used_mb")
            if used is not None:
                self.remote_peak_memory_mb = max(self.remote_peak_memory_mb, int(used))
        except Exception:
            pass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record real workstation runtime metrics.")
    parser.add_argument("--app-url", default="http://127.0.0.1:7860")
    parser.add_argument("--case-id", default="scoliosis")
    parser.add_argument(
        "--blind-read",
        default="PA chest radiograph. Mild thoracic spinal curvature. No focal opacity.",
    )
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    parser.add_argument(
        "--gpu-baseline-mb",
        type=int,
        default=0,
        help="GPU memory already used by the host before the application starts.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/runtime_benchmark.json"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()

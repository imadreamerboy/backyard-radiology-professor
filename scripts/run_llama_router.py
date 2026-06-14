from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    args = _parse_args()
    command = [
        args.binary,
        "--models-preset",
        str(args.preset.resolve()),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--metrics",
        "--models-max",
        "1",
        "--api-key",
        args.api_key,
    ]
    print(" ".join(command))
    if args.detach:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        with args.log.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        args.pid.write_text(str(process.pid), encoding="utf-8")
        print(f"llama.cpp router PID {process.pid}; log: {args.log}")
        return
    subprocess.run(command, check=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the unified llama.cpp model router.")
    parser.add_argument("--binary", default="llama-server")
    parser.add_argument("--preset", type=Path, default=Path("runtime/models.ini"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--api-key", default="local")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--log", type=Path, default=Path("outputs/llama-router.log"))
    parser.add_argument("--pid", type=Path, default=Path("outputs/llama-router.pid"))
    return parser.parse_args()


if __name__ == "__main__":
    main()

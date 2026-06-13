from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

try:
    from run_vllm_nemotron import DEFAULT_MODEL, DEFAULT_SERVED_MODEL, _ensure_reasoning_parser
except ModuleNotFoundError:
    from scripts.run_vllm_nemotron import DEFAULT_MODEL, DEFAULT_SERVED_MODEL, _ensure_reasoning_parser


DEFAULT_VENV = "~/.cache/radiology-trainer-vllm/venv"
DEFAULT_LOG = "~/.cache/radiology-trainer-vllm/server.log"
DEFAULT_PID = "~/.cache/radiology-trainer-vllm/server.pid"


def main() -> None:
    args = _parse_args()
    parser_path = _ensure_reasoning_parser(args.parser_dir.resolve())
    parser_path_wsl = _windows_path_to_wsl(parser_path)
    bash = _build_bash(args, parser_path_wsl)
    subprocess.run(["wsl", "-d", args.distro, "--", "bash", "-lc", bash], check=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run quantized Nemotron with vLLM inside WSL.")
    parser.add_argument("--distro", default=os.getenv("RAD_TRAINER_WSL_DISTRO", "Ubuntu"))
    parser.add_argument("--venv", default=os.getenv("RAD_TRAINER_WSL_VLLM_VENV", DEFAULT_VENV))
    parser.add_argument("--model", default=os.getenv("RAD_TRAINER_VLLM_MODEL", DEFAULT_MODEL))
    parser.add_argument(
        "--served-model-name",
        default=os.getenv("RAD_TRAINER_VLLM_SERVED_MODEL", DEFAULT_SERVED_MODEL),
    )
    parser.add_argument("--port", type=int, default=int(os.getenv("RAD_TRAINER_VLLM_PORT", "8000")))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--attention-backend", default="TRITON_ATTN")
    parser.add_argument("--parser-dir", type=Path, default=Path("outputs/vllm"))
    parser.add_argument("--log-path", default=DEFAULT_LOG)
    parser.add_argument("--pid-path", default=DEFAULT_PID)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--use-flashinfer-sampler", action="store_true")
    parser.add_argument("--setup-only", action="store_true")
    parser.add_argument("--no-reasoning-parser", action="store_true")
    return parser.parse_args()


def _build_bash(args: argparse.Namespace, parser_path_wsl: str) -> str:
    command = [
        _shell_path(_join_path(args.venv, "bin/vllm")),
        "serve",
        _quote(args.model),
        "--served-model-name",
        _quote(args.served_model_name),
        "--host",
        _quote(args.host),
        "--port",
        str(args.port),
        "--dtype",
        "auto",
        "--trust-remote-code",
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--tensor-parallel-size",
        "1",
        "--attention-backend",
        _quote(args.attention_backend),
        "--mamba-ssm-cache-dtype",
        "float32",
        "--kv-cache-dtype",
        "fp8",
    ]
    if args.enforce_eager:
        command.append("--enforce-eager")
    if not args.no_reasoning_parser:
        command.extend(
            [
                "--reasoning-parser-plugin",
                _quote(parser_path_wsl),
                "--reasoning-parser",
                "nano_v3",
            ]
        )
    serve = " ".join(command)
    setup = (
        "set -e; "
        f"mkdir -p {_shell_path(_posix_parent(args.venv))}; "
        f"if [ ! -x {_shell_path(_join_path(args.venv, 'bin/vllm'))} ]; then "
        f"uv venv --python 3.10 {_shell_path(args.venv)}; "
        f"uv pip install --python {_shell_path(_join_path(args.venv, 'bin/python'))} 'vllm>=0.15.1'; "
        "fi; "
        f"export PATH={_shell_path(_join_path(args.venv, 'bin'))}:"
        f"{_shell_path(_join_path(args.venv, 'lib/python3.10/site-packages/nvidia/cu13/bin'))}:\"$PATH\"; "
        + ("export VLLM_USE_FLASHINFER_SAMPLER=0; " if not args.use_flashinfer_sampler else "")
        + f"{_shell_path(_join_path(args.venv, 'bin/vllm'))} --version"
    )
    if args.setup_only:
        return setup
    if args.detach:
        return (
            setup
            + "; "
            + f"mkdir -p {_shell_path(_posix_parent(args.log_path))}; "
            + f"nohup {serve} > {_shell_path(args.log_path)} 2>&1 & "
            + "sleep 1; "
            + f"pgrep -f {_quote('vllm serve ' + args.model)} | tail -1 > {_shell_path(args.pid_path)}; "
            + f"cat {_shell_path(args.pid_path)}"
        )
    return setup + "; exec " + serve


def _windows_path_to_wsl(path: Path) -> str:
    drive = path.drive.rstrip(":").lower()
    if not drive:
        return path.as_posix()
    rest = path.resolve().as_posix().split(":", maxsplit=1)[1].lstrip("/")
    return f"/mnt/{drive}/{rest}"


def _posix_parent(path: str) -> str:
    return path.rstrip("/").rsplit("/", maxsplit=1)[0]


def _join_path(base: str, suffix: str) -> str:
    return base.rstrip("/") + "/" + suffix.lstrip("/")


def _shell_path(path: str) -> str:
    if path == "~":
        return '"$HOME"'
    if path.startswith("~/"):
        return '"$HOME"/' + _quote(path[2:])
    return _quote(path)


def _quote(value: str) -> str:
    return "'" + value.replace("'", "'\\''") + "'"


if __name__ == "__main__":
    main()

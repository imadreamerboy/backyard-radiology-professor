from __future__ import annotations

import argparse
import os
import subprocess
import urllib.request
from pathlib import Path


DEFAULT_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8"
DEFAULT_SERVED_MODEL = "nemotron3-nano-4B-FP8"
PARSER_URL = (
    "https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16/resolve/main/"
    "nano_v3_reasoning_parser.py"
)


def main() -> None:
    args = _parse_args()
    cache_dir = args.cache_dir.resolve()
    parser_dir = args.parser_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    parser_dir.mkdir(parents=True, exist_ok=True)

    command = _build_docker_command(args, cache_dir, parser_dir)
    print(" ".join(command))
    subprocess.run(command, check=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run quantized Nemotron through vLLM Docker.")
    parser.add_argument("--model", default=os.getenv("RAD_TRAINER_VLLM_MODEL", DEFAULT_MODEL))
    parser.add_argument(
        "--served-model-name",
        default=os.getenv("RAD_TRAINER_VLLM_SERVED_MODEL", DEFAULT_SERVED_MODEL),
    )
    parser.add_argument("--image", default=os.getenv("RAD_TRAINER_VLLM_IMAGE", "vllm/vllm-openai:latest"))
    parser.add_argument("--port", type=int, default=int(os.getenv("RAD_TRAINER_VLLM_PORT", "8000")))
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--attention-backend", default="TRITON_ATTN")
    parser.add_argument("--container-name", default="radiology-trainer-vllm")
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/hf-cache"))
    parser.add_argument("--parser-dir", type=Path, default=Path("outputs/vllm"))
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--use-flashinfer-sampler", action="store_true")
    parser.add_argument("--no-reasoning-parser", action="store_true")
    return parser.parse_args()


def _build_docker_command(args: argparse.Namespace, cache_dir: Path, parser_dir: Path) -> list[str]:
    command = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "--ipc=host",
        "--name",
        args.container_name,
        "-p",
        f"{args.port}:8000",
        "-v",
        f"{cache_dir}:/root/.cache/huggingface",
        "-v",
        f"{parser_dir}:/workspace/nemotron:ro",
    ]
    if args.detach:
        command.append("-d")
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        command.extend(["-e", f"HF_TOKEN={hf_token}"])
        command.extend(["-e", f"HUGGING_FACE_HUB_TOKEN={hf_token}"])
    if not args.use_flashinfer_sampler:
        command.extend(["-e", "VLLM_USE_FLASHINFER_SAMPLER=0"])

    command.extend(
        [
            args.image,
            "--model",
            args.model,
            "--served-model-name",
            args.served_model_name,
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
            args.attention_backend,
            "--mamba_ssm_cache_dtype",
            "float32",
            "--kv-cache-dtype",
            "fp8",
        ]
    )
    if args.enforce_eager:
        command.append("--enforce-eager")
    if not args.no_reasoning_parser:
        parser_file = _ensure_reasoning_parser(parser_dir)
        command.extend(
            [
                "--reasoning-parser-plugin",
                f"/workspace/nemotron/{parser_file.name}",
                "--reasoning-parser",
                "nano_v3",
            ]
        )
    return command


def _ensure_reasoning_parser(parser_dir: Path) -> Path:
    parser_path = parser_dir / "nano_v3_reasoning_parser.py"
    if parser_path.exists():
        return parser_path
    urllib.request.urlretrieve(PARSER_URL, parser_path)
    return parser_path


if __name__ == "__main__":
    main()

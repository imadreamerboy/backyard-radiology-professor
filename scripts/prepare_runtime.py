from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from huggingface_hub import hf_hub_download
from radiology_trainer.runtime_manifest import (
    LOCALIZER_REPO,
    LOCALIZER_REVISION,
    PROFESSOR_REPO,
    PROFESSOR_REVISION,
    XRAYDAR_CODE_REVISION,
    XRAYDAR_REPO,
    XRAYDAR_REVISION,
)


MODELS = (
    (
        PROFESSOR_REPO,
        PROFESSOR_REVISION,
        "medgemma-27b-it-Q4_K_M.gguf",
        "medgemma-professor",
    ),
    (
        PROFESSOR_REPO,
        PROFESSOR_REVISION,
        "mmproj-F16.gguf",
        "medgemma-professor",
    ),
    (
        LOCALIZER_REPO,
        LOCALIZER_REVISION,
        "medgemma-1.5-4b-it-Q4_K_M.gguf",
        "medgemma-localizer",
    ),
    (
        LOCALIZER_REPO,
        LOCALIZER_REVISION,
        "mmproj-F16.gguf",
        "medgemma-localizer",
    ),
)
XRAYDAR_CODE_REPO_URL = "https://github.com/gmontana/xraydar-cv"
XRAYDAR_COMMIT = XRAYDAR_CODE_REVISION
XRAYDAR_HF_REPO = XRAYDAR_REPO
XRAYDAR_HF_REVISION = XRAYDAR_REVISION
XRAYDAR_WEIGHT_FILES = ("model_best.pth.tar", "model_TranslatorCVLogitsToUrgency_fcs.pth.tar")


def main() -> None:
    args = _parse_args()
    if not args.skip_llm:
        _prepare_models(args.model_dir)
    if not args.skip_xraydar:
        _prepare_xraydar(args.xraydar_dir)
    print("Runtime assets are ready.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download llama.cpp and X-Raydar runtime assets.")
    parser.add_argument("--model-dir", type=Path, default=Path("outputs/llama-models"))
    parser.add_argument("--xraydar-dir", type=Path, default=Path("outputs/xraydar-cv"))
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--skip-xraydar", action="store_true")
    return parser.parse_args()


def _prepare_models(root: Path) -> None:
    for repo_id, revision, filename, subdir in MODELS:
        target_dir = root / subdir
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / filename
        if target.exists() and target.stat().st_size > 1_000_000:
            print(f"Model exists: {target}")
            continue
        source = Path(
            hf_hub_download(
                repo_id=repo_id,
                revision=revision,
                filename=filename,
                local_dir=target_dir,
            )
        )
        if source.resolve() != target.resolve():
            raise RuntimeError(f"Unexpected Hugging Face download path: {source}")
        print(f"Downloaded: {target}")


def _prepare_xraydar(backend_dir: Path) -> None:
    if not (backend_dir / "src" / "model_20210820_XNet38MS").exists():
        backend_dir.parent.mkdir(parents=True, exist_ok=True)
        backend_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", str(backend_dir)], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(backend_dir),
                "remote",
                "add",
                "origin",
                XRAYDAR_CODE_REPO_URL,
            ],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(backend_dir),
                "fetch",
                "--depth",
                "1",
                "origin",
                XRAYDAR_COMMIT,
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(backend_dir), "checkout", "--detach", "FETCH_HEAD"],
            check=True,
        )
    for size in (299, 512, 1024):
        target_dir = (
            backend_dir
            / "src"
            / "model_20210820_XNet38MS"
            / "model_weights"
            / f"direct_multi93_is{size}_Rv10_pre00_imagenet"
        )
        target_dir.mkdir(parents=True, exist_ok=True)
        for filename in XRAYDAR_WEIGHT_FILES:
            target = target_dir / filename
            if target.exists() and target.stat().st_size > 1_000:
                continue
            source = Path(
                hf_hub_download(
                    repo_id=XRAYDAR_HF_REPO,
                    revision=XRAYDAR_HF_REVISION,
                    filename=f"cv/is{size}/{filename}",
                )
            )
            shutil.copy2(source, target)


if __name__ == "__main__":
    main()

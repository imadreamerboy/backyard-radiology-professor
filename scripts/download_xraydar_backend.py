from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from huggingface_hub import hf_hub_download


REPO_URL = "https://github.com/gmontana/xraydar-cv"
HF_REPO_ID = "dnamodel/xraydar-cv"
WEIGHT_FILES = ["model_best.pth.tar", "model_TranslatorCVLogitsToUrgency_fcs.pth.tar"]
SIZES = [299, 512, 1024]


def main() -> None:
    args = _parse_args()
    backend_dir = args.backend_dir.resolve()
    _ensure_repo(backend_dir)
    _download_weights(backend_dir)
    print(f"X-Raydar backend ready: {backend_dir}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare local X-Raydar backend files.")
    parser.add_argument(
        "--backend-dir",
        type=Path,
        default=Path("outputs/xraydar-cv"),
        help="Local clone path for gmontana/xraydar-cv.",
    )
    return parser.parse_args()


def _ensure_repo(backend_dir: Path) -> None:
    if (backend_dir / "src" / "model_20210820_XNet38MS").exists():
        return
    backend_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", REPO_URL, str(backend_dir)],
        check=True,
    )


def _download_weights(backend_dir: Path) -> None:
    for size in SIZES:
        target = (
            backend_dir
            / "src"
            / "model_20210820_XNet38MS"
            / "model_weights"
            / f"direct_multi93_is{size}_Rv10_pre00_imagenet"
        )
        target.mkdir(parents=True, exist_ok=True)
        for filename in WEIGHT_FILES:
            source = Path(hf_hub_download(HF_REPO_ID, f"cv/is{size}/{filename}"))
            shutil.copy2(source, target / filename)


if __name__ == "__main__":
    main()


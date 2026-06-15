from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


DEFAULT_REPO = "build-small-hackathon/backyard-radiology-professor"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--backend-url", required=True)
    parser.add_argument(
        "--token-file",
        default=str(Path.home() / ".cache" / "huggingface" / "token"),
    )
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN") or _read_token(Path(args.token_file))
    api = HfApi(token=token)
    repo_id = args.repo_id
    api.add_space_variable(
        repo_id=repo_id,
        key="RAD_TRAINER_REMOTE_BACKEND_URL",
        value=args.backend_url.rstrip("/"),
    )
    if token:
        api.add_space_secret(repo_id=repo_id, key="HF_TOKEN", value=token)
    commit = api.upload_folder(
        repo_id=repo_id,
        repo_type="space",
        folder_path=".",
        commit_message="Configure Modal-backed Space proxy",
        ignore_patterns=[
            ".git/*",
            ".venv/*",
            "outputs/*",
            "data/*",
            "__pycache__/*",
            ".pytest_cache/*",
            ".ruff_cache/*",
            ".env",
            ".env.*",
            "*.pyc",
        ],
    )
    print(commit.commit_url)


def _read_token(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from radiology_trainer.domain import AnalysisSession
from radiology_trainer.study import StudyImageRecord, StudyRecord


class SessionStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def directory(self, session_id: str) -> Path:
        return self.root / session_id

    def save(self, public: AnalysisSession, study: StudyRecord) -> None:
        directory = self.directory(public.id)
        images_dir = directory / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        for image_id, image in study.images.items():
            target = images_dir / f"{image_id}.npy"
            if not target.exists():
                _atomic_numpy_write(target, image.pixels)
        _atomic_text_write(
            directory / "session.json",
            public.model_dump_json(),
        )

    def load(self, session_id: str) -> tuple[AnalysisSession, StudyRecord, Path]:
        directory = self.directory(session_id)
        public = AnalysisSession.model_validate_json(
            (directory / "session.json").read_text(encoding="utf-8")
        )
        images = {
            item.id: StudyImageRecord(
                public=item,
                pixels=np.load(directory / "images" / f"{item.id}.npy", allow_pickle=False),
                invert_default=(
                    item.metadata.photometric_interpretation.upper() == "MONOCHROME1"
                ),
            )
            for item in public.study.images
        }
        return public, StudyRecord(public=public.study, images=images), directory

    def delete(self, session_id: str) -> None:
        shutil.rmtree(self.directory(session_id), ignore_errors=True)

    def expired_ids(self, threshold: datetime) -> list[str]:
        expired: list[str] = []
        for metadata_path in self.root.glob("*/session.json"):
            try:
                payload = json.loads(metadata_path.read_text(encoding="utf-8"))
                updated_at = datetime.fromisoformat(str(payload["updated_at"]).replace("Z", "+00:00"))
                if updated_at.astimezone(UTC) < threshold:
                    expired.append(metadata_path.parent.name)
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                expired.append(metadata_path.parent.name)
        return expired


def _atomic_text_write(path: Path, content: str) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_numpy_write(path: Path, pixels: np.ndarray) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, pixels, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)

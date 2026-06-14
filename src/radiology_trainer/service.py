from __future__ import annotations

import base64
import shutil
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

from PIL import Image

from radiology_trainer.adapters.demo import DemoChestEvidenceModel, DemoTutorModel
from radiology_trainer.adapters.llama_client import LlamaCppClient
from radiology_trainer.adapters.medgemma import MedGemmaVisionTool
from radiology_trainer.adapters.professor import MedGemmaProfessorModel
from radiology_trainer.adapters.xraydar import XRaydarEvidenceModel
from radiology_trainer.cases import get_demo_case
from radiology_trainer.config import AppConfig
from radiology_trainer.domain import (
    AnalysisResult,
    AnalysisSession,
    ChatMessage,
    ModelRun,
    StudentRead,
)
from radiology_trainer.learning import build_scorecard
from radiology_trainer.reporting import format_session_note
from radiology_trainer.runtime_manifest import LLAMA_CPP_BUILD, PROFESSOR_REPO
from radiology_trainer.study import StudyRecord, load_study


@dataclass
class _SessionRecord:
    public: AnalysisSession
    study: StudyRecord
    work_dir: Path


class TrainerService:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._sessions: dict[str, _SessionRecord] = {}
        self._sessions_lock = threading.RLock()
        self._gpu_lock = threading.Lock()
        self._queue_lock = threading.Lock()
        self._queue_depth = 0
        self._workspace = Path(tempfile.mkdtemp(prefix="radiology-trainer-"))
        self._client = LlamaCppClient(
            base_url=config.llama_base_url,
            api_key=config.llama_api_key,
            timeout_seconds=config.model_timeout_seconds,
        )
        self._xraydar: XRaydarEvidenceModel | None = None
        self._localizer: MedGemmaVisionTool | None = None
        self._professor: MedGemmaProfessorModel | None = None

    @property
    def queue_depth(self) -> int:
        with self._queue_lock:
            return self._queue_depth

    def create_demo_session(self, case_id: str) -> AnalysisSession:
        case = get_demo_case(case_id, self.config.xraydar_backend_dir)
        path = Path(case.image_path)
        if not path.exists():
            raise FileNotFoundError("Demo data is not prepared.")
        return self._create_session(
            title=case.title,
            source="demo",
            paths=[path],
            case_id=case.id,
            reference=case.reference,
        )

    def create_upload_session(
        self,
        uploads: list[tuple[str, bytes]],
    ) -> AnalysisSession:
        if not uploads:
            raise ValueError("Upload at least one image, DICOM file, or ZIP study.")
        total = sum(len(content) for _, content in uploads)
        if total > self.config.max_upload_mb * 1024 * 1024:
            raise ValueError(f"Uploads may not exceed {self.config.max_upload_mb} MB.")
        session_id = uuid4().hex
        work_dir = self._workspace / session_id
        work_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for index, (filename, content) in enumerate(uploads):
            safe_name = Path(filename or f"upload-{index}").name
            target = work_dir / f"{index:02d}-{safe_name}"
            target.write_bytes(content)
            paths.append(target)
        return self._create_session(
            session_id=session_id,
            title=Path(uploads[0][0] or "Uploaded chest study").stem,
            source="upload",
            paths=paths,
            work_dir=work_dir,
        )

    def get_session(self, session_id: str) -> AnalysisSession:
        return self._record(session_id).public

    def delete_session(self, session_id: str) -> None:
        with self._sessions_lock:
            record = self._sessions.pop(session_id, None)
        if record:
            shutil.rmtree(record.work_dir, ignore_errors=True)

    def render_image(
        self,
        session_id: str,
        image_id: str,
        *,
        center: float | None = None,
        width: float | None = None,
        invert: bool = False,
    ) -> Image.Image:
        record = self._record(session_id)
        try:
            image = record.study.images[image_id]
        except KeyError as exc:
            raise KeyError("Unknown study image.") from exc
        return image.render(center=center, width=width, invert=invert)

    def analyze_stream(
        self,
        session_id: str,
        observation: str,
    ) -> Iterator[dict[str, Any]]:
        blind_read = observation.strip()
        if not blind_read:
            raise ValueError("Commit a blind interpretation before revealing evidence.")
        record = self._record(session_id)
        record.public.status = "analyzing"
        record.public.blind_read = blind_read
        record.public.updated_at = datetime.now(UTC)
        yield {"type": "session", "status": "analyzing"}

        try:
            with self._gpu_slot() as queue_position:
                yield {
                    "type": "queue",
                    "position": queue_position,
                    "message": "GPU inference slot acquired.",
                }
                primary_record = record.study.primary()
                primary = primary_record.render()
                evidence_model = self._evidence_model()
                yield {
                    "type": "stage",
                    "stage": "xraydar",
                    "status": "running",
                    "message": "Running independent X-Raydar classification.",
                }
                evidence = evidence_model.analyze(primary)
                for region in evidence.regions:
                    if region.image_id is None:
                        region.image_id = primary_record.public.id
                yield {
                    "type": "stage",
                    "stage": "xraydar",
                    "status": "complete",
                    "model_run": (
                        evidence.model_runs[-1].model_dump(mode="json")
                        if evidence.model_runs
                        else None
                    ),
                }

                if self.config.enable_medical_vlm and self.config.model_mode != "demo":
                    yield from self._run_localizer(primary, primary_record.public.id, evidence)
                    yield {
                        "type": "stage",
                        "stage": "professor-load",
                        "status": "running",
                        "message": "Loading MedGemma 27B professor.",
                    }
                    self._switch_model(
                        unload=self.config.localizer_model,
                        load=self.config.professor_model,
                    )
                    yield {
                        "type": "stage",
                        "stage": "professor-load",
                        "status": "complete",
                    }

                student_read = StudentRead(observation=blind_read)
                scorecard = build_scorecard(evidence, student_read)
                reference = record.study.public.reference
                yield {
                    "type": "stage",
                    "stage": "professor",
                    "status": "running",
                    "message": "Professor is comparing the committed read with the evidence.",
                }
                tutor = self._coach(
                    evidence=evidence,
                    student_read=student_read,
                    images=self._professor_images(record.study),
                    reference=reference,
                )
                yield {
                    "type": "stage",
                    "stage": "professor",
                    "status": "complete",
                    "model_run": (
                        tutor.model_run.model_dump(mode="json") if tutor.model_run else None
                    ),
                }
                result = AnalysisResult(
                    session_id=session_id,
                    case_id=record.study.public.case_id,
                    primary_image_id=primary_record.public.id,
                    image_data_url=_image_data_url(primary),
                    evidence=evidence,
                    scorecard=scorecard,
                    tutor=tutor,
                    reference=reference,
                    session_note=format_session_note(
                        evidence, scorecard, student_read, tutor
                    ),
                )
                assistant = ChatMessage(
                    role="assistant",
                    content=tutor.summary,
                    model_run=tutor.model_run,
                    evidence_sources=["X-Raydar", "MedGemma 1.5 4B", "MedGemma 27B"],
                    region_ids=[item.id for item in evidence.regions],
                )
                record.public.result = result
                record.public.messages = [assistant]
                record.public.status = "complete"
                record.public.updated_at = datetime.now(UTC)
                yield {
                    "type": "complete",
                    "session": record.public.model_dump(mode="json"),
                }
        except Exception as exc:
            record.public.status = "error"
            record.public.updated_at = datetime.now(UTC)
            yield {"type": "error", "message": str(exc)}

    def chat_stream(
        self,
        session_id: str,
        question: str,
    ) -> Iterator[dict[str, Any]]:
        prompt = question.strip()
        if not prompt:
            raise ValueError("Enter a question for the professor.")
        record = self._record(session_id)
        if not record.public.result:
            raise ValueError("Complete the blind-read analysis before starting chat.")
        user_message = ChatMessage(role="user", content=prompt)
        record.public.messages.append(user_message)
        yield {"type": "message", "message": user_message.model_dump(mode="json")}

        chunks: list[str] = []
        metrics: dict[str, Any] = {}
        try:
            with self._gpu_slot() as queue_position:
                yield {"type": "queue", "position": queue_position}
                if self.config.model_mode == "demo":
                    text = (
                        "In this demo, compare the question with the committed blind read and "
                        "the independently listed evidence. Focus on observation, interpretation, "
                        "and uncertainty as separate steps."
                    )
                    for chunk in text.split(" "):
                        value = f"{chunk} "
                        chunks.append(value)
                        yield {"type": "delta", "content": value}
                    metrics = {"latency_ms": 1}
                else:
                    self._switch_model(
                        unload=self.config.localizer_model,
                        load=self.config.professor_model,
                    )
                    professor = self._professor_model()
                    for content, stream_metrics in professor.stream_reply(
                        evidence=record.public.result.evidence,
                        blind_read=record.public.blind_read,
                        question=prompt,
                        history=record.public.messages[:-1],
                        images=self._professor_images(record.study),
                        reference=record.study.public.reference,
                    ):
                        if content:
                            chunks.append(content)
                            yield {"type": "delta", "content": content}
                        if stream_metrics:
                            metrics = stream_metrics
        except Exception as exc:
            yield {"type": "error", "message": str(exc)}
            return

        run = ModelRun(
            role="professor-chat",
            model_id=(
                self.config.professor_model
                if self.config.model_mode != "demo"
                else "demo-professor"
            ),
            model_source=(
                PROFESSOR_REPO if self.config.model_mode != "demo" else "demo"
            ),
            model_revision=(
                self.config.professor_revision
                if self.config.model_mode != "demo"
                else ""
            ),
            runtime="llama.cpp" if self.config.model_mode != "demo" else "demo",
            runtime_revision=(
                LLAMA_CPP_BUILD if self.config.model_mode != "demo" else ""
            ),
            status="ok",
            latency_ms=int(metrics.get("latency_ms") or 0),
            time_to_first_token_ms=metrics.get("time_to_first_token_ms"),
            tokens_per_second=metrics.get("tokens_per_second"),
        )
        assistant = ChatMessage(
            role="assistant",
            content="".join(chunks).strip(),
            model_run=run,
            evidence_sources=["image", "X-Raydar", "MedGemma localization"],
        )
        record.public.messages.append(assistant)
        record.public.updated_at = datetime.now(UTC)
        yield {"type": "complete", "message": assistant.model_dump(mode="json")}

    def _create_session(
        self,
        *,
        title: str,
        source: str,
        paths: list[Path],
        session_id: str | None = None,
        work_dir: Path | None = None,
        case_id: str | None = None,
        reference=None,
    ) -> AnalysisSession:
        self._cleanup_expired()
        identifier = session_id or uuid4().hex
        directory = work_dir or self._workspace / identifier
        directory.mkdir(parents=True, exist_ok=True)
        study = load_study(
            study_id=uuid4().hex,
            title=title,
            source=source,
            paths=paths,
            work_dir=directory,
            case_id=case_id,
            reference=reference,
            max_files=self.config.max_study_files,
            max_uncompressed_mb=self.config.max_study_uncompressed_mb,
        )
        for item in study.public.images:
            item.image_url = f"/api/sessions/{identifier}/images/{item.id}"
        now = datetime.now(UTC)
        public = AnalysisSession(
            id=identifier,
            status="ready",
            study=study.public,
            created_at=now,
            updated_at=now,
        )
        with self._sessions_lock:
            self._sessions[identifier] = _SessionRecord(
                public=public,
                study=study,
                work_dir=directory,
            )
        return public

    def _record(self, session_id: str) -> _SessionRecord:
        self._cleanup_expired()
        with self._sessions_lock:
            try:
                return self._sessions[session_id]
            except KeyError as exc:
                raise KeyError("Unknown or expired session.") from exc

    def _cleanup_expired(self) -> None:
        threshold = datetime.now(UTC) - timedelta(minutes=self.config.session_ttl_minutes)
        with self._sessions_lock:
            expired = [
                session_id
                for session_id, record in self._sessions.items()
                if record.public.updated_at < threshold
            ]
        for session_id in expired:
            self.delete_session(session_id)

    @contextmanager
    def _gpu_slot(self) -> Iterator[int]:
        with self._queue_lock:
            self._queue_depth += 1
            position = self._queue_depth
        try:
            with self._gpu_lock:
                yield position
        finally:
            with self._queue_lock:
                self._queue_depth = max(0, self._queue_depth - 1)

    def _evidence_model(self):
        if self.config.model_mode == "demo":
            return DemoChestEvidenceModel()
        if self._xraydar is None:
            self._xraydar = XRaydarEvidenceModel(
                backend_dir=Path(self.config.xraydar_backend_dir),
                device_name=self.config.xraydar_device,
                model_revision=self.config.xraydar_revision,
                offload_after_inference=True,
            )
        return self._xraydar

    def _professor_model(self) -> MedGemmaProfessorModel:
        if self._professor is None:
            self._professor = MedGemmaProfessorModel(
                config=self.config,
                client=self._client,
            )
        return self._professor

    def _coach(self, *, evidence, student_read, images, reference):
        if self.config.model_mode == "demo":
            return DemoTutorModel().coach(evidence, student_read)
        return self._professor_model().coach(
            evidence,
            student_read,
            images=images,
            reference=reference,
        )

    def _run_localizer(self, image: Image.Image, image_id: str, evidence) -> Iterator[dict]:
        yield {
            "type": "stage",
            "stage": "localizer-load",
            "status": "running",
            "message": "Loading MedGemma 1.5 4B localizer.",
        }
        self._switch_model(
            unload=self.config.professor_model,
            load=self.config.localizer_model,
        )
        yield {
            "type": "stage",
            "stage": "localizer-load",
            "status": "complete",
        }
        if self._localizer is None:
            self._localizer = MedGemmaVisionTool(
                client=self._client,
                model_id=self.config.localizer_model,
                model_revision=self.config.localizer_revision,
            )
        observations, regions, uncertainty, run = self._localizer.analyze(image, evidence)
        for region in regions:
            region.image_id = image_id
        evidence.observations.extend(observations)
        evidence.regions.extend(regions)
        evidence.agreement_notes.extend(uncertainty)
        evidence.model_runs.append(run)
        if run.status == "error":
            evidence.model_notes.append(f"MedGemma localization rejected: {run.detail}")
        yield {
            "type": "stage",
            "stage": "localizer",
            "status": run.status,
            "model_run": run.model_dump(mode="json"),
            "message": run.detail,
        }

    def _switch_model(self, *, unload: str, load: str) -> None:
        try:
            self._client.unload_model(unload)
        except Exception:
            pass
        self._client.load_model(load)

    @staticmethod
    def _professor_images(study: StudyRecord) -> list[Image.Image]:
        primary = study.primary()
        images = [primary.render()]
        lateral = next(
            (
                item
                for item in study.ordered()
                if item.public.id != primary.public.id
                and "LAT" in item.public.projection.upper()
            ),
            None,
        )
        if lateral:
            images.append(lateral.render())
        return images


def _image_data_url(image: Image.Image) -> str:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"

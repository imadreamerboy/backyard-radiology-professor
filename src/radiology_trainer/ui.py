from __future__ import annotations

import json
import os
import subprocess
from io import BytesIO
from pathlib import Path
from typing import Any, Iterator

import gradio as gr
from fastapi import File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from radiology_trainer.adapters.llama_client import LlamaCppClient
from radiology_trainer.cases import demo_cases
from radiology_trainer.config import AppConfig
from radiology_trainer.runtime_manifest import (
    LLAMA_CPP_BUILD,
    LOCALIZER_QUANTIZATION,
    LOCALIZER_REPO,
    PROFESSOR_QUANTIZATION,
    PROFESSOR_REPO,
    XRAYDAR_CODE_REVISION,
    XRAYDAR_REPO,
)
from radiology_trainer.service import TrainerService


ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = ROOT / "static"


class AnalyzeRequest(BaseModel):
    observation: str = Field(min_length=1, max_length=8000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


class FeedbackRequest(BaseModel):
    session_id: str | None = None
    case_id: str | None = None
    rating: int = Field(ge=1, le=5)
    comment: str = Field(default="", max_length=1000)


def create_server(config: AppConfig | None = None) -> gr.Server:
    cfg = config or AppConfig.from_env()
    service = TrainerService(cfg)
    server = gr.Server(
        title="Backyard Radiology Professor",
        description="Educational chest radiograph practice workstation",
        docs_url="/api/docs",
        redoc_url=None,
    )
    server.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @server.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @server.get("/api/cases")
    def cases() -> list[dict[str, Any]]:
        return [
            {
                "id": case.id,
                "title": case.title,
                "difficulty": case.difficulty,
                "available": Path(case.image_path).exists(),
                "reference_source": case.reference.source,
            }
            for case in demo_cases(cfg.xraydar_backend_dir)
        ]

    @server.post("/api/sessions")
    async def create_session(
        case_id: str | None = Form(default=None),
        files: list[UploadFile] = File(default=[]),
    ) -> JSONResponse:
        try:
            if case_id:
                session = service.create_demo_session(case_id)
            else:
                uploads = [
                    (upload.filename or "upload", await upload.read())
                    for upload in files
                    if upload.filename
                ]
                session = service.create_upload_session(uploads)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Unknown demo case.") from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return JSONResponse(session.model_dump(mode="json"))

    @server.get("/api/sessions/{session_id}")
    def get_session(session_id: str) -> JSONResponse:
        try:
            session = service.get_session(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return JSONResponse(session.model_dump(mode="json"))

    @server.delete("/api/sessions/{session_id}")
    def delete_session(session_id: str) -> dict[str, bool]:
        service.delete_session(session_id)
        return {"deleted": True}

    @server.get("/api/sessions/{session_id}/images/{image_id}")
    def study_image(
        session_id: str,
        image_id: str,
        center: float | None = None,
        width: float | None = None,
        invert: bool = False,
    ) -> Response:
        try:
            image = service.render_image(
                session_id,
                image_id,
                center=center,
                width=width,
                invert=invert,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        with BytesIO() as buffer:
            image.save(buffer, format="PNG")
            return Response(
                content=buffer.getvalue(),
                media_type="image/png",
                headers={"Cache-Control": "private, max-age=60"},
            )

    @server.post("/api/sessions/{session_id}/analyze")
    def analyze_session(session_id: str, request: AnalyzeRequest) -> StreamingResponse:
        try:
            events = service.analyze_stream(session_id, request.observation)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return StreamingResponse(
            _sse(events),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @server.post("/api/sessions/{session_id}/chat")
    def chat_session(session_id: str, request: ChatRequest) -> StreamingResponse:
        try:
            events = service.chat_stream(session_id, request.message)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return StreamingResponse(
            _sse(events),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @server.get("/api/status")
    def status() -> dict[str, Any]:
        required = [cfg.professor_model, cfg.localizer_model]
        model_states: list[dict[str, Any]] = []
        detail = ""
        if cfg.model_mode == "demo":
            runtime_status = "demo"
        else:
            llama = LlamaCppClient(
                base_url=cfg.llama_base_url,
                api_key=cfg.llama_api_key,
                timeout_seconds=10,
            )
            try:
                models = llama.list_models()
                known = {str(item.get("id", "")) for item in models}
                model_states = [
                    {
                        "id": str(item.get("id", "")),
                        "status": (item.get("status") or {}).get("value", "unknown"),
                        "architecture": item.get("architecture") or {},
                    }
                    for item in models
                ]
                missing = set(required) - known
                runtime_status = "ready" if not missing else "loading"
                if missing:
                    detail = f"Waiting for model presets: {', '.join(sorted(missing))}"
            except Exception as exc:
                runtime_status = "unavailable"
                detail = str(exc)
        return {
            "mode": cfg.model_mode,
            "runtime": "llama.cpp",
            "runtime_revision": LLAMA_CPP_BUILD,
            "runtime_status": runtime_status,
            "models": model_states,
            "model_revisions": [
                {
                    "id": cfg.professor_model,
                    "source": PROFESSOR_REPO,
                    "revision": cfg.professor_revision,
                    "quantization": PROFESSOR_QUANTIZATION,
                },
                {
                    "id": cfg.localizer_model,
                    "source": LOCALIZER_REPO,
                    "revision": cfg.localizer_revision,
                    "quantization": LOCALIZER_QUANTIZATION,
                },
            ],
            "required_models": required,
            "models_max": 1,
            "queue_depth": service.queue_depth,
            "xraydar_available": (
                cfg.model_mode == "demo" or Path(cfg.xraydar_backend_dir).exists()
            ),
            "xraydar": {
                "source": XRAYDAR_REPO,
                "weights_revision": cfg.xraydar_revision,
                "code_revision": XRAYDAR_CODE_REVISION,
            },
            "gpu": _gpu_status(),
            "detail": detail,
        }

    @server.post("/api/feedback")
    def feedback(request: FeedbackRequest) -> dict[str, Any]:
        return {
            "accepted": True,
            "session_id": request.session_id,
            "case_id": request.case_id,
            "rating": request.rating,
            "comment": request.comment.strip(),
            "persisted": False,
        }

    # Kept while command-line validators migrate to the session API.
    @server.post("/api/analyze")
    async def compatibility_analyze(
        case_id: str | None = Form(default=None),
        observation: str = Form(...),
        question: str = Form(default=""),
        image: UploadFile | None = File(default=None),
    ) -> JSONResponse:
        try:
            if case_id:
                session = service.create_demo_session(case_id)
            elif image:
                session = service.create_upload_session(
                    [(image.filename or "upload", await image.read())]
                )
            else:
                raise ValueError("Select a case or upload an image.")
            completed = None
            for event in service.analyze_stream(session.id, observation):
                if event.get("type") == "error":
                    raise RuntimeError(str(event.get("message")))
                if event.get("type") == "complete":
                    completed = service.get_session(session.id).result
            if completed is None:
                raise RuntimeError("Analysis did not return a completed result.")
            if question.strip():
                for _ in service.chat_stream(session.id, question):
                    pass
            return JSONResponse(completed.model_dump(mode="json"))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @server.api(name="runtime_status", queue=False)
    def runtime_status() -> str:
        return f"{cfg.model_mode}: llama.cpp + MedGemma + X-Raydar"

    return server


def launch_server(config: AppConfig | None = None) -> None:
    server = create_server(config)
    server.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        show_error=True,
        max_file_size=f"{(config or AppConfig.from_env()).max_upload_mb}mb",
        _frontend=False,
    )


def _sse(events: Iterator[dict[str, Any]]) -> Iterator[str]:
    try:
        for event in events:
            event_type = str(event.get("type", "message"))
            yield f"event: {event_type}\ndata: {json.dumps(event, ensure_ascii=True)}\n\n"
    except Exception as exc:
        payload = json.dumps({"type": "error", "message": str(exc)}, ensure_ascii=True)
        yield f"event: error\ndata: {payload}\n\n"


def _gpu_status() -> dict[str, Any] | None:
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        name, used, total, utilization = [item.strip() for item in output.splitlines()[0].split(",")]
        return {
            "name": name,
            "memory_used_mb": int(used),
            "memory_total_mb": int(total),
            "utilization_percent": int(utilization),
        }
    except Exception:
        return None

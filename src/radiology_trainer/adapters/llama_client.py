from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Iterator

import requests
from PIL import Image


@dataclass(frozen=True)
class LlamaCppClient:
    base_url: str
    api_key: str = "local"
    timeout_seconds: float = 180.0

    def list_models(self) -> list[dict[str, Any]]:
        response = requests.get(
            f"{self._router_url()}/models",
            headers=self._headers(),
            timeout=min(self.timeout_seconds, 30),
        )
        response.raise_for_status()
        return list(response.json().get("data", []))

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float = 0.0,
        json_schema: dict[str, Any] | None = None,
        enable_thinking: bool | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if json_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_response",
                    "strict": True,
                    "schema": json_schema,
                },
            }
        if enable_thinking is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}

        response = requests.post(
            f"{self.base_url.rstrip('/')}/chat/completions",
            headers=self._headers(),
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        message = response.json()["choices"][0]["message"]
        return str(message.get("content") or message.get("reasoning") or "").strip()

    def stream_chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float = 0.2,
    ) -> Iterator[tuple[str, dict[str, Any]]]:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        started = time.perf_counter()
        first_token_at: float | None = None
        usage: dict[str, Any] = {}
        with requests.post(
            f"{self.base_url.rstrip('/')}/chat/completions",
            headers=self._headers(),
            json=payload,
            timeout=self.timeout_seconds,
            stream=True,
        ) as response:
            response.raise_for_status()
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line or not raw_line.startswith("data:"):
                    continue
                data = raw_line[5:].strip()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                usage = event.get("usage") or usage
                choices = event.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                content = str(delta.get("content") or "")
                if content:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    yield content, {}

        elapsed = max(time.perf_counter() - started, 0.001)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        yield "", {
            "latency_ms": int(elapsed * 1000),
            "time_to_first_token_ms": (
                int((first_token_at - started) * 1000) if first_token_at else None
            ),
            "tokens_per_second": completion_tokens / elapsed if completion_tokens else None,
        }

    def load_model(self, model: str) -> None:
        if self._model_status(model) == "loaded":
            return
        response = requests.post(
            f"{self._router_url()}/models/load",
            headers=self._headers(),
            json={"model": model},
            timeout=self.timeout_seconds,
        )
        if response.status_code == 400 and self._model_status(model) == "loaded":
            return
        response.raise_for_status()

    def unload_model(self, model: str) -> None:
        if self._model_status(model) != "loaded":
            return
        response = requests.post(
            f"{self._router_url()}/models/unload",
            headers=self._headers(),
            json={"model": model},
            timeout=min(self.timeout_seconds, 60),
        )
        if response.status_code == 400 and self._model_status(model) != "loaded":
            return
        if response.status_code not in {200, 404, 409}:
            response.raise_for_status()

    def image_url(self, image: Image.Image) -> str:
        buffer = BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _router_url(self) -> str:
        value = self.base_url.rstrip("/")
        return value[:-3] if value.endswith("/v1") else value

    def _model_status(self, model: str) -> str | None:
        for item in self.list_models():
            if item.get("id") != model:
                continue
            status = item.get("status")
            if isinstance(status, dict):
                return str(status.get("value") or "")
            return str(status or "")
        return None

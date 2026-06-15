from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import requests
from fastapi import HTTPException, Request
from fastapi.responses import Response, StreamingResponse


HOP_BY_HOP_HEADERS = {
    "connection",
    "content-encoding",
    "content-length",
    "host",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


class RemoteBackendProxy:
    def __init__(self, base_url: str, *, timeout_seconds: float = 1800.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    async def forward(self, path: str, request: Request) -> Response:
        url = f"{self.base_url}/api/{path}"
        try:
            upstream = requests.request(
                request.method,
                url,
                params=list(request.query_params.multi_items()),
                data=(await request.body()) or None,
                headers=_forward_headers(request.headers),
                stream=True,
                timeout=(20, self.timeout_seconds),
            )
        except requests.RequestException as exc:
            raise HTTPException(status_code=502, detail=f"Remote backend unavailable: {exc}") from exc

        response_headers = _response_headers(upstream.headers)
        content_type = upstream.headers.get("content-type", "")
        if content_type.startswith("text/event-stream"):
            return StreamingResponse(
                _stream_response(upstream),
                status_code=upstream.status_code,
                media_type=content_type,
                headers=response_headers,
            )

        try:
            content = upstream.content
        finally:
            upstream.close()
        return Response(
            content=content,
            status_code=upstream.status_code,
            media_type=content_type or None,
            headers=response_headers,
        )


def _forward_headers(headers: Any) -> dict[str, str]:
    return {
        key: value
        for key, value in headers.items()
        if key.lower() not in HOP_BY_HOP_HEADERS
    }


def _response_headers(headers: Any) -> dict[str, str]:
    return {
        key: value
        for key, value in headers.items()
        if key.lower() not in HOP_BY_HOP_HEADERS
    }


def _stream_response(response: requests.Response) -> Iterator[bytes]:
    try:
        for chunk in response.iter_content(chunk_size=None):
            if chunk:
                yield chunk
    finally:
        response.close()

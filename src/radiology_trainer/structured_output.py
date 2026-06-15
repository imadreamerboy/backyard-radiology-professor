from __future__ import annotations

import json
import re
from typing import TypeVar

from pydantic import BaseModel


T = TypeVar("T", bound=BaseModel)


def parse_json_model(text: str, model_type: type[T]) -> T:
    candidate = _extract_json(text)
    return model_type.model_validate(json.loads(candidate))


def parse_json_value(text: str):
    return json.loads(_extract_json(text))


def _extract_json(text: str) -> str:
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1).strip()

    candidates = [
        (cleaned.find("{"), cleaned.rfind("}")),
        (cleaned.find("["), cleaned.rfind("]")),
    ]
    candidates = [(start, end) for start, end in candidates if start >= 0 and end > start]
    if candidates:
        start, end = min(candidates, key=lambda item: item[0])
        return cleaned[start : end + 1]

    raise ValueError("Model response did not contain JSON.")

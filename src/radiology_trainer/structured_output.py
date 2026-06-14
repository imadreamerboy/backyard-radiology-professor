from __future__ import annotations

import json
import re
from typing import TypeVar

from pydantic import BaseModel


T = TypeVar("T", bound=BaseModel)


def parse_json_model(text: str, model_type: type[T]) -> T:
    candidate = _extract_json(text)
    return model_type.model_validate(json.loads(candidate))


def _extract_json(text: str) -> str:
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1).strip()

    object_start = cleaned.find("{")
    object_end = cleaned.rfind("}")
    if object_start >= 0 and object_end > object_start:
        return cleaned[object_start : object_end + 1]

    raise ValueError("Model response did not contain a JSON object.")

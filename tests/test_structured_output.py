from __future__ import annotations

import pytest

from radiology_trainer.adapters.medgemma import _RawBox, _convert_box
from radiology_trainer.structured_output import _extract_json


def test_extract_json_accepts_fenced_output() -> None:
    assert _extract_json('```json\n{"ok": true}\n```') == '{"ok": true}'


def test_extract_json_rejects_plain_text() -> None:
    with pytest.raises(ValueError):
        _extract_json("No structured output")


def test_medgemma_box_converts_yx_coordinates() -> None:
    box = _convert_box(
        _RawBox(box_2d=[100, 200, 700, 800], label="heart"),
        (1000, 1000),
    )
    assert box.x1 == 0.2
    assert box.y1 == 0.1
    assert box.x2 == 0.8
    assert box.y2 == 0.7


def test_medgemma_box_removes_square_padding() -> None:
    box = _convert_box(
        _RawBox(box_2d=[200, 200, 800, 800], label="heart"),
        (800, 1000),
    )
    assert box.x1 == pytest.approx(0.125)
    assert box.y1 == pytest.approx(0.2)
    assert box.x2 == pytest.approx(0.875)
    assert box.y2 == pytest.approx(0.8)


def test_medgemma_box_rejects_reversed_coordinates() -> None:
    with pytest.raises(ValueError):
        _convert_box(
            _RawBox(box_2d=[700, 800, 100, 200], label="heart"),
            (1000, 1000),
        )

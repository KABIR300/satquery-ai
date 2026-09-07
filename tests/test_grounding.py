import json

import numpy as np
import pytest

from satquery.backends.base import OutputError
from satquery.grounding import parse_optical, validate_mask


def response(coords, space="normalized", **extra):
    return json.dumps(
        {
            "answer": "Candidate",
            "coordinate_space": space,
            "boxes": [{"label": "region", "bbox": coords}],
            **extra,
        }
    )


@pytest.mark.parametrize(
    "space,coords",
    [
        ("normalized", [0.1, 0.2, 0.5, 0.8]),
        ("normalized_1000", [100, 200, 500, 800]),
        ("pixels", [20, 20, 100, 80]),
    ],
)
def test_coordinate_spaces(space, coords):
    prediction = parse_optical(response(coords, space), 200, 100, "test")
    box = prediction.boxes[0]
    assert (box.x1, box.y1, box.x2, box.y2) == (20, 20, 100, 80)


@pytest.mark.parametrize(
    "coords",
    [
        [0.5, 0.2, 0.1, 0.8],
        [-0.1, 0, 0.5, 0.8],
        [0, 0, 1.1, 1],
        [0, 0, 0, 1],
        [0, 0, float("nan"), 1],
        ["0", 0, 1, 1],
        [False, 0, 1, 1],
        [0, 1],
    ],
)
def test_bad_boxes_rejected(coords):
    with pytest.raises(OutputError):
        parse_optical(response(coords), 100, 100, "test")


@pytest.mark.parametrize(
    "text",
    [
        "",
        "Some prose",
        "{}",
        "[]",
        '{"answer":"x","boxes":[]}',
        '{"answer":"","coordinate_space":"pixels","boxes":[]}',
    ],
)
def test_bad_model_output(text):
    with pytest.raises(OutputError):
        parse_optical(text, 100, 100, "test")


def test_confidence_ignored_and_json_fence():
    prediction = parse_optical(
        "```json\n" + response([0, 0, 1, 1], confidence=0.999) + "\n```", 100, 100, "test"
    )
    assert "ignored" in prediction.warnings[0]
    assert not hasattr(prediction, "confidence")


def test_mask_handling():
    mask = np.zeros((5, 6), bool)
    assert validate_mask(mask, (5, 6)) is mask
    with pytest.raises(OutputError):
        validate_mask(mask.astype(np.uint8), (5, 6))
    with pytest.raises(OutputError):
        validate_mask(mask, (6, 5))
    mask[1, 1] = True
    with pytest.raises(OutputError):
        validate_mask(mask, (5, 6), np.zeros((5, 6), bool))

"""Fail-closed parsing and spatial output validation."""

import json
import math

import numpy as np

from satquery.backends.base import OpticalPrediction, OutputError
from satquery.schemas import Box


def validate_boxes(boxes: list[Box], width: int, height: int) -> list[Box]:
    if len(boxes) > 100:
        raise OutputError("Too many bounding boxes (maximum 100).")
    for box in boxes:
        if not isinstance(box, Box) or not (0 <= box.x1 < box.x2 <= width and 0 <= box.y1 < box.y2 <= height):
            raise OutputError("A bounding box lies outside the image or is malformed.")
    return boxes


def parse_optical(text: str, width: int, height: int, backend: str) -> OpticalPrediction:
    if not isinstance(text, str) or not text.strip() or len(text) > 50_000:
        raise OutputError("Empty or excessively long model response.")
    cleaned = text.strip()
    if cleaned.startswith("```json\n") and cleaned.endswith("```"):
        cleaned = cleaned[8:-3].strip()
    elif cleaned.startswith("```\n") and cleaned.endswith("```"):
        cleaned = cleaned[4:-3].strip()
    try:
        data = json.loads(cleaned)
        if (
            not isinstance(data, dict)
            or not isinstance(data.get("answer"), str)
            or not data["answer"].strip()
        ):
            raise ValueError("Response must contain a nonempty answer string.")
        if len(data["answer"]) > 5000:
            raise ValueError("Answer is too long.")
        space = data.get("coordinate_space")
        if space not in {"pixels", "normalized", "normalized_1000"}:
            raise ValueError("Explicit coordinate_space is required.")
        raw_boxes = data.get("boxes")
        if not isinstance(raw_boxes, list) or len(raw_boxes) > 100:
            raise ValueError("boxes must be a list of at most 100 regions.")
        boxes = []
        for raw in raw_boxes:
            if not isinstance(raw, dict) or not isinstance(raw.get("label"), str):
                raise ValueError("Each box requires a label and bbox.")
            coords = raw.get("bbox")
            if not isinstance(coords, list) or len(coords) != 4:
                raise ValueError("bbox requires four coordinates.")
            if any(type(v) not in {int, float} or not math.isfinite(v) for v in coords):
                raise ValueError("Coordinates must be finite numbers, not strings or booleans.")
            divisor = 1 if space == "normalized" else 1000
            if space != "pixels":
                if any(v < 0 or v > divisor for v in coords):
                    raise ValueError("Normalized coordinate is outside its declared range.")
                coords = [
                    coords[0] * width / divisor,
                    coords[1] * height / divisor,
                    coords[2] * width / divisor,
                    coords[3] * height / divisor,
                ]
            boxes.append(Box(x1=coords[0], y1=coords[1], x2=coords[2], y2=coords[3], label=raw["label"]))
        validate_boxes(boxes, width, height)
    except (ValueError, TypeError, KeyError) as exc:
        raise OutputError(f"Invalid structured model output: {exc}") from exc
    notes = ["Model self-reported confidence was ignored."] if "confidence" in data else []
    return OpticalPrediction(data["answer"].strip(), boxes, backend, warnings=notes)


def validate_mask(mask: np.ndarray, shape: tuple[int, int], valid: np.ndarray | None = None):
    if not isinstance(mask, np.ndarray) or mask.shape != shape or mask.dtype != bool:
        raise OutputError("Mask must be a boolean array matching the image grid.")
    if valid is not None and (valid.shape != shape or valid.dtype != bool or np.any(mask & ~valid)):
        raise OutputError("Mask contains changes outside the valid data footprint.")
    return mask

"""Render validated spatial evidence, without asking the UI to parse model text."""

import io

import numpy as np
from PIL import Image, ImageDraw

from satquery.grounding import validate_boxes, validate_mask


def overlay(rgb: np.ndarray, boxes=(), mask: np.ndarray | None = None) -> Image.Image:
    canvas = rgb.copy()
    h, w = canvas.shape[:2]
    if mask is not None:
        validate_mask(mask, (h, w))
        canvas[mask] = (0.5 * canvas[mask] + 0.5 * np.array([255, 96, 77])).astype(np.uint8)
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    validate_boxes(list(boxes), w, h)
    for box in boxes:
        draw.rectangle((box.x1, box.y1, min(box.x2, w - 1), min(box.y2, h - 1)), outline="#5ED3BB", width=2)
        label = box.label[:64]
        left, top = max(0, int(box.x1)), max(0, int(box.y1) - 16)
        draw.rectangle((left, top, min(w, left + len(label) * 7 + 8), top + 16), fill="#102329")
        draw.text((left + 3, top + 1), label, fill="white")
    return image


def png_bytes(image: Image.Image | np.ndarray) -> bytes:
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()

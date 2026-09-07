"""Tiny procedural fixtures, explicitly synthetic, with known changes and georeferencing."""

import json
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw
from rasterio.enums import ColorInterp
from rasterio.transform import from_origin


def generate_samples(directory: str | Path) -> Path:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(26167)
    base = np.clip(rng.normal(0, 10, (256, 256, 1)) + np.array([95, 126, 79]), 0, 255).astype(np.uint8)
    image = Image.fromarray(base)
    draw = ImageDraw.Draw(image)
    draw.polygon(
        [(25, 0), (72, 0), (99, 95), (63, 180), (84, 255), (43, 255), (29, 161), (60, 85)], fill=(40, 95, 150)
    )
    draw.line([(115, 0), (115, 256)], fill=(177, 174, 158), width=9)
    draw.line([(0, 151), (255, 151)], fill=(177, 174, 158), width=7)
    for x in (143, 175, 207):
        for y in (22, 55, 88):
            draw.rectangle((x, y, x + 18, y + 19), fill=(188, 177, 156))
    before = np.array(image)
    kinds = {
        "flood": ((85, 172, 155, 230), (32, 82, 148)),
        "vegetation": ((157, 173, 227, 231), (174, 126, 75)),
        "urban": ((151, 171, 218, 224), (202, 193, 180)),
        "no_change": (None, None),
    }
    cases = []
    for kind, (rect, color) in kinds.items():
        after = before.copy()
        truth = np.zeros((256, 256), np.uint8)
        if rect:
            x1, y1, x2, y2 = rect
            after[y1:y2, x1:x2] = color
            truth[y1:y2, x1:x2] = 255
        for phase, pixels, date in (("before", before, "2026-01-01"), ("after", after, "2026-01-15")):
            Image.fromarray(pixels).save(root / f"{kind}_{phase}.png")
            with rasterio.open(
                root / f"{kind}_{phase}.tif",
                "w",
                driver="GTiff",
                width=256,
                height=256,
                count=3,
                dtype="uint8",
                crs="EPSG:32643",
                transform=from_origin(500000, 2200000, 10, 10),
                compress="deflate",
            ) as ds:
                ds.write(pixels.transpose(2, 0, 1))
                ds.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
                ds.update_tags(
                    sensor="SYNTHETIC_OPTICAL",
                    modality="optical",
                    synthetic="true",
                    acquisition_date=date,
                    source="Procedural test fixture; dates/location are fictional",
                )
        Image.fromarray(truth).save(root / f"{kind}_mask.png")
        cases.append(
            {
                "id": f"synthetic-{kind}",
                "scene_id": kind,
                "synthetic": True,
                "images": [f"{kind}_before.tif", f"{kind}_after.tif"],
                "question": "Show appearance differences between these images.",
                "task": "change",
                "expected_status": "OK",
                "mask": f"{kind}_mask.png",
                "expected_boxes": [list(rect)] if rect else [],
            }
        )
    manifest = root / "eval.jsonl"
    manifest.write_text("\n".join(json.dumps(case) for case in cases) + "\n", encoding="utf-8")
    (root / "README.md").write_text(
        "# Synthetic demonstration pack\n\n"
        "All pixels are procedural fixtures created by `python -m satquery samples`. "
        "These are NOT satellite observations. Flood/vegetation/urban names describe fixture intent, "
        "not detected classes. Coordinates and dates are fictional, included to test metadata preservation. "
        "Ground-truth masks cover the deliberately modified rectangles. No-change repeats the same pixels. "
        "No benchmark accuracy or real-world claims may be inferred from this pack.\n",
        encoding="utf-8",
    )
    return manifest

"""An evidence bundle: result JSON, rendered overlay, masks and real CRS exports."""

import io
import json
import zipfile

import numpy as np
from affine import Affine
from rasterio.io import MemoryFile

from satquery.schemas import PipelineOutput
from satquery.visualization import overlay, png_bytes


def evidence_bundle(output: PipelineOutput) -> bytes:
    stream = io.BytesIO()
    result = output.result
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("result.json", result.model_dump_json(indent=2))
        archive.writestr(
            "README.txt",
            "SatQuery AI evidence export\n"
            "Read result.json for status, backend, provenance, metadata and limitations.\n"
            "Coordinates and masks refer to the loaded preview/window, not necessarily full resolution.\n"
            "Difference values are normalized appearance differences, NOT probabilities.\n"
            "valid.png: 255=valid, 0=nodata. change.tif: 0=no change, 1=change, 255=nodata.\n",
        )
        if output.images:
            target = output.images[-1]
            archive.writestr(
                "overlay.png", png_bytes(overlay(target.rgb, result.bounding_boxes, output.mask))
            )
            for i, image in enumerate(output.images):
                archive.writestr(f"source-preview-{i + 1}.png", png_bytes(image.rgb))
            if output.mask is not None:
                archive.writestr("change.png", png_bytes(output.mask.astype(np.uint8) * 255))
                archive.writestr("valid.png", png_bytes(output.valid_mask.astype(np.uint8) * 255))
                values = io.BytesIO()
                np.save(values, output.difference, allow_pickle=False)
                archive.writestr("difference.npy", values.getvalue())
                archive.writestr("difference.png", png_bytes((output.difference * 255).astype(np.uint8)))
                meta = target.metadata
                if meta.crs and meta.preview_transform:
                    arr = np.where(output.valid_mask, output.mask.astype(np.uint8), 255).astype(np.uint8)
                    with MemoryFile() as memory:
                        with memory.open(
                            driver="GTiff",
                            width=arr.shape[1],
                            height=arr.shape[0],
                            count=1,
                            dtype="uint8",
                            nodata=255,
                            crs=meta.crs,
                            transform=Affine(*meta.preview_transform),
                            compress="deflate",
                        ) as dataset:
                            dataset.write(arr, 1)
                            dataset.update_tags(method=result.backend, evidence_status=result.status.value)
                        archive.writestr("change.tif", memory.read())
        if result.geospatial:
            collection = {"type": "FeatureCollection", "features": [g["wgs84"] for g in result.geospatial]}
            archive.writestr("regions.geojson", json.dumps(collection, indent=2, allow_nan=False))
    return stream.getvalue()

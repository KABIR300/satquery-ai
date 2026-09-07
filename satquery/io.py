"""Bounded optical/raster loading with a transform for the actual preview grid."""

import hashlib
import io
import math
import re
import warnings
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageOps, UnidentifiedImageError
from rasterio.enums import ColorInterp, Resampling
from rasterio.io import MemoryFile
from rasterio.windows import Window

from satquery.config import Settings
from satquery.schemas import ImageMetadata, LoadedImage


class InputError(ValueError):
    pass


def _size(width: int, height: int, limit: int) -> tuple[int, int]:
    ratio = min(1, limit / max(width, height))
    return max(1, round(width * ratio)), max(1, round(height * ratio))


def sensor_metadata(tags: dict[str, str]) -> tuple[str | None, str, str | None]:
    lower = {k.lower(): v for k, v in tags.items()}
    sensor = next(
        (
            lower[k]
            for k in ("sensor", "sensor_id", "satellite", "platform", "spacecraft_name")
            if lower.get(k)
        ),
        None,
    )
    text = " ".join(
        str(v)
        for k, v in lower.items()
        if k
        in {
            "sensor",
            "sensor_id",
            "satellite",
            "platform",
            "spacecraft_name",
            "modality",
            "instrument",
            "polarization",
            "polarisation",
        }
    ).lower()
    if re.search(r"sentinel[- _]?1|\bsar\b|\bc[- ]?sar\b|\b(vv|vh|hh|hv)\b", text):
        modality = "sar"
    elif re.search(r"sentinel[- _]?2|landsat|\boptical\b|\bmsi\b", text):
        modality = "optical"
    else:
        modality = "unknown"
    date = next(
        (lower[k] for k in ("acquisition_date", "datetime", "date_acquired", "sensing_time") if lower.get(k)),
        None,
    )
    return sensor, modality, date


def _display(data: np.ndarray, valid: np.ndarray) -> np.ndarray:
    channels = data if data.shape[2] >= 3 else np.repeat(data[:, :, :1], 3, axis=2)
    out = np.zeros((*valid.shape, 3), dtype=np.uint8)
    for i in range(3):
        band = channels[:, :, i]
        # SAR may already be dB; percentile stretching is safe without assuming intensity/amplitude.
        values = band[valid]
        if not values.size:
            continue
        low, high = np.percentile(values, [2, 98])
        if high <= low:
            low, high = float(values.min()), float(values.max())
        if high > low:
            out[:, :, i] = np.clip((band - low) / (high - low) * 255, 0, 255).astype(np.uint8)
        else:
            out[:, :, i] = 127
    out[~valid] = 0
    return out


def load_image(
    source: str | Path | bytes,
    settings: Settings,
    *,
    name: str | None = None,
    modality: str = "auto",
    window: tuple[int, int, int, int] | None = None,
) -> LoadedImage:
    if modality not in {"auto", "optical", "sar"}:
        raise InputError("Modality must be auto, optical, or sar.")
    path = Path(source) if isinstance(source, (str, Path)) else None
    if path is not None:
        if not path.is_file():
            raise InputError("Input file does not exist.")
        source_name = path.name
        size = path.stat().st_size
    elif isinstance(source, bytes):
        size = len(source)
        source_name = Path(name or "upload").name
    else:
        raise InputError("Supply a local path or uploaded bytes.")
    suffix = Path(source_name).suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        raise InputError("Supported files: PNG, JPG/JPEG, TIFF/GeoTIFF.")
    # Local TIFF paths are streamed; compressed uploads/ordinary images are bounded.
    if size == 0 or (
        size > settings.max_upload_mb * 1024**2 and not (path is not None and suffix in {".tif", ".tiff"})
    ):
        raise InputError(f"Empty file or upload exceeds {settings.max_upload_mb} MiB.")
    try:
        if suffix in {".tif", ".tiff"}:
            return _load_raster(path or source, source_name, settings, modality, window)
        if window is not None:
            raise InputError("Window reads require a TIFF raster.")
        data = path.read_bytes() if path else source
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as opened:
                if opened.format not in {"PNG", "JPEG"}:
                    raise InputError("Image contents do not match a supported optical format.")
                if opened.width * opened.height > settings.max_source_pixels:
                    raise InputError("Image is too large to decode safely; use a TIFF window.")
                if min(opened.size) < 16:
                    raise InputError("Images must be at least 16 pixels in both dimensions.")
                oriented = ImageOps.exif_transpose(opened)
                width, height = oriented.size
                exif = opened.getexif()
                target = _size(width, height, settings.max_image_size)
                rgba = oriented.convert("RGBA").resize(target, Image.Resampling.BILINEAR)
                arr = np.asarray(rgba)
                valid = arr[:, :, 3] > 0
                rgb = arr[:, :, :3].copy()
                rgb[~valid] = 0
        metadata = ImageMetadata(
            source=source_name,
            source_sha256=hashlib.sha256(data).hexdigest(),
            width=width,
            height=height,
            preview_width=target[0],
            preview_height=target[1],
            band_count=3,
            modality="optical" if modality == "auto" else modality,
            acquisition_date=str(exif[36867]) if exif.get(36867) else None,
            dtype="uint8",
            selected_bands=[1, 2, 3],
            visualization="RGB, EXIF orientation applied",
            tags={"exif_date_source": "DateTimeOriginal"} if exif.get(36867) else {},
        )
        return LoadedImage(
            rgb,
            rgb.astype(np.float32) / 255,
            valid,
            metadata,
            [
                "No geospatial reference; coordinates are preview pixels.",
                "EXIF capture time, if present, is not independently verified satellite metadata.",
            ],
        )
    except InputError:
        raise
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        rasterio.errors.RasterioError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise InputError(f"Cannot decode image: {type(exc).__name__}. Check the file and bands.") from exc


def _load_raster(source, name, settings, modality, window):
    notes = []
    with ExitStack() as stack:
        stack.enter_context(rasterio.Env(GDAL_CACHEMAX=64 * 1024**2, GDAL_NUM_THREADS="1"))
        if isinstance(source, bytes):
            mem = stack.enter_context(MemoryFile(source))
            ds = stack.enter_context(mem.open())
            digest = hashlib.sha256(source).hexdigest()
        else:
            ds = stack.enter_context(rasterio.open(source))
            digest = None  # Avoid an additional full-file scan of a potentially huge raster.
        if ds.count == 0 or min(ds.width, ds.height) < 16:
            raise InputError("Raster is empty or dimensions are below 16 pixels.")
        if ds.driver != "GTiff":
            raise InputError("Only TIFF/GeoTIFF raster contents are supported.")
        tags = ds.tags()
        sensor, detected, date = sensor_metadata(tags)
        if detected == "sar" and modality == "optical":
            raise InputError("Metadata identifies SAR; it cannot be forced through the optical backend.")
        selected_modality = detected if modality == "auto" else modality
        if selected_modality == "unknown":
            notes.append("Sensor/modality missing: treating the preview as optical only; verify input type.")
        if selected_modality == "sar":
            bands = [1]
            notes.append("SAR percentile preview only; no radiometric calibration or SAR interpretation.")
        elif settings.rgb_bands:
            bands = list(settings.rgb_bands)
        elif all(c in ds.colorinterp for c in (ColorInterp.red, ColorInterp.green, ColorInterp.blue)):
            bands = [
                ds.colorinterp.index(c) + 1 for c in (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
            ]
        elif ds.count >= 3:
            bands = [1, 2, 3]
            notes.append("RGB assignment defaults to bands 1/2/3; set rgb_bands for this product.")
        else:
            bands = [1]
            notes.append("Single-band grayscale preview; multispectral semantics are unavailable.")
        if max(bands) > ds.count:
            raise InputError(f"Requested bands {bands} exceed raster band count {ds.count}.")
        win = Window(0, 0, ds.width, ds.height)
        if window is not None:
            if len(window) != 4 or any(type(v) is not int for v in window):
                raise InputError("Window is integer (column, row, width, height).")
            col, row, width, height = window
            if (
                min(col, row) < 0
                or min(width, height) < 16
                or col + width > ds.width
                or row + height > ds.height
            ):
                raise InputError("Window must be within the raster and at least 16 by 16 pixels.")
            win = Window(*window)
        out_w, out_h = _size(int(win.width), int(win.height), settings.max_image_size)
        raw = ds.read(
            bands,
            window=win,
            out_shape=(len(bands), out_h, out_w),
            masked=True,
            out_dtype="float32",
            resampling=Resampling.bilinear,
        )
        valid = ~np.ma.getmaskarray(raw).any(axis=0) & np.isfinite(raw.data).all(axis=0)
        if not valid.any():
            raise InputError("Raster contains no valid finite pixels in this window.")
        analysis = raw.filled(0).transpose(1, 2, 0).astype(np.float32)
        scales = [float(ds.scales[i - 1]) for i in bands]
        offsets = [float(ds.offsets[i - 1]) for i in bands]
        analysis = analysis * np.array(scales, np.float32) + np.array(offsets, np.float32)
        valid &= np.isfinite(analysis).all(axis=2)
        if not valid.any():
            raise InputError("Band scaling produced no finite valid data.")
        analysis[~valid] = 0
        rgb = _display(analysis, valid)
        tr = ds.window_transform(win) * ds.transform.scale(win.width / out_w, win.height / out_h)
        georeferenced = ds.crs is not None and abs(tr.determinant) > 1e-15
        if not georeferenced:
            notes.append("No valid CRS/affine reference; geospatial output is unavailable.")
        nodata = [v if v is None or math.isfinite(v) else str(v) for v in ds.nodatavals]
        metadata = ImageMetadata(
            source=name,
            source_sha256=digest,
            width=ds.width,
            height=ds.height,
            preview_width=out_w,
            preview_height=out_h,
            band_count=ds.count,
            sensor=sensor,
            modality=selected_modality,
            acquisition_date=date,
            crs=ds.crs.to_string() if georeferenced else None,
            source_transform=tuple(ds.transform)[:6],
            preview_transform=tuple(tr)[:6] if georeferenced else None,
            window=window,
            band_descriptions=list(ds.descriptions),
            selected_bands=bands,
            nodata=nodata,
            tags=tags,
            band_tags=[ds.tags(i) for i in bands],
            dtype=ds.dtypes[bands[0] - 1],
            scales=scales,
            offsets=offsets,
            synthetic=tags.get("synthetic", "").lower() == "true",
            visualization="Per-band 2–98 percentile preview; analysis uses original scaled values.",
        )
        if metadata.synthetic:
            notes.append("Synthetic fixture: pixel values, location and dates are demonstration data.")
        return LoadedImage(rgb, analysis, valid, metadata, notes)


def iter_tiles(path: str | Path, settings: Settings):
    """Yield bounded LoadedImages, preserving each tile's offset and scale."""
    with rasterio.open(path) as ds:
        width, height = ds.width, ds.height
    for row in range(0, height, settings.tile_size):
        for col in range(0, width, settings.tile_size):
            w, h = min(settings.tile_size, width - col), min(settings.tile_size, height - row)
            # Include narrow edges in a full-sized overlapping final tile.
            if w < 16:
                col, w = max(0, width - 16), min(16, width)
            if h < 16:
                row, h = max(0, height - 16), min(16, height)
            yield load_image(path, settings, window=(col, row, w, h))

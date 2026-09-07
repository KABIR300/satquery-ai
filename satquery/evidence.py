"""Measurable input checks, conservative alignment gating, uncalibrated evidence score."""

from dataclasses import dataclass, field
from datetime import datetime

import cv2
import numpy as np
from affine import Affine

from satquery.config import Settings
from satquery.schemas import Evidence, LoadedImage


@dataclass
class Quality:
    usable: bool
    score: float
    signals: dict
    warnings: list[str] = field(default_factory=list)


@dataclass
class Alignment:
    accepted: bool
    score: float
    signals: dict
    warnings: list[str] = field(default_factory=list)


def assess_quality(image: LoadedImage) -> Quality:
    fraction = float(image.valid.mean())
    pixels = image.rgb[image.valid].astype(np.float32) / 255
    if not pixels.size:
        return Quality(False, 0, {"valid_fraction": 0}, ["No valid image pixels."])
    brightness = float(pixels.mean())
    # Spatial contrast, not differences between constant R/G/B channels.
    luminance = pixels @ np.array([0.299, 0.587, 0.114], np.float32)
    contrast = float(luminance.std())
    notes = []
    if fraction < 0.1:
        notes.append("Less than 10% of the selected image contains valid data.")
    if brightness < 0.015:
        notes.append("Image is extremely dark; select a better tile or visualization.")
    if brightness > 0.985:
        notes.append("Image is extremely bright; cloud/saturation may obscure evidence.")
    if contrast < 0.01:
        notes.append("Image has almost no spatial contrast; reliable grounding is unavailable.")
    return Quality(
        not notes,
        0 if notes else min(1, fraction),
        {"valid_fraction": fraction, "mean_brightness": brightness, "luminance_std": contrast},
        notes,
    )


def _date(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def check_alignment(
    before: LoadedImage, after: LoadedImage, settings: Settings, user_confirmed: bool = False
) -> Alignment:
    signals = {"user_confirmed_coregistration": user_confirmed}
    notes = []

    def reject(reason):
        return Alignment(False, 0, signals, [reason])

    if before.rgb.shape != after.rgb.shape or before.analysis.shape != after.analysis.shape:
        return reject(
            "Image/band dimensions differ. Supply co-registered scenes with matching band selections."
        )
    a, b = before.metadata, after.metadata
    if a.sensor and b.sensor and a.sensor.casefold() != b.sensor.casefold():
        return reject("Sensors differ; the baseline requires comparable observations from the same sensor.")
    if (
        a.selected_bands != b.selected_bands
        or a.dtype != b.dtype
        or a.scales != b.scales
        or a.offsets != b.offsets
    ):
        return reject(
            "Band selection, dtype, or scale/offset differs; harmonize radiometry before comparison."
        )
    if bool(a.crs) != bool(b.crs):
        return reject("Only one image has a geospatial reference; provide a consistently registered pair.")
    if a.crs and b.crs:
        from rasterio.crs import CRS

        if CRS.from_user_input(a.crs) != CRS.from_user_input(b.crs):
            return reject("Coordinate systems differ. Reproject both observations to the same grid.")
        if not a.preview_transform or not b.preview_transform:
            return reject("Missing affine transform for a georeferenced image.")
        mapping = ~Affine(*a.preview_transform) * Affine(*b.preview_transform)
        if not np.allclose(tuple(mapping)[:6], (1, 0, 0, 0, 1, 0), atol=1e-3, rtol=0):
            return reject("Raster grids do not match; co-register and resample before analysis.")
        signals["matching_geospatial_grid"] = True
    elif (a.width, a.height) != (b.width, b.height):
        return reject("Original image dimensions differ. Equal preview sizes do not establish alignment.")
    else:
        signals["matching_geospatial_grid"] = False
        notes.append("No geospatial reference. Translation screening does not establish georegistration.")
    da, db = _date(a.acquisition_date), _date(b.acquisition_date)
    if da and db and da > db:
        return reject("Before date is later than after date; swap the input images.")
    if not da or not db:
        notes.append("Acquisition dates missing or unparseable; temporal order is user-supplied.")
    elif da == db:
        notes.append("The two acquisition dates are identical; verify temporal order.")
    common = before.valid & after.valid
    signals["common_valid_fraction"] = float(common.mean())
    if common.mean() < 0.5:
        return reject("Less than half the image has shared valid data; select an overlapping valid tile.")
    gray_a = cv2.cvtColor(before.rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray_b = cv2.cvtColor(after.rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    # Ignore nodata consistently. This test only screens translation, not rotation/parallax.
    gray_a[~common] = float(gray_a[common].mean())
    gray_b[~common] = float(gray_b[common].mean())
    if np.array_equal(before.analysis[common], after.analysis[common]):
        signals.update(translation_x_px=0.0, translation_y_px=0.0, identical_valid_data=True)
        return Alignment(True, 1, signals, notes + ["The valid image data are identical."])
    h, w = common.shape
    window = cv2.createHanningWindow((w, h), cv2.CV_32F)
    shift, response = cv2.phaseCorrelate(gray_a, gray_b, window)
    if not np.isfinite([*shift, response]).all():
        return reject("Alignment could not be measured. Supply better textured, co-registered imagery.")
    signals.update(
        translation_x_px=float(shift[0]),
        translation_y_px=float(shift[1]),
        phase_correlation_response=float(response),
    )
    if (
        response >= settings.alignment_min_response
        and max(abs(shift[0]), abs(shift[1])) > settings.alignment_max_shift
    ):
        return reject("Images appear shifted. Use georegistered imagery; the baseline does not auto-align.")
    if response < settings.alignment_min_response:
        if not user_confirmed:
            return reject(
                "Alignment is uncertain (weak phase correlation). Use georegistered imagery or "
                "explicitly attest that registration was checked independently."
            )
        notes.append("Alignment accepted on user attestation; automated screening was inconclusive.")
        return Alignment(True, 0.5, signals, notes)
    notes.append(
        "Translation screen passed; rotation, parallax, cloud and seasonal effects remain unchecked."
    )
    return Alignment(True, 0.8, signals, notes)


def evidence_score(
    images: list[LoadedImage],
    qualities: list[Quality],
    *,
    spatial_valid: bool,
    alignment: Alignment | None = None,
    demonstration: bool = False,
) -> Evidence:
    quality = min(q.score for q in qualities)
    completeness = float(
        np.mean(
            [
                sum([bool(i.metadata.sensor), bool(i.metadata.acquisition_date), bool(i.metadata.crs)]) / 3
                for i in images
            ]
        )
    )
    signals = {
        "input_quality": quality,
        "metadata_completeness": completeness,
        "structured_output_valid": True,
        "spatial_evidence_valid": spatial_valid,
        "semantic_correctness_verified": False,
    }
    if alignment:
        signals.update(alignment.signals)
        signals["alignment_check"] = alignment.score
    if demonstration:
        signals["demonstration"] = True
        return Evidence(
            score=None, interpretation="MOCK: no model inference or confidence measurement.", signals=signals
        )
    # Weights sum to 1 for either task. The score reports checks, never semantic accuracy.
    score = (
        0.4 * quality
        + 0.1 * completeness
        + 0.2
        + 0.2 * float(spatial_valid)
        + 0.1 * (alignment.score if alignment else float(spatial_valid))
    )
    return Evidence(score=round(score, 2), signals=signals)

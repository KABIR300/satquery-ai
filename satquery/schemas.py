"""Model-independent serializable contracts; coordinates refer to the loaded preview."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Status(str, Enum):
    OK = "OK"
    LOW_EVIDENCE = "LOW_EVIDENCE"
    NEED_BETTER_INPUT = "NEED_BETTER_INPUT"
    INVALID_INPUT = "INVALID_INPUT"
    INVALID_OUTPUT = "INVALID_OUTPUT"
    UNSUPPORTED = "UNSUPPORTED"


class TaskType(str, Enum):
    OPTICAL_QA = "OPTICAL_QA"
    CHANGE_DETECTION = "CHANGE_DETECTION"
    SAR = "SAR"
    UNSUPPORTED = "UNSUPPORTED"


class Box(Contract):
    x1: float = Field(ge=0)
    y1: float = Field(ge=0)
    x2: float = Field(gt=0)
    y2: float = Field(gt=0)
    label: str = Field(min_length=1, max_length=160)
    coordinate_space: Literal["pixels"] = "pixels"

    @model_validator(mode="after")
    def ordered(self):
        if self.x1 >= self.x2 or self.y1 >= self.y2:
            raise ValueError("Box requires x1 < x2 and y1 < y2.")
        return self


class ImageMetadata(Contract):
    source: str
    source_sha256: str | None = None
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    preview_width: int = Field(gt=0)
    preview_height: int = Field(gt=0)
    band_count: int = Field(gt=0)
    sensor: str | None = None
    modality: Literal["optical", "sar", "unknown"] = "unknown"
    acquisition_date: str | None = None
    crs: str | None = None
    source_transform: tuple[float, float, float, float, float, float] | None = None
    preview_transform: tuple[float, float, float, float, float, float] | None = None
    window: tuple[int, int, int, int] | None = None
    band_descriptions: list[str | None] = Field(default_factory=list)
    selected_bands: list[int] = Field(default_factory=list)
    nodata: list[float | str | None] = Field(default_factory=list)
    tags: dict[str, str] = Field(default_factory=dict)
    band_tags: list[dict[str, str]] = Field(default_factory=list)
    dtype: str | None = None
    scales: list[float] = Field(default_factory=list)
    offsets: list[float] = Field(default_factory=list)
    visualization: str = "RGB"
    synthetic: bool = False


class Evidence(Contract):
    score: float | None = Field(default=None, ge=0, le=1)
    interpretation: str = "Engineering evidence checks; not a probability of correctness."
    signals: dict[str, float | bool | str | None] = Field(default_factory=dict)


class MaskInfo(Contract):
    name: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    positive_pixels: int = Field(ge=0)
    valid_pixels: int = Field(ge=0)
    representation: str = "Binary mask; exported as PNG and georeferenced TIFF when available."

    @model_validator(mode="after")
    def consistent_counts(self):
        if not 0 <= self.positive_pixels <= self.valid_pixels <= self.width * self.height:
            raise ValueError("Mask counts must satisfy positive <= valid <= width * height.")
        return self


class Result(Contract):
    schema_version: str = "1.0"
    answer: str = Field(min_length=1)
    task_type: TaskType
    status: Status
    evidence: Evidence = Field(default_factory=Evidence)
    bounding_boxes: list[Box] = Field(default_factory=list)
    masks: list[MaskInfo] = Field(default_factory=list)
    image_metadata: list[ImageMetadata] = Field(default_factory=list)
    backend: str = "none"
    warnings: list[str] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)
    geospatial: list[dict[str, Any]] = Field(default_factory=list)
    statistics: dict[str, float | int | str] = Field(default_factory=dict)
    latency_seconds: float = Field(default=0, ge=0)


@dataclass
class LoadedImage:
    rgb: np.ndarray
    analysis: np.ndarray
    valid: np.ndarray
    metadata: ImageMetadata
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self):
        h, w = self.rgb.shape[:2]
        if self.rgb.shape != (h, w, 3) or self.rgb.dtype != np.uint8:
            raise ValueError("Preview must be uint8 RGB.")
        if self.analysis.ndim != 3 or self.analysis.shape[:2] != (h, w):
            raise ValueError("Analysis bands must share the preview grid.")
        if self.valid.shape != (h, w) or self.valid.dtype != bool:
            raise ValueError("Validity mask must be boolean and match the preview.")
        if (w, h) != (self.metadata.preview_width, self.metadata.preview_height):
            raise ValueError("Metadata does not match the preview dimensions.")
        if not np.isfinite(self.analysis[self.valid]).all():
            raise ValueError("Valid pixels must be finite.")


@dataclass
class PipelineOutput:
    result: Result
    images: list[LoadedImage] = field(default_factory=list)
    mask: np.ndarray | None = None
    difference: np.ndarray | None = None
    valid_mask: np.ndarray | None = None

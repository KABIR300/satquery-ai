"""One validated configuration boundary for both execution profiles."""

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    profile: Literal["local", "cloud"] = "local"
    backend: Literal["mock", "qwen"] = "mock"
    device: Literal["auto", "cpu", "cuda"] = "auto"
    dtype: Literal["auto", "float32", "float16", "bfloat16"] = "auto"
    quantization: Literal["none", "4bit"] = "none"
    model_id: str = Field(default="Qwen/Qwen2.5-VL-3B-Instruct", min_length=1)
    model_revision: str = Field(default="main", min_length=1)
    model_cache: Path = Path(".cache/models")
    allow_model_download: bool = False
    max_image_size: int = Field(default=1024, ge=64, le=2048)
    max_source_pixels: int = Field(default=40_000_000, ge=4096, le=100_000_000)
    max_upload_mb: int = Field(default=64, ge=1, le=256)
    tile_size: int = Field(default=512, ge=64, le=2048)
    rgb_bands: tuple[int, int, int] | None = None
    max_new_tokens: int = Field(default=384, ge=32, le=2048)
    max_question_chars: int = Field(default=2000, ge=32, le=8000)
    max_visual_pixels: int = Field(default=512 * 28 * 28, ge=4 * 28 * 28, le=2048 * 28 * 28)
    do_sample: bool = False
    temperature: float = Field(default=0.2, gt=0, le=2)
    change_threshold: float = Field(default=0.15, gt=0, lt=1)
    min_region_pixels: int = Field(default=16, ge=1, le=4096)
    evidence_threshold: float = Field(default=0.7, ge=0, le=1)
    alignment_max_shift: float = Field(default=2, ge=0.1, le=20)
    alignment_min_response: float = Field(default=0.25, gt=0, le=1)

    @model_validator(mode="after")
    def compatible(self):
        if self.rgb_bands and any(b < 1 for b in self.rgb_bands):
            raise ValueError("RGB bands are one-based positive indices.")
        if self.quantization == "4bit" and (self.device == "cpu" or self.dtype == "float32"):
            raise ValueError("4-bit inference requires CUDA and float16/bfloat16 compute.")
        if self.device == "cpu" and self.dtype in {"float16", "bfloat16"}:
            raise ValueError("The portable CPU path uses float32.")
        return self


def load_settings(path: str | Path | None = None, **overrides) -> Settings:
    selected = path or os.environ.get("SATQUERY_CONFIG")
    data = {}
    if selected:
        with Path(selected).open(encoding="utf-8") as stream:
            try:
                data = yaml.safe_load(stream) or {}
            except yaml.YAMLError as exc:
                raise ValueError(f"Malformed YAML configuration: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("Configuration must be a YAML mapping.")
    for field in Settings.model_fields:
        key = f"SATQUERY_{field.upper()}"
        if key in os.environ:
            try:
                data[field] = yaml.safe_load(os.environ[key])
            except yaml.YAMLError as exc:
                raise ValueError(f"Malformed configuration environment variable {key}.") from exc
    data.update({key: value for key, value in overrides.items() if value is not None})
    return Settings.model_validate(data)

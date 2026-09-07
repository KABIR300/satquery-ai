from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from satquery.schemas import Box, LoadedImage


class BackendUnavailable(RuntimeError):
    pass


class OutputError(ValueError):
    pass


@dataclass
class OpticalPrediction:
    answer: str
    boxes: list[Box]
    backend: str
    demonstration: bool = False
    warnings: list[str] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)


@dataclass
class ChangePrediction:
    mask: np.ndarray
    difference: np.ndarray
    valid: np.ndarray
    boxes: list[Box]
    backend: str
    statistics: dict
    warnings: list[str] = field(default_factory=list)


class VisionLanguageBackend(Protocol):
    name: str

    def analyze(self, image: LoadedImage, question: str) -> OpticalPrediction: ...


class ChangeDetectionBackend(Protocol):
    name: str

    def analyze(self, before: LoadedImage, after: LoadedImage) -> ChangePrediction: ...


class SARBackend(Protocol):
    name: str

    def analyze(self, images: list[LoadedImage], question: str) -> OpticalPrediction: ...

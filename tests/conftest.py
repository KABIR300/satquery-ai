from pathlib import Path

import numpy as np
import pytest

from satquery.config import Settings
from satquery.io import load_image
from satquery.samples import generate_samples


@pytest.fixture(scope="session")
def pack(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("samples")
    generate_samples(root)
    return root


@pytest.fixture
def settings():
    return Settings()


@pytest.fixture
def pair(pack, settings):
    return [load_image(pack / f"flood_{p}.tif", settings) for p in ("before", "after")]


@pytest.fixture
def textured_png(tmp_path):
    from PIL import Image

    path = tmp_path / "texture.png"
    Image.fromarray(np.random.default_rng(10).integers(40, 220, (96, 128, 3), dtype=np.uint8)).save(path)
    return path

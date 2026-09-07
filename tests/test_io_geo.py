import io

import numpy as np
import pytest
import rasterio
from PIL import Image
from rasterio.transform import from_origin

from satquery.config import Settings
from satquery.geo import box_to_geo, pixel_to_world
from satquery.io import InputError, iter_tiles, load_image, sensor_metadata
from satquery.schemas import Box


def test_png_bytes(textured_png, settings):
    image = load_image(textured_png.read_bytes(), settings, name="upload.png")
    assert image.rgb.shape == (96, 128, 3)
    assert image.metadata.crs is None
    assert image.metadata.acquisition_date is None
    assert len(image.metadata.source_sha256) == 64


def test_transparency(settings):
    pixels = np.full((32, 32, 4), 150, np.uint8)
    pixels[:, :, 3] = 255
    pixels[:16, :, 3] = 0
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    image = load_image(buffer.getvalue(), settings, name="alpha.png")
    assert image.valid.sum() == 512
    assert not image.rgb[:16].any()


@pytest.mark.parametrize(
    "data,name",
    [(b"", "empty.png"), (b"garbage", "bad.jpg"), (b"123", "file.txt"), (b"II*\x00bad", "bad.tif")],
)
def test_invalid_files(data, name, settings):
    with pytest.raises(InputError):
        load_image(data, settings, name=name)


def test_size_limits(textured_png):
    with pytest.raises(InputError, match="too large"):
        load_image(textured_png, Settings(max_source_pixels=4096))
    with pytest.raises(InputError, match="exceeds"):
        load_image(b"0" * (1024**2 + 1), Settings(max_upload_mb=1), name="large.png")


def test_geotiff_metadata_and_resized_transform(pack):
    image = load_image(pack / "flood_before.tif", Settings(max_image_size=128))
    m = image.metadata
    assert m.crs == "EPSG:32643"
    assert (m.width, m.preview_width, m.band_count) == (256, 128, 3)
    assert m.synthetic
    assert pixel_to_world(m, 1, 1) == (500020, 2199980)
    assert pixel_to_world(m, 0, 0, center=True) == (500010, 2199990)
    assert m.sensor == "SYNTHETIC_OPTICAL" and m.acquisition_date == "2026-01-01"


def test_window_and_resize(pack):
    image = load_image(pack / "flood_before.tif", Settings(max_image_size=64), window=(32, 64, 128, 128))
    assert pixel_to_world(image.metadata, 0, 0) == (500320, 2199360)
    assert pixel_to_world(image.metadata, 64, 64) == (501600, 2198080)


@pytest.mark.parametrize("window", [(-1, 0, 20, 20), (0, 0, 500, 500), (0, 0, 4, 4), (0.0, 0, 20, 20)])
def test_invalid_window(window, pack, settings):
    with pytest.raises(InputError):
        load_image(pack / "flood_before.tif", settings, window=window)


def test_wgs84_and_rotated_polygon(pair):
    meta = pair[0].metadata.model_copy(update={"preview_transform": (10, 2, 500000, 1, -10, 2200000)})
    assert pixel_to_world(meta, 2, 3) == (500026, 2199972)
    feature = box_to_geo(Box(x1=0, y1=0, x2=20, y2=30, label="test"), meta)
    ring = feature["wgs84"]["geometry"]["coordinates"][0]
    assert ring[0] == ring[-1]
    assert ring[0][0] == pytest.approx(75, abs=0.001)
    assert 19 < ring[0][1] < 21
    assert feature["source_crs"] == "EPSG:32643"


@pytest.mark.parametrize(
    "x,y,center", [(257, 0, False), (0, -1, False), (float("nan"), 0, False), (256, 256, True)]
)
def test_invalid_pixels(x, y, center, pair):
    with pytest.raises(ValueError):
        pixel_to_world(pair[0].metadata, x, y, center=center)


def test_no_crs_conversion(textured_png, settings):
    with pytest.raises(ValueError):
        pixel_to_world(load_image(textured_png, settings).metadata, 0, 0)


def test_nodata_and_missing_bands(tmp_path, settings):
    path = tmp_path / "nodata.tif"
    arr = np.full((1, 32, 32), 100, np.float32)
    arr[:, :10] = np.nan
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=32,
        height=32,
        count=1,
        dtype="float32",
        nodata=np.nan,
        crs="EPSG:4326",
        transform=from_origin(70, 20, 0.01, 0.01),
    ) as ds:
        ds.write(arr)
        ds.update_tags(sensor="Sentinel-1", acquisition_date="2026-01-01")
    image = load_image(path, settings)
    assert image.metadata.modality == "sar"
    assert image.metadata.nodata == ["nan"]
    assert not image.valid[:10].any()
    assert image.valid[10:].all()
    with pytest.raises(InputError, match="cannot be forced"):
        load_image(path, settings, modality="optical")
    with rasterio.open(path, "r+") as ds:
        ds.update_tags(sensor="Sentinel-2")
    with pytest.raises(InputError, match="band count"):
        load_image(path, Settings(rgb_bands=(3, 2, 1)))


def test_all_nodata_rejected(tmp_path, settings):
    path = tmp_path / "empty.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=32,
        height=32,
        count=1,
        dtype="uint8",
        nodata=0,
        crs="EPSG:4326",
        transform=from_origin(70, 20, 0.01, 0.01),
    ) as ds:
        ds.write(np.zeros((1, 32, 32), np.uint8))
    with pytest.raises(InputError, match="no valid"):
        load_image(path, settings)


def test_tiling(pack):
    tiles = list(iter_tiles(pack / "urban_before.tif", Settings(tile_size=128)))
    assert len(tiles) == 4
    assert pixel_to_world(tiles[-1].metadata, 0, 0) == (501280, 2198720)


def test_sensor_missing_values():
    assert sensor_metadata({}) == (None, "unknown", None)
    assert sensor_metadata({"platform": "Sentinel-1A"})[1] == "sar"
    assert sensor_metadata({"spacecraft_name": "Sentinel-2B"})[1] == "optical"

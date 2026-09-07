"""Explicit preview pixel edges to source CRS and optional WGS84 longitude/latitude."""

import math

from affine import Affine
from rasterio.warp import transform

from satquery.schemas import Box, ImageMetadata


def pixel_to_world(metadata: ImageMetadata, x: float, y: float, *, center=False, wgs84=False):
    if not metadata.crs or not metadata.preview_transform:
        raise ValueError("A valid CRS and preview affine transform are required.")
    if not (math.isfinite(x) and math.isfinite(y)):
        raise ValueError("Pixel coordinates must be finite.")
    if not (0 <= x <= metadata.preview_width and 0 <= y <= metadata.preview_height):
        raise ValueError("Coordinates are outside the preview.")
    if center and (x >= metadata.preview_width or y >= metadata.preview_height):
        raise ValueError("A pixel center must be inside the image.")
    offset = 0.5 if center else 0
    wx, wy = Affine(*metadata.preview_transform) * (x + offset, y + offset)
    if wgs84:
        xs, ys = transform(metadata.crs, "EPSG:4326", [wx], [wy])
        wx, wy = xs[0], ys[0]
    if not (math.isfinite(wx) and math.isfinite(wy)):
        raise ValueError("Coordinate transformation produced non-finite coordinates.")
    return wx, wy


def box_to_geo(box: Box, metadata: ImageMetadata):
    corners = [(box.x1, box.y1), (box.x2, box.y1), (box.x2, box.y2), (box.x1, box.y2)]
    corners.append(corners[0])
    source = [pixel_to_world(metadata, x, y) for x, y in corners]
    geographic = [pixel_to_world(metadata, x, y, wgs84=True) for x, y in corners]
    return {
        "label": box.label,
        "source_crs": metadata.crs,
        "source_polygon_xy": source,
        "wgs84": {
            "type": "Feature",
            "properties": {"label": box.label},
            "geometry": {"type": "Polygon", "coordinates": [geographic]},
        },
    }

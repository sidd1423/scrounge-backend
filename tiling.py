"""
Split a fridge photo into tiles for detection.

Two tiling passes run together:
  1. A 3x3 overlapping grid (tile_image) — good general coverage.
  2. Full-height vertical strips (tile_vertical_strips) — door shelves hold
     tall, narrow items (condiment bottles, jugs) standing upright. A square
     grid slices a tall bottle into 2-3 horizontal pieces, so no single grid
     tile ever shows the whole bottle or its full label. Full-height strips
     guarantee that most upright bottles appear complete in at least one tile.

Both passes' tiles get merged into one flat list before detection — the
downstream pipeline doesn't care which pass a tile came from, only that
items get properly deduplicated later (which merge.py already handles for
overlapping/duplicate detections of the same physical item).
"""

from __future__ import annotations

import io
from dataclasses import dataclass

from PIL import Image

GRID_ROWS = 3
GRID_COLS = 3
GRID_OVERLAP_FRACTION = 0.15  # each grid tile extends 15% further into its neighbor

VERTICAL_STRIPS = 3
VERTICAL_STRIP_OVERLAP_FRACTION = 0.2  # slightly more overlap so a bottle near a strip boundary still lands whole in one strip

# If a crop's longer side is smaller than this after cropping, upscale it.
# Small crops sent to a vision model lose exactly the fine detail (texture,
# subtle color differences, label text) needed to tell e.g. sliced cheese
# from deli meat apart, or read a condiment label — upscaling doesn't
# invent detail, but it does stop the image from being needlessly tiny
# before the model even sees it.
MIN_TILE_DIMENSION = 512


@dataclass
class Tile:
    tile_id: int
    image_bytes: bytes
    box: tuple[int, int, int, int]  # (left, top, right, bottom) in original image


def _upscale_if_small(crop: Image.Image, min_dimension: int = MIN_TILE_DIMENSION) -> Image.Image:
    """Upscale a crop so its longer side is at least min_dimension, preserving aspect ratio."""
    longer_side = max(crop.width, crop.height)
    if longer_side >= min_dimension:
        return crop
    scale = min_dimension / longer_side
    new_size = (round(crop.width * scale), round(crop.height * scale))
    return crop.resize(new_size, Image.LANCZOS)


def _save_crop(crop: Image.Image, box: tuple[int, int, int, int], jpeg_quality: int,
               min_tile_dimension: int) -> Tile:
    crop = _upscale_if_small(crop, min_tile_dimension)
    buf = io.BytesIO()
    crop.save(buf, format="JPEG", quality=jpeg_quality)
    return Tile(tile_id=-1, image_bytes=buf.getvalue(), box=box)  # tile_id assigned by caller


def tile_image(image_bytes: bytes, rows: int = GRID_ROWS, cols: int = GRID_COLS,
                overlap_fraction: float = GRID_OVERLAP_FRACTION, jpeg_quality: int = 90,
                min_tile_dimension: int = MIN_TILE_DIMENSION) -> list[Tile]:
    """
    Split one fridge photo into rows*cols overlapping grid tiles.
    Good general-purpose coverage for items that fit comfortably within
    one grid cell (produce, most shelf items). For tall door-shelf items,
    see tile_vertical_strips — use both together via tile_photo().
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = image.size

    base_tile_w = width / cols
    base_tile_h = height / rows
    overlap_w = base_tile_w * overlap_fraction
    overlap_h = base_tile_h * overlap_fraction

    tiles: list[Tile] = []
    for row in range(rows):
        for col in range(cols):
            left = max(0, col * base_tile_w - overlap_w)
            top = max(0, row * base_tile_h - overlap_h)
            right = min(width, (col + 1) * base_tile_w + overlap_w)
            bottom = min(height, (row + 1) * base_tile_h + overlap_h)
            box = (int(left), int(top), int(right), int(bottom))
            crop = image.crop(box)
            tiles.append(_save_crop(crop, box, jpeg_quality, min_tile_dimension))

    return tiles


def tile_vertical_strips(image_bytes: bytes, strips: int = VERTICAL_STRIPS,
                          overlap_fraction: float = VERTICAL_STRIP_OVERLAP_FRACTION,
                          jpeg_quality: int = 90, min_tile_dimension: int = MIN_TILE_DIMENSION) -> list[Tile]:
    """
    Split one fridge photo into `strips` full-height vertical strips.
    Each strip spans the entire height of the photo, so a tall upright
    item (condiment bottle, milk jug, water pitcher) standing in a door
    shelf is captured whole — and its label is readable — in whichever
    strip it falls into, rather than being cut into horizontal pieces
    by the square grid.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = image.size

    base_strip_w = width / strips
    overlap_w = base_strip_w * overlap_fraction

    tiles: list[Tile] = []
    for col in range(strips):
        left = max(0, col * base_strip_w - overlap_w)
        right = min(width, (col + 1) * base_strip_w + overlap_w)
        box = (int(left), 0, int(right), height)
        crop = image.crop(box)
        tiles.append(_save_crop(crop, box, jpeg_quality, min_tile_dimension))

    return tiles


def tile_photo(image_bytes: bytes, **kwargs) -> list[Tile]:
    """
    Combined tiling for one photo: grid tiles + vertical strip tiles,
    with unique sequential tile_ids across both sets. This is what
    tile_multiple_images() calls per photo — use this directly if you're
    only tiling a single image.
    """
    grid_kwargs = {k: v for k, v in kwargs.items() if k not in ("strips", "overlap_fraction")}
    grid_tiles = tile_image(image_bytes, **grid_kwargs)

    strip_kwargs = {}
    if "min_tile_dimension" in kwargs:
        strip_kwargs["min_tile_dimension"] = kwargs["min_tile_dimension"]
    if "jpeg_quality" in kwargs:
        strip_kwargs["jpeg_quality"] = kwargs["jpeg_quality"]
    strip_tiles = tile_vertical_strips(image_bytes, **strip_kwargs)

    all_tiles = grid_tiles + strip_tiles
    for i, t in enumerate(all_tiles):
        t.tile_id = i
    return all_tiles


def tile_multiple_images(images: list[bytes], **kwargs) -> list[Tile]:
    """
    Tile several fridge photos (e.g. multiple angles) using the combined
    grid + vertical-strip approach, returning one flat list of tiles with
    globally unique tile_id values, so downstream reconcile/merge logic
    doesn't need to know which source photo a tile came from.
    """
    all_tiles: list[Tile] = []
    next_id = 0
    for image_bytes in images:
        tiles = tile_photo(image_bytes, **kwargs)
        for t in tiles:
            t.tile_id = next_id
            all_tiles.append(t)
            next_id += 1
    return all_tiles
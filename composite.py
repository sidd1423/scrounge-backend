"""
Combine several tile crops into one labeled grid montage image, so
detection can send far fewer API calls (one per montage instead of one
per tile) — this is the main lever for both latency and free-tier
quota usage, since request count is what's actually expensive/slow on
OpenRouter's free models, not image size.

Each cell in the montage is labeled with its original tile_id in a
high-contrast corner box, so the model's response can reference "cell 4"
and that maps directly back to the tile_id used everywhere else in the
pipeline (reconciliation, OCR matching, etc.) — no separate ID scheme.
"""

from __future__ import annotations

import io
import math

from PIL import Image, ImageDraw, ImageFont, ImageOps

CELL_SIZE = 420  # each tile is fit into a CELL_SIZE x CELL_SIZE square
GUTTER = 6  # px of border between cells, helps the model see cell boundaries
LABEL_BOX_SIZE = 56  # size of the number label box drawn in each cell's corner


def _load_label_font(size: int) -> ImageFont.ImageFont:
    """Try for a real truetype font (bundled with Pillow) so labels are legible; fall back to default bitmap font."""
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()


_LABEL_FONT = _load_label_font(32)


def _fit_into_cell(image: Image.Image, cell_size: int = CELL_SIZE) -> Image.Image:
    """Resize (preserving aspect ratio) and pad with a neutral background to fill a square cell exactly."""
    fitted = ImageOps.contain(image, (cell_size, cell_size))
    canvas = Image.new("RGB", (cell_size, cell_size), color=(30, 30, 30))
    offset = ((cell_size - fitted.width) // 2, (cell_size - fitted.height) // 2)
    canvas.paste(fitted, offset)
    return canvas


def _draw_cell_label(canvas: Image.Image, tile_id: int) -> None:
    """Draw a high-contrast numbered label in the top-left corner of a cell."""
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, LABEL_BOX_SIZE, LABEL_BOX_SIZE], fill=(255, 220, 0))
    text = str(tile_id)
    # Center the text in the label box (basic centering; exact metrics vary by font but this is close enough at this size)
    bbox = draw.textbbox((0, 0), text, font=_LABEL_FONT)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    text_x = (LABEL_BOX_SIZE - text_w) // 2 - bbox[0]
    text_y = (LABEL_BOX_SIZE - text_h) // 2 - bbox[1]
    draw.text((text_x, text_y), text, fill=(0, 0, 0), font=_LABEL_FONT)


def build_montage(tiles: list[tuple[int, bytes]], cell_size: int = CELL_SIZE,
                   jpeg_quality: int = 88) -> tuple[bytes, list[int]]:
    """
    Build one grid montage image from a list of (tile_id, image_bytes) tiles.
    Returns (montage_jpeg_bytes, tile_ids_in_order). The grid is laid out
    as close to square as possible (e.g. 4 tiles -> 2x2, 6 tiles -> 3x2).

    Keep the number of tiles per montage modest (4-6) — cramming too many
    into one image shrinks each cell below the detail needed for the
    fine-grained disambiguation (cheese vs. deli meat, reading a partial
    label) the detection prompt relies on.
    """
    n = len(tiles)
    if n == 0:
        raise ValueError("build_montage requires at least one tile")

    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    montage_w = cols * cell_size + (cols + 1) * GUTTER
    montage_h = rows * cell_size + (rows + 1) * GUTTER
    montage = Image.new("RGB", (montage_w, montage_h), color=(0, 0, 0))  # black gutter = clear cell boundaries

    tile_ids: list[int] = []
    for i, (tile_id, image_bytes) in enumerate(tiles):
        row, col = divmod(i, cols)
        cell_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        cell_canvas = _fit_into_cell(cell_image, cell_size)
        _draw_cell_label(cell_canvas, tile_id)

        x = GUTTER + col * (cell_size + GUTTER)
        y = GUTTER + row * (cell_size + GUTTER)
        montage.paste(cell_canvas, (x, y))
        tile_ids.append(tile_id)

    buf = io.BytesIO()
    montage.save(buf, format="JPEG", quality=jpeg_quality)
    return buf.getvalue(), tile_ids


def group_tiles(tiles: list[tuple[int, bytes]], group_size: int = 4) -> list[list[tuple[int, bytes]]]:
    """Split a flat tile list into chunks, each chunk becoming one montage/API call."""
    return [tiles[i:i + group_size] for i in range(0, len(tiles), group_size)]
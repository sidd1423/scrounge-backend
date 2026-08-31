"""
OCR step using local Tesseract, run per tile.

Requires:
  - System install of the Tesseract binary (see README for Windows steps)
  - pip install pytesseract pillow

Most tiles will produce empty/garbage OCR text (produce, empty shelving,
reflections) — that's expected. ocr_resolve.py already treats empty/
unmatched OCR text as "unknown" and merge.py's reconcile_tile() just
passes visual detections through unchanged when there's nothing useful
from OCR.
"""

from __future__ import annotations

import io
import logging

import pytesseract
from PIL import Image

from tiling import Tile

logger = logging.getLogger(__name__)

# On Windows, if tesseract isn't on PATH, uncomment and set this to your
# install location, e.g.:
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

MIN_TEXT_LENGTH = 3  # ignore OCR noise shorter than this


def ocr_tile(tile: Tile) -> str:
    """
    Run OCR on one tile's image bytes. Returns cleaned text, or "" if
    nothing usable was found (short-circuits ocr_resolve.py to "unknown"
    without an LLM call).
    """
    try:
        image = Image.open(io.BytesIO(tile.image_bytes))
        raw_text = pytesseract.image_to_string(image)
    except Exception as e:
        logger.warning("Tile %s: OCR failed: %s", tile.tile_id, e)
        return ""

    cleaned = " ".join(raw_text.split())  # collapse whitespace/newlines
    if len(cleaned) < MIN_TEXT_LENGTH:
        return ""
    return cleaned


def ocr_tiles_batch(tiles: list[Tile]) -> list[tuple[int, str]]:
    """Returns (tile_id, ocr_text) pairs, matching the shape resolve_ocr_batch() expects."""
    return [(tile.tile_id, ocr_tile(tile)) for tile in tiles]

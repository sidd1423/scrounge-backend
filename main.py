"""
Scrounge detection service.

Run locally with:
    uvicorn main:app --reload --port 8000

Then POST one or more fridge photos (multipart/form-data, field name
"images") to http://localhost:8000/detect-multi

Example curl:
    curl -X POST http://localhost:8000/detect-multi \\
        -F "images=@fridge1.jpg" -F "images=@fridge2.jpg"
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from tiling import tile_multiple_images
from detection import detect_tiles_batch, DetectedItem
from ocr import ocr_tiles_batch
from ocr_resolve import resolve_ocr_batch
from merge import reconcile_all
from cross_tile_merge import merge_cross_tile
from ingredient_vocab import UNKNOWN, match_text_to_vocab


def recover_unknown_visual_items(items: list[DetectedItem]) -> list[DetectedItem]:
    """
    The vocab-constrained detection prompt sometimes returns "unknown" even
    when raw_guess clearly names something in (or close to) the vocab —
    e.g. the model sees "mixed greens" and won't force-fit it to "lettuce"
    on its own, or sees "tortillas" (which IS in the vocab) but isn't
    visually confident enough to commit to it live.

    This re-matches raw_guess against the vocab/aliases after the fact.
    Recovered items get a slightly reduced confidence relative to a
    same-named live match, since this is a looser, after-the-fact match.
    """
    recovered = []
    for item in items:
        if item.name != UNKNOWN or not item.raw_guess:
            recovered.append(item)
            continue

        matched_name = match_text_to_vocab(item.raw_guess)
        if matched_name:
            recovered.append(DetectedItem(
                name=matched_name,
                confidence=max(item.confidence, 0.6),
                raw_guess=item.raw_guess,
                source=item.source,
                tile_id=item.tile_id,
            ))
        else:
            recovered.append(item)
    return recovered

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Scrounge Detection Service")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/detect-multi")
async def detect_multi(images: list[UploadFile] = File(...)):
    """
    Accepts one or more fridge photos, tiles each into a 3x3 overlapping
    grid, runs visual detection (GPT-4o-mini) and OCR (Tesseract) on every
    tile, reconciles visual vs. OCR per tile, then sends the reconciled
    set through a cross-tile merge/dedupe LLM call.
    """
    if not images:
        return JSONResponse(status_code=400, content={"error": "No images provided"})

    image_bytes_list = [await img.read() for img in images]

    tiles = tile_multiple_images(image_bytes_list)
    logger.info("Tiled %d photo(s) into %d tiles", len(image_bytes_list), len(tiles))

    visual_items = detect_tiles_batch([(t.tile_id, t.image_bytes) for t in tiles])
    logger.info("Visual detection produced %d raw items", len(visual_items))

    visual_items = recover_unknown_visual_items(visual_items)

    ocr_results = ocr_tiles_batch(tiles)
    ocr_items = resolve_ocr_batch(ocr_results)
    logger.info("OCR resolution produced %d items", len(ocr_items))

    reconciled = reconcile_all(visual_items, ocr_items)
    logger.info("Reconciled to %d items before cross-tile merge", len(reconciled))
    for item in reconciled:
        logger.info(
            "  RECONCILED tile=%s name=%s conf=%.2f source=%s raw_guess=%r",
            item.tile_id, item.name, item.confidence, item.source, item.raw_guess,
        )

    final_ingredients = merge_cross_tile(reconciled)
    logger.info("Final merged ingredient list: %d items", len(final_ingredients))
    for ing in final_ingredients:
        logger.info("  FINAL name=%s conf=%.2f", ing["name"], ing["confidence"])

    return {"ingredients": final_ingredients}
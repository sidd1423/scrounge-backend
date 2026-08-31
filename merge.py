"""
Merge visual detections (detection.py) and OCR detections (ocr_resolve.py)
into a single ingredient list.

Two stages:

1. `reconcile_tile()` — for a single tile/crop, decide between the visual
   guess and the OCR guess when both exist. OCR wins when it resolves
   on-vocabulary with reasonable confidence, since it's reading the actual
   package rather than inferring from appearance.

2. `build_cross_tile_merge_prompt()` — after reconciliation, hand the
   consolidated per-tile items to your EXISTING cross-tile LLM merge/dedupe
   step (the one that already collapses duplicates seen across overlapping
   tiles). This just changes what you feed that step: instead of raw names,
   it gets vocab-clean names annotated with source and confidence, so the
   merge LLM can make better dedup calls (e.g. prefer the OCR-sourced name
   when two tiles disagree on the same physical item).

Wire order in your /detect-multi handler:
    visual_items = detect_tiles_batch(tiles)
    ocr_items    = resolve_ocr_batch(ocr_results)
    reconciled   = reconcile_all(visual_items, ocr_items)
    final_list   = your_existing_merge_llm_call(build_cross_tile_merge_prompt(reconciled))
"""

from __future__ import annotations

from collections import defaultdict

from detection import DetectedItem
from ingredient_vocab import UNKNOWN

# OCR overrides visual when OCR resolved on-vocabulary at/above this confidence.
OCR_OVERRIDE_THRESHOLD = 0.7


def reconcile_tile(visual_items: list[DetectedItem], ocr_item: DetectedItem | None) -> list[DetectedItem]:
    """
    Reconcile the visual detection(s) for one tile against that tile's OCR
    result (if any). Most tiles will have no OCR text (produce, non-labeled
    items) — in that case visual_items pass through unchanged.
    """
    if ocr_item is None or ocr_item.name == UNKNOWN:
        return visual_items

    if ocr_item.confidence < OCR_OVERRIDE_THRESHOLD:
        # OCR too unsure to trust over vision; keep both, let cross-tile
        # merge step see both signals rather than silently dropping OCR.
        return visual_items + [ocr_item]

    # OCR is confident and on-vocabulary: prefer it, but keep visual items
    # that named something clearly different (e.g. OCR read a label on a
    # jar that's sitting next to, not on, the detected item) rather than
    # discarding potentially-separate items.
    kept_visual = [
        v for v in visual_items
        if v.name != UNKNOWN and v.name != ocr_item.name and v.confidence >= 0.5
    ]
    return [ocr_item] + kept_visual


def reconcile_all(
    visual_items: list[DetectedItem],
    ocr_items: list[DetectedItem],
) -> list[DetectedItem]:
    """Group by tile_id and reconcile each tile independently."""
    visual_by_tile: dict[int | None, list[DetectedItem]] = defaultdict(list)
    for item in visual_items:
        visual_by_tile[item.tile_id].append(item)

    ocr_by_tile: dict[int | None, DetectedItem] = {item.tile_id: item for item in ocr_items}

    reconciled: list[DetectedItem] = []
    all_tile_ids = set(visual_by_tile.keys()) | set(ocr_by_tile.keys())
    for tile_id in all_tile_ids:
        tile_visual = visual_by_tile.get(tile_id, [])
        tile_ocr = ocr_by_tile.get(tile_id)
        reconciled.extend(reconcile_tile(tile_visual, tile_ocr))

    # Drop unknowns with very low confidence before they hit the merge LLM —
    # they add noise without adding information. Keep unknowns with a
    # meaningful raw_guess, since those are exactly the signal you want for
    # vocabulary expansion later.
    return [
        item for item in reconciled
        if not (item.name == UNKNOWN and item.confidence < 0.3 and not item.raw_guess)
    ]


def build_cross_tile_merge_prompt(reconciled_items: list[DetectedItem]) -> str:
    """
    Render reconciled items as the input block for your existing cross-tile
    merge/dedupe LLM call. Swap whatever you currently pass into that
    call's prompt for this block.
    """
    lines = []
    for item in reconciled_items:
        tile_str = f"tile {item.tile_id}" if item.tile_id is not None else "tile ?"
        lines.append(
            f'- name="{item.name}", confidence={item.confidence:.2f}, '
            f'source={item.source}, {tile_str}, raw_guess="{item.raw_guess}"'
        )
    items_block = "\n".join(lines) if lines else "(no items detected)"

    return f"""Below are ingredient detections from overlapping crops of the same fridge photo(s).
Items from different tiles may refer to the same physical object.

{items_block}

Instructions:
- Deduplicate items that likely refer to the same physical object across tiles.
- When two detections of the same physical item disagree on name, prefer the one with source="ocr" \
if its confidence is reasonable — it's reading the actual package rather than inferring from appearance.
- Drop items where name is "unknown" and no reasonable ingredient can be inferred from raw_guess.
- Return a final deduplicated ingredient list as JSON: {{"ingredients": [{{"name": str, "confidence": float}}]}}
"""

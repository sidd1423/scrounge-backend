"""
The cross-tile merge/dedupe LLM call.

merge.py builds the prompt (build_cross_tile_merge_prompt); this module
actually sends it to GPT-4o-mini and parses the final ingredient list back
out. Kept separate from merge.py so merge.py has no API dependency and can
be unit tested purely on prompt construction / reconciliation logic.
"""

from __future__ import annotations

import logging

from llm_client import chat_completion_with_fallback, extract_json, response_format_kwarg
from detection import DetectedItem
from merge import build_cross_tile_merge_prompt

logger = logging.getLogger(__name__)

MERGE_SYSTEM_PROMPT = (
    "You deduplicate and finalize a food ingredient list from overlapping "
    "detections of the same fridge photo(s). Follow the instructions given "
    "in the user message exactly. Return strict JSON only, no markdown fences."
)


def merge_cross_tile(reconciled_items: list[DetectedItem]) -> list[dict]:
    """
    Send the reconciled per-tile items to the configured LLM for final
    cross-tile dedup, and return the final ingredient list as
    [{"name": str, "confidence": float}, ...].

    Falls back to a naive local dedup (keep highest-confidence entry per
    name) if the LLM call fails, so a single API hiccup doesn't zero out
    the whole detection result for the user.
    """
    if not reconciled_items:
        return []

    user_prompt = build_cross_tile_merge_prompt(reconciled_items)

    try:
        response = chat_completion_with_fallback(
            messages=[
                {"role": "system", "content": MERGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            **response_format_kwarg(),
        )
        parsed = extract_json(response.choices[0].message.content)
        ingredients = parsed.get("ingredients", [])
        # basic shape validation
        return [
            {"name": str(i["name"]).strip().lower(), "confidence": float(i.get("confidence", 0.0))}
            for i in ingredients
            if "name" in i
        ]
    except Exception as e:
        logger.error("Cross-tile merge call failed, falling back to naive local dedup: %s", e)
        return _naive_local_dedup(reconciled_items)


def _naive_local_dedup(reconciled_items: list[DetectedItem]) -> list[dict]:
    """Fallback: keep the highest-confidence detection per unique name, skip unknowns."""
    best_by_name: dict[str, DetectedItem] = {}
    for item in reconciled_items:
        if item.name == "unknown":
            continue
        existing = best_by_name.get(item.name)
        if existing is None or item.confidence > existing.confidence:
            best_by_name[item.name] = item
    return [{"name": name, "confidence": item.confidence} for name, item in best_by_name.items()]
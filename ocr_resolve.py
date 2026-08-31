"""
Resolve raw OCR text (from packaging) onto the shared ingredient vocabulary.

This sits between your existing OCR step and the merge step: OCR gives you
noisy text like "KRAFT SHREDDED CHEDDAR CHEESE 8OZ", and this module turns
that into a vocab-clean {"name": "cheddar cheese", "confidence": 0.9}.

Tries alias lookup first (cheap, deterministic, no API call) for every
tile. Anything that doesn't alias-match gets resolved in a SINGLE batched
LLM call for the whole photo, instead of one call per unresolved tile —
this is a meaningful chunk of the total API call count on a rate/quota-
limited free tier, since most photos have several tiles with no OCR text
at all (produce, unlabeled items) that would otherwise all trigger their
own wasted fallback call.
"""

from __future__ import annotations

import logging

from llm_client import chat_completion_with_fallback, extract_json, response_format_kwarg
from ingredient_vocab import UNKNOWN, match_text_to_vocab, vocab_prompt_block
from detection import DetectedItem

logger = logging.getLogger(__name__)

BATCH_RESOLVE_SYSTEM_PROMPT = """You map raw text scanned off food packaging to ingredient names, for \
several separate items at once. Each item has an id and its raw OCR text.

For each item, choose ONLY from this list:
{vocab}

If nothing on the list reasonably matches an item's text, return "unknown" for that item.

Return strict JSON only, no markdown fences, matching this shape:
{{"results": [{{"id": <id from input>, "name": "<vocab entry or 'unknown'>", "confidence": 0.0}}]}}
"""


def _alias_resolve(ocr_text: str, tile_id: int | None) -> DetectedItem | None:
    """Try the free, local alias/vocab match. Returns None if nothing matched (needs LLM fallback)."""
    if not ocr_text or not ocr_text.strip():
        return DetectedItem(name=UNKNOWN, confidence=0.0, raw_guess="", source="ocr", tile_id=tile_id)

    alias_match = match_text_to_vocab(ocr_text)
    if alias_match:
        return DetectedItem(
            name=alias_match,
            confidence=0.9,  # alias/substring matches are treated as high-confidence
            raw_guess=ocr_text,
            source="ocr",
            tile_id=tile_id,
        )
    return None


def resolve_ocr_text(ocr_text: str, tile_id: int | None = None, use_llm_fallback: bool = True) -> DetectedItem:
    """
    Resolve a single OCR text blob. Kept for cases where you have just one
    text to resolve outside the main batch flow — resolve_ocr_batch is the
    one actually used by main.py, since it batches the LLM fallback.
    """
    alias_result = _alias_resolve(ocr_text, tile_id)
    if alias_result is not None:
        return alias_result

    if not use_llm_fallback:
        return DetectedItem(name=UNKNOWN, confidence=0.0, raw_guess=ocr_text, source="ocr", tile_id=tile_id)

    batch_result = _resolve_batch_via_llm([(tile_id, ocr_text)])
    return batch_result.get(tile_id, DetectedItem(name=UNKNOWN, confidence=0.0, raw_guess=ocr_text, source="ocr", tile_id=tile_id))


def _resolve_batch_via_llm(pending: list[tuple[int, str]]) -> dict[int, DetectedItem]:
    """
    Send all unresolved (tile_id, ocr_text) pairs in ONE LLM call. Returns
    a dict keyed by tile_id — callers should treat a missing key as
    "resolution failed for this one, fall back to unknown" rather than
    letting one bad entry break the whole batch.
    """
    if not pending:
        return {}

    input_block = "\n".join(f'- id={tid}, text="{text}"' for tid, text in pending)

    try:
        response = chat_completion_with_fallback(
            messages=[
                {"role": "system", "content": BATCH_RESOLVE_SYSTEM_PROMPT.format(vocab=vocab_prompt_block())},
                {"role": "user", "content": input_block},
            ],
            temperature=0.1,
            **response_format_kwarg(),
        )
        parsed = extract_json(response.choices[0].message.content)
    except Exception as e:
        logger.warning("Batched OCR resolve failed for %d items: %s", len(pending), e)
        return {}

    text_by_id = dict(pending)
    resolved: dict[int, DetectedItem] = {}
    for entry in parsed.get("results", []):
        try:
            tile_id = int(entry.get("id"))
        except (TypeError, ValueError):
            continue
        if tile_id not in text_by_id:
            continue
        name = str(entry.get("name", UNKNOWN)).strip().lower()
        confidence = float(entry.get("confidence", 0.0))
        resolved[tile_id] = DetectedItem(
            name=name, confidence=confidence, raw_guess=text_by_id[tile_id], source="ocr", tile_id=tile_id,
        )
    return resolved


def resolve_ocr_batch(ocr_results: list[tuple[int, str]]) -> list[DetectedItem]:
    """
    `ocr_results` is a list of (tile_id, ocr_text) pairs from your existing
    OCR step. Alias-matches everything possible locally (no API calls),
    then resolves everything else in a SINGLE batched LLM call instead of
    one call per unresolved tile.
    """
    results: list[DetectedItem] = []
    pending: list[tuple[int, str]] = []

    for tile_id, text in ocr_results:
        alias_result = _alias_resolve(text, tile_id)
        if alias_result is not None:
            results.append(alias_result)
        else:
            pending.append((tile_id, text))

    if pending:
        resolved = _resolve_batch_via_llm(pending)
        for tile_id, text in pending:
            results.append(resolved.get(
                tile_id,
                DetectedItem(name=UNKNOWN, confidence=0.0, raw_guess=text, source="ocr", tile_id=tile_id),
            ))

    return results
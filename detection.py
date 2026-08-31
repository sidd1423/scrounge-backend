"""
Per-tile ingredient detection against the constrained vocabulary.

Drop-in replacement for whatever function currently sends a tile crop to
GPT-4o-mini in your /detect-multi flow. Swap your existing call site to
use `detect_tile()` instead, then pass its output into merge.py alongside
ocr_resolve.py's output.

Requires: openai>=1.0
"""

from __future__ import annotations

import base64
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from llm_client import chat_completion_with_fallback, extract_json, response_format_kwarg
from ingredient_vocab import UNKNOWN, vocab_prompt_block
from composite import build_montage, group_tiles

logger = logging.getLogger(__name__)

DETECTION_SYSTEM_PROMPT = """You are identifying food ingredients in a photo montage. The image is a grid of \
separate crops taken from inside a refrigerator, arranged in cells with a black border between them. Each \
cell has a bold number in a yellow box in its top-left corner — that number identifies the cell.

Examine EACH numbered cell independently and identify any food ingredient(s) visible in it. Cells are \
unrelated crops from different parts of the fridge — do not assume adjacent cells show the same item unless \
they visually clearly do.

Choose each item ONLY from this list:
{vocab}

Disambiguation guidance for commonly confused items — look closely at color and texture before deciding:
- Sliced cheese vs. deli meat: cheese slices are typically pale yellow, white, or orange with a \
smooth, uniform, semi-translucent appearance and clean straight edges. Deli meat (ham, turkey, \
salami) is pink, red, or tan, often with visible grain, marbling, or a slightly glossy/wet surface, \
and slices are often folded or ruffled rather than perfectly flat. If genuinely unsure which one, \
say so directly in raw_guess (e.g. "pale sliced item, could be cheese or deli meat") rather than \
guessing confidently.
- Yogurt vs. sour cream vs. cottage cheese: check the container shape/label color if visible — \
these are frequently confused in generic tubs. Note any visible label text in raw_guess.
- Bell pepper vs. tomato: bell peppers have a glossy, ridged/lobed shape with a stem indent; \
tomatoes are rounder and smoother.
- Broccoli vs. cauliflower: color is the main cue — green means broccoli, white/pale means cauliflower.
- Green onion vs. celery: green onion is thin (pencil-width) with a white base; celery stalks are \
wider and ribbed.
- Cheese (block, sliced, or shredded) is a common fridge item — look carefully for pale yellow/white/orange \
blocks or slices, often in plastic wrap or a resealable bag, before concluding a cell has no food in it.

Condiment/door-shelf items: some cells may show a full-height crop of a door shelf with bottles/jars \
standing upright, possibly cut off at the top or bottom edge — identify anyway from whatever is visible \
(cap color, bottle shape, body color, partial label text) rather than marking it unknown just because \
it's not fully in frame. Common condiment visual cues:
- Ketchup: red bottle, often with a flip-top cap.
- Mustard: yellow bottle or jar, sometimes a squeeze bottle with a pointed cap.
- Mayonnaise: white/cream contents, often in a jar or squeeze bottle with a blue or white label.
- Soy sauce: small dark glass or plastic bottle, often with a red or green cap.
- Hot sauce: tall thin bottle, often red or orange contents, wide variety of cap colors.
- BBQ sauce: dark brown/reddish contents, typically a squeeze bottle or jar with a wide label.
- Salad dressing: often a tall bottle with visible separated layers (oil/vinegar) or creamy contents.
- Pickles/olives/capers: clear glass jar, contents visible through the glass (green/olive-colored).
If any text is visible on a label even partially, include it verbatim in raw_guess — this is \
matched against known brand/product aliases downstream even when it's not a full match to the list.

Rules:
- Do NOT force an item onto the vocabulary list just because the list is there. Most cells in this montage \
will show shelving, packaging you can't identify, a partial/blurry item, or genuinely nothing recognizable — \
for those cells, the correct response is an empty "items" list. Only report an item when you can actually \
see something that resembles a specific food or bottle/container in that cell. Inventing a plausible-sounding \
item from the list (e.g. reporting "tortillas" or "strawberries" because they're on the list, not because \
you can see them) is a serious error — it is far better to under-report than to guess an item into existence.
- If an item is visible but does not match anything on the list, or you are not confident, set "name" to "unknown". \
This is different from seeing nothing at all — "unknown" means "something food-like is here but I can't place it," \
not "let me pick something from the list anyway."
- Always fill "raw_guess" with what you actually see, even when name is "unknown" or when you are confident \
— this lets us catch ingredients that should be added to the list later. Include distinguishing color/shape/ \
texture detail in raw_guess whenever the item was ambiguous, not just its generic category.
- If a cell shows no food item at all (e.g. it's shelving, a wall, an empty container), omit that cell \
from the response entirely, or include it with an empty "items" list. This is the MOST COMMON correct \
answer for many cells — do not treat an empty result as a failure to avoid.
- Confidence is your own calibrated 0.0-1.0 estimate, not just 1.0 for everything. When you use the \
disambiguation guidance above and are still not fully sure, reflect that with a lower confidence \
(0.4-0.6) rather than a high one.
- The "cell" number in your response MUST exactly match the number shown in that cell's yellow label box.

Return strict JSON only, no markdown fences, matching this shape:
{{"cells": [{{"cell": <number from the yellow label box>, "items": [{{"name": "<vocab entry or 'unknown'>", \
"confidence": 0.0, "raw_guess": "<what you see>"}}]}}]}}
"""


@dataclass
class DetectedItem:
    name: str
    confidence: float
    raw_guess: str
    source: str = "visual"
    tile_id: int | None = None

    def to_dict(self) -> dict:
        d = {"name": self.name, "confidence": self.confidence,
             "raw_guess": self.raw_guess, "source": self.source}
        if self.tile_id is not None:
            d["tile"] = self.tile_id
        return d


def _encode_image(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


def detect_montage(tiles: list[tuple[int, bytes]]) -> list[DetectedItem]:
    """
    Build one labeled grid montage from several tiles and send it as a
    SINGLE vision API call, asking the model to identify items per
    numbered cell. This is the main lever for cutting API call count —
    a 12-tile photo that used to need 12 calls now needs ~3 (with
    group_size=4), which matters a lot on a rate/quota-limited free tier.

    Never raises on a malformed response — logs and returns an empty
    list instead, so one bad montage doesn't kill the whole batch.
    """
    montage_bytes, tile_ids_in_montage = build_montage(tiles)
    b64 = _encode_image(montage_bytes)
    system_prompt = DETECTION_SYSTEM_PROMPT.format(vocab=vocab_prompt_block())
    valid_ids = set(tile_ids_in_montage)

    try:
        response = chat_completion_with_fallback(
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Identify the food ingredient(s) in each numbered cell "
                                                  f"({', '.join(str(t) for t in tile_ids_in_montage)})."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ],
                },
            ],
            temperature=0.1,
            **response_format_kwarg(),
        )
        raw = response.choices[0].message.content
        parsed = extract_json(raw)
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        logger.warning("Montage %s: failed to parse detection response: %s", tile_ids_in_montage, e)
        return []
    except Exception as e:
        logger.error("Montage %s: detection call failed: %s", tile_ids_in_montage, e)
        return []

    items = []
    for cell_entry in parsed.get("cells", []):
        try:
            cell_tile_id = int(cell_entry.get("cell"))
        except (TypeError, ValueError):
            logger.warning("Montage %s: response had a non-numeric cell id, skipping: %r",
                            tile_ids_in_montage, cell_entry.get("cell"))
            continue
        if cell_tile_id not in valid_ids:
            logger.warning("Montage %s: response referenced unknown cell %s, skipping",
                            tile_ids_in_montage, cell_tile_id)
            continue
        for raw_item in cell_entry.get("items", []):
            name = str(raw_item.get("name", UNKNOWN)).strip().lower()
            confidence = float(raw_item.get("confidence", 0.0))
            raw_guess = str(raw_item.get("raw_guess", "")).strip()
            items.append(DetectedItem(
                name=name,
                confidence=confidence,
                raw_guess=raw_guess,
                source="visual",
                tile_id=cell_tile_id,
            ))
    return items


def detect_tiles_batch(tiles: list[tuple[int, bytes]], group_size: int = 4, max_workers: int = 3) -> list[DetectedItem]:
    """
    Detect ingredients across all tiles using grouped montage calls
    instead of one call per tile. Groups run concurrently (a modest
    max_workers, since free-tier rate limits apply per-account, not
    per-model — too much concurrency just trades waiting for 429s).

    group_size trades off two things: larger groups mean fewer API calls
    (good for speed/quota) but smaller cells per montage (worse for fine
    detail — the cheese-vs-deli-meat kind of disambiguation). 4 is a
    reasonable default; drop to 2-3 if you see accuracy regress on small
    or ambiguous items, raise it if quota is still tight.
    """
    groups = group_tiles(tiles, group_size=group_size)
    all_items: list[DetectedItem] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(detect_montage, group): [t[0] for t in group] for group in groups}
        for future in as_completed(futures):
            tile_ids = futures[future]
            try:
                all_items.extend(future.result())
            except Exception as e:
                logger.error("Montage %s: unexpected error in detection thread: %s", tile_ids, e)
    return all_items
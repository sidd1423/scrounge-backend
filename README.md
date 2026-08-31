# Scrounge Detection Service

Fridge photo(s) → ingredient list, using vision-model detection + local
Tesseract OCR, against a constrained ingredient vocabulary.

## How it works (current architecture)

```
Fridge photo(s)
   │
   ├─► tiling.py — split into a 3x3 overlapping grid + 3 full-height
   │   vertical strips (strips catch tall door-shelf bottles the grid
   │   would otherwise slice into pieces)
   │
   ├─► composite.py — group tiles (default: 4 per group) into a single
   │   labeled montage image per group, instead of one image per tile
   │
   ├─► detection.py — send each montage as ONE vision API call; the
   │   model identifies items per numbered cell, constrained to
   │   ingredient_vocab.py's list
   │
   ├─► ocr.py — local Tesseract OCR on each individual tile (free, no
   │   API call — reads text off packaging/labels)
   │
   ├─► ocr_resolve.py — match OCR text to the vocab locally (aliases),
   │   and batch everything unresolved into ONE LLM call for the whole
   │   photo (not one call per tile)
   │
   ├─► merge.py — reconcile visual vs. OCR detections per tile (OCR
   │   wins on packaged items when confident), build the final merge prompt
   │
   └─► cross_tile_merge.py — ONE LLM call to deduplicate across
       overlapping tiles/montages and produce the final ingredient list
```

Per photo, this is roughly **4-5 total API calls** (a handful of montage
detection calls + at most 1 batched OCR-resolve call + 1 final merge
call) — down from an early version of this pipeline that made ~25 calls
per photo (one per tile).

## Setup (Windows / PyCharm)

1. **Install Tesseract OCR** (the binary, not just the Python wrapper):
   Download and run the installer from
   https://github.com/UB-Mannheim/tesseract/wiki
   Default install path is usually `C:\Program Files\Tesseract-OCR\tesseract.exe`.
   If `pytesseract` can't find it automatically, open `ocr.py` and uncomment/set:
   ```python
   pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
   ```

2. **Create a virtual environment and install Python deps** (in PyCharm's terminal):
   ```
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Set an API key** — pick ONE of the two:

   **Free tier (OpenRouter):**
   ```
   $env:OPENROUTER_API_KEY="sk-or-..."
   ```
   Uses free vision models (NVIDIA Nemotron / Google Gemma variants — see
   "Switching models" below). Free, but noticeably less accurate than
   GPT-4o-mini, rate-limited (~20 requests/min), and capped at 50
   requests/day unless you've added $10+ of OpenRouter credit (which
   raises the cap to 1000/day at no extra per-call cost).

   **Paid (OpenAI, more accurate):**
   ```
   $env:OPENAI_API_KEY="sk-..."
   ```
   Uses GPT-4o-mini. Given the current ~4-5 calls/photo, this now costs
   only a few cents per photo — much cheaper than it was before the
   montage/batching changes.

   If both are set, OpenRouter is used (it's checked first). Set via
   Windows System Properties → Environment Variables for a permanent
   setting, or a `.env` file + `python-dotenv` if you prefer.

## Running

```
uvicorn main:app --reload --port 8000
```

Check it's alive:
```
curl.exe http://localhost:8000/health
```
(Use `curl.exe` specifically on Windows — plain `curl` is aliased to
PowerShell's `Invoke-WebRequest`, which doesn't understand `-X`/`-F`.)

Send a fridge photo (or several):
```
curl.exe -X POST http://localhost:8000/detect-multi -F "images=@fridge1.jpg" -F "images=@fridge2.jpg"
```

Response:
```json
{"ingredients": [{"name": "milk", "confidence": 0.9}, {"name": "cheese", "confidence": 0.85}]}
```

From your Flutter app, point the camera screen's multipart upload at
`http://<your-machine-ip>:8000/detect-multi` (use your machine's actual
LAN IP, not `localhost`, when testing from a phone/emulator).

## Reading the logs

`main.py` logs the reconciled item list (post visual+OCR reconciliation,
pre cross-tile merge) and the final list, one line per item — this is
the fastest way to diagnose an accuracy issue:

```
RECONCILED tile=3 name=yogurt conf=0.90 source=visual raw_guess='yogurt container'
FINAL name=yogurt conf=0.90
```

- **Item missing entirely** → check if it shows up in RECONCILED with
  `name=unknown` and a `raw_guess` describing it correctly (vocab/alias
  gap — add an entry to `ingredient_vocab.py`) or if it's absent from
  RECONCILED altogether (the model never saw/reported it at all — a
  recognition or montage-attribution issue).
- **Wrong item reported** → the model misidentified it; check
  `raw_guess` for what it actually thought it saw. If it's a plausible
  visual confusion (e.g. cheese vs. deli meat), add disambiguation
  guidance to `DETECTION_SYSTEM_PROMPT` in `detection.py`. If it's an
  invented item that isn't in the photo at all, that's a model-quality/
  hallucination issue — see the free-tier note below.
- **9→3 style collapse** (many raw detections, few final items) → check
  whether the RECONCILED items really were duplicates of the same
  physical object (correct) or genuinely different items that
  `cross_tile_merge.py` over-merged (a prompt-tuning issue in
  `merge.py`'s `build_cross_tile_merge_prompt`).

## Known trade-offs

- **Free-tier models hallucinate vocab items more than GPT-4o-mini.**
  Smaller free vision models sometimes force-fit a plausible item from
  the constrained vocabulary list even when nothing matching is actually
  in the photo, and can also miss real items entirely. `detection.py`'s
  prompt explicitly warns against this, but it's a real capability gap,
  not something a prompt can fully close. If accuracy matters more than
  cost, switch to `OPENAI_API_KEY` (GPT-4o-mini) — it's now cheap enough
  per photo that this is a reasonable default.
- **Montage group size (`group_size` in `detect_tiles_batch`, currently
  4) trades API-call count against per-item detail.** Larger groups =
  fewer/cheaper calls but smaller cells per image (harder for the model
  to make fine-grained calls). Drop to 2-3 if accuracy on small/ambiguous
  items matters more than call count.
- **OpenRouter's free model roster changes without notice.** `llm_client.py`
  tries a list of candidate free models and falls back automatically if
  one is delisted (a 404) or consistently too slow/rate-limited. If ALL
  candidates eventually stop working, check
  https://openrouter.ai/models?max_price=0 for current free vision models
  and set `OPENROUTER_MODEL=<model-id>` to override.

## File overview

| File | Role |
|---|---|
| `ingredient_vocab.py` | Shared vocabulary + aliases — no dependencies |
| `tiling.py` | Splits a fridge photo into a 3x3 grid + 3 full-height vertical strips |
| `composite.py` | Builds labeled grid montages from groups of tiles |
| `llm_client.py` | Picks OpenRouter (free) or OpenAI (paid) based on env vars; free-model fallback list; robust JSON extraction; request timeout |
| `detection.py` | Sends montages to the vision model, constrained to the vocab, parsed back per-cell |
| `ocr.py` | Local Tesseract text extraction per tile (no API call) |
| `ocr_resolve.py` | Maps OCR text to the vocab (alias match, then one batched LLM call for the rest) |
| `merge.py` | Reconciles visual vs. OCR per tile; builds the cross-tile merge prompt |
| `cross_tile_merge.py` | Sends the merge prompt to the LLM; naive local dedup as fallback |
| `main.py` | FastAPI app — the `/detect-multi` endpoint wiring all of the above, with per-item logging |

## Tuning knobs

- `tiling.py`: `GRID_ROWS`/`GRID_COLS`/`GRID_OVERLAP_FRACTION`,
  `VERTICAL_STRIPS`/`VERTICAL_STRIP_OVERLAP_FRACTION`, `MIN_TILE_DIMENSION`
  (crops smaller than this get upscaled before detection).
- `detection.py`: `detect_tiles_batch(..., group_size=4, max_workers=3)` —
  montage size and call concurrency.
- `merge.py`: `OCR_OVERRIDE_THRESHOLD` — how confident OCR needs to be
  before it overrides a visual guess for the same tile.
- `llm_client.py`: `FREE_VISION_MODEL_CANDIDATES`, `REQUEST_TIMEOUT_SECONDS`.
- `ingredient_vocab.py`: extend `INGREDIENT_VOCAB` and `INGREDIENT_ALIASES`
  as `raw_guess` values in the logs reveal gaps.

## Known gaps / next steps

- No caching/rate-limiting beyond the SDK-level timeout — repeated
  identical photos re-run the full pipeline.
- SigLIP verification (discussed early on, not implemented) would give a
  second independent signal to catch visual/OCR disagreement, at the
  cost of another model dependency — only worth it if disagreement turns
  out to be common in practice.
- Nothing reads inside opaque/covered containers — items in a covered
  bowl or foil won't be identified by any part of this pipeline.
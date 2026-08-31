"""
Shared LLM client configuration for the detection pipeline.

Supports two backends, chosen by which env var is set:

  - OPENROUTER_API_KEY set -> uses OpenRouter (default model: free-tier
    Qwen2.5-VL). This is the free option — see README for setup and the
    current free-tier model slug, since OpenRouter's free model names
    shift over time.
  - Otherwise falls back to OPENAI_API_KEY -> uses OpenAI directly
    (gpt-4o-mini), the original paid setup.

All three call sites (detection.py, ocr_resolve.py, cross_tile_merge.py)
import `client` and `DEFAULT_MODEL` from here instead of constructing
their own OpenAI() instance, so switching backends is a one-place change.
"""

from __future__ import annotations

import json
import logging
import os
import re

from openai import OpenAI

logger = logging.getLogger(__name__)

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# OpenRouter's free-tier model roster changes often — models get delisted
# with no notice (this is exactly what happened to the Qwen2.5-VL free
# endpoints). Rather than hardcode one slug, try these in order and
# remember whichever one actually works. Override with the
# OPENROUTER_MODEL env var to force a specific model, or check
# https://openrouter.ai/models?max_price=0 for the current free vision
# roster if all of these eventually get delisted too.
FREE_VISION_MODEL_CANDIDATES = [
    os.environ.get("OPENROUTER_MODEL"),  # explicit override, if set
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "nvidia/nemotron-nano-12b-v2-vl:free",
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
]
FREE_VISION_MODEL_CANDIDATES = [m for m in FREE_VISION_MODEL_CANDIDATES if m]

OPENAI_DEFAULT_MODEL = "gpt-4o-mini"

REQUEST_TIMEOUT_SECONDS = 40  # fail fast — the SDK default (600s) is far too long for a per-tile call
CLIENT_MAX_RETRIES = 1  # SDK-level auto-retry on transient errors; keep low so our own fallback loop controls total wait time

if OPENROUTER_API_KEY:
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=CLIENT_MAX_RETRIES,
    )
    DEFAULT_MODEL = FREE_VISION_MODEL_CANDIDATES[0]
    USING_OPENROUTER = True
elif OPENAI_API_KEY:
    client = OpenAI(
        api_key=OPENAI_API_KEY,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=CLIENT_MAX_RETRIES,
    )
    DEFAULT_MODEL = OPENAI_DEFAULT_MODEL
    USING_OPENROUTER = False
else:
    raise RuntimeError(
        "No API key found. Set OPENROUTER_API_KEY (free tier) or OPENAI_API_KEY "
        "(paid) as an environment variable before starting the server."
    )

# Once a candidate is confirmed working (or a different one is found to
# work after a 404), remember it here so we stop retrying dead models on
# every single call.
_working_model: str | None = None


def chat_completion_with_fallback(**kwargs):
    """
    Wraps client.chat.completions.create() with automatic fallback across
    FREE_VISION_MODEL_CANDIDATES when using OpenRouter and a model returns
    a 404 (delisted) error. Pass all normal chat.completions.create()
    kwargs except `model` — the model is chosen/retried internally.

    On OpenAI (non-OpenRouter), this just calls through once with
    OPENAI_DEFAULT_MODEL — no fallback list applies there.
    """
    global _working_model

    if not USING_OPENROUTER:
        return client.chat.completions.create(model=OPENAI_DEFAULT_MODEL, **kwargs)

    candidates = [_working_model] if _working_model else []
    candidates += [m for m in FREE_VISION_MODEL_CANDIDATES if m != _working_model]

    last_error = None
    for model in candidates:
        try:
            response = client.chat.completions.create(model=model, **kwargs)
            _working_model = model
            return response
        except Exception as e:
            last_error = e
            err_str = str(e)
            is_404 = "404" in err_str or "No endpoints found" in err_str
            is_timeout = "timeout" in err_str.lower() or "timed out" in err_str.lower()
            is_connection_error = "connection" in err_str.lower()
            is_rate_limited = "429" in err_str or "rate limit" in err_str.lower()
            if is_404 or is_timeout or is_connection_error or is_rate_limited:
                logger.warning("Model %s failed (%s), trying next candidate", model, err_str[:150])
                continue
            raise  # genuine errors (auth, malformed request, etc.) shouldn't trigger silent fallback

    raise RuntimeError(
        f"All free-tier vision model candidates failed (tried: {candidates}). "
        f"OpenRouter's free roster may have changed — check "
        f"https://openrouter.ai/models?max_price=0 for current options and "
        f"set OPENROUTER_MODEL to override. Last error: {last_error}"
    )


def extract_json(raw_text: str) -> dict:
    """
    Parse a JSON object out of a model response, tolerating models that
    don't strictly honor response_format (common on some OpenRouter-hosted
    models). Tries, in order:
      1. Direct json.loads on the whole string.
      2. Stripping ```json ... ``` or ``` ... ``` markdown fences.
      3. Extracting the first {...} block found in the text.
    Raises json.JSONDecodeError if none of these work, so callers can
    handle it the same way they handle any other parse failure.
    """
    text = raw_text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        return json.loads(brace_match.group(0))  # let this raise if it still fails

    raise json.JSONDecodeError("No JSON object found in response", text, 0)


def response_format_kwarg() -> dict:
    """
    OpenAI reliably supports response_format={"type": "json_object"}.
    Not all OpenRouter-hosted models honor it the same way, and some
    reject the parameter outright — omit it for OpenRouter and rely on
    extract_json() plus explicit "return JSON only" prompt instructions
    instead.
    """
    return {} if USING_OPENROUTER else {"response_format": {"type": "json_object"}}
# Description: LLM response parsing utilities shared across PDP consumers.
# Description: Fence stripping and JSON extraction with graceful fallbacks.

from __future__ import annotations

import json
import logging

log = logging.getLogger(__name__)


def strip_markdown_fences(text: str) -> str:
    """Strip markdown code fences from LLM responses.

    LLMs frequently wrap JSON in fences even when the prompt requests raw JSON.
    Handles opening fences with language tags (```json), plain fences (```),
    and closing fences independently.

    Args:
        text: Raw LLM response text.

    Returns:
        Text with outer code fences removed.
    """
    stripped = text.strip()
    if not stripped:
        return stripped
    if stripped.startswith("```"):
        newline_idx = stripped.find("\n")
        stripped = stripped[newline_idx + 1 :] if newline_idx != -1 else stripped[3:]
    if stripped.endswith("```"):
        stripped = stripped[:-3]
    return stripped.strip()


def parse_llm_json(text: str) -> dict | list | str:
    """Parse LLM response as JSON with fence-stripping and raw_decode fallback.

    Three-step parsing chain:
    1. Strip markdown fences, try json.loads()
    2. Fall back to json.JSONDecoder().raw_decode() for trailing text (e.g., mcporter stderr)
    3. Return the original text as-is if unparseable

    Args:
        text: Raw LLM response text, possibly fenced or with trailing content.

    Returns:
        Parsed dict/list on success, or the original text (str) if unparseable.
    """
    stripped = strip_markdown_fences(text)
    if not stripped:
        return text

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    try:
        decoder = json.JSONDecoder()
        result, _ = decoder.raw_decode(stripped)
        return result
    except (json.JSONDecodeError, ValueError):
        log.debug("Could not parse LLM response as JSON, returning raw text")
        return text

"""Text normalization helpers."""

from __future__ import annotations

import json
import re
from typing import Any

_DASH_TRANSLATION = str.maketrans(
    {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
    }
)


def is_missing(value: Any) -> bool:
    """Return True when a scalar should be treated as missing."""

    if value is None:
        return True
    if isinstance(value, float):
        try:
            return value != value
        except Exception:
            return False
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "na", "n/a", "none", "null"}


def clean_text(value: Any) -> str | None:
    """Convert a scalar into a stripped string, or None when missing."""

    if is_missing(value):
        return None
    return str(value).strip()


def normalize_column_name(value: Any) -> str:
    """Normalize a column header to a canonical lookup token."""

    text = clean_text(value) or ""
    text = text.translate(_DASH_TRANSLATION).lower()
    text = re.sub(r"[\s/_\-()]+", "", text)
    text = text.replace(".", "")
    return text


def normalize_cas(value: Any) -> str | None:
    """Normalize CAS strings while preserving invalid formats for QC."""

    text = clean_text(value)
    if text is None:
        return None
    text = text.translate(_DASH_TRANSLATION)
    text = re.sub(r"\s+", "", text)
    return text


def json_dumps(value: Any) -> str:
    """Serialize simple Python structures in a stable, compact way."""

    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def append_note(existing: Any, note: str | None) -> str | None:
    """Append a note segment to an existing note string."""

    current = clean_text(existing)
    addition = clean_text(note)
    if not addition:
        return current
    if not current:
        return addition
    if addition in current:
        return current
    return f"{current}; {addition}"

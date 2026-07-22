"""Limited obvious-identifier guard for synthetic prototype requests.

Regular expressions cannot establish de-identification. The API therefore rejects
clear matches instead of claiming that it can safely anonymize arbitrary data.
"""

from __future__ import annotations

import re

_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b", re.IGNORECASE),
    "phone": re.compile(
        r"(?<!\d)(?:(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]\d{3}[-.\s]\d{4}|(?:\+?86[-.\s]?)?1[3-9]\d{9})(?!\d)"
    ),
    "ssn": re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"),
    "ip_address": re.compile(
        r"(?<!\d)(?:25[0-5]|2[0-4]\d|1?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}(?!\d)"
    ),
    "medical_record_id": re.compile(
        r"\b(?:mrn|medical\s+record(?:\s+number)?|patient\s+id|member\s+id)\s*[:#-]?\s*[a-z0-9][a-z0-9-]{3,}\b",
        re.IGNORECASE,
    ),
}


def find_obvious_identifiers(text: str) -> list[str]:
    """Return stable identifier category names without returning matched values."""

    return sorted(
        name for name, pattern in _PATTERNS.items() if pattern.search(text or "")
    )

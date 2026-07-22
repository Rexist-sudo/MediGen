"""Combined Presidio and deterministic direct-identifier guard."""

from __future__ import annotations

import re
from functools import lru_cache

import structlog
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider

logger = structlog.get_logger(__name__)

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
    "labeled_name": re.compile(
        r"(?:(?:姓名|患者姓名|name)\s*[:：]\s*|patient\s+named\s+)[\u4e00-\u9fffA-Za-z][\u4e00-\u9fffA-Za-z .'-]{1,40}",
        re.IGNORECASE,
    ),
}

_PRESIDIO_ENTITIES = [
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "US_SSN",
    "IP_ADDRESS",
    "MEDICAL_LICENSE",
    "US_DRIVER_LICENSE",
    "CREDIT_CARD",
    "IBAN_CODE",
    "URL",
]

_ENTITY_NAMES = {
    "EMAIL_ADDRESS": "email",
    "PHONE_NUMBER": "phone",
    "US_SSN": "ssn",
    "IP_ADDRESS": "ip_address",
    "MEDICAL_LICENSE": "medical_license",
    "US_DRIVER_LICENSE": "driver_license",
    "CREDIT_CARD": "credit_card",
    "IBAN_CODE": "bank_account",
    "URL": "url",
}


@lru_cache(maxsize=1)
def get_presidio_analyzer() -> AnalyzerEngine:
    configuration = {
        "nlp_engine_name": "spacy",
        "models": [
            {
                "lang_code": "en",
                "model_name": "en_core_web_sm",
            }
        ],
    }
    nlp_engine = NlpEngineProvider(nlp_configuration=configuration).create_engine()
    return AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])


def presidio_is_ready() -> bool:
    try:
        analyzer = get_presidio_analyzer()
        analyzer.analyze(text="Synthetic readiness text.", language="en", entities=["EMAIL_ADDRESS"])
        return True
    except Exception as exc:
        logger.warning("presidio.readiness_failed", error_type=type(exc).__name__)
        return False


def find_obvious_identifiers(text: str) -> list[str]:
    """Return stable identifier categories without exposing matched values."""

    value = text or ""
    findings = {
        name for name, pattern in _PATTERNS.items() if pattern.search(value)
    }
    try:
        results = get_presidio_analyzer().analyze(
            text=value,
            language="en",
            entities=_PRESIDIO_ENTITIES,
            score_threshold=0.65,
        )
        findings.update(
            _ENTITY_NAMES[result.entity_type]
            for result in results
            if result.entity_type in _ENTITY_NAMES
        )
    except Exception as exc:
        logger.warning("presidio.scan_failed", error_type=type(exc).__name__)
    if "labeled_name" in findings:
        findings.remove("labeled_name")
        findings.add("person_name")
    return sorted(findings)

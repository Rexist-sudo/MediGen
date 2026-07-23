"""Shared test bootstrap for the source-layout MediGen package."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


PYTHON_ROOT = Path(__file__).resolve().parents[1]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))


@pytest.fixture
def catalog_path() -> Path:
    return PYTHON_ROOT / "data" / "recommendation" / "knowledge_topics.jsonl"


@pytest.fixture
def topic_store(catalog_path):
    from src.services.recommendation.topic_store import TopicStore

    return TopicStore.from_jsonl(catalog_path)


def pytest_collection_modifyitems(config, items) -> None:
    """Keep heavyweight profiles opt-in while making ``-m`` self-sufficient."""

    expression = str(config.getoption("markexpr") or "")
    for marker_name in ("model", "integration"):
        if marker_name in expression:
            continue
        reason = f"run explicitly with pytest -m {marker_name}"
        skip = pytest.mark.skip(reason=reason)
        for item in items:
            if item.get_closest_marker(marker_name) is not None:
                item.add_marker(skip)

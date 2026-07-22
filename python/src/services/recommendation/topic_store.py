"""Validated JSONL fallback catalog for education topics."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from ...models.recommendation import KnowledgeTopic


class TopicStoreError(RuntimeError):
    """The fixed topic catalog could not be loaded safely."""


def resolve_topic_path(configured_path: str) -> Path:
    path = Path(configured_path)
    if path.is_absolute():
        return path.resolve()
    python_root = Path(__file__).resolve().parents[3]
    return (python_root / path).resolve()


class TopicStore:
    def __init__(self, topics: list[KnowledgeTopic]):
        self._topics = tuple(topics)
        self._by_id = {topic.topic_id: topic for topic in topics}

    @classmethod
    def from_jsonl(cls, path: str) -> "TopicStore":
        resolved = resolve_topic_path(path)
        topics: list[KnowledgeTopic] = []
        seen: set[str] = set()
        try:
            with resolved.open("r", encoding="utf-8") as handle:
                for line_number, raw_line in enumerate(handle, start=1):
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        topic = KnowledgeTopic.model_validate(json.loads(line))
                    except (json.JSONDecodeError, ValidationError) as exc:
                        raise TopicStoreError(
                            f"invalid topic record at line {line_number}"
                        ) from exc
                    if topic.topic_id in seen:
                        raise TopicStoreError(
                            f"duplicate topic_id at line {line_number}"
                        )
                    seen.add(topic.topic_id)
                    topics.append(topic)
        except OSError as exc:
            raise TopicStoreError("topic catalog is unavailable") from exc

        if not topics:
            raise TopicStoreError("topic catalog is empty")
        if not any(topic.status in {"active", "prototype"} for topic in topics):
            raise TopicStoreError("topic catalog has no active topics")
        return cls(topics)

    def list_active(self) -> list[KnowledgeTopic]:
        return [
            topic
            for topic in self._topics
            if topic.status in {"active", "prototype"}
        ]

    def get(self, topic_id: str) -> KnowledgeTopic | None:
        return self._by_id.get(topic_id)

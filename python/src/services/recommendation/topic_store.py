"""Validated topic catalog and stable Direct-SID token registry."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ...models.recommendation import KnowledgeTopic
from .ranker_protocol import CONTROL_TOKENS, ModelArtifactError


class TopicStoreError(RuntimeError):
    """The fixed topic catalog could not be loaded safely."""


def resolve_topic_path(configured_path: str | Path) -> Path:
    path = Path(configured_path)
    if path.is_absolute():
        return path.resolve()
    python_root = Path(__file__).resolve().parents[3]
    return (python_root / path).resolve()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class TopicStore:
    def __init__(self, topics: list[KnowledgeTopic], *, source_path: Path | None = None):
        if not topics:
            raise TopicStoreError("topic catalog is empty")
        ids = [topic.topic_id for topic in topics]
        tokens = [topic.topic_token for topic in topics]
        if len(ids) != len(set(ids)):
            raise TopicStoreError("duplicate topic_id")
        if len(tokens) != len(set(tokens)):
            raise TopicStoreError("duplicate topic_token")
        if not any(topic.status == "active" for topic in topics):
            raise TopicStoreError("topic catalog has no active topics")

        self._topics = tuple(topics)
        self._by_id = {topic.topic_id: topic for topic in topics}
        self._by_token = {topic.topic_token: topic for topic in topics}
        self.source_path = source_path
        canonical_rows = [
            topic.model_dump(mode="json")
            for topic in sorted(topics, key=lambda item: item.topic_id)
        ]
        self._catalog_sha256 = canonical_json_sha256(canonical_rows)

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        *,
        token_map_path: str | Path | None = None,
        require_token_map: bool = True,
    ) -> "TopicStore":
        resolved = resolve_topic_path(path)
        topics: list[KnowledgeTopic] = []
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
                    topics.append(topic)
        except OSError as exc:
            raise TopicStoreError("topic catalog is unavailable") from exc

        store = cls(topics, source_path=resolved)
        mapping_path = (
            resolve_topic_path(token_map_path)
            if token_map_path is not None
            else resolved.with_name("topic_token_map.json")
        )
        if mapping_path.exists():
            store.validate_token_map(mapping_path)
        elif require_token_map:
            raise TopicStoreError("topic token map is unavailable")
        return store

    def validate_token_map(self, path: str | Path) -> None:
        resolved = resolve_topic_path(path)
        try:
            payload = json.loads(resolved.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TopicStoreError("topic token map is invalid") from exc
        mapping = payload.get("topic_id_to_token") if isinstance(payload, dict) else None
        if not isinstance(mapping, dict):
            raise TopicStoreError("topic token map has no topic_id_to_token")
        clean = {str(key): str(value) for key, value in mapping.items()}
        if clean != self.topic_id_to_token():
            raise TopicStoreError("topic token map does not match catalog")

    def list_active(self) -> list[KnowledgeTopic]:
        return [topic for topic in self._topics if topic.status == "active"]

    def list_all(self) -> list[KnowledgeTopic]:
        return list(self._topics)

    def get_by_id(
        self,
        topic_id: str,
        *,
        include_inactive: bool = False,
    ) -> KnowledgeTopic | None:
        topic = self._by_id.get(topic_id)
        if topic is None:
            return None
        if not include_inactive and topic.status != "active":
            return None
        return topic

    def get(self, topic_id: str) -> KnowledgeTopic | None:
        """Compatibility alias for callers that need an active topic."""

        return self.get_by_id(topic_id)

    def get_by_token(self, topic_token: str) -> KnowledgeTopic | None:
        return self._by_token.get(topic_token)

    def topic_id_to_token(self) -> dict[str, str]:
        return {
            topic.topic_id: topic.topic_token
            for topic in sorted(self._topics, key=lambda item: item.topic_id)
        }

    def token_to_topic_id(self) -> dict[str, str]:
        return {
            topic.topic_token: topic.topic_id
            for topic in sorted(self._topics, key=lambda item: item.topic_token)
        }

    def catalog_sha256(self) -> str:
        return self._catalog_sha256

    def validate_tokenizer(self, tokenizer) -> None:
        for token in [*self.token_to_topic_id(), *CONTROL_TOKENS]:
            token_id = tokenizer.convert_tokens_to_ids(token)
            encoded = tokenizer.encode(token, add_special_tokens=False)
            if token_id is None or token_id == tokenizer.unk_token_id:
                raise ModelArtifactError("unknown_topic_token")
            if encoded != [token_id]:
                raise ModelArtifactError("topic_token_not_single_id")

    def token_map_payload(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "topic_id_to_token": self.topic_id_to_token(),
        }

"""Validate and order interaction history against the complete topic registry."""

from __future__ import annotations

from datetime import timezone

from ...models.recommendation import UserHistoryContext
from .ranker_protocol import NormalizedHistory
from .topic_store import TopicStore


class HistoryNormalizer:
    def __init__(self, topic_store: TopicStore, max_history: int):
        self._topic_store = topic_store
        self._max_history = max(0, max_history)

    def normalize(
        self,
        history_context: UserHistoryContext | None,
    ) -> NormalizedHistory:
        source = list(history_context.interactions if history_context else ())
        valid = [
            item
            for item in source
            if self._topic_store.get_by_id(item.topic_id, include_inactive=True)
            is not None
        ]
        dropped = len(source) - len(valid)
        warnings: list[str] = []
        if dropped:
            warnings.append("unknown_history_topic_ignored")

        timed = [item for item in valid if item.occurred_at is not None]
        untimed = [item for item in valid if item.occurred_at is None]
        if timed and untimed:
            warnings.append("history_missing_timestamps")

        def timestamp(item) -> float:
            occurred_at = item.occurred_at
            if occurred_at is None:
                return 0.0
            if occurred_at.tzinfo is None:
                occurred_at = occurred_at.replace(tzinfo=timezone.utc)
            return occurred_at.timestamp()

        if timed:
            timed = sorted(timed, key=timestamp)
            valid = [*timed, *untimed]
        if self._max_history == 0:
            valid = []
        else:
            valid = valid[-self._max_history :]

        return NormalizedHistory(
            interactions=tuple(valid),
            valid_count=len(valid),
            dropped_unknown_count=dropped,
            warnings=tuple(warnings),
        )


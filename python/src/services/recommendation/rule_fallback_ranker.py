"""Deterministic availability fallback for already-approved candidates."""

from __future__ import annotations

from time import perf_counter

from ...models.recommendation import KnowledgeTopic, TopicCategory
from .candidate_policy import clinical_signals
from .ranker_protocol import RankerInput, RankerResult
from .topic_store import TopicStore


class RuleFallbackRanker:
    STRATEGY = "rule_v1_fallback"

    def __init__(self, topic_store: TopicStore | None = None):
        self._topic_store = topic_store

    def rank(self, ranker_input: RankerInput) -> RankerResult:
        started = perf_counter()
        preferences = ranker_input.preferences
        preferred = set(preferences.preferred_categories if preferences else ())
        events_by_topic: dict[str, set[str]] = {}
        for interaction in ranker_input.history.interactions:
            events_by_topic.setdefault(interaction.topic_id, set()).add(
                interaction.event_type
            )
        positive_categories: set[TopicCategory] = set()
        for interaction in ranker_input.history.interactions:
            if interaction.event_type not in {"helpful", "save"}:
                continue
            topic = None
            if self._topic_store is not None:
                topic = self._topic_store.get_by_id(
                    interaction.topic_id,
                    include_inactive=True,
                )
            if topic is None:
                topic = next(
                    (
                        item
                        for item in ranker_input.candidates
                        if item.topic_id == interaction.topic_id
                    ),
                    None,
                )
            if topic is not None:
                positive_categories.add(topic.category)

        scored: list[tuple[int, KnowledgeTopic]] = []
        for topic in ranker_input.candidates:
            code, diagnosis, test, medication = clinical_signals(
                ranker_input.context,
                topic,
            )
            score = topic.priority
            score += 10 if code else 0
            score += 6 if test else 0
            score += 5 if diagnosis else 0
            score += 3 if medication else 0
            score += 3 if topic.category in preferred else 0
            if (
                topic.category in positive_categories
                and topic.topic_id not in events_by_topic
            ):
                score += 2
            if preferences:
                score += int(preferences.preferred_depth == topic.depth)
                score += int(preferences.preferred_format == topic.format)
                score += int(
                    preferences.max_reading_minutes is not None
                    and topic.estimated_reading_minutes
                    <= preferences.max_reading_minutes
                )
            topic_events = events_by_topic.get(topic.topic_id, set())
            score -= 3 if "view" in topic_events else 0
            score -= 2 if topic_events.intersection({"helpful", "save"}) else 0
            scored.append((score, topic))

        scored.sort(key=lambda item: (-item[0], item[1].topic_id))
        limit = max(0, min(ranker_input.top_k, len(scored)))
        return RankerResult(
            topic_ids=tuple(item.topic_id for _, item in scored[:limit]),
            strategy_used=self.STRATEGY,
            inference_ms=round((perf_counter() - started) * 1000, 3),
        )


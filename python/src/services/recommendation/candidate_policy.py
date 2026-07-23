"""Safety and eligibility policy executed before any learned ranker."""

from __future__ import annotations

import unicodedata

from ...models.recommendation import (
    KnowledgeTopic,
    RecommendationContext,
    TopicCategory,
    UserPreferenceContext,
)
from .ranker_protocol import (
    CandidatePolicyResult,
    NormalizedHistory,
    TopicMatchSignals,
)
from .topic_store import TopicStore


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "")
    return " ".join(value.casefold().strip().split())


def _contains_match(context_values: list[str], topic_values: list[str]) -> bool:
    context = [normalize(item) for item in context_values if normalize(item)]
    topic = [normalize(item) for item in topic_values if normalize(item)]
    return any(
        right in left or left in right
        for left in context
        for right in topic
        if left and right
    )


def clinical_signals(
    context: RecommendationContext,
    topic: KnowledgeTopic,
) -> tuple[bool, bool, bool, bool]:
    context_codes = [normalize(code).replace(" ", "") for code in context.diagnosis_codes]
    topic_codes = [normalize(code).replace(" ", "") for code in topic.related_codes]
    code_match = any(
        context_code.startswith(topic_code)
        for context_code in context_codes
        for topic_code in topic_codes
        if context_code and topic_code
    )
    return (
        code_match,
        _contains_match(context.diagnosis_terms, topic.related_terms),
        _contains_match(context.recommended_tests, topic.related_tests),
        topic.category == TopicCategory.MEDICATION_SAFETY
        and _contains_match(context.medication_names, topic.related_medications),
    )


class CandidatePolicy:
    def __init__(self, topic_store: TopicStore, *, allow_prototype: bool = False):
        self._topic_store = topic_store
        self._allow_prototype = allow_prototype

    def apply(
        self,
        *,
        context: RecommendationContext,
        preferences: UserPreferenceContext | None,
        history: NormalizedHistory,
        recalled_topics: list[KnowledgeTopic],
        all_active_topics: list[KnowledgeTopic],
        top_k: int,
        max_candidates: int,
    ) -> CandidatePolicyResult:
        top_k = max(1, min(3, top_k))
        max_candidates = max(1, max_candidates)
        preferred = set(preferences.preferred_categories if preferences else ())
        excluded_categories = set(
            preferences.excluded_categories if preferences else ()
        )
        events_by_topic: dict[str, set[str]] = {}
        for interaction in history.interactions:
            events_by_topic.setdefault(interaction.topic_id, set()).add(
                interaction.event_type
            )
        negative_ids = {
            topic_id
            for topic_id, events in events_by_topic.items()
            if events.intersection({"dismiss", "not_helpful"})
        }
        positive_categories = {
            topic.category
            for interaction in history.interactions
            if interaction.event_type in {"save", "helpful"}
            and (
                topic := self._topic_store.get_by_id(
                    interaction.topic_id,
                    include_inactive=True,
                )
            )
            is not None
        }

        recalled: list[KnowledgeTopic] = []
        seen: set[str] = set()
        for raw_topic in recalled_topics:
            if raw_topic.topic_id in seen:
                continue
            registry_topic = self._topic_store.get_by_id(
                raw_topic.topic_id,
                include_inactive=True,
            )
            if registry_topic is None:
                continue
            if registry_topic.status != "active" and not (
                self._allow_prototype and registry_topic.status == "prototype"
            ):
                continue
            seen.add(registry_topic.topic_id)
            recalled.append(registry_topic)

        signals: dict[str, TopicMatchSignals] = {}

        def signals_for(topic: KnowledgeTopic) -> TopicMatchSignals:
            code, diagnosis, test, medication = clinical_signals(context, topic)
            item = TopicMatchSignals(
                code_match=code,
                diagnosis_match=diagnosis,
                test_match=test,
                medication_match=medication,
                preferred_category_match=topic.category in preferred,
                preferred_depth_match=bool(
                    preferences
                    and preferences.preferred_depth == topic.depth
                ),
                preferred_format_match=bool(
                    preferences
                    and preferences.preferred_format == topic.format
                ),
                reading_time_match=bool(
                    preferences
                    and preferences.max_reading_minutes is not None
                    and topic.estimated_reading_minutes
                    <= preferences.max_reading_minutes
                ),
                viewed_before="view" in events_by_topic.get(topic.topic_id, set()),
                positive_same_category=(
                    topic.category in positive_categories
                    and topic.topic_id not in events_by_topic
                ),
            )
            signals[topic.topic_id] = item
            return item

        pinned: list[KnowledgeTopic] = []
        rankable: list[KnowledgeTopic] = []
        excluded_ids: set[str] = set(negative_ids)
        warnings: list[str] = list(history.warnings)

        for topic in recalled:
            item_signals = signals_for(topic)
            if topic.general_fallback or not item_signals.clinical_match:
                continue
            if topic.topic_id in negative_ids:
                excluded_ids.add(topic.topic_id)
                continue
            safety_pin = topic.mandatory_safety and item_signals.strong_safety_match
            if topic.category in excluded_categories and not safety_pin:
                excluded_ids.add(topic.topic_id)
                continue
            if safety_pin:
                pinned.append(topic)
                if topic.category in excluded_categories:
                    warnings.append("mandatory_safety_override")
            else:
                rankable.append(topic)

        pinned.sort(key=lambda item: (-item.priority, item.topic_id))
        pinned = pinned[:top_k]
        pinned_ids = {topic.topic_id for topic in pinned}

        fallback_pool: list[KnowledgeTopic] = []
        active_by_id = {topic.topic_id: topic for topic in all_active_topics}
        for topic in active_by_id.values():
            if topic.status != "active" or not topic.general_fallback:
                continue
            if topic.topic_id in negative_ids or topic.category in excluded_categories:
                excluded_ids.add(topic.topic_id)
                continue
            if topic.topic_id in pinned_ids:
                continue
            signals_for(topic)
            fallback_pool.append(topic)
        fallback_pool.sort(key=lambda item: (-item.priority, item.topic_id))

        if not context.demo_safe:
            warnings.append("unsafe_context")
            for topic in fallback_pool:
                if len(pinned) >= top_k:
                    break
                pinned.append(topic)
            rankable = []
        else:
            rankable_ids = {topic.topic_id for topic in rankable}
            required = max(0, top_k - len(pinned))
            for topic in fallback_pool:
                if len(rankable) >= required:
                    break
                if topic.topic_id not in rankable_ids:
                    rankable.append(topic)
                    rankable_ids.add(topic.topic_id)

        rankable = [
            topic
            for topic in rankable
            if topic.topic_id not in pinned_ids
            and topic.topic_id not in excluded_ids
        ][:max_candidates]
        return CandidatePolicyResult(
            pinned_topics=tuple(pinned),
            rankable_topics=tuple(rankable),
            signals_by_topic_id=signals,
            warnings=tuple(dict.fromkeys(warnings)),
            excluded_topic_ids=tuple(sorted(excluded_ids)),
        )


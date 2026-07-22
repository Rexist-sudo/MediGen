"""Stable rule-based ranking behind the MiniOneRec-Lite-compatible contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ...models.recommendation import (
    KnowledgeTopic,
    RecommendationContext,
    TopicCategory,
    TopicInteraction,
    UserPreferenceContext,
)


def normalize(value: str) -> str:
    return " ".join((value or "").casefold().strip().split())


@dataclass(frozen=True)
class MatchSignals:
    code: bool = False
    test: bool = False
    diagnosis: bool = False
    medication: bool = False

    @property
    def clinical(self) -> bool:
        return self.code or self.test or self.diagnosis or self.medication

    @property
    def strong_safety(self) -> bool:
        return self.code or self.test or self.diagnosis


@dataclass
class RankingResult:
    topics: list[KnowledgeTopic]
    candidate_count: int
    signals_by_topic: dict[str, MatchSignals] = field(default_factory=dict)
    safety_overrides: list[str] = field(default_factory=list)


class RecommendationRanker(Protocol):
    def rank(
        self,
        *,
        context: RecommendationContext,
        topics: list[KnowledgeTopic],
        preferences: UserPreferenceContext | None,
        history: list[TopicInteraction],
        top_k: int,
    ) -> list[KnowledgeTopic]: ...


class RuleRecommendationRanker:
    """Deterministic rule ranker with stable topic-id tie breaking."""

    def rank(
        self,
        *,
        context: RecommendationContext,
        topics: list[KnowledgeTopic],
        preferences: UserPreferenceContext | None,
        history: list[TopicInteraction],
        top_k: int,
    ) -> list[KnowledgeTopic]:
        return self.rank_detailed(
            context=context,
            topics=topics,
            preferences=preferences,
            history=history,
            top_k=top_k,
        ).topics

    def rank_detailed(
        self,
        *,
        context: RecommendationContext,
        topics: list[KnowledgeTopic],
        preferences: UserPreferenceContext | None,
        history: list[TopicInteraction],
        top_k: int,
    ) -> RankingResult:
        top_k = max(1, min(3, top_k))
        topic_by_id = {topic.topic_id: topic for topic in topics}
        events_by_topic: dict[str, set[str]] = {}
        for interaction in history:
            events_by_topic.setdefault(interaction.topic_id, set()).add(
                interaction.event_type
            )

        excluded = set(preferences.excluded_categories if preferences else [])
        preferred = set(preferences.preferred_categories if preferences else [])
        negative_ids = {
            topic_id
            for topic_id, events in events_by_topic.items()
            if events.intersection({"dismiss", "not_helpful"})
        }
        consumed_ids = set(events_by_topic)
        helpful_categories = {
            topic_by_id[item.topic_id].category
            for item in history
            if item.event_type in {"helpful", "save"}
            and item.topic_id in topic_by_id
        }

        scored: list[tuple[int, KnowledgeTopic]] = []
        signals_by_topic: dict[str, MatchSignals] = {}
        safety_overrides: list[str] = []

        for topic in topics:
            if topic.status not in {"active", "prototype"} or topic.general_fallback:
                continue
            signals = self._match(context, topic)
            if not signals.clinical:
                continue
            override = topic.mandatory_safety and signals.strong_safety
            if topic.category in excluded and not override:
                continue
            if topic.topic_id in negative_ids and not override:
                continue
            if override and (
                topic.category in excluded or topic.topic_id in negative_ids
            ):
                safety_overrides.append(topic.topic_id)

            signals_by_topic[topic.topic_id] = signals
            scored.append(
                (
                    self._score(
                        topic=topic,
                        signals=signals,
                        preferences=preferences,
                        preferred=preferred,
                        events=events_by_topic.get(topic.topic_id, set()),
                        helpful_categories=helpful_categories,
                        consumed_ids=consumed_ids,
                    ),
                    topic,
                )
            )

        if len(scored) < top_k:
            already = {topic.topic_id for _, topic in scored}
            for topic in topics:
                if (
                    topic.status not in {"active", "prototype"}
                    or not topic.general_fallback
                    or topic.topic_id in already
                    or topic.category in excluded
                    or topic.topic_id in negative_ids
                ):
                    continue
                signals = self._match(context, topic)
                signals_by_topic[topic.topic_id] = signals
                scored.append(
                    (
                        self._score(
                            topic=topic,
                            signals=signals,
                            preferences=preferences,
                            preferred=preferred,
                            events=events_by_topic.get(topic.topic_id, set()),
                            helpful_categories=helpful_categories,
                            consumed_ids=consumed_ids,
                        ),
                        topic,
                    )
                )

        scored.sort(key=lambda item: (-item[0], item[1].topic_id))
        return RankingResult(
            topics=[topic for _, topic in scored[:top_k]],
            candidate_count=len(scored),
            signals_by_topic=signals_by_topic,
            safety_overrides=sorted(set(safety_overrides)),
        )

    @staticmethod
    def _contains_match(context_values: list[str], topic_values: list[str]) -> bool:
        normalized_context = [normalize(item) for item in context_values if normalize(item)]
        normalized_topics = [normalize(item) for item in topic_values if normalize(item)]
        return any(
            topic_value in context_value or context_value in topic_value
            for context_value in normalized_context
            for topic_value in normalized_topics
        )

    @classmethod
    def _match(
        cls,
        context: RecommendationContext,
        topic: KnowledgeTopic,
    ) -> MatchSignals:
        context_codes = [normalize(code).replace(" ", "") for code in context.diagnosis_codes]
        topic_codes = [normalize(code).replace(" ", "") for code in topic.related_codes]
        code_match = any(
            context_code.startswith(topic_code)
            for context_code in context_codes
            for topic_code in topic_codes
            if context_code and topic_code
        )
        return MatchSignals(
            code=code_match,
            test=cls._contains_match(context.recommended_tests, topic.related_tests),
            diagnosis=cls._contains_match(context.diagnosis_terms, topic.related_terms),
            medication=(
                topic.category == TopicCategory.MEDICATION_SAFETY
                and cls._contains_match(
                    context.medication_names,
                    topic.related_medications,
                )
            ),
        )

    @staticmethod
    def _score(
        *,
        topic: KnowledgeTopic,
        signals: MatchSignals,
        preferences: UserPreferenceContext | None,
        preferred: set[TopicCategory],
        events: set[str],
        helpful_categories: set[TopicCategory],
        consumed_ids: set[str],
    ) -> int:
        score = topic.priority
        score += 10 if signals.code else 0
        score += 6 if signals.test else 0
        score += 5 if signals.diagnosis else 0
        score += 3 if signals.medication else 0
        score += 3 if topic.category in preferred else 0
        if topic.category in helpful_categories and topic.topic_id not in consumed_ids:
            score += 2
        if preferences:
            score += int(preferences.preferred_depth == topic.depth)
            score += int(preferences.preferred_format == topic.format)
            score += int(
                preferences.max_reading_minutes is not None
                and topic.estimated_reading_minutes
                <= preferences.max_reading_minutes
            )
        score -= 3 if "view" in events else 0
        score -= 2 if events.intersection({"helpful", "save"}) else 0
        return score

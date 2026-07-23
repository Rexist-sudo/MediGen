"""Characterize the rule ranking behavior at the locked repository baseline."""

from __future__ import annotations

from src.models.recommendation import (
    KnowledgeTopic,
    RecommendationContext,
    TopicInteraction,
    UserPreferenceContext,
)
from src.services.recommendation.ranker import RuleRecommendationRanker


def topic(
    topic_id: str,
    *,
    category: str = "disease_basics",
    codes: list[str] | None = None,
    terms: list[str] | None = None,
    tests: list[str] | None = None,
    mandatory: bool = False,
    fallback: bool = False,
    priority: int = 0,
) -> KnowledgeTopic:
    return KnowledgeTopic(
        topic_id=topic_id,
        topic_token=f"<MED_TOPIC_9{sum(ord(char) for char in topic_id):04d}>",
        title=topic_id,
        category=category,
        depth="beginner",
        format="brief",
        estimated_reading_minutes=2,
        related_codes=codes or [],
        related_terms=terms or [],
        related_tests=tests or [],
        summary="A reviewed educational summary.",
        source_label="catalog",
        safety_note="Reviewed safety note.",
        mandatory_safety=mandatory,
        general_fallback=fallback,
        priority=priority,
    )


def rank(
    topics: list[KnowledgeTopic],
    *,
    context: RecommendationContext,
    preferences: UserPreferenceContext | None = None,
    history: list[TopicInteraction] | None = None,
    top_k: int = 3,
):
    return RuleRecommendationRanker().rank_detailed(
        context=context,
        topics=topics,
        preferences=preferences,
        history=history or [],
        top_k=top_k,
    )


def test_code_prefix_test_and_diagnosis_signals_are_characterized() -> None:
    topics = [
        topic("code", codes=["E11"]),
        topic("test", category="test_explanation", tests=["HbA1c"]),
        topic("term", terms=["diabetes"]),
    ]
    result = rank(
        topics,
        context=RecommendationContext(
            diagnosis_codes=["E11.9"],
            diagnosis_terms=["Type 2 diabetes"],
            recommended_tests=["HbA1c blood test"],
            demo_safe=True,
        ),
    )

    assert [item.topic_id for item in result.topics] == ["code", "test", "term"]
    assert result.signals_by_topic["code"].code is True
    assert result.signals_by_topic["test"].test is True
    assert result.signals_by_topic["term"].diagnosis is True


def test_preference_exclusion_history_penalties_and_stable_tie_break() -> None:
    topics = [
        topic("a", codes=["E11"]),
        topic("b", codes=["E11"]),
        topic("preferred", category="test_explanation", codes=["E11"]),
        topic("excluded", category="warning_signs", codes=["E11"]),
    ]
    result = rank(
        topics,
        context=RecommendationContext(diagnosis_codes=["E11"], demo_safe=True),
        preferences=UserPreferenceContext(
            preferred_categories=["test_explanation"],
            excluded_categories=["warning_signs"],
        ),
        history=[TopicInteraction(topic_id="b", event_type="view")],
    )

    assert [item.topic_id for item in result.topics] == ["preferred", "a", "b"]
    assert "excluded" not in result.signals_by_topic


def test_negative_feedback_filters_topic_and_general_topics_fill_slots() -> None:
    topics = [
        topic("clinical", codes=["E11"]),
        topic("dismissed", codes=["E11"]),
        topic("fallback_a", fallback=True, priority=2),
        topic("fallback_b", fallback=True, priority=1),
    ]
    result = rank(
        topics,
        context=RecommendationContext(diagnosis_codes=["E11"], demo_safe=True),
        history=[TopicInteraction(topic_id="dismissed", event_type="dismiss")],
    )

    assert [item.topic_id for item in result.topics] == [
        "clinical",
        "fallback_a",
        "fallback_b",
    ]
    assert result.candidate_count == 3


def test_mandatory_safety_override_is_recorded() -> None:
    warning = topic(
        "warning",
        category="warning_signs",
        codes=["I21"],
        mandatory=True,
        priority=10,
    )
    result = rank(
        [warning],
        context=RecommendationContext(diagnosis_codes=["I21.0"], demo_safe=True),
        preferences=UserPreferenceContext(excluded_categories=["warning_signs"]),
        top_k=1,
    )

    assert [item.topic_id for item in result.topics] == ["warning"]
    assert result.safety_overrides == ["warning"]

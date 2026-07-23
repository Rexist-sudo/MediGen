from __future__ import annotations

from src.models.recommendation import (
    RecommendationContext,
    TopicInteraction,
    UserHistoryContext,
    UserPreferenceContext,
)
from src.services.recommendation.candidate_policy import CandidatePolicy
from src.services.recommendation.history_normalizer import HistoryNormalizer


def apply(topic_store, *, code="I21.0", preferences=None, history=None, safe=True):
    normalized = HistoryNormalizer(topic_store, 20).normalize(history)
    return CandidatePolicy(topic_store).apply(
        context=RecommendationContext(diagnosis_codes=[code], demo_safe=safe),
        preferences=preferences,
        history=normalized,
        recalled_topics=topic_store.list_all(),
        all_active_topics=topic_store.list_active(),
        top_k=3,
        max_candidates=20,
    )


def test_mandatory_safety_is_pinned_before_rankable_topics(topic_store) -> None:
    result = apply(topic_store)
    assert [item.topic_id for item in result.pinned_topics] == [
        "myocardial_infarction_warning_signs"
    ]
    assert "ecg_and_troponin_explanation" in {
        item.topic_id for item in result.rankable_topics
    }


def test_category_exclusion_is_hard_and_safety_override_is_explicit(topic_store) -> None:
    result = apply(
        topic_store,
        preferences=UserPreferenceContext(
            excluded_categories=["warning_signs", "test_explanation"]
        ),
    )
    assert [item.topic_id for item in result.pinned_topics] == [
        "myocardial_infarction_warning_signs"
    ]
    assert "mandatory_safety_override" in result.warnings
    assert "ecg_and_troponin_explanation" in result.excluded_topic_ids


def test_negative_feedback_remains_a_hard_exclusion(topic_store) -> None:
    result = apply(
        topic_store,
        history=UserHistoryContext(
            interactions=[
                TopicInteraction(
                    topic_id="myocardial_infarction_warning_signs",
                    event_type="dismiss",
                )
            ]
        ),
    )
    selected = {
        item.topic_id
        for item in (*result.pinned_topics, *result.rankable_topics)
    }
    assert "myocardial_infarction_warning_signs" not in selected


def test_unsafe_context_returns_fixed_safe_path_without_rankable_topics(topic_store) -> None:
    result = apply(topic_store, code="I50", safe=False)
    assert result.rankable_topics == ()
    assert result.pinned_topics[0].topic_id == "heart_failure_warning_signs"
    assert "unsafe_context" in result.warnings


def test_unknown_recalled_topic_is_dropped_and_candidate_limit_is_enforced(
    topic_store,
) -> None:
    unknown = topic_store.list_all()[0].model_copy(
        update={
            "topic_id": "unknown_graph_topic",
            "topic_token": "<MED_TOPIC_9999>",
        }
    )
    result = CandidatePolicy(topic_store).apply(
        context=RecommendationContext(diagnosis_codes=["E11.9"], demo_safe=True),
        preferences=None,
        history=HistoryNormalizer(topic_store, 20).normalize(None),
        recalled_topics=[unknown, *topic_store.list_all()],
        all_active_topics=topic_store.list_active(),
        top_k=3,
        max_candidates=1,
    )
    assert len(result.rankable_topics) == 1
    assert "unknown_graph_topic" not in {
        topic.topic_id for topic in (*result.pinned_topics, *result.rankable_topics)
    }


def test_inactive_and_prototype_statuses_follow_policy(topic_store) -> None:
    from src.services.recommendation.topic_store import TopicStore

    topics = topic_store.list_all()
    topics[0] = topics[0].model_copy(update={"status": "inactive"})
    topics[1] = topics[1].model_copy(update={"status": "prototype"})
    store = TopicStore(topics)
    common = {
        "context": RecommendationContext(
            diagnosis_codes=["J18.1"],
            recommended_tests=["chest X-ray"],
            demo_safe=True,
        ),
        "preferences": None,
        "history": HistoryNormalizer(store, 20).normalize(None),
        "recalled_topics": store.list_all(),
        "all_active_topics": store.list_active(),
        "top_k": 3,
        "max_candidates": 20,
    }
    production = CandidatePolicy(store).apply(**common)
    production_ids = {
        topic.topic_id
        for topic in (*production.pinned_topics, *production.rankable_topics)
    }
    assert "pneumonia_basics" not in production_ids
    assert "chest_xray_explanation" not in production_ids

    test_policy = CandidatePolicy(store, allow_prototype=True).apply(**common)
    assert "chest_xray_explanation" in {
        topic.topic_id for topic in test_policy.rankable_topics
    }

"""Behavior contracts for the deterministic recommendation post-processor."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from src.models.recommendation import (
    KnowledgeTopic,
    RecommendationContext,
    TopicInteraction,
    UserHistoryContext,
    UserPreferenceContext,
)
from src.services.recommendation.ranker import RuleRecommendationRanker
from src.services.recommendation.service import RecommendationService
from src.services.recommendation.topic_store import TopicStore, resolve_topic_path

TOPIC_PATH = resolve_topic_path("./data/recommendation/knowledge_topics.jsonl")


def _clinical_result(
    *,
    disease: str = "Type 2 diabetes mellitus",
    code: str = "E11.9",
    tests: list[str] | None = None,
    demo_safe: bool = True,
    include_audit: bool = True,
) -> dict:
    result = {
        "diagnosis": {
            "primary_diagnosis": {"disease_name": disease},
            "differential_list": [],
            "recommended_tests": tests if tests is not None else ["HbA1c"],
        },
        "coding_result": {
            "primary_icd10": {"code": code},
            "secondary_icd10_codes": [],
        },
        "treatment_plan": {"medications": []},
    }
    if include_audit:
        result["audit_result"] = {"demo_safe": demo_safe}
    return result


def _service(path: Path = TOPIC_PATH, *, enabled: bool = True) -> RecommendationService:
    return RecommendationService(
        topic_path=str(path),
        ranker=RuleRecommendationRanker(),
        enabled=enabled,
        max_history_interactions=20,
    )


def _topic(topic_id: str, *, priority: int = 0) -> KnowledgeTopic:
    return KnowledgeTopic(
        topic_id=topic_id,
        title=topic_id,
        category="test_explanation",
        depth="beginner",
        format="brief",
        estimated_reading_minutes=2,
        related_tests=["demo test"],
        summary="Prototype summary.",
        source_label="MVP prototype content; not medically reviewed",
        safety_note="Prototype only.",
        priority=priority,
        status="prototype",
    )


def test_topic_store_contract() -> None:
    store = TopicStore.from_jsonl(str(TOPIC_PATH))
    topics = store.list_active()
    assert 12 <= len(topics) <= 15
    assert len({topic.topic_id for topic in topics}) == len(topics)
    assert all(topic.summary and topic.safety_note for topic in topics)
    assert all("not medically reviewed" in topic.source_label for topic in topics)
    assert all(topic.status == "prototype" for topic in topics)


def test_no_preferences_or_history_returns_only_store_topics() -> None:
    store = TopicStore.from_jsonl(str(TOPIC_PATH))
    valid_ids = {topic.topic_id for topic in store.list_active()}
    result = _service().recommend_after_analysis(
        clinical_result=_clinical_result(),
        user_preferences=None,
        user_history_context=None,
        top_k=3,
    )
    ids = [item.topic_id for item in result.recommendations]
    assert result.recommendation_status == "ok"
    assert 1 <= len(ids) <= 3
    assert len(ids) == len(set(ids))
    assert set(ids).issubset(valid_ids)
    assert {"diabetes_basics", "hba1c_test_explanation"}.intersection(ids)


def test_excluded_category_takes_precedence() -> None:
    preferences = UserPreferenceContext(
        preferred_categories=["test_explanation"],
        excluded_categories=["test_explanation"],
    )
    result = _service().recommend_after_analysis(
        clinical_result=_clinical_result(),
        user_preferences=preferences,
        user_history_context=None,
    )
    assert all(
        item.category.value != "test_explanation"
        for item in result.recommendations
    )
    assert "excluded_category_takes_precedence" in result.warnings


def test_dismiss_and_not_helpful_exclude_normal_topics() -> None:
    history = UserHistoryContext(
        interactions=[
            TopicInteraction(topic_id="hba1c_test_explanation", event_type="dismiss"),
            TopicInteraction(topic_id="diabetes_basics", event_type="not_helpful"),
        ]
    )
    result = _service().recommend_after_analysis(
        clinical_result=_clinical_result(),
        user_preferences=None,
        user_history_context=history,
    )
    ids = {item.topic_id for item in result.recommendations}
    assert "hba1c_test_explanation" not in ids
    assert "diabetes_basics" not in ids


def test_view_downweights_a_topic() -> None:
    ranker = RuleRecommendationRanker()
    topics = [_topic("a_topic"), _topic("b_topic")]
    context = RecommendationContext(recommended_tests=["demo test"], demo_safe=True)
    baseline = ranker.rank(
        context=context,
        topics=topics,
        preferences=None,
        history=[],
        top_k=2,
    )
    viewed = ranker.rank(
        context=context,
        topics=topics,
        preferences=None,
        history=[TopicInteraction(topic_id="a_topic", event_type="view")],
        top_k=2,
    )
    assert [item.topic_id for item in baseline] == ["a_topic", "b_topic"]
    assert [item.topic_id for item in viewed] == ["b_topic", "a_topic"]


def test_helpful_history_boosts_unconsumed_same_category() -> None:
    ranker = RuleRecommendationRanker()
    topics = [_topic("consumed", priority=1), _topic("new_topic")]
    context = RecommendationContext(recommended_tests=["demo test"], demo_safe=True)
    ranked = ranker.rank(
        context=context,
        topics=topics,
        preferences=None,
        history=[TopicInteraction(topic_id="consumed", event_type="helpful")],
        top_k=2,
    )
    assert [item.topic_id for item in ranked] == ["new_topic", "consumed"]


def test_unknown_history_is_ignored_with_warning() -> None:
    result = _service().recommend_after_analysis(
        clinical_result=_clinical_result(),
        user_preferences=None,
        user_history_context=UserHistoryContext(
            interactions=[
                TopicInteraction(topic_id="unknown_topic", event_type="view")
            ]
        ),
    )
    assert result.valid_history_count == 0
    assert result.history_used is False
    assert "unknown_history_topic_ignored" in result.warnings


def test_same_input_is_fully_deterministic() -> None:
    service = _service()
    history = UserHistoryContext(
        interactions=[
            TopicInteraction(topic_id="diabetes_basics", event_type="view")
        ]
    )
    kwargs = {
        "clinical_result": _clinical_result(),
        "user_preferences": UserPreferenceContext(
            preferred_categories=["test_explanation"],
            preferred_depth="beginner",
        ),
        "user_history_context": history,
        "top_k": 3,
    }
    first = service.recommend_after_analysis(**kwargs)
    second = service.recommend_after_analysis(**kwargs)
    assert first.model_dump(mode="json") == second.model_dump(mode="json")


def test_store_failure_degrades_without_mutating_clinical_result(tmp_path: Path) -> None:
    missing = tmp_path / "missing.jsonl"
    clinical = _clinical_result()
    before = deepcopy(clinical)
    result = _service(missing).recommend_after_analysis(
        clinical_result=clinical,
        user_preferences=None,
        user_history_context=None,
    )
    assert result.recommendation_status == "degraded"
    assert result.strategy_used == "none"
    assert result.recommendations == []
    assert result.warnings == ["recommendation_unavailable"]
    assert clinical == before


def test_disabled_service_is_stable() -> None:
    result = _service(enabled=False).recommend_after_analysis(
        clinical_result=_clinical_result(),
        user_preferences=None,
        user_history_context=None,
    )
    assert result.recommendation_status == "disabled"
    assert result.strategy_used == "none"
    assert result.recommendations == []


def test_mandatory_safety_can_override_exclusion_on_strong_match() -> None:
    result = _service().recommend_after_analysis(
        clinical_result=_clinical_result(
            disease="Myocardial infarction",
            code="I21.9",
            tests=["ECG", "troponin"],
        ),
        user_preferences=UserPreferenceContext(
            excluded_categories=["warning_signs"]
        ),
        user_history_context=UserHistoryContext(
            interactions=[
                TopicInteraction(
                    topic_id="myocardial_infarction_warning_signs",
                    event_type="dismiss",
                )
            ]
        ),
    )
    ids = [item.topic_id for item in result.recommendations]
    assert "myocardial_infarction_warning_signs" in ids
    assert "mandatory_safety_override" in result.warnings


def test_failed_audit_limits_results_and_marks_degraded() -> None:
    result = _service().recommend_after_analysis(
        clinical_result=_clinical_result(demo_safe=False),
        user_preferences=None,
        user_history_context=None,
    )
    store = TopicStore.from_jsonl(str(TOPIC_PATH))
    allowed = {
        topic.topic_id
        for topic in store.list_active()
        if topic.mandatory_safety or topic.general_fallback
    }
    assert result.recommendation_status == "degraded"
    assert {item.topic_id for item in result.recommendations}.issubset(allowed)
    assert "audit_not_demo_safe" in result.warnings


def test_missing_audit_warns_but_keeps_local_recommendations() -> None:
    result = _service().recommend_after_analysis(
        clinical_result=_clinical_result(include_audit=False),
        user_preferences=None,
        user_history_context=None,
    )
    assert result.recommendation_status == "ok"
    assert result.recommendations
    assert "audit_result_missing" in result.warnings


def test_reasons_never_contain_treatment_instructions() -> None:
    result = _service().recommend_after_analysis(
        clinical_result=_clinical_result(),
        user_preferences=None,
        user_history_context=None,
    )
    forbidden = ("dosage", "dose", "stop medication", "change medication", "加药", "停药", "换药", "剂量")
    assert all(
        not any(word in item.reason.casefold() for word in forbidden)
        for item in result.recommendations
    )

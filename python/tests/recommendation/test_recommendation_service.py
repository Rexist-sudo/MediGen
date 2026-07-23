from __future__ import annotations

import json

import pytest

from src.models.recommendation import UserPreferenceContext
from src.services.recommendation.model_loader import ModelReadiness
from src.services.recommendation.ranker_protocol import RankerResult
from src.services.recommendation.service import RecommendationService


class FakeRouter:
    def __init__(self, *, strategy="mini_onerec_mvp", fallback_reason=None):
        self.strategy = strategy
        self.fallback_reason = fallback_reason

    def rank(self, ranker_input):
        return RankerResult(
            topic_ids=tuple(
                topic.topic_id
                for topic in ranker_input.candidates[: ranker_input.top_k]
            ),
            strategy_used=self.strategy,
            model_version=("model-v1" if self.strategy == "mini_onerec_mvp" else None),
            inference_ms=12.5,
            fallback_reason=self.fallback_reason,
        )

    def rank_fallback(self, ranker_input, *, reason):
        self.strategy = "rule_v1_fallback"
        self.fallback_reason = reason
        return self.rank(ranker_input)

    def readiness(self, *, load=False):
        return ModelReadiness(
            configured=True,
            artifact_valid=True,
            loaded=True,
            device="cpu",
            dtype="float32",
            model_version="model-v1",
            last_failure_code=None,
        )


def clinical_result(*, safe=True):
    return {
        "diagnosis": {
            "primary_diagnosis": {"disease_name": "Type 2 diabetes"},
            "differential_list": [],
            "recommended_tests": ["HbA1c"],
        },
        "coding_result": {
            "primary_icd10": {"code": "E11.9"},
            "secondary_icd10_codes": [],
        },
        "treatment_plan": {"medications": []},
        "audit_result": {"demo_safe": safe},
    }


def test_model_path_and_content_strategy_are_reported_separately(topic_store) -> None:
    service = RecommendationService(
        topic_store=topic_store,
        ranker_router=FakeRouter(),
        enabled=True,
        max_candidates=20,
    )
    result = service.recommend_after_analysis(
        clinical_result=clinical_result(),
        user_preferences=UserPreferenceContext(
            preferred_categories=["test_explanation"]
        ),
        user_history_context=None,
        top_k=3,
    )
    assert result.recommendation_status == "ok"
    assert result.ranking_strategy_used == "mini_onerec_mvp"
    assert result.content_strategy_used == "catalog_fallback"
    assert result.strategy_used == "mini_onerec_mvp"
    assert result.model_ready is True
    assert result.model_version == "model-v1"
    assert 1 <= len(result.recommendations) <= 3
    assert len({item.topic_id for item in result.recommendations}) == len(
        result.recommendations
    )


def test_rule_fallback_is_degraded_without_losing_cards(topic_store) -> None:
    service = RecommendationService(
        topic_store=topic_store,
        ranker_router=FakeRouter(
            strategy="rule_v1_fallback",
            fallback_reason="inference_failed",
        ),
        enabled=True,
        max_candidates=20,
    )
    result = service.recommend_after_analysis(
        clinical_result=clinical_result(),
        user_preferences=None,
        user_history_context=None,
        top_k=3,
    )
    assert result.recommendation_status == "degraded"
    assert result.ranking_strategy_used == "rule_v1_fallback"
    assert result.fallback_reason == "inference_failed"
    assert result.recommendations


def test_unsafe_context_never_invokes_ranker_and_returns_fixed_cards(topic_store) -> None:
    class NoCallRouter(FakeRouter):
        def rank(self, ranker_input):
            raise AssertionError("unsafe context entered learned ranker")

    service = RecommendationService(
        topic_store=topic_store,
        ranker_router=NoCallRouter(),
        enabled=True,
        max_candidates=20,
    )
    result = service.recommend_after_analysis(
        clinical_result=clinical_result(safe=False),
        user_preferences=None,
        user_history_context=None,
        top_k=3,
    )
    assert result.recommendation_status == "degraded"
    assert result.ranking_strategy_used == "none"
    assert result.fallback_reason == "unsafe_context"
    assert result.recommendations


@pytest.mark.parametrize(
    ("ranking_strategy", "content_mode", "expected_content"),
    [
        ("mini_onerec_mvp", "success", "deepseek_generated"),
        ("mini_onerec_mvp", "disabled", "catalog_fallback"),
        ("rule_v1_fallback", "success", "deepseek_generated"),
        ("rule_v1_fallback", "failure", "catalog_fallback"),
    ],
)
def test_ranking_and_content_strategies_remain_independent(
    topic_store,
    ranking_strategy,
    content_mode,
    expected_content,
) -> None:
    class ContentClient:
        def invoke_json(self, *, user_prompt, response_model, **_kwargs):
            if content_mode == "failure":
                raise RuntimeError("injected content failure")
            supplied = json.loads(user_prompt)["candidate_topics"]
            return response_model.model_validate(
                {
                    "cards": [
                        {
                            "topic_id": item["topic_id"],
                            "summary": (
                                "这是一段经过固定测试构造的医学教育说明，用于确认正文生成状态与主题排序状态分别记录，"
                                "并保留目录主题编号、来源和安全提示。"
                            ),
                        }
                        for item in supplied
                    ]
                }
            )

    fallback_reason = (
        "inference_failed" if ranking_strategy == "rule_v1_fallback" else None
    )
    service = RecommendationService(
        topic_store=topic_store,
        ranker_router=FakeRouter(
            strategy=ranking_strategy,
            fallback_reason=fallback_reason,
        ),
        enabled=True,
        max_candidates=20,
        content_client_factory=lambda: ContentClient(),
        generate_content=content_mode != "disabled",
    )
    result = service.recommend_after_analysis(
        clinical_result=clinical_result(),
        user_preferences=None,
        user_history_context=None,
        top_k=3,
    )
    assert result.ranking_strategy_used == ranking_strategy
    assert result.content_strategy_used == expected_content
    assert result.recommendations
    assert {
        card.content_source for card in result.recommendations
    } == {expected_content}
    if content_mode == "failure":
        assert "education_content_generation_fallback" in result.warnings
        assert result.fallback_reason == "inference_failed"


def test_pinned_safety_precedes_model_topics_and_graph_failure_uses_catalog(
    topic_store,
) -> None:
    class FailingProvider:
        use_neo4j = True

        def find_education_topics(self, _context):
            raise RuntimeError("injected graph outage")

    result = RecommendationService(
        topic_store=topic_store,
        ranker_router=FakeRouter(),
        enabled=True,
        max_candidates=20,
        topic_provider=FailingProvider(),
    ).recommend_after_analysis(
        clinical_result={
            "diagnosis": {
                "primary_diagnosis": {"disease_name": "myocardial infarction"},
                "differential_list": [],
                "recommended_tests": ["ECG", "troponin"],
            },
            "coding_result": {
                "primary_icd10": {"code": "I21.0"},
                "secondary_icd10_codes": [],
            },
            "treatment_plan": {"medications": []},
            "audit_result": {"demo_safe": True},
        },
        user_preferences=None,
        user_history_context=None,
        top_k=3,
    )
    assert result.ranking_strategy_used == "mini_onerec_mvp"
    assert result.recommendations[0].topic_id == (
        "myocardial_infarction_warning_signs"
    )
    assert result.recommendations[0].reason.startswith(
        "该主题与本次结构化临床结果相关"
    )
    assert result.candidate_source == "local_catalog"
    assert "knowledge_graph_catalog_fallback" in result.warnings
    assert len(result.recommendations) == 3


def test_unexpected_service_error_is_bounded_to_recommendation_result(topic_store) -> None:
    class BrokenContextBuilder:
        def build(self, _clinical_result):
            raise RuntimeError("injected context error")

    result = RecommendationService(
        topic_store=topic_store,
        ranker_router=FakeRouter(),
        enabled=True,
        max_candidates=20,
        context_builder=BrokenContextBuilder(),
    ).recommend_after_analysis(
        clinical_result=clinical_result(),
        user_preferences=None,
        user_history_context=None,
        top_k=3,
    )
    assert result.recommendation_status == "degraded"
    assert result.ranking_strategy_used == "none"
    assert result.recommendations == []
    assert result.warnings == ["recommendation_unavailable"]

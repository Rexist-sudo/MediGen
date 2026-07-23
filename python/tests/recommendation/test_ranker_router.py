from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.models.recommendation import RecommendationContext
from src.services.recommendation.history_normalizer import HistoryNormalizer
from src.services.recommendation.ranker_protocol import (
    InvalidModelOutputError,
    ModelArtifactError,
    ModelArtifactMissingError,
    ModelInferenceError,
    ModelLoadError,
    ModelNotReadyError,
    RankerInput,
    RankerResult,
    RecommendationRankerError,
)
from src.services.recommendation.ranker_router import RankerRouter


class StubRanker:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = 0

    def rank(self, _ranker_input):
        self.calls += 1
        if self.error:
            raise self.error
        return self.result


def input_for(topic_store):
    topic = topic_store.get_by_id("diabetes_basics")
    assert topic is not None
    return RankerInput(
        context=RecommendationContext(demo_safe=True),
        preferences=None,
        history=HistoryNormalizer(topic_store, 20).normalize(None),
        candidates=(topic,),
        already_selected_topic_ids=(),
        top_k=1,
    )


def settings(strategy="auto", fallback=True):
    return SimpleNamespace(
        recommendation_ranker=strategy,
        minionerec_enabled=True,
        recommendation_rule_fallback_enabled=fallback,
    )


def test_auto_uses_primary_when_ready(topic_store) -> None:
    primary = StubRanker(
        RankerResult(
            topic_ids=("diabetes_basics",),
            strategy_used="mini_onerec_mvp",
            model_version="v1",
        )
    )
    fallback = StubRanker(
        RankerResult(topic_ids=("diabetes_basics",), strategy_used="rule_v1_fallback")
    )
    router = RankerRouter(
        primary=primary,
        fallback=fallback,
        settings=settings(),
        model_loader=SimpleNamespace(readiness=lambda **_kwargs: None),
    )
    assert router.rank(input_for(topic_store)).strategy_used == "mini_onerec_mvp"
    assert primary.calls == 1
    assert fallback.calls == 0


def test_missing_artifact_maps_to_rule_fallback(topic_store) -> None:
    primary = StubRanker(error=ModelArtifactMissingError())
    fallback = StubRanker(
        RankerResult(topic_ids=("diabetes_basics",), strategy_used="rule_v1_fallback")
    )
    router = RankerRouter(
        primary=primary,
        fallback=fallback,
        settings=settings(),
        model_loader=SimpleNamespace(readiness=lambda **_kwargs: None),
    )
    result = router.rank(input_for(topic_store))
    assert result.strategy_used == "rule_v1_fallback"
    assert result.fallback_reason == "artifact_missing"


def test_fallback_disabled_propagates_controlled_failure(topic_store) -> None:
    router = RankerRouter(
        primary=StubRanker(error=ModelArtifactMissingError()),
        fallback=StubRanker(),
        settings=settings(fallback=False),
        model_loader=SimpleNamespace(readiness=lambda **_kwargs: None),
    )
    with pytest.raises(RecommendationRankerError, match="artifact_missing"):
        router.rank(input_for(topic_store))


def test_no_history_still_calls_primary(topic_store) -> None:
    primary = StubRanker(
        RankerResult(
            topic_ids=("diabetes_basics",),
            strategy_used="mini_onerec_mvp",
            model_version="v1",
        )
    )
    router = RankerRouter(
        primary=primary,
        fallback=StubRanker(),
        settings=settings(),
        model_loader=SimpleNamespace(readiness=lambda **_kwargs: None),
    )
    router.rank(input_for(topic_store))
    assert primary.calls == 1


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (ModelArtifactError(), "artifact_incompatible"),
        (ModelLoadError(), "model_load_failed"),
        (ModelNotReadyError(), "model_not_ready"),
        (ModelInferenceError(), "inference_failed"),
        (ModelInferenceError("cuda_oom"), "cuda_oom"),
        (ModelNotReadyError("concurrency_busy"), "concurrency_busy"),
        (InvalidModelOutputError(), "invalid_model_output"),
    ],
)
def test_controlled_primary_failures_keep_fixed_fallback_reason(
    topic_store,
    error,
    reason,
) -> None:
    fallback = StubRanker(
        RankerResult(topic_ids=("diabetes_basics",), strategy_used="rule_v1_fallback")
    )
    router = RankerRouter(
        primary=StubRanker(error=error),
        fallback=fallback,
        settings=settings(),
        model_loader=SimpleNamespace(readiness=lambda **_kwargs: None),
    )
    result = router.rank(input_for(topic_store))
    assert result.strategy_used == "rule_v1_fallback"
    assert result.fallback_reason == reason
    assert "ranker_fallback" in result.warnings


def test_rule_profile_never_calls_primary(topic_store) -> None:
    primary = StubRanker(error=AssertionError("primary called"))
    fallback = StubRanker(
        RankerResult(topic_ids=("diabetes_basics",), strategy_used="rule_v1_fallback")
    )
    router = RankerRouter(
        primary=primary,
        fallback=fallback,
        settings=settings(strategy="rule_v1"),
        model_loader=SimpleNamespace(readiness=lambda **_kwargs: None),
    )
    result = router.rank(input_for(topic_store))
    assert result.strategy_used == "rule_v1_fallback"
    assert result.fallback_reason == "model_disabled"
    assert primary.calls == 0


def test_unsafe_context_never_calls_primary(topic_store) -> None:
    ranker_input = input_for(topic_store)
    ranker_input = RankerInput(
        context=ranker_input.context.model_copy(update={"demo_safe": False}),
        preferences=ranker_input.preferences,
        history=ranker_input.history,
        candidates=ranker_input.candidates,
        already_selected_topic_ids=(),
        top_k=1,
    )
    primary = StubRanker(error=AssertionError("primary called"))
    fallback = StubRanker(
        RankerResult(topic_ids=("diabetes_basics",), strategy_used="rule_v1_fallback")
    )
    router = RankerRouter(
        primary=primary,
        fallback=fallback,
        settings=settings(),
        model_loader=SimpleNamespace(readiness=lambda **_kwargs: None),
    )
    result = router.rank(ranker_input)
    assert result.fallback_reason == "unsafe_context"
    assert primary.calls == 0

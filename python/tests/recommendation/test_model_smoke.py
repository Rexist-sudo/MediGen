from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from src.config.settings import Settings
from src.models.recommendation import (
    RecommendationContext,
    TopicInteraction,
    UserHistoryContext,
)
from src.services.recommendation.history_normalizer import HistoryNormalizer
from src.services.recommendation.minionerec_ranker import MiniOneRecRanker
from src.services.recommendation.model_loader import MiniOneRecModelLoader
from src.services.recommendation.ranker_protocol import RankerInput
from src.services.recommendation.ranker_router import RankerRouter
from src.services.recommendation.rule_fallback_ranker import RuleFallbackRanker
from src.services.recommendation.topic_store import TopicStore


pytestmark = pytest.mark.model
PYTHON_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = PYTHON_ROOT / "artifacts" / "minionerec-mvp" / "v1"


@pytest.fixture(scope="module")
def real_stack():
    settings = Settings(
        _env_file=None,
        recommendation_ranker="minionerec",
        recommendation_rule_fallback_enabled=False,
        minionerec_enabled=True,
        minionerec_model_version="minionerec-mvp-direct-sid-v1",
        minionerec_base_model_path=str(
            PYTHON_ROOT / "artifacts" / "base-models" / "qwen2.5-0.5b"
        ),
        minionerec_artifact_path=str(ARTIFACT),
        minionerec_device="auto",
        minionerec_dtype="auto",
        minionerec_readiness_strict=True,
    )
    store = TopicStore.from_jsonl(
        PYTHON_ROOT / "data" / "recommendation" / "knowledge_topics.jsonl"
    )
    loader = MiniOneRecModelLoader(settings=settings, topic_store=store)
    loaded = loader.load()
    ranker = MiniOneRecRanker(
        model_loader=loader,
        topic_store=store,
        inference_semaphore=threading.Semaphore(1),
        semaphore_wait_seconds=1.0,
        max_input_tokens=1024,
    )
    return settings, store, loader, loaded, ranker


def _ranker_input(
    store: TopicStore,
    *,
    top_k: int = 3,
    history_context: UserHistoryContext | None = None,
) -> RankerInput:
    candidate_ids = (
        "diabetes_basics",
        "hba1c_test_explanation",
        "follow_up_checklist",
    )
    candidates = tuple(store.get_by_id(topic_id) for topic_id in candidate_ids)
    assert all(topic is not None for topic in candidates)
    return RankerInput(
        context=RecommendationContext(
            diagnosis_codes=["E11.9"],
            diagnosis_terms=["type 2 diabetes"],
            recommended_tests=["HbA1c"],
            demo_safe=True,
        ),
        preferences=None,
        history=HistoryNormalizer(store, 20).normalize(history_context),
        candidates=tuple(topic for topic in candidates if topic is not None),
        already_selected_topic_ids=(),
        top_k=top_k,
    )


def test_real_artifact_loads_offline_and_registers_single_id_tokens(real_stack) -> None:
    _settings, store, loader, loaded, _ranker = real_stack
    readiness = loader.readiness(load=False)
    assert readiness.artifact_valid is True
    assert readiness.loaded is True
    assert loaded.manifest.model_version == "minionerec-mvp-direct-sid-v1"
    store.validate_tokenizer(loaded.tokenizer)
    for topic in store.list_all():
        token_id = loaded.tokenizer.convert_tokens_to_ids(topic.topic_token)
        assert loaded.tokenizer.encode(
            topic.topic_token,
            add_special_tokens=False,
        ) == [token_id]


def test_real_adapter_ranks_top3_on_cold_start(real_stack) -> None:
    _settings, store, _loader, _loaded, ranker = real_stack
    ranker_input = _ranker_input(store)
    result = ranker.rank(ranker_input)
    assert result.strategy_used == "mini_onerec_mvp"
    assert result.model_version == "minionerec-mvp-direct-sid-v1"
    assert result.fallback_reason is None
    assert len(result.topic_ids) == 3
    assert len(set(result.topic_ids)) == 3
    assert set(result.topic_ids) == {
        topic.topic_id for topic in ranker_input.candidates
    }
    assert result.inference_ms is not None and result.inference_ms > 0
    assert len(result.diagnostics["prompt_sha256"]) == 3


def test_fallback_disabled_router_uses_real_primary(real_stack) -> None:
    settings, store, loader, _loaded, ranker = real_stack
    router = RankerRouter(
        primary=ranker,
        fallback=RuleFallbackRanker(store),
        settings=settings,
        model_loader=loader,
    )
    result = router.rank(_ranker_input(store, top_k=1))
    assert result.strategy_used == "mini_onerec_mvp"
    assert result.model_version
    assert result.fallback_reason is None
    assert len(result.topic_ids) == 1


def test_real_checkpoint_uses_history_and_changes_a_fixed_pair(real_stack) -> None:
    _settings, store, _loader, _loaded, ranker = real_stack
    cold = ranker.rank(_ranker_input(store))
    history = ranker.rank(
        _ranker_input(
            store,
            history_context=UserHistoryContext(
                interactions=[
                    TopicInteraction(topic_id="diabetes_basics", event_type="view"),
                    TopicInteraction(
                        topic_id="chest_xray_explanation",
                        event_type="helpful",
                    ),
                ]
            ),
        )
    )
    assert cold.diagnostics["prompt_sha256"][0] != history.diagnostics[
        "prompt_sha256"
    ][0]
    assert cold.topic_ids[0] != history.topic_ids[0]


def test_packaged_metrics_prove_adapter_history_and_structural_gates() -> None:
    metrics = json.loads((ARTIFACT / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["AdapterEffect"] == "PASS"
    assert metrics["AdapterMaxCandidateLogitChange"] > 1e-5
    assert metrics["HistoryPairCount"] >= 20
    assert metrics["HistoryPairFlipCount"] >= 1
    assert metrics["CandidateOrderPairCount"] >= 1
    assert metrics["FirstCandidateCopyRate"] < 1.0
    assert metrics["ValidTopicRate"] == 1.0
    for name in (
        "DuplicateTopicRate",
        "CandidateViolationRate",
        "ExcludedCategoryViolationRate",
        "NegativeFeedbackViolationRate",
        "MandatorySafetyOrderingViolationRate",
        "UnknownTokenRate",
    ):
        assert metrics[name] == 0.0


def test_recorded_cpu_and_gpu_smokes_are_real_and_offline() -> None:
    for device in ("cpu", "cuda"):
        report_path = ARTIFACT / f"smoke_report_{device}.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["status"] == "pass"
        assert report["offline"] is True
        assert report["device"] == device
        assert len(report["cold_start"]["topic_ids"]) == 3
        assert len(report["history_case"]["topic_ids"]) == 3
        assert report["load_ms"] > 0

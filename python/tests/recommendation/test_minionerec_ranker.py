from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from src.models.recommendation import RecommendationContext
from src.services.recommendation.history_normalizer import HistoryNormalizer
from src.services.recommendation.minionerec_ranker import MiniOneRecRanker
from src.services.recommendation.ranker_protocol import (
    ModelInferenceError,
    ModelNotReadyError,
    RankerInput,
)


class FakeTokenizer:
    unk_token_id = 0

    def encode(self, text, *, add_special_tokens=False):
        return list(range(max(1, len(text) // 5)))


class FakeLoader:
    def __init__(self):
        self.loaded = SimpleNamespace(
            model=object(),
            tokenizer=FakeTokenizer(),
            device="cpu",
            manifest=SimpleNamespace(
                max_input_tokens=1024,
                model_version="real-test-v1",
            ),
        )
        self.calls = 0
        self.failure_codes = []

    def load(self):
        self.calls += 1
        return self.loaded

    def mark_failure(self, code):
        self.failure_codes.append(code)


class SequencedDecoder:
    def __init__(self, sequence):
        self.sequence = list(sequence)
        self.allowed = []

    def select_one(self, *, prompt, allowed_topic_tokens):
        assert prompt.endswith("<NEXT_TOPIC>")
        self.allowed.append(tuple(allowed_topic_tokens))
        desired = self.sequence.pop(0)
        assert desired in allowed_topic_tokens
        return desired, 1.0


def test_iterative_top3_removes_selected_tokens_and_uses_cold_start(topic_store) -> None:
    candidates = tuple(
        topic_store.get_by_id(item)
        for item in (
            "diabetes_basics",
            "hba1c_test_explanation",
            "follow_up_checklist",
        )
    )
    candidates = tuple(item for item in candidates if item is not None)
    decoder = SequencedDecoder(
        [
            "<MED_TOPIC_0012>",
            "<MED_TOPIC_0011>",
            "<MED_TOPIC_0014>",
        ]
    )
    loader = FakeLoader()
    ranker = MiniOneRecRanker(
        model_loader=loader,
        topic_store=topic_store,
        inference_semaphore=threading.Semaphore(1),
        semaphore_wait_seconds=0.1,
        max_input_tokens=1024,
        decoder_factory=lambda _loaded: decoder,
    )
    result = ranker.rank(
        RankerInput(
            context=RecommendationContext(
                diagnosis_codes=["E11"],
                demo_safe=True,
            ),
            preferences=None,
            history=HistoryNormalizer(topic_store, 20).normalize(None),
            candidates=candidates,
            already_selected_topic_ids=(),
            top_k=3,
        )
    )
    assert result.topic_ids == (
        "hba1c_test_explanation",
        "diabetes_basics",
        "follow_up_checklist",
    )
    assert result.strategy_used == "mini_onerec_mvp"
    assert result.model_version == "real-test-v1"
    assert loader.calls == 1
    assert len(decoder.allowed[0]) == 3
    assert len(decoder.allowed[1]) == 2
    assert len(decoder.allowed[2]) == 1


def _single_input(topic_store):
    topic = topic_store.get_by_id("diabetes_basics")
    assert topic is not None
    return RankerInput(
        context=RecommendationContext(demo_safe=True),
        preferences=None,
        history=HistoryNormalizer(topic_store, 20).normalize(None),
        candidates=(topic,),
        already_selected_topic_ids=("myocardial_infarction_warning_signs",),
        top_k=1,
    )


def test_semaphore_busy_is_a_controlled_failure(topic_store) -> None:
    semaphore = threading.Semaphore(1)
    assert semaphore.acquire(timeout=0.1)
    ranker = MiniOneRecRanker(
        model_loader=FakeLoader(),
        topic_store=topic_store,
        inference_semaphore=semaphore,
        semaphore_wait_seconds=0.0,
        max_input_tokens=1024,
    )
    try:
        with pytest.raises(ModelNotReadyError, match="concurrency_busy"):
            ranker.rank(_single_input(topic_store))
    finally:
        semaphore.release()


def test_oom_is_mapped_and_semaphore_is_released(topic_store) -> None:
    OutOfMemoryError = type("OutOfMemoryError", (RuntimeError,), {})

    class OOMDecoder:
        def select_one(self, **_kwargs):
            raise OutOfMemoryError("injected")

    loader = FakeLoader()
    semaphore = threading.Semaphore(1)
    ranker = MiniOneRecRanker(
        model_loader=loader,
        topic_store=topic_store,
        inference_semaphore=semaphore,
        semaphore_wait_seconds=0.1,
        max_input_tokens=1024,
        decoder_factory=lambda _loaded: OOMDecoder(),
    )
    with pytest.raises(ModelInferenceError, match="cuda_oom"):
        ranker.rank(_single_input(topic_store))
    assert loader.failure_codes == ["cuda_oom"]
    assert semaphore.acquire(timeout=0.1)
    semaphore.release()


def test_unsafe_and_empty_candidates_do_not_load_model(topic_store) -> None:
    loader = FakeLoader()
    ranker = MiniOneRecRanker(
        model_loader=loader,
        topic_store=topic_store,
        inference_semaphore=threading.Semaphore(1),
        semaphore_wait_seconds=0.1,
        max_input_tokens=1024,
    )
    single = _single_input(topic_store)
    unsafe = RankerInput(
        context=single.context.model_copy(update={"demo_safe": False}),
        preferences=None,
        history=single.history,
        candidates=single.candidates,
        already_selected_topic_ids=(),
        top_k=1,
    )
    with pytest.raises(ModelNotReadyError, match="unsafe_context"):
        ranker.rank(unsafe)
    empty = RankerInput(
        context=single.context,
        preferences=None,
        history=single.history,
        candidates=(),
        already_selected_topic_ids=(),
        top_k=1,
    )
    with pytest.raises(ModelNotReadyError, match="no_rankable_candidates"):
        ranker.rank(empty)
    assert loader.calls == 0

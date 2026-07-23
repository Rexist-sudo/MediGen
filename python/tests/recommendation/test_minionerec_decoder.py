from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.services.recommendation.minionerec_decoder import DirectCandidateTokenScorer
from src.services.recommendation.ranker_protocol import ModelInferenceError


torch = pytest.importorskip("torch")


class FakeTokenizer:
    unk_token_id = 0

    def __init__(self):
        self.ids = {"<MED_TOPIC_0001>": 1, "<MED_TOPIC_0002>": 2}

    def convert_tokens_to_ids(self, token):
        return self.ids.get(token, 0)

    def encode(self, text, *, add_special_tokens=False):
        if text in self.ids:
            return [self.ids[text]]
        return [5, 6]

    def __call__(self, text, **_kwargs):
        return {"input_ids": torch.tensor([[5, 6]]), "attention_mask": torch.ones((1, 2), dtype=torch.long)}


class FakeModel:
    def generate(self, **_kwargs):
        raise AssertionError("generate must not be used")

    def __call__(self, **_kwargs):
        logits = torch.zeros((1, 2, 8))
        logits[0, -1, 7] = 100.0
        logits[0, -1, 1] = 2.0
        logits[0, -1, 2] = 4.0
        return SimpleNamespace(logits=logits)


def test_decoder_selects_only_the_best_allowed_token() -> None:
    scorer = DirectCandidateTokenScorer(FakeModel(), FakeTokenizer(), "cpu", 20)
    token, logit = scorer.select_one(
        prompt="x<NEXT_TOPIC>",
        allowed_topic_tokens=["<MED_TOPIC_0001>", "<MED_TOPIC_0002>"],
    )
    assert token == "<MED_TOPIC_0002>"
    assert logit == 4.0


def test_decoder_rejects_empty_or_unknown_candidates() -> None:
    scorer = DirectCandidateTokenScorer(FakeModel(), FakeTokenizer(), "cpu", 20)
    with pytest.raises(ModelInferenceError, match="empty_allowed_tokens"):
        scorer.select_one(prompt="x<NEXT_TOPIC>", allowed_topic_tokens=[])
    with pytest.raises(ModelInferenceError, match="unknown_candidate_token"):
        scorer.select_one(
            prompt="x<NEXT_TOPIC>",
            allowed_topic_tokens=["<MED_TOPIC_9999>"],
        )


def test_decoder_rejects_split_sid_and_missing_prompt_tail() -> None:
    class SplitTokenizer(FakeTokenizer):
        def encode(self, text, *, add_special_tokens=False):
            if text == "<MED_TOPIC_0001>":
                return [1, 1]
            return super().encode(text, add_special_tokens=add_special_tokens)

    scorer = DirectCandidateTokenScorer(FakeModel(), SplitTokenizer(), "cpu", 20)
    with pytest.raises(ModelInferenceError, match="candidate_token_not_single_id"):
        scorer.select_one(
            prompt="x<NEXT_TOPIC>",
            allowed_topic_tokens=["<MED_TOPIC_0001>"],
        )
    with pytest.raises(ModelInferenceError, match="prompt_tail_missing"):
        scorer.select_one(
            prompt="missing tail",
            allowed_topic_tokens=["<MED_TOPIC_0002>"],
        )


def test_argmax_tie_is_stable_in_allowed_order() -> None:
    class TieModel(FakeModel):
        def __call__(self, **_kwargs):
            logits = torch.zeros((1, 2, 8))
            logits[0, -1, 1] = 4.0
            logits[0, -1, 2] = 4.0
            return SimpleNamespace(logits=logits)

    scorer = DirectCandidateTokenScorer(TieModel(), FakeTokenizer(), "cpu", 20)
    token, _ = scorer.select_one(
        prompt="x<NEXT_TOPIC>",
        allowed_topic_tokens=["<MED_TOPIC_0002>", "<MED_TOPIC_0001>"],
    )
    assert token == "<MED_TOPIC_0002>"

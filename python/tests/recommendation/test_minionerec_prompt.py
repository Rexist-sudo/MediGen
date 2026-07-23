from __future__ import annotations

import pytest

from src.models.recommendation import (
    RecommendationContext,
    TopicInteraction,
    UserHistoryContext,
    UserPreferenceContext,
)
from src.services.recommendation.history_normalizer import HistoryNormalizer
from src.services.recommendation.minionerec_prompt import MiniOneRecPromptBuilder
from src.services.recommendation.ranker_protocol import ModelInferenceError, RankerInput


def test_prompt_is_canonical_bounded_and_uses_global_history(topic_store) -> None:
    candidates = tuple(
        topic_store.get_by_id(topic_id)
        for topic_id in (
            "diabetes_basics",
            "hba1c_test_explanation",
            "follow_up_checklist",
        )
    )
    candidates = tuple(item for item in candidates if item is not None)
    history = HistoryNormalizer(topic_store, 20).normalize(
        UserHistoryContext(
            interactions=[
                TopicInteraction(
                    topic_id="chest_xray_explanation",
                    event_type="helpful",
                )
            ]
        )
    )
    ranker_input = RankerInput(
        context=RecommendationContext(
            diagnosis_codes=[" e11.9 "],
            diagnosis_terms=["type 2|diabetes"],
            recommended_tests=["HbA1c=check"],
            demo_safe=True,
        ),
        preferences=UserPreferenceContext(
            preferred_categories=["test_explanation"],
            preferred_depth="beginner",
            preferred_format="bullet_points",
            max_reading_minutes=3,
        ),
        history=history,
        candidates=candidates,
        already_selected_topic_ids=(),
        top_k=3,
    )
    prompt = MiniOneRecPromptBuilder().build(
        ranker_input=ranker_input,
        selected_topic_tokens=["<MED_TOPIC_0012>"],
        history_topic_tokens=topic_store.topic_id_to_token(),
    )

    assert prompt.splitlines() == [
        "<REC>",
        "DX_CODES=E11.9",
        "DX_TERMS=type_2_diabetes",
        "TESTS=HbA1c_check",
        "MEDICATIONS=NONE",
        "PREFERRED_CATEGORIES=test_explanation",
        "EXCLUDED_CATEGORIES=NONE",
        "DEPTH=beginner",
        "FORMAT=bullet_points",
        "MAX_READING_MINUTES=3",
        "HISTORY=<EV_HELPFUL><MED_TOPIC_0002>",
        "CANDIDATES=<MED_TOPIC_0011>|<MED_TOPIC_0012>|<MED_TOPIC_0014>",
        "SELECTED=<MED_TOPIC_0012>",
        "NEXT=",
        "</REC>",
        "<NEXT_TOPIC>",
    ]
    assert "patient_description" not in prompt
    assert prompt == MiniOneRecPromptBuilder().build(
        ranker_input=ranker_input,
        selected_topic_tokens=["<MED_TOPIC_0012>"],
        history_topic_tokens=topic_store.topic_id_to_token(),
    )


def test_no_history_and_no_selected_tokens_are_explicit(topic_store) -> None:
    topic = topic_store.get_by_id("pneumonia_basics")
    assert topic is not None
    ranker_input = RankerInput(
        context=RecommendationContext(demo_safe=True),
        preferences=None,
        history=HistoryNormalizer(topic_store, 20).normalize(None),
        candidates=(topic,),
        already_selected_topic_ids=(),
        top_k=1,
    )
    prompt = MiniOneRecPromptBuilder().build(
        ranker_input=ranker_input,
        selected_topic_tokens=[],
        history_topic_tokens=topic_store.topic_id_to_token(),
    )
    assert "HISTORY=<NO_HISTORY>" in prompt
    assert "SELECTED=<NO_SELECTED>" in prompt
    assert prompt.endswith("<NEXT_TOPIC>")


class CharacterTokenizer:
    def encode(self, text, *, add_special_tokens=False):
        return list(range(len(text)))


def test_prompt_sorts_candidate_tokens_and_trims_old_history(topic_store) -> None:
    candidate_ids = [
        "follow_up_checklist",
        "diabetes_basics",
        "hba1c_test_explanation",
    ]
    candidates = tuple(topic_store.get_by_id(item) for item in candidate_ids)
    assert all(item is not None for item in candidates)
    interactions = [
        TopicInteraction(topic_id=topic.topic_id, event_type="view")
        for topic in topic_store.list_all()
    ]
    ranker_input = RankerInput(
        context=RecommendationContext(diagnosis_codes=["E11.9"], demo_safe=True),
        preferences=None,
        history=HistoryNormalizer(topic_store, 20).normalize(
            UserHistoryContext(interactions=interactions)
        ),
        candidates=tuple(item for item in candidates if item is not None),
        already_selected_topic_ids=(),
        top_k=3,
    )
    builder = MiniOneRecPromptBuilder(
        tokenizer=CharacterTokenizer(),
        max_input_tokens=400,
    )
    prompt = builder.build(
        ranker_input=ranker_input,
        selected_topic_tokens=[],
        history_topic_tokens=topic_store.topic_id_to_token(),
    )
    assert "CANDIDATES=<MED_TOPIC_0011>|<MED_TOPIC_0012>|<MED_TOPIC_0014>" in prompt
    assert builder.token_count(prompt) <= 400
    assert prompt.endswith("<NEXT_TOPIC>")
    assert prompt.count("<EV_VIEW>") < len(interactions)


def test_prompt_rejects_context_that_cannot_fit_after_trimming(topic_store) -> None:
    topic = topic_store.get_by_id("diabetes_basics")
    assert topic is not None
    ranker_input = RankerInput(
        context=RecommendationContext(
            diagnosis_terms=[f"term-{index}-" + "x" * 128 for index in range(20)],
            demo_safe=True,
        ),
        preferences=None,
        history=HistoryNormalizer(topic_store, 20).normalize(None),
        candidates=(topic,),
        already_selected_topic_ids=(),
        top_k=1,
    )
    with pytest.raises(ModelInferenceError, match="prompt_too_long"):
        MiniOneRecPromptBuilder(
            tokenizer=CharacterTokenizer(),
            max_input_tokens=128,
        ).build(
            ranker_input=ranker_input,
            selected_topic_tokens=[],
            history_topic_tokens=topic_store.topic_id_to_token(),
        )

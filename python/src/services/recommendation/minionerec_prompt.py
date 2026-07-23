"""Canonical prompt serialization shared by training and online inference."""

from __future__ import annotations

import unicodedata

from .ranker_protocol import (
    EVENT_TOKENS,
    ModelInferenceError,
    RankerInput,
)


class MiniOneRecPromptBuilder:
    def __init__(self, *, tokenizer=None, max_input_tokens: int = 1024):
        self._tokenizer = tokenizer
        self._max_input_tokens = max_input_tokens

    @staticmethod
    def _clean(value: object, *, code: bool = False) -> str:
        text = unicodedata.normalize("NFKC", str(value or ""))
        for character in "|=<>\r\n\t":
            text = text.replace(character, " ")
        text = "_".join(text.strip().split())[:128]
        return text.upper() if code else text

    @classmethod
    def _join(cls, values, *, code: bool = False) -> str:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            cleaned = cls._clean(value, code=code)
            key = cleaned.casefold()
            if cleaned and key not in seen:
                result.append(cleaned)
                seen.add(key)
        return "|".join(result) if result else "NONE"

    def build(
        self,
        *,
        ranker_input: RankerInput,
        selected_topic_tokens: list[str],
        preserve_candidate_order: bool = False,
        history_topic_tokens: dict[str, str] | None = None,
    ) -> str:
        # History can reference any globally registered topic, so temporarily
        # resolve it through the explicit registry supplied by the caller.
        interactions = list(ranker_input.history.interactions)

        def serialize(include_optional_preferences: bool = True) -> str:
            preferences = ranker_input.preferences
            topic_tokens = [topic.topic_token for topic in ranker_input.candidates]
            if not preserve_candidate_order:
                topic_tokens.sort()
            history_parts = []
            for item in interactions:
                token = (history_topic_tokens or {}).get(item.topic_id)
                if token and item.event_type in EVENT_TOKENS:
                    history_parts.append(EVENT_TOKENS[item.event_type] + token)
            preferred_categories = (
                [item.value for item in preferences.preferred_categories]
                if preferences
                else []
            )
            excluded_categories = (
                [item.value for item in preferences.excluded_categories]
                if preferences
                else []
            )
            depth = preferences.preferred_depth if preferences else None
            content_format = preferences.preferred_format if preferences else None
            reading_minutes = preferences.max_reading_minutes if preferences else None
            if not include_optional_preferences:
                depth = None
                content_format = None
                reading_minutes = None
            return "\n".join(
                [
                    "<REC>",
                    f"DX_CODES={self._join(ranker_input.context.diagnosis_codes, code=True)}",
                    f"DX_TERMS={self._join(ranker_input.context.diagnosis_terms)}",
                    f"TESTS={self._join(ranker_input.context.recommended_tests)}",
                    f"MEDICATIONS={self._join(ranker_input.context.medication_names)}",
                    f"PREFERRED_CATEGORIES={self._join(preferred_categories)}",
                    f"EXCLUDED_CATEGORIES={self._join(excluded_categories)}",
                    f"DEPTH={self._clean(depth) if depth else 'NONE'}",
                    f"FORMAT={self._clean(content_format) if content_format else 'NONE'}",
                    (
                        f"MAX_READING_MINUTES={reading_minutes}"
                        if reading_minutes is not None
                        else "MAX_READING_MINUTES=NONE"
                    ),
                    f"HISTORY={'|'.join(history_parts) if history_parts else '<NO_HISTORY>'}",
                    f"CANDIDATES={'|'.join(topic_tokens)}",
                    (
                        f"SELECTED={'|'.join(selected_topic_tokens)}"
                        if selected_topic_tokens
                        else "SELECTED=<NO_SELECTED>"
                    ),
                    "NEXT=",
                    "</REC>",
                    "<NEXT_TOPIC>",
                ]
            )

        prompt = serialize()
        if self._tokenizer is None:
            return prompt
        while interactions and self.token_count(prompt) > self._max_input_tokens:
            interactions.pop(0)
            prompt = serialize()
        if self.token_count(prompt) > self._max_input_tokens:
            prompt = serialize(include_optional_preferences=False)
        if self.token_count(prompt) > self._max_input_tokens:
            raise ModelInferenceError("prompt_too_long")
        if not prompt.endswith("<NEXT_TOPIC>"):
            raise ModelInferenceError("prompt_tail_missing")
        return prompt

    def token_count(self, prompt: str) -> int:
        if self._tokenizer is None:
            return 0
        return len(self._tokenizer.encode(prompt, add_special_tokens=False))

    @classmethod
    def build_topic_alignment_prompt(cls, topic) -> str:
        return "\n".join(
            [
                "<TOPIC_META>",
                f"TITLE={cls._clean(topic.title)}",
                f"CATEGORY={topic.category.value}",
                f"TERMS={cls._join(topic.related_terms)}",
                f"TESTS={cls._join(topic.related_tests)}",
                "TARGET=",
                "</TOPIC_META>",
                "<TOPIC_SID>",
            ]
        )

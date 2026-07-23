"""Internal contracts shared by local recommendation rankers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ...models.recommendation import (
    KnowledgeTopic,
    RecommendationContext,
    TopicInteraction,
    UserPreferenceContext,
)


EVENT_TOKENS: dict[str, str] = {
    "view": "<EV_VIEW>",
    "save": "<EV_SAVE>",
    "helpful": "<EV_HELPFUL>",
    "dismiss": "<EV_DISMISS>",
    "not_helpful": "<EV_NOT_HELPFUL>",
}
CONTROL_TOKENS: tuple[str, ...] = (
    *EVENT_TOKENS.values(),
    "<NO_HISTORY>",
    "<NO_SELECTED>",
    "<REC>",
    "</REC>",
    "<NEXT_TOPIC>",
    "<TOPIC_META>",
    "</TOPIC_META>",
    "<TOPIC_SID>",
)


@dataclass(frozen=True)
class NormalizedHistory:
    interactions: tuple[TopicInteraction, ...] = ()
    valid_count: int = 0
    dropped_unknown_count: int = 0
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class TopicMatchSignals:
    code_match: bool = False
    diagnosis_match: bool = False
    test_match: bool = False
    medication_match: bool = False
    preferred_category_match: bool = False
    preferred_depth_match: bool = False
    preferred_format_match: bool = False
    reading_time_match: bool = False
    viewed_before: bool = False
    positive_same_category: bool = False

    @property
    def clinical_match(self) -> bool:
        return any(
            (
                self.code_match,
                self.diagnosis_match,
                self.test_match,
                self.medication_match,
            )
        )

    @property
    def strong_safety_match(self) -> bool:
        return self.code_match or self.diagnosis_match or self.test_match


@dataclass(frozen=True)
class CandidatePolicyResult:
    pinned_topics: tuple[KnowledgeTopic, ...]
    rankable_topics: tuple[KnowledgeTopic, ...]
    signals_by_topic_id: dict[str, TopicMatchSignals]
    warnings: tuple[str, ...] = ()
    excluded_topic_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class RankerInput:
    context: RecommendationContext
    preferences: UserPreferenceContext | None
    history: NormalizedHistory
    candidates: tuple[KnowledgeTopic, ...]
    already_selected_topic_ids: tuple[str, ...]
    top_k: int


@dataclass(frozen=True)
class RankerResult:
    topic_ids: tuple[str, ...]
    strategy_used: str
    model_version: str | None = None
    inference_ms: float | None = None
    warnings: tuple[str, ...] = ()
    fallback_reason: str | None = None
    diagnostics: dict[str, object] = field(default_factory=dict)


class RecommendationRanker(Protocol):
    def rank(self, ranker_input: RankerInput) -> RankerResult:
        """Return an ordered subset of the supplied candidates."""


class RecommendationRankerError(RuntimeError):
    """Base class for controlled ranking failures."""

    default_code = "inference_failed"

    def __init__(self, code: str | None = None):
        self.code = code or self.default_code
        super().__init__(self.code)


class ModelArtifactError(RecommendationRankerError):
    default_code = "artifact_incompatible"


class ModelArtifactMissingError(ModelArtifactError):
    default_code = "artifact_missing"


class ModelNotReadyError(RecommendationRankerError):
    default_code = "model_not_ready"


class ModelLoadError(RecommendationRankerError):
    default_code = "model_load_failed"


class ModelInferenceError(RecommendationRankerError):
    default_code = "inference_failed"


class InvalidModelOutputError(RecommendationRankerError):
    default_code = "invalid_model_output"


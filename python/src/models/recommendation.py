"""Pydantic contracts for topic ranking and educational content cards."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class TopicCategory(str, Enum):
    DISEASE_BASICS = "disease_basics"
    TEST_EXPLANATION = "test_explanation"
    MEDICATION_SAFETY = "medication_safety"
    LIFESTYLE_EDUCATION = "lifestyle_education"
    FOLLOW_UP_EDUCATION = "follow_up_education"
    CARE_PROCESS = "care_process"
    WARNING_SIGNS = "warning_signs"


class UserPreferenceContext(BaseModel):
    preferred_categories: list[TopicCategory] = Field(default_factory=list)
    excluded_categories: list[TopicCategory] = Field(default_factory=list)
    preferred_depth: Literal["beginner", "standard"] | None = None
    preferred_format: Literal[
        "brief",
        "bullet_points",
        "step_by_step",
        "question_answer",
    ] | None = None
    max_reading_minutes: int | None = Field(default=None, ge=1, le=10)


class TopicInteraction(BaseModel):
    topic_id: str = Field(min_length=1, max_length=100)
    event_type: Literal[
        "view",
        "save",
        "helpful",
        "dismiss",
        "not_helpful",
    ]
    occurred_at: datetime | None = None


class UserHistoryContext(BaseModel):
    interactions: list[TopicInteraction] = Field(
        default_factory=list,
        max_length=20,
    )


class KnowledgeTopic(BaseModel):
    topic_id: str = Field(min_length=1, max_length=100)
    topic_token: str = Field(pattern=r"^<MED_TOPIC_[0-9]{4,}>$")
    title: str = Field(min_length=1)
    category: TopicCategory
    depth: Literal["beginner", "standard"]
    format: Literal[
        "brief",
        "bullet_points",
        "step_by_step",
        "question_answer",
    ]
    estimated_reading_minutes: int = Field(ge=1, le=10)

    related_codes: list[str] = Field(default_factory=list)
    related_terms: list[str] = Field(default_factory=list)
    related_tests: list[str] = Field(default_factory=list)
    related_medications: list[str] = Field(default_factory=list)

    summary: str = Field(min_length=1)
    source_label: str = Field(min_length=1)
    source_url: str | None = None
    safety_note: str = Field(min_length=1)

    mandatory_safety: bool = False
    general_fallback: bool = False
    priority: int = 0
    status: Literal["active", "prototype", "inactive"] = "active"

    @field_validator("source_label")
    @classmethod
    def normalize_source_label(cls, value: str) -> str:
        return value.strip()


class GeneratedEducationCard(BaseModel):
    topic_id: str = Field(min_length=1, max_length=100)
    summary: str = Field(min_length=40, max_length=1200)


class GeneratedEducationContent(BaseModel):
    cards: list[GeneratedEducationCard] = Field(min_length=1, max_length=3)


class KnowledgeRecommendation(BaseModel):
    rank: int = Field(ge=1, le=3)
    topic_id: str
    title: str
    category: TopicCategory
    reason: str
    summary: str
    source_label: str
    source_url: str | None = None
    safety_note: str
    content_source: Literal["deepseek_generated", "catalog_fallback"] = (
        "catalog_fallback"
    )
    content_depth: Literal["beginner", "standard"] = "beginner"


class EducationRecommendationResult(BaseModel):
    recommendation_status: Literal["ok", "degraded", "disabled"]
    # Compatibility projection retained for one API generation.
    strategy_used: str = "none"
    ranking_strategy_used: Literal[
        "mini_onerec_mvp",
        "rule_v1_fallback",
        "none",
    ] = "none"
    content_strategy_used: Literal[
        "deepseek_generated",
        "catalog_fallback",
        "none",
    ] = "none"
    model_version: str | None = None
    model_ready: bool = False
    fallback_reason: Literal[
        "model_disabled",
        "artifact_missing",
        "artifact_incompatible",
        "model_load_failed",
        "model_not_ready",
        "unsafe_context",
        "no_rankable_candidates",
        "inference_failed",
        "cuda_oom",
        "invalid_model_output",
        "concurrency_busy",
    ] | None = None
    ranker_inference_ms: float | None = None
    candidate_source: Literal["neo4j", "local_catalog", "none"] = "none"
    candidate_cache_status: Literal["hit", "miss", "offline", "none"] = "none"
    content_cache_status: Literal["hit", "miss", "fallback", "none"] = "none"
    history_used: bool = False
    valid_history_count: int = 0
    candidate_count: int = 0
    recommendations: list[KnowledgeRecommendation] = Field(
        default_factory=list,
        max_length=3,
    )
    warnings: list[str] = Field(default_factory=list)


class RecommendationContext(BaseModel):
    diagnosis_terms: list[str] = Field(default_factory=list)
    diagnosis_codes: list[str] = Field(default_factory=list)
    recommended_tests: list[str] = Field(default_factory=list)
    medication_names: list[str] = Field(default_factory=list)
    demo_safe: bool = True

"""Pydantic contracts for deterministic educational recommendations."""

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
    status: Literal["prototype", "inactive"] = "prototype"

    @field_validator("source_label")
    @classmethod
    def require_honest_source_label(cls, value: str) -> str:
        if "not medically reviewed" not in value.casefold():
            raise ValueError("prototype topics must state that they are not medically reviewed")
        return value


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


class EducationRecommendationResult(BaseModel):
    recommendation_status: Literal["ok", "degraded", "disabled"]
    strategy_used: Literal["rule_v1", "none"]
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
    demo_safe: bool = False

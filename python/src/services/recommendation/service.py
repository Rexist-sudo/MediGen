"""Fault-isolated recommendation service executed after the clinical graph."""

from __future__ import annotations

from datetime import timezone
from functools import cached_property, lru_cache

import structlog

from ...config.settings import get_settings
from ...models.recommendation import (
    EducationRecommendationResult,
    KnowledgeRecommendation,
    KnowledgeTopic,
    RecommendationContext,
    TopicInteraction,
    UserHistoryContext,
    UserPreferenceContext,
)
from .ranker import MatchSignals, RuleRecommendationRanker
from .topic_store import TopicStore, resolve_topic_path

logger = structlog.get_logger(__name__)


def _unique_strings(values: list[object]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        cleaned = value.strip()
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result


def _dict(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def build_recommendation_context(clinical_result: dict) -> RecommendationContext:
    diagnosis = _dict(clinical_result.get("diagnosis"))
    primary = _dict(diagnosis.get("primary_diagnosis"))
    differential = [
        _dict(item)
        for item in diagnosis.get("differential_list", [])
        if isinstance(item, dict)
    ]

    coding = _dict(clinical_result.get("coding_result"))
    primary_code = _dict(coding.get("primary_icd10"))
    secondary_codes = [
        _dict(item)
        for item in coding.get("secondary_icd10_codes", [])
        if isinstance(item, dict)
    ]

    treatment = _dict(clinical_result.get("treatment_plan"))
    medications = [
        _dict(item)
        for item in treatment.get("medications", [])
        if isinstance(item, dict)
    ]
    audit = _dict(clinical_result.get("audit_result"))

    return RecommendationContext(
        diagnosis_terms=_unique_strings(
            [primary.get("disease_name")]
            + [item.get("disease_name") for item in differential]
        ),
        diagnosis_codes=_unique_strings(
            [primary_code.get("code")]
            + [item.get("code") for item in secondary_codes]
        ),
        recommended_tests=_unique_strings(diagnosis.get("recommended_tests", [])),
        medication_names=_unique_strings(
            [
                name
                for medication in medications
                for name in (
                    medication.get("generic_name"),
                    medication.get("drug_name"),
                )
            ]
        ),
        demo_safe=bool(audit.get("demo_safe", False)),
    )


class RecommendationService:
    def __init__(
        self,
        *,
        topic_path: str,
        ranker: RuleRecommendationRanker,
        enabled: bool,
        max_history_interactions: int = 20,
    ):
        self.topic_path = topic_path
        self.ranker = ranker
        self.enabled = enabled
        self.max_history_interactions = max(
            0,
            min(20, max_history_interactions),
        )

    @cached_property
    def topic_store(self) -> TopicStore:
        return TopicStore.from_jsonl(self.topic_path)

    def is_store_ready(self) -> bool:
        try:
            return bool(self.topic_store.list_active())
        except Exception as exc:
            logger.warning(
                "recommendation.readiness_failed",
                error_type=type(exc).__name__,
            )
            return False

    def recommend_after_analysis(
        self,
        *,
        clinical_result: dict,
        user_preferences: UserPreferenceContext | None,
        user_history_context: UserHistoryContext | None,
        top_k: int = 3,
    ) -> EducationRecommendationResult:
        if not self.enabled:
            return EducationRecommendationResult(
                recommendation_status="disabled",
                strategy_used="none",
            )

        try:
            topics = self.topic_store.list_active()
            warnings: list[str] = []
            preferences = self._normalize_preferences(user_preferences, warnings)
            history = self._normalize_history(
                user_history_context,
                topics,
                warnings,
            )
            context = build_recommendation_context(clinical_result)

            audit_present = isinstance(clinical_result.get("audit_result"), dict)
            degraded_for_audit = audit_present and not context.demo_safe
            if not audit_present:
                warnings.append("audit_result_missing")
            elif degraded_for_audit:
                warnings.append("audit_not_demo_safe")
                topics = [
                    topic
                    for topic in topics
                    if topic.mandatory_safety or topic.general_fallback
                ]

            ranking = self.ranker.rank_detailed(
                context=context,
                topics=topics,
                preferences=preferences,
                history=history,
                top_k=max(1, min(3, top_k)),
            )
            if ranking.safety_overrides:
                warnings.append("mandatory_safety_override")
            if not ranking.topics:
                warnings.append("no_recommendation_candidates")

            recommendations = [
                self._render(
                    topic=topic,
                    rank=index,
                    signals=ranking.signals_by_topic.get(
                        topic.topic_id,
                        MatchSignals(),
                    ),
                    preferences=preferences,
                    history=history,
                    topics=topics,
                )
                for index, topic in enumerate(ranking.topics, start=1)
            ]
            logger.info(
                "recommendation.success",
                candidate_count=ranking.candidate_count,
                result_count=len(recommendations),
                status="degraded" if degraded_for_audit else "ok",
            )
            return EducationRecommendationResult(
                recommendation_status=(
                    "degraded" if degraded_for_audit else "ok"
                ),
                strategy_used="rule_v1",
                history_used=bool(history),
                valid_history_count=len(history),
                candidate_count=ranking.candidate_count,
                recommendations=recommendations,
                warnings=warnings,
            )
        except Exception as exc:
            logger.warning(
                "recommendation.degraded",
                error_type=type(exc).__name__,
            )
            return EducationRecommendationResult(
                recommendation_status="degraded",
                strategy_used="none",
                warnings=["recommendation_unavailable"],
            )

    @staticmethod
    def _normalize_preferences(
        preferences: UserPreferenceContext | None,
        warnings: list[str],
    ) -> UserPreferenceContext | None:
        if preferences is None:
            return None
        excluded = set(preferences.excluded_categories)
        overlap = excluded.intersection(preferences.preferred_categories)
        if overlap:
            warnings.append("excluded_category_takes_precedence")
        return preferences.model_copy(
            update={
                "preferred_categories": [
                    category
                    for category in preferences.preferred_categories
                    if category not in excluded
                ]
            }
        )

    def _normalize_history(
        self,
        history_context: UserHistoryContext | None,
        topics: list[KnowledgeTopic],
        warnings: list[str],
    ) -> list[TopicInteraction]:
        interactions = list(history_context.interactions if history_context else [])
        valid_topic_ids = {topic.topic_id for topic in topics}
        valid: list[TopicInteraction] = []
        unknown_found = False
        for interaction in interactions:
            if interaction.topic_id not in valid_topic_ids:
                unknown_found = True
                continue
            valid.append(interaction)
        if unknown_found:
            warnings.append("unknown_history_topic_ignored")

        if any(item.occurred_at is not None for item in valid):
            indexed = list(enumerate(valid))

            def time_key(item: tuple[int, TopicInteraction]) -> tuple[float, int]:
                index, interaction = item
                occurred_at = interaction.occurred_at
                if occurred_at is None:
                    return (float("-inf"), index)
                if occurred_at.tzinfo is None:
                    occurred_at = occurred_at.replace(tzinfo=timezone.utc)
                return (occurred_at.timestamp(), index)

            valid = [item for _, item in sorted(indexed, key=time_key)]

        if self.max_history_interactions == 0:
            return []
        return valid[-self.max_history_interactions :]

    @staticmethod
    def _render(
        *,
        topic: KnowledgeTopic,
        rank: int,
        signals: MatchSignals,
        preferences: UserPreferenceContext | None,
        history: list[TopicInteraction],
        topics: list[KnowledgeTopic],
    ) -> KnowledgeRecommendation:
        helpful_categories = {
            next(
                (
                    candidate.category
                    for candidate in topics
                    if candidate.topic_id == interaction.topic_id
                ),
                None,
            )
            for interaction in history
            if interaction.event_type in {"helpful", "save"}
        }
        if topic.mandatory_safety and signals.strong_safety:
            reason = "这是与当前上下文相关的安全提醒主题。"
        elif topic.category in helpful_categories and not any(
            item.topic_id == topic.topic_id for item in history
        ):
            reason = "你曾对同类内容标记为有帮助，因此优先展示该未阅读主题。"
        elif signals.test:
            reason = "该主题与当前建议检查相关。"
        elif signals.code or signals.diagnosis:
            reason = "该主题与当前诊断类别相关。"
        else:
            reason = "这是用于软件原型演示的通用教育主题。"

        preference_matches: list[str] = []
        if preferences and preferences.preferred_depth == topic.depth:
            preference_matches.append("阅读深度")
        if preferences and preferences.preferred_format == topic.format:
            preference_matches.append("内容格式")
        if preference_matches:
            reason = reason.rstrip("。") + "，并符合你的" + "和".join(preference_matches) + "偏好。"

        return KnowledgeRecommendation(
            rank=rank,
            topic_id=topic.topic_id,
            title=topic.title,
            category=topic.category,
            reason=reason,
            summary=topic.summary,
            source_label=topic.source_label,
            source_url=topic.source_url,
            safety_note=topic.safety_note,
        )


@lru_cache(maxsize=1)
def get_recommendation_service() -> RecommendationService:
    settings = get_settings()
    return RecommendationService(
        topic_path=str(resolve_topic_path(settings.recommendation_topic_path)),
        ranker=RuleRecommendationRanker(),
        enabled=settings.recommendation_enabled,
        max_history_interactions=settings.max_history_interactions,
    )

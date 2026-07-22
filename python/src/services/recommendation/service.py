"""Fault-isolated recommendation service executed after the clinical graph."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import timezone
from functools import cached_property, lru_cache

import structlog

from ...config.settings import get_settings
from ...models.recommendation import (
    EducationRecommendationResult,
    GeneratedEducationContent,
    KnowledgeRecommendation,
    KnowledgeTopic,
    RecommendationContext,
    TopicInteraction,
    UserHistoryContext,
    UserPreferenceContext,
)
from ..deepseek_client import (
    DeepSeekOutputError,
    JSONClient,
    get_json_client,
)
from ..graphrag_service import GraphRAGService, get_graphrag_service
from ..redis_service import RedisService, get_redis_service
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
        topic_provider: GraphRAGService | None = None,
        content_client_factory: Callable[[], JSONClient] | None = None,
        cache: RedisService | None = None,
        generate_content: bool = False,
    ):
        self.topic_path = topic_path
        self.ranker = ranker
        self.enabled = enabled
        self.max_history_interactions = max(
            0,
            min(20, max_history_interactions),
        )
        self.topic_provider = topic_provider
        self.content_client_factory = content_client_factory
        self.cache = cache
        self.generate_content = generate_content

    @cached_property
    def topic_store(self) -> TopicStore:
        return TopicStore.from_jsonl(self.topic_path)

    def is_store_ready(self) -> bool:
        try:
            local_ready = bool(self.topic_store.list_active())
            if self.topic_provider and self.topic_provider.use_neo4j:
                return local_ready and self.topic_provider.is_ready()
            return local_ready
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
            warnings: list[str] = []
            preferences = self._normalize_preferences(user_preferences, warnings)
            context = build_recommendation_context(clinical_result)
            topics, candidate_source, candidate_cache_status = self._candidate_topics(
                context,
                warnings,
            )
            history = self._normalize_history(
                user_history_context,
                topics,
                warnings,
            )
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
            strategy = "rule_v1"
            content_cache_status = "none"
            if (
                recommendations
                and self.generate_content
                and self.content_client_factory is not None
            ):
                try:
                    recommendations, content_cache_status = self._generate_content(
                        recommendations=recommendations,
                        context=context,
                        depth=(
                            preferences.preferred_depth
                            if preferences and preferences.preferred_depth
                            else "beginner"
                        ),
                    )
                    strategy = "rule_v1_deepseek"
                except Exception as exc:
                    logger.warning(
                        "recommendation.content_generation_failed",
                        error_type=type(exc).__name__,
                    )
                    warnings.append("education_content_generation_fallback")
                    content_cache_status = "fallback"
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
                strategy_used=strategy,
                candidate_source=candidate_source,
                candidate_cache_status=candidate_cache_status,
                content_cache_status=content_cache_status,
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

    def _candidate_topics(
        self,
        context: RecommendationContext,
        warnings: list[str],
    ) -> tuple[list[KnowledgeTopic], str, str]:
        if self.topic_provider is None:
            return self.topic_store.list_active(), "local_catalog", "offline"
        try:
            result = self.topic_provider.find_education_topics(
                context.model_dump(mode="json", exclude={"demo_safe"})
            )
            topics = [KnowledgeTopic.model_validate(item) for item in result["topics"]]
            if topics:
                return topics, "neo4j", result["cache_status"]
            warnings.append("knowledge_graph_candidates_empty")
        except Exception as exc:
            logger.warning(
                "recommendation.graph_retrieval_failed",
                error_type=type(exc).__name__,
            )
            warnings.append("knowledge_graph_catalog_fallback")
        return self.topic_store.list_active(), "local_catalog", "offline"

    def _generate_content(
        self,
        *,
        recommendations: list[KnowledgeRecommendation],
        context: RecommendationContext,
        depth: str,
    ) -> tuple[list[KnowledgeRecommendation], str]:
        cache_payload = {
            "topic_ids": [item.topic_id for item in recommendations],
            "depth": depth,
            "context": context.model_dump(mode="json"),
        }
        digest = hashlib.sha256(
            json.dumps(
                cache_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        cache_key = f"medigen:education:{digest}"
        generated_payload = self.cache.get_json(cache_key) if self.cache else None
        cache_status = "hit" if isinstance(generated_payload, dict) else "miss"

        if isinstance(generated_payload, dict):
            generated = GeneratedEducationContent.model_validate(generated_payload)
        else:
            depth_instruction = (
                "Use plain language, define clinical terms, and keep each card between 100 and 180 Chinese characters."
                if depth == "beginner"
                else "Use clinically precise language with mechanisms, interpretation boundaries, and practical follow-up points; keep each card between 220 and 420 Chinese characters."
            )
            generated = self.content_client_factory().invoke_json(
                task_name="education_content",
                system_prompt=(
                    "Generate Chinese patient-education card content for the exact candidate topics supplied. "
                    "Return JSON only with shape {\"cards\":[{\"topic_id\":\"...\",\"summary\":\"...\"}]}. "
                    "Keep every topic_id unchanged and return one card per supplied topic. "
                    "Use only the bounded clinical context and general medical knowledge. Do not diagnose, prescribe, "
                    "specify individualized doses, or add patient facts. Use direct, clear system copy without "
                    "development-stage terminology or contrastive '不是...而是' phrasing. "
                    + depth_instruction
                ),
                user_prompt=json.dumps(
                    {
                        "content_depth": depth,
                        "clinical_context": context.model_dump(mode="json"),
                        "candidate_topics": [
                            {
                                "topic_id": item.topic_id,
                                "title": item.title,
                                "category": item.category.value,
                                "selection_reason": item.reason,
                            }
                            for item in recommendations
                        ],
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                response_model=GeneratedEducationContent,
                max_tokens=1800,
            )
            if self.cache:
                self.cache.set_json(cache_key, generated.model_dump(mode="json"))

        by_topic = {item.topic_id: item.summary for item in generated.cards}
        expected = {item.topic_id for item in recommendations}
        if not expected.issubset(by_topic):
            raise DeepSeekOutputError("generated education cards are incomplete")
        depth_label = "入门" if depth == "beginner" else "标准"
        return [
            item.model_copy(
                update={
                    "summary": by_topic[item.topic_id],
                    "source_label": f"DeepSeek 生成 · {depth_label}内容",
                    "content_source": "deepseek_generated",
                    "content_depth": depth,
                }
            )
            for item in recommendations
        ], cache_status

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
            reason = "该主题与当前结果中的健康教育需求相关。"

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
        topic_provider=get_graphrag_service(),
        content_client_factory=(
            get_json_client
            if settings.llm_backend == "deepseek"
            and settings.recommendation_generate_content
            else None
        ),
        cache=(
            get_redis_service()
            if settings.llm_backend == "deepseek"
            else None
        ),
        generate_content=(
            settings.llm_backend == "deepseek"
            and settings.recommendation_generate_content
        ),
    )

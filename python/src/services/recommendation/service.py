"""Fault-isolated recommendation orchestration after the clinical graph."""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections.abc import Callable
from functools import lru_cache

import structlog

from ...config.settings import get_settings
from ...models.recommendation import (
    EducationRecommendationResult,
    GeneratedEducationContent,
    KnowledgeRecommendation,
    KnowledgeTopic,
    RecommendationContext,
    UserHistoryContext,
    UserPreferenceContext,
)
from ..deepseek_client import DeepSeekOutputError, JSONClient, get_json_client
from ..graphrag_service import GraphRAGService, get_graphrag_service
from ..redis_service import RedisService, get_redis_service
from .candidate_policy import CandidatePolicy
from .card_renderer import CardRenderer
from .context_builder import (
    RecommendationContextBuilder,
    build_recommendation_context,
)
from .history_normalizer import HistoryNormalizer
from .minionerec_ranker import MiniOneRecRanker
from .model_loader import MiniOneRecModelLoader, ModelReadiness
from .output_validator import RecommendationOutputValidator
from .ranker_protocol import (
    InvalidModelOutputError,
    RankerInput,
    RankerResult,
)
from .ranker_router import RankerRouter
from .rule_fallback_ranker import RuleFallbackRanker
from .topic_store import TopicStore, resolve_topic_path


logger = structlog.get_logger(__name__)
UNSAFE_DYNAMIC_CONTENT = re.compile(
    r"(?:\b\d+(?:\.\d+)?\s*(?:mg|g|ml|μg|mcg)\b|"
    r"自行(?:停药|换药|加量|减量)|嚼服.{0,12}\d)",
    re.IGNORECASE,
)


def _compatibility_strategy(ranking: str, content: str) -> str:
    if ranking == "mini_onerec_mvp":
        return (
            "mini_onerec_mvp_deepseek"
            if content == "deepseek_generated"
            else "mini_onerec_mvp"
        )
    if ranking == "rule_v1_fallback":
        return (
            "rule_v1_fallback_deepseek"
            if content == "deepseek_generated"
            else "rule_v1_fallback"
        )
    return "none"


class RecommendationService:
    def __init__(
        self,
        *,
        topic_store: TopicStore,
        ranker_router: RankerRouter,
        enabled: bool,
        max_candidates: int,
        context_builder: RecommendationContextBuilder | None = None,
        history_normalizer: HistoryNormalizer | None = None,
        candidate_policy: CandidatePolicy | None = None,
        output_validator: RecommendationOutputValidator | None = None,
        card_renderer: CardRenderer | None = None,
        topic_provider: GraphRAGService | None = None,
        content_client_factory: Callable[[], JSONClient] | None = None,
        cache: RedisService | None = None,
        generate_content: bool = False,
    ):
        self.topic_store = topic_store
        self.ranker_router = ranker_router
        self.enabled = enabled
        self.max_candidates = max_candidates
        self.context_builder = context_builder or RecommendationContextBuilder()
        self.history_normalizer = history_normalizer or HistoryNormalizer(
            topic_store,
            20,
        )
        self.candidate_policy = candidate_policy or CandidatePolicy(topic_store)
        self.output_validator = output_validator or RecommendationOutputValidator(
            topic_store
        )
        self.card_renderer = card_renderer or CardRenderer()
        self.topic_provider = topic_provider
        self.content_client_factory = content_client_factory
        self.cache = cache
        self.generate_content = generate_content

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

    def model_readiness(self, *, load: bool = False) -> ModelReadiness:
        return self.ranker_router.readiness(load=load)

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
                ranking_strategy_used="none",
                content_strategy_used="none",
            )

        top_k = max(1, min(3, top_k))
        try:
            warnings: list[str] = []
            preferences = self._normalize_preferences(
                user_preferences,
                warnings,
            )
            context = self.context_builder.build(clinical_result)
            recalled, candidate_source, candidate_cache_status = (
                self._candidate_topics(context, warnings)
            )
            history = self.history_normalizer.normalize(user_history_context)
            policy = self.candidate_policy.apply(
                context=context,
                preferences=preferences,
                history=history,
                recalled_topics=recalled,
                all_active_topics=self.topic_store.list_active(),
                top_k=top_k,
                max_candidates=self.max_candidates,
            )
            warnings.extend(policy.warnings)
            remaining_slots = max(0, top_k - len(policy.pinned_topics))
            ranker_input = RankerInput(
                context=context,
                preferences=preferences,
                history=history,
                candidates=policy.rankable_topics,
                already_selected_topic_ids=tuple(
                    topic.topic_id for topic in policy.pinned_topics
                ),
                top_k=remaining_slots,
            )
            if remaining_slots and policy.rankable_topics:
                ranker_result = self.ranker_router.rank(ranker_input)
            else:
                reason = None
                if not context.demo_safe:
                    reason = "unsafe_context"
                elif remaining_slots and not policy.rankable_topics:
                    reason = "no_rankable_candidates"
                ranker_result = RankerResult(
                    topic_ids=(),
                    strategy_used="none",
                    fallback_reason=reason,
                )
            try:
                selected = self.output_validator.validate(
                    ranker_result=ranker_result,
                    policy_result=policy,
                    top_k=top_k,
                )
            except InvalidModelOutputError:
                ranker_result = self.ranker_router.rank_fallback(
                    ranker_input,
                    reason="invalid_model_output",
                )
                selected = self.output_validator.validate(
                    ranker_result=ranker_result,
                    policy_result=policy,
                    top_k=top_k,
                )

            warnings.extend(ranker_result.warnings)
            if not selected:
                warnings.append("no_recommendation_candidates")
            recommendations = self.card_renderer.render(
                topics=selected,
                signals_by_topic_id=policy.signals_by_topic_id,
                preferences=preferences,
                history=history,
            )
            content_strategy = "catalog_fallback" if recommendations else "none"
            content_cache_status = "none"
            content_failed = False
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
                    content_strategy = "deepseek_generated"
                except Exception as exc:
                    logger.warning(
                        "recommendation.content_generation_failed",
                        error_type=type(exc).__name__,
                    )
                    warnings.append("education_content_generation_fallback")
                    content_cache_status = "fallback"
                    content_failed = True

            readiness = self.model_readiness(load=False)
            ranking_strategy = (
                ranker_result.strategy_used
                if ranker_result.strategy_used
                in {"mini_onerec_mvp", "rule_v1_fallback"}
                else "none"
            )
            fallback_reason = ranker_result.fallback_reason
            degraded = bool(
                fallback_reason
                or not context.demo_safe
                or content_failed
            )
            model_ready = (
                ranking_strategy == "mini_onerec_mvp"
                or (readiness.artifact_valid and readiness.loaded)
            )
            logger.info(
                "recommendation.success",
                candidate_count=(
                    len(policy.pinned_topics) + len(policy.rankable_topics)
                ),
                result_count=len(recommendations),
                ranking_strategy=ranking_strategy,
                fallback_reason=fallback_reason,
                history_count=history.valid_count,
                inference_ms=ranker_result.inference_ms,
            )
            return EducationRecommendationResult(
                recommendation_status="degraded" if degraded else "ok",
                strategy_used=_compatibility_strategy(
                    ranking_strategy,
                    content_strategy,
                ),
                ranking_strategy_used=ranking_strategy,
                content_strategy_used=content_strategy,
                model_version=(
                    ranker_result.model_version or readiness.model_version
                ),
                model_ready=model_ready,
                fallback_reason=fallback_reason,
                ranker_inference_ms=ranker_result.inference_ms,
                candidate_source=candidate_source,
                candidate_cache_status=candidate_cache_status,
                content_cache_status=content_cache_status,
                history_used=bool(history.interactions),
                valid_history_count=history.valid_count,
                candidate_count=(
                    len(policy.pinned_topics) + len(policy.rankable_topics)
                ),
                recommendations=recommendations,
                warnings=list(dict.fromkeys(warnings)),
            )
        except Exception as exc:
            logger.warning(
                "recommendation.degraded",
                error_type=type(exc).__name__,
            )
            return EducationRecommendationResult(
                recommendation_status="degraded",
                strategy_used="none",
                ranking_strategy_used="none",
                content_strategy_used="none",
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
            topics: list[KnowledgeTopic] = []
            seen: set[str] = set()
            for raw in result.get("topics", []):
                if not isinstance(raw, dict):
                    continue
                topic_id = raw.get("topic_id")
                if not isinstance(topic_id, str) or topic_id in seen:
                    continue
                topic = self.topic_store.get_by_id(topic_id)
                if topic is not None:
                    topics.append(topic)
                    seen.add(topic_id)
            if topics:
                return topics, "neo4j", str(result.get("cache_status", "miss"))
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
        cache_key = f"medigen:education:v2:{digest}"
        generated_payload = self.cache.get_json(cache_key) if self.cache else None
        cache_status = "hit" if isinstance(generated_payload, dict) else "miss"

        if isinstance(generated_payload, dict):
            generated = GeneratedEducationContent.model_validate(generated_payload)
        else:
            depth_instruction = (
                "Use plain language, define clinical terms, and keep each card "
                "between 100 and 180 Chinese characters."
                if depth == "beginner"
                else "Use clinically precise language with mechanisms, interpretation "
                "boundaries, and practical follow-up points; keep each card between "
                "220 and 420 Chinese characters."
            )
            generated = self.content_client_factory().invoke_json(
                task_name="education_content",
                system_prompt=(
                    "Generate Chinese educational text for the exact reviewed topics. "
                    "Return JSON with shape {\"cards\":[{\"topic_id\":\"...\","
                    "\"summary\":\"...\"}]}. Keep every topic_id unchanged and "
                    "return one card per supplied topic. Do not diagnose, prescribe, "
                    "name an individualized dose, direct medication changes, or add "
                    "patient facts. "
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
        if any(UNSAFE_DYNAMIC_CONTENT.search(by_topic[item]) for item in expected):
            raise DeepSeekOutputError("generated education content failed safety policy")
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


@lru_cache(maxsize=1)
def get_recommendation_service() -> RecommendationService:
    settings = get_settings()
    store = TopicStore.from_jsonl(
        resolve_topic_path(settings.recommendation_topic_path)
    )
    loader = MiniOneRecModelLoader(settings=settings, topic_store=store)
    fallback = RuleFallbackRanker(store)
    primary = MiniOneRecRanker(
        model_loader=loader,
        topic_store=store,
        inference_semaphore=threading.Semaphore(
            settings.minionerec_inference_concurrency
        ),
        semaphore_wait_seconds=settings.minionerec_semaphore_wait_seconds,
        max_input_tokens=settings.minionerec_max_input_tokens,
    )
    router = RankerRouter(
        primary=primary,
        fallback=fallback,
        settings=settings,
        model_loader=loader,
    )
    return RecommendationService(
        topic_store=store,
        ranker_router=router,
        enabled=settings.recommendation_enabled,
        max_candidates=settings.minionerec_max_candidates,
        history_normalizer=HistoryNormalizer(
            store,
            settings.minionerec_max_history,
        ),
        topic_provider=get_graphrag_service(),
        content_client_factory=(
            get_json_client
            if settings.llm_backend == "deepseek"
            and settings.recommendation_generate_content
            else None
        ),
        cache=(
            get_redis_service() if settings.llm_backend == "deepseek" else None
        ),
        generate_content=(
            settings.llm_backend == "deepseek"
            and settings.recommendation_generate_content
        ),
    )


__all__ = [
    "RecommendationService",
    "build_recommendation_context",
    "get_recommendation_service",
]

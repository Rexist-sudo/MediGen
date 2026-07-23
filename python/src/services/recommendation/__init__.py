"""Safe recommendation package with lazy public imports for model tooling."""

from __future__ import annotations


__all__ = [
    "MiniOneRecRanker",
    "RankerRouter",
    "RecommendationRanker",
    "RecommendationService",
    "RuleFallbackRanker",
    "TopicStore",
    "TopicStoreError",
    "get_recommendation_service",
    "resolve_topic_path",
]


def __getattr__(name: str):
    if name in {"RecommendationService", "get_recommendation_service"}:
        from .service import RecommendationService, get_recommendation_service

        return {
            "RecommendationService": RecommendationService,
            "get_recommendation_service": get_recommendation_service,
        }[name]
    if name == "MiniOneRecRanker":
        from .minionerec_ranker import MiniOneRecRanker

        return MiniOneRecRanker
    if name == "RankerRouter":
        from .ranker_router import RankerRouter

        return RankerRouter
    if name == "RecommendationRanker":
        from .ranker_protocol import RecommendationRanker

        return RecommendationRanker
    if name == "RuleFallbackRanker":
        from .rule_fallback_ranker import RuleFallbackRanker

        return RuleFallbackRanker
    if name in {"TopicStore", "TopicStoreError", "resolve_topic_path"}:
        from .topic_store import TopicStore, TopicStoreError, resolve_topic_path

        return {
            "TopicStore": TopicStore,
            "TopicStoreError": TopicStoreError,
            "resolve_topic_path": resolve_topic_path,
        }[name]
    raise AttributeError(name)

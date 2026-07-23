"""Final enforcement of topic eligibility and ranking invariants."""

from __future__ import annotations

from ...models.recommendation import KnowledgeTopic
from .ranker_protocol import (
    CandidatePolicyResult,
    InvalidModelOutputError,
    RankerResult,
)
from .topic_store import TopicStore


class RecommendationOutputValidator:
    def __init__(self, topic_store: TopicStore):
        self._topic_store = topic_store

    def validate(
        self,
        *,
        ranker_result: RankerResult,
        policy_result: CandidatePolicyResult,
        top_k: int,
    ) -> list[KnowledgeTopic]:
        top_k = max(1, min(3, top_k))
        rankable_ids = {topic.topic_id for topic in policy_result.rankable_topics}
        excluded_ids = set(policy_result.excluded_topic_ids)
        model_ids = list(ranker_result.topic_ids)
        if len(model_ids) != len(set(model_ids)):
            raise InvalidModelOutputError()
        if any(topic_id not in rankable_ids for topic_id in model_ids):
            raise InvalidModelOutputError()
        if any(topic_id in excluded_ids for topic_id in model_ids):
            raise InvalidModelOutputError()
        expected_slots = max(0, top_k - len(policy_result.pinned_topics))
        if expected_slots and rankable_ids and not model_ids:
            raise InvalidModelOutputError()
        if len(model_ids) > expected_slots:
            raise InvalidModelOutputError()
        if ranker_result.strategy_used == "mini_onerec_mvp" and not ranker_result.model_version:
            raise InvalidModelOutputError()

        selected_ids = [topic.topic_id for topic in policy_result.pinned_topics]
        selected_ids.extend(model_ids)
        if len(selected_ids) != len(set(selected_ids)):
            raise InvalidModelOutputError()
        selected: list[KnowledgeTopic] = []
        for topic_id in selected_ids[:top_k]:
            topic = self._topic_store.get_by_id(topic_id)
            if topic is None or topic.status != "active":
                raise InvalidModelOutputError()
            selected.append(topic)
        return selected


"""Render fixed catalog metadata and deterministic recommendation reasons."""

from __future__ import annotations

from ...models.recommendation import (
    KnowledgeRecommendation,
    KnowledgeTopic,
    UserPreferenceContext,
)
from .ranker_protocol import (
    NormalizedHistory,
    TopicMatchSignals,
)


class CardRenderer:
    def render(
        self,
        *,
        topics: list[KnowledgeTopic],
        signals_by_topic_id: dict[str, TopicMatchSignals],
        preferences: UserPreferenceContext | None,
        history: NormalizedHistory,
    ) -> list[KnowledgeRecommendation]:
        return [
            self._render_one(
                topic=topic,
                rank=index,
                signals=signals_by_topic_id.get(
                    topic.topic_id,
                    TopicMatchSignals(),
                ),
                preferences=preferences,
                history=history,
            )
            for index, topic in enumerate(topics, start=1)
        ]

    @staticmethod
    def _render_one(
        *,
        topic: KnowledgeTopic,
        rank: int,
        signals: TopicMatchSignals,
        preferences: UserPreferenceContext | None,
        history: NormalizedHistory,
    ) -> KnowledgeRecommendation:
        if topic.mandatory_safety and signals.strong_safety_match:
            reason = "该主题与本次结构化临床结果相关，按安全优先顺序展示。"
        elif signals.test_match:
            reason = "该主题与本次建议检查相关。"
        elif signals.code_match or signals.diagnosis_match:
            reason = "该主题与本次诊断类别相关。"
        elif signals.medication_match:
            reason = "该主题与本次用药信息核对相关。"
        elif signals.positive_same_category:
            reason = "历史反馈显示你关注同类内容，本次展示相关的新主题。"
        else:
            reason = "该主题适合作为本次健康教育补充材料。"

        matches: list[str] = []
        if preferences and preferences.preferred_depth == topic.depth:
            matches.append("阅读深度")
        if preferences and preferences.preferred_format == topic.format:
            matches.append("内容格式")
        if matches:
            reason = reason.rstrip("。") + "，并符合你的" + "和".join(matches) + "偏好。"
        if topic.mandatory_safety and topic.category in set(
            preferences.excluded_categories if preferences else ()
        ):
            reason = reason.rstrip("。") + "；该安全主题优先于普通类别偏好。"

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
            content_depth=(
                preferences.preferred_depth
                if preferences and preferences.preferred_depth
                else topic.depth
            ),
        )


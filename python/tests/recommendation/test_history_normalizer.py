from __future__ import annotations

from datetime import datetime, timezone

from src.models.recommendation import TopicInteraction, UserHistoryContext
from src.services.recommendation.history_normalizer import HistoryNormalizer


def test_global_registry_keeps_history_outside_request_candidates(topic_store) -> None:
    history = UserHistoryContext(
        interactions=[
            TopicInteraction(topic_id="pneumonia_basics", event_type="helpful"),
            TopicInteraction(topic_id="does_not_exist", event_type="view"),
        ]
    )
    result = HistoryNormalizer(topic_store, 20).normalize(history)

    assert [item.topic_id for item in result.interactions] == ["pneumonia_basics"]
    assert result.dropped_unknown_count == 1
    assert result.warnings == ("unknown_history_topic_ignored",)


def test_timed_events_sort_before_stable_untimed_events(topic_store) -> None:
    history = UserHistoryContext(
        interactions=[
            TopicInteraction(topic_id="diabetes_basics", event_type="view"),
            TopicInteraction(
                topic_id="hba1c_test_explanation",
                event_type="helpful",
                occurred_at=datetime(2026, 2, 1, tzinfo=timezone.utc),
            ),
            TopicInteraction(
                topic_id="pneumonia_basics",
                event_type="view",
                occurred_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            ),
            TopicInteraction(topic_id="follow_up_checklist", event_type="save"),
        ]
    )
    result = HistoryNormalizer(topic_store, 20).normalize(history)

    assert [item.topic_id for item in result.interactions] == [
        "pneumonia_basics",
        "hba1c_test_explanation",
        "diabetes_basics",
        "follow_up_checklist",
    ]
    assert "history_missing_timestamps" in result.warnings


def test_truncation_keeps_the_20_most_recent_events(topic_store) -> None:
    topic_ids = list(topic_store.topic_id_to_token())
    interactions = [
        TopicInteraction(
            topic_id=topic_ids[index % len(topic_ids)],
            event_type="view",
            occurred_at=datetime(2026, 1, index + 1, tzinfo=timezone.utc),
        )
        for index in range(25)
    ]
    history = UserHistoryContext.model_construct(interactions=interactions)
    result = HistoryNormalizer(topic_store, 20).normalize(history)
    assert result.valid_count == 20
    assert result.interactions[0].occurred_at == datetime(
        2026,
        1,
        6,
        tzinfo=timezone.utc,
    )
    assert result.interactions[-1].occurred_at == datetime(
        2026,
        1,
        25,
        tzinfo=timezone.utc,
    )


def test_inactive_registered_topic_remains_valid_history(topic_store) -> None:
    from src.services.recommendation.topic_store import TopicStore

    topics = topic_store.list_all()
    inactive_id = topics[0].topic_id
    topics[0] = topics[0].model_copy(update={"status": "inactive"})
    store = TopicStore(topics)
    result = HistoryNormalizer(store, 20).normalize(
        UserHistoryContext(
            interactions=[
                TopicInteraction(topic_id=inactive_id, event_type="helpful")
            ]
        )
    )
    assert result.valid_count == 1
    assert result.interactions[0].topic_id == inactive_id

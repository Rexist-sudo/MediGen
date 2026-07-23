from __future__ import annotations

import json

import pytest

from src.services.recommendation.ranker_protocol import ModelArtifactError
from src.services.recommendation.topic_store import TopicStore, TopicStoreError


EXPECTED = {
    "pneumonia_basics": "<MED_TOPIC_0001>",
    "chest_xray_explanation": "<MED_TOPIC_0002>",
    "myocardial_infarction_warning_signs": "<MED_TOPIC_0003>",
    "ecg_and_troponin_explanation": "<MED_TOPIC_0004>",
    "hypothyroidism_basics": "<MED_TOPIC_0005>",
    "thyroid_function_test_explanation": "<MED_TOPIC_0006>",
    "heart_failure_daily_monitoring": "<MED_TOPIC_0007>",
    "heart_failure_warning_signs": "<MED_TOPIC_0008>",
    "appendicitis_care_process": "<MED_TOPIC_0009>",
    "abdominal_pain_warning_signs": "<MED_TOPIC_0010>",
    "diabetes_basics": "<MED_TOPIC_0011>",
    "hba1c_test_explanation": "<MED_TOPIC_0012>",
    "medication_safety_basics": "<MED_TOPIC_0013>",
    "follow_up_checklist": "<MED_TOPIC_0014>",
    "when_to_seek_urgent_help": "<MED_TOPIC_0015>",
}


def test_fixed_token_map_and_canonical_hash(topic_store) -> None:
    assert len(topic_store.list_all()) == 15
    assert topic_store.topic_id_to_token() == dict(sorted(EXPECTED.items()))
    assert topic_store.catalog_sha256() == (
        "d21429f89d91fbf1ddb47e5dac470ca0dc506b21bdba31052512329697ce8c0e"
    )


def test_duplicate_token_fails(tmp_path, catalog_path) -> None:
    rows = [json.loads(line) for line in catalog_path.read_text(encoding="utf-8").splitlines()]
    rows[1]["topic_token"] = rows[0]["topic_token"]
    path = tmp_path / "topics.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    with pytest.raises(TopicStoreError, match="duplicate topic_token"):
        TopicStore.from_jsonl(path, require_token_map=False)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda rows: rows[1].update(topic_id=rows[0]["topic_id"]), "duplicate topic_id"),
        (lambda rows: rows[0].update(topic_token="MED_TOPIC_1"), "invalid topic record"),
    ],
)
def test_duplicate_id_and_invalid_token_fail(
    tmp_path,
    catalog_path,
    mutation,
    message,
) -> None:
    rows = [
        json.loads(line)
        for line in catalog_path.read_text(encoding="utf-8").splitlines()
    ]
    mutation(rows)
    path = tmp_path / "topics.jsonl"
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )
    with pytest.raises(TopicStoreError, match=message):
        TopicStore.from_jsonl(path, require_token_map=False)


def test_hash_ignores_order_whitespace_and_newline_style(
    tmp_path,
    catalog_path,
    topic_store,
) -> None:
    rows = [
        json.loads(line)
        for line in catalog_path.read_text(encoding="utf-8").splitlines()
    ]
    path = tmp_path / "topics.jsonl"
    path.write_text(
        "\r\n\r\n".join(
            json.dumps(row, ensure_ascii=False, separators=(", ", ": "))
            for row in reversed(rows)
        )
        + "\r\n",
        encoding="utf-8",
    )
    rebuilt = TopicStore.from_jsonl(path, require_token_map=False)
    assert rebuilt.catalog_sha256() == topic_store.catalog_sha256()


def test_token_map_mismatch_fails(tmp_path, catalog_path) -> None:
    catalog_copy = tmp_path / "topics.jsonl"
    catalog_copy.write_text(catalog_path.read_text(encoding="utf-8"), encoding="utf-8")
    mapping = {
        "schema_version": 1,
        "topic_id_to_token": {"pneumonia_basics": "<MED_TOPIC_9999>"},
    }
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(json.dumps(mapping), encoding="utf-8")
    with pytest.raises(TopicStoreError, match="does not match catalog"):
        TopicStore.from_jsonl(catalog_copy, token_map_path=mapping_path)


def test_inactive_topic_keeps_its_registry_token(topic_store) -> None:
    topics = topic_store.list_all()
    topics[0] = topics[0].model_copy(update={"status": "inactive"})
    rebuilt = TopicStore(topics)
    assert rebuilt.get_by_id(topics[0].topic_id) is None
    assert rebuilt.get_by_id(topics[0].topic_id, include_inactive=True) is not None
    assert rebuilt.topic_id_to_token()[topics[0].topic_id] == topics[0].topic_token


class FakeTokenizer:
    unk_token_id = 0

    def __init__(self, *, split_token: str | None = None):
        self.split_token = split_token
        self.ids: dict[str, int] = {}

    def convert_tokens_to_ids(self, token: str) -> int:
        if token not in self.ids:
            self.ids[token] = len(self.ids) + 1
        return self.ids[token]

    def encode(self, token: str, *, add_special_tokens: bool) -> list[int]:
        token_id = self.convert_tokens_to_ids(token)
        return [token_id, token_id] if token == self.split_token else [token_id]


def test_tokenizer_requires_every_token_to_be_single_id(topic_store) -> None:
    topic_store.validate_tokenizer(FakeTokenizer())
    with pytest.raises(ModelArtifactError, match="topic_token_not_single_id"):
        topic_store.validate_tokenizer(FakeTokenizer(split_token="<MED_TOPIC_0001>"))


def test_tokenizer_rejects_an_unregistered_control_token(topic_store) -> None:
    class MissingTokenizer(FakeTokenizer):
        unk_token_id = 0

        def convert_tokens_to_ids(self, token: str) -> int:
            if token == "<NEXT_TOPIC>":
                return self.unk_token_id
            return super().convert_tokens_to_ids(token)

        def encode(self, token: str, *, add_special_tokens: bool) -> list[int]:
            token_id = self.convert_tokens_to_ids(token)
            return [token_id]

    with pytest.raises(ModelArtifactError, match="unknown_topic_token"):
        topic_store.validate_tokenizer(MissingTokenizer())

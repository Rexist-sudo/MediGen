from __future__ import annotations

import json

import pytest

from src.services.recommendation.model_manifest import (
    sha256_directory,
    sha256_file,
    validate_artifact_layout,
)
from src.services.recommendation.ranker_protocol import EVENT_TOKENS, ModelArtifactError


def make_artifact(tmp_path, topic_store):
    artifact = tmp_path / "artifact"
    base = tmp_path / "base"
    adapter = artifact / "adapter"
    tokenizer = artifact / "tokenizer"
    adapter.mkdir(parents=True)
    tokenizer.mkdir(parents=True)
    base.mkdir()
    (base / "base_model_manifest.json").write_text(
        json.dumps(
            {
                "model_id": "Qwen/Qwen2.5-0.5B",
                "resolved_revision": "abcdef1234567",
            }
        ),
        encoding="utf-8",
    )
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"real-adapter-fixture")
    (tokenizer / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    token_map = topic_store.token_map_payload()
    (artifact / "topic_token_map.json").write_text(
        json.dumps(token_map, sort_keys=True), encoding="utf-8"
    )
    (artifact / "catalog_snapshot.sha256").write_text(
        topic_store.catalog_sha256(), encoding="utf-8"
    )
    (artifact / "data_manifest.json").write_text(
        json.dumps(
            {
                "dataset_sha256": "a" * 64,
                "seed": 42,
                "catalog_sha256": topic_store.catalog_sha256(),
                "topic_token_map_sha256": sha256_file(
                    artifact / "topic_token_map.json"
                ),
            }
        ),
        encoding="utf-8",
    )
    (artifact / "training_config.yaml").write_text("schema_version: 1\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "model_version": "v1",
        "profile": "direct_sid",
        "base_model_id": "Qwen/Qwen2.5-0.5B",
        "base_model_revision": "abcdef1234567",
        "adapter_sha256": sha256_directory(adapter),
        "tokenizer_sha256": sha256_directory(tokenizer),
        "topic_token_map_sha256": sha256_file(artifact / "topic_token_map.json"),
        "catalog_sha256": topic_store.catalog_sha256(),
        "dataset_sha256": "a" * 64,
        "dataset_seed": 42,
        "topic_count": len(topic_store.list_all()),
        "event_tokens": [
            *EVENT_TOKENS.values(),
            "<NO_HISTORY>",
            "<NO_SELECTED>",
        ],
        "max_input_tokens": 1024,
        "max_history": 20,
        "max_candidates": 20,
        "python_version": "3.11.15",
        "torch_version": "2.7.1+cu128",
        "transformers_version": "4.53.2",
        "peft_version": "0.16.0",
        "created_at": "2026-07-23T00:00:00Z",
        "training_config_sha256": sha256_file(artifact / "training_config.yaml"),
    }
    (artifact / "model_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return artifact, base


def test_valid_layout_passes_and_corrupt_adapter_fails(tmp_path, topic_store) -> None:
    artifact, base = make_artifact(tmp_path, topic_store)
    manifest = validate_artifact_layout(
        artifact_path=artifact,
        base_model_path=base,
        topic_store=topic_store,
        expected_model_version="v1",
        max_input_tokens=1024,
        max_history=20,
        max_candidates=20,
    )
    assert manifest.model_version == "v1"
    (artifact / "adapter" / "adapter_model.safetensors").write_bytes(b"corrupt")
    with pytest.raises(ModelArtifactError):
        validate_artifact_layout(
            artifact_path=artifact,
            base_model_path=base,
            topic_store=topic_store,
            expected_model_version="v1",
            max_input_tokens=1024,
            max_history=20,
            max_candidates=20,
        )


def _validate(artifact, base, topic_store):
    return validate_artifact_layout(
        artifact_path=artifact,
        base_model_path=base,
        topic_store=topic_store,
        expected_model_version="v1",
        max_input_tokens=1024,
        max_history=20,
        max_candidates=20,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 2),
        ("model_version", ""),
        ("topic_count", 14),
        ("event_tokens", ["<EV_VIEW>"]),
        ("max_input_tokens", 1025),
        ("max_history", 21),
        ("max_candidates", 21),
        ("catalog_sha256", "b" * 64),
        ("topic_token_map_sha256", "b" * 64),
    ],
)
def test_incompatible_manifest_fields_fail(
    tmp_path,
    topic_store,
    field,
    value,
) -> None:
    artifact, base = make_artifact(tmp_path, topic_store)
    manifest_path = artifact / "model_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ModelArtifactError):
        _validate(artifact, base, topic_store)


def test_dataset_and_base_snapshot_bindings_are_checked(tmp_path, topic_store) -> None:
    artifact, base = make_artifact(tmp_path, topic_store)
    data_path = artifact / "data_manifest.json"
    data = json.loads(data_path.read_text(encoding="utf-8"))
    data["seed"] = 7
    data_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ModelArtifactError):
        _validate(artifact, base, topic_store)

    artifact, base = make_artifact(tmp_path / "second", topic_store)
    base_path = base / "base_model_manifest.json"
    base_manifest = json.loads(base_path.read_text(encoding="utf-8"))
    base_manifest["resolved_revision"] = "different123456"
    base_path.write_text(json.dumps(base_manifest), encoding="utf-8")
    with pytest.raises(ModelArtifactError):
        _validate(artifact, base, topic_store)

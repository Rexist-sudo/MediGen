"""Model artifact schema, hashing, and compatibility validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from .ranker_protocol import (
    EVENT_TOKENS,
    ModelArtifactError,
    ModelArtifactMissingError,
)
from .topic_store import TopicStore


PLACEHOLDER_MARKERS = ("ACTUAL", "TODO", "RESOLVE_AND_PIN")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_directory(path: Path) -> str:
    inventory = [
        {
            "path": item.relative_to(path).as_posix(),
            "sha256": sha256_file(item),
            "size": item.stat().st_size,
        }
        for item in sorted(path.rglob("*"))
        if item.is_file()
    ]
    if not inventory:
        raise ModelArtifactMissingError()
    payload = json.dumps(
        inventory,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class ModelArtifactManifest(BaseModel):
    schema_version: Literal[1]
    model_version: str = Field(min_length=1)
    profile: Literal["direct_sid"]
    base_model_id: str = Field(min_length=1)
    base_model_revision: str = Field(min_length=7)
    adapter_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tokenizer_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    topic_token_map_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    catalog_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_seed: int
    topic_count: int = Field(ge=1)
    event_tokens: list[str]
    max_input_tokens: int = Field(ge=1)
    max_history: int = Field(ge=0)
    max_candidates: int = Field(ge=1)
    python_version: str
    torch_version: str
    transformers_version: str
    peft_version: str
    created_at: str
    training_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_device: str | None = None
    training_dtype: str | None = None

    @classmethod
    def load(cls, path: Path) -> "ModelArtifactManifest":
        if not path.is_file():
            raise ModelArtifactMissingError()
        try:
            raw = path.read_text(encoding="utf-8")
            if any(marker in raw for marker in PLACEHOLDER_MARKERS):
                raise ModelArtifactError()
            return cls.model_validate_json(raw)
        except ModelArtifactError:
            raise
        except (OSError, ValidationError, ValueError) as exc:
            raise ModelArtifactError() from exc


def validate_artifact_layout(
    *,
    artifact_path: Path,
    base_model_path: Path,
    topic_store: TopicStore,
    expected_model_version: str,
    max_input_tokens: int,
    max_history: int,
    max_candidates: int,
) -> ModelArtifactManifest:
    if not artifact_path.is_dir() or not base_model_path.is_dir():
        raise ModelArtifactMissingError()
    manifest = ModelArtifactManifest.load(artifact_path / "model_manifest.json")
    if manifest.model_version != expected_model_version:
        raise ModelArtifactError()

    adapter_path = artifact_path / "adapter"
    tokenizer_path = artifact_path / "tokenizer"
    token_map_path = artifact_path / "topic_token_map.json"
    catalog_snapshot_path = artifact_path / "catalog_snapshot.sha256"
    data_manifest_path = artifact_path / "data_manifest.json"
    training_config_path = artifact_path / "training_config.yaml"
    base_manifest_path = base_model_path / "base_model_manifest.json"
    if not (adapter_path / "adapter_config.json").is_file():
        raise ModelArtifactMissingError()
    if not any(adapter_path.glob("*.safetensors")) and not any(
        adapter_path.glob("*.bin")
    ):
        raise ModelArtifactMissingError()
    if not (tokenizer_path / "tokenizer_config.json").is_file():
        raise ModelArtifactMissingError()
    required_files = (
        token_map_path,
        catalog_snapshot_path,
        data_manifest_path,
        training_config_path,
        base_manifest_path,
    )
    if not all(path.is_file() for path in required_files):
        raise ModelArtifactMissingError()

    if sha256_directory(adapter_path) != manifest.adapter_sha256:
        raise ModelArtifactError()
    if sha256_directory(tokenizer_path) != manifest.tokenizer_sha256:
        raise ModelArtifactError()
    if sha256_file(token_map_path) != manifest.topic_token_map_sha256:
        raise ModelArtifactError()
    if manifest.catalog_sha256 != topic_store.catalog_sha256():
        raise ModelArtifactError()
    if catalog_snapshot_path.read_text(encoding="utf-8").strip() != manifest.catalog_sha256:
        raise ModelArtifactError()
    try:
        packaged_map = json.loads(token_map_path.read_text(encoding="utf-8"))
        dataset_manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
        base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelArtifactError() from exc
    if packaged_map.get("topic_id_to_token") != topic_store.topic_id_to_token():
        raise ModelArtifactError()
    if dataset_manifest.get("dataset_sha256") != manifest.dataset_sha256:
        raise ModelArtifactError()
    if dataset_manifest.get("seed") != manifest.dataset_seed:
        raise ModelArtifactError()
    if dataset_manifest.get("catalog_sha256") != manifest.catalog_sha256:
        raise ModelArtifactError()
    if (
        dataset_manifest.get("topic_token_map_sha256")
        != manifest.topic_token_map_sha256
    ):
        raise ModelArtifactError()
    if base_manifest.get("resolved_revision") != manifest.base_model_revision:
        raise ModelArtifactError()
    if base_manifest.get("model_id") != manifest.base_model_id:
        raise ModelArtifactError()
    if sha256_file(training_config_path) != manifest.training_config_sha256:
        raise ModelArtifactError()
    if manifest.topic_count != len(topic_store.list_all()):
        raise ModelArtifactError()
    expected_event_tokens = [
        *EVENT_TOKENS.values(),
        "<NO_HISTORY>",
        "<NO_SELECTED>",
    ]
    if manifest.event_tokens != expected_event_tokens:
        raise ModelArtifactError()
    if manifest.max_input_tokens > max_input_tokens:
        raise ModelArtifactError()
    if manifest.max_history > max_history:
        raise ModelArtifactError()
    if manifest.max_candidates > max_candidates:
        raise ModelArtifactError()
    return manifest

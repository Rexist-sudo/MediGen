from __future__ import annotations

import json
from pathlib import Path

import yaml

from recommendation_model.build_synthetic_dataset import build_dataset
from recommendation_model.validate_dataset import validate


def test_committed_dataset_passes_all_structural_gates() -> None:
    report = validate("recommendation_model/config.yaml")
    assert report["status"] == "pass"
    assert report["scenario_count"] == 600
    assert report["counterfactual_pair_count"] >= 40
    assert report["contains_real_phi"] is False


def test_seed_42_rebuild_has_stable_file_hashes(tmp_path) -> None:
    python_root = Path(__file__).resolve().parents[2]
    source = yaml.safe_load(
        (python_root / "recommendation_model" / "config.yaml").read_text(
            encoding="utf-8"
        )
    )
    source["data"]["train_path"] = str(tmp_path / "train.jsonl")
    source["data"]["validation_path"] = str(tmp_path / "validation.jsonl")
    source["data"]["test_path"] = str(tmp_path / "test.jsonl")
    source["data"]["manifest_path"] = str(tmp_path / "data_manifest.json")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(source, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    first = build_dataset(config_path, seed=42)
    second = build_dataset(config_path, seed=42)
    assert first["files"] == second["files"]
    assert first["dataset_sha256"] == second["dataset_sha256"]
    assert json.loads((tmp_path / "data_manifest.json").read_text())["seed"] == 42

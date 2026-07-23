"""Package a trained adapter, tokenizer, hashes, and immutable compatibility metadata."""

# ruff: noqa: E402 -- standalone execution adds the source root before imports.

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from .common import load_config, resolve_path, sha256_file
except ImportError:
    from common import load_config, resolve_path, sha256_file  # type: ignore[no-redef]

from src.services.recommendation.model_manifest import sha256_directory
from src.services.recommendation.ranker_protocol import EVENT_TOKENS
from src.services.recommendation.topic_store import TopicStore


def _safe_output(path: Path) -> None:
    allowed = (ROOT / "artifacts" / "minionerec-mvp").resolve()
    if not path.resolve().is_relative_to(allowed):
        raise ValueError("artifact output must stay under artifacts/minionerec-mvp")


def _sanitize_adapter_metadata(adapter_dir: Path, config: dict) -> None:
    """Remove machine-local paths and generated placeholders from PEFT metadata."""

    adapter_config_path = adapter_dir / "adapter_config.json"
    adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    adapter_config["base_model_name_or_path"] = config["model"]["base_model_id"]
    adapter_config["revision"] = config["model"]["base_model_revision"]
    adapter_config_path.write_text(
        json.dumps(adapter_config, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    (adapter_dir / "README.md").write_text(
        "\n".join(
            [
                "---",
                f"base_model: {config['model']['base_model_id']}",
                "library_name: peft",
                "tags:",
                "- lora",
                "- direct-sid-ranking",
                "---",
                "",
                "# MediGen MiniOneRec Direct-SID LoRA",
                "",
                "This adapter ranks the fixed MediGen education-topic SID catalog. ",
                "It is loaded with the pinned local base-model revision and the ",
                "tokenizer packaged beside the adapter.",
                "",
                f"- Model version: `{config['artifact']['model_version']}`",
                f"- Base model: `{config['model']['base_model_id']}`",
                f"- Base revision: `{config['model']['base_model_revision']}`",
                "- Output: constrained candidate Topic SID tokens",
                "- Intended use: local education-topic ranking inside MediGen",
                "",
                "The model does not generate clinical diagnoses, prescriptions, ",
                "dosages, or education-card prose.",
                "",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )


def package(config_path: str | Path, checkpoint: str | Path | None) -> dict:
    config = load_config(config_path)
    run_dir = resolve_path(config["training"]["output_dir"])
    adapter_source = resolve_path(checkpoint) if checkpoint else run_dir / "final_adapter"
    tokenizer_source = run_dir / "tokenizer"
    output = resolve_path(config["artifact"]["output_dir"])
    _safe_output(output)
    if not (adapter_source / "adapter_config.json").is_file():
        raise FileNotFoundError("trained adapter is missing")
    if not (tokenizer_source / "tokenizer_config.json").is_file():
        raise FileNotFoundError("trained tokenizer is missing")
    output.mkdir(parents=True, exist_ok=True)
    for destination in (output / "adapter", output / "tokenizer"):
        if destination.exists():
            shutil.rmtree(destination)
    shutil.copytree(adapter_source, output / "adapter")
    _sanitize_adapter_metadata(output / "adapter", config)
    shutil.copytree(tokenizer_source, output / "tokenizer")

    token_map_source = resolve_path(config["tokens"]["token_map_path"])
    data_manifest_source = resolve_path(config["data"]["manifest_path"])
    training_config_source = resolve_path(config_path)
    shutil.copy2(token_map_source, output / "topic_token_map.json")
    shutil.copy2(data_manifest_source, output / "data_manifest.json")
    shutil.copy2(training_config_source, output / "training_config.yaml")
    store = TopicStore.from_jsonl(config["tokens"]["catalog_path"])
    (output / "catalog_snapshot.sha256").write_text(
        store.catalog_sha256() + "\n",
        encoding="utf-8",
        newline="\n",
    )
    base_manifest = json.loads(
        (
            resolve_path(config["model"]["local_base_model_path"])
            / "base_model_manifest.json"
        ).read_text(encoding="utf-8")
    )
    data_manifest = json.loads(
        data_manifest_source.read_text(encoding="utf-8")
    )
    training_summary_path = run_dir / "training_summary.json"
    training_summary = json.loads(
        training_summary_path.read_text(encoding="utf-8")
    )
    metrics_source = run_dir / "metrics.json"
    if not metrics_source.is_file():
        raise FileNotFoundError("metrics.json is required before packaging")
    metrics = json.loads(metrics_source.read_text(encoding="utf-8"))
    if metrics.get("AdapterEffect") != "PASS":
        raise ValueError("adapter effect gate did not pass")
    for metric_name, expected in {
        "ValidTopicRate": 1.0,
        "DuplicateTopicRate": 0.0,
        "CandidateViolationRate": 0.0,
        "ExcludedCategoryViolationRate": 0.0,
        "NegativeFeedbackViolationRate": 0.0,
        "MandatorySafetyOrderingViolationRate": 0.0,
        "UnknownTokenRate": 0.0,
    }.items():
        if metrics.get(metric_name) != expected:
            raise ValueError(f"structural evaluation gate failed: {metric_name}")
    if int(metrics.get("HistoryPairCount", 0)) < 20:
        raise ValueError("history pair coverage is below 20")
    if int(metrics.get("HistoryPairFlipCount", 0)) < 1:
        raise ValueError("real adapter did not change any fixed history pair")
    if int(metrics.get("CandidateOrderPairCount", 0)) < 1:
        raise ValueError("candidate-order pair coverage is absent")
    if float(metrics.get("FirstCandidateCopyRate", 1.0)) >= 1.0:
        raise ValueError("candidate-order test indicates first-item copying")
    shutil.copy2(metrics_source, output / "metrics.json")

    manifest = {
        "schema_version": 1,
        "model_version": config["artifact"]["model_version"],
        "profile": "direct_sid",
        "base_model_id": config["model"]["base_model_id"],
        "base_model_revision": base_manifest["resolved_revision"],
        "adapter_sha256": sha256_directory(output / "adapter"),
        "tokenizer_sha256": sha256_directory(output / "tokenizer"),
        "topic_token_map_sha256": sha256_file(output / "topic_token_map.json"),
        "catalog_sha256": store.catalog_sha256(),
        "dataset_sha256": data_manifest["dataset_sha256"],
        "dataset_seed": data_manifest["seed"],
        "topic_count": len(store.list_all()),
        "event_tokens": [
            *EVENT_TOKENS.values(),
            "<NO_HISTORY>",
            "<NO_SELECTED>",
        ],
        "max_input_tokens": int(config["model"]["max_input_tokens"]),
        "max_history": 20,
        "max_candidates": 20,
        "python_version": platform.python_version(),
        "torch_version": importlib.metadata.version("torch"),
        "transformers_version": importlib.metadata.version("transformers"),
        "peft_version": importlib.metadata.version("peft"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_config_sha256": sha256_file(output / "training_config.yaml"),
        "training_device": training_summary["device"],
        "training_dtype": training_summary["dtype"],
    }
    (output / "model_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    shutil.copy2(training_summary_path, output / "training_summary.json")
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="recommendation_model/config.yaml")
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    package(args.config, args.checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

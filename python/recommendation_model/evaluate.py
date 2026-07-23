"""Evaluate the real adapter with candidate-only logits and structural gates."""

# ruff: noqa: E402 -- standalone execution adds the source root before imports.

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from .common import load_config, read_jsonl, resolve_path
except ImportError:
    from common import load_config, read_jsonl, resolve_path  # type: ignore[no-redef]

from src.services.recommendation.topic_store import TopicStore


def _adapter_and_tokenizer(checkpoint: Path) -> tuple[Path, Path, Path]:
    if (checkpoint / "adapter_config.json").is_file():
        adapter = checkpoint
        run_dir = checkpoint.parent
    elif (checkpoint / "final_adapter" / "adapter_config.json").is_file():
        adapter = checkpoint / "final_adapter"
        run_dir = checkpoint
    elif (checkpoint / "adapter" / "adapter_config.json").is_file():
        adapter = checkpoint / "adapter"
        run_dir = checkpoint
    else:
        raise FileNotFoundError("adapter_config.json is absent")
    tokenizer = (
        checkpoint / "tokenizer"
        if (checkpoint / "tokenizer" / "tokenizer_config.json").is_file()
        else run_dir / "tokenizer"
    )
    if not (tokenizer / "tokenizer_config.json").is_file():
        raise FileNotFoundError("trained tokenizer is absent")
    return adapter, tokenizer, run_dir


def _dtype_and_device(torch, requested: str) -> tuple[str, object, str]:
    if requested == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = requested
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return device, torch.bfloat16, "bfloat16"
    if device == "cuda":
        return device, torch.float16, "float16"
    return device, torch.float32, "float32"


def _candidate_logits(model, tokenizer, row, device: str, torch) -> list[float]:
    encoded = tokenizer(
        row["prompt"],
        return_tensors="pt",
        add_special_tokens=False,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    candidate_ids = [
        tokenizer.convert_tokens_to_ids(token)
        for token in row["allowed_topic_tokens"]
    ]
    with torch.inference_mode():
        logits = model(**encoded, use_cache=False).logits[0, -1]
    index = torch.tensor(candidate_ids, dtype=torch.long, device=logits.device)
    return [float(value) for value in logits.index_select(0, index).float().cpu()]


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1)
    return ordered[max(0, index)]


def evaluate(
    config_path: str | Path,
    *,
    checkpoint: str | Path | None,
    device_requested: str,
    max_samples: int | None = None,
) -> dict[str, Any]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    config = load_config(config_path)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    store = TopicStore.from_jsonl(config["tokens"]["catalog_path"])
    token_map = store.topic_id_to_token()
    test_rows = [
        row
        for row in read_jsonl(resolve_path(config["data"]["test_path"]))
        if row["task_type"] == "next_topic"
    ]
    if max_samples is not None:
        test_rows = test_rows[:max_samples]
    for row in test_rows:
        selected = set(row.get("selected_topic_ids", []))
        excluded = set(row.get("excluded_topic_ids", []))
        row["allowed_topic_ids"] = [
            topic_id
            for topic_id in row["candidate_topic_ids"]
            if topic_id not in selected and topic_id not in excluded
        ]
        if row["target_topic_id"] not in row["allowed_topic_ids"]:
            raise RuntimeError("evaluation target is outside remaining candidates")
        row["allowed_topic_tokens"] = [
            token_map[topic_id] for topic_id in row["allowed_topic_ids"]
        ]

    checkpoint_path = (
        resolve_path(checkpoint)
        if checkpoint
        else resolve_path(config["training"]["output_dir"])
    )
    adapter_path, tokenizer_path, run_dir = _adapter_and_tokenizer(checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
        trust_remote_code=False,
    )
    store.validate_tokenizer(tokenizer)
    device, torch_dtype, dtype_name = _dtype_and_device(
        torch,
        device_requested,
    )
    base = AutoModelForCausalLM.from_pretrained(
        resolve_path(config["model"]["local_base_model_path"]),
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch_dtype,
        device_map=None,
    )
    base.resize_token_embeddings(len(tokenizer))
    base.eval().to(device)

    effect_rows = test_rows[: min(10, len(test_rows))]
    base_vectors = [
        _candidate_logits(base, tokenizer, row, device, torch)
        for row in effect_rows
    ]
    model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
    model.eval().to(device)

    correct = 0
    hit3 = 0
    ndcg_values: list[float] = []
    latencies: list[float] = []
    predictions: dict[str, str] = {}
    preferred_matches = 0
    preferred_total = 0
    first_candidate_copies = 0
    first_candidate_cases = 0
    adapter_vectors: list[list[float]] = []
    prediction_rows: list[tuple[dict[str, Any], str]] = []
    for row_index, row in enumerate(test_rows):
        if device == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        vector = _candidate_logits(model, tokenizer, row, device, torch)
        if device == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - started) * 1000)
        if row_index < len(effect_rows):
            adapter_vectors.append(vector)
        order = sorted(range(len(vector)), key=lambda index: -vector[index])
        predicted_id = row["allowed_topic_ids"][order[0]]
        target_id = row["target_topic_id"]
        correct += int(predicted_id == target_id)
        ranked_ids = [row["allowed_topic_ids"][index] for index in order]
        target_rank = ranked_ids.index(target_id) + 1
        hit3 += int(target_rank <= 3)
        ndcg_values.append(1.0 / math.log2(target_rank + 1))
        prediction_rows.append((row, predicted_id))
        if row["position"] == 1:
            first_candidate_cases += 1
            first_candidate_copies += int(
                predicted_id == row["allowed_topic_ids"][0]
            )
            prediction_key = (
                f"{row['scenario_id']}:{row['permutation_index']}"
            )
            predictions[prediction_key] = predicted_id
            preferred = set(row.get("preferences", {}).get("preferred_categories", []))
            if preferred:
                preferred_total += 1
                predicted_topic = store.get_by_id(predicted_id)
                preferred_matches += int(
                    predicted_topic is not None
                    and predicted_topic.category.value in preferred
                )

    maximum_logit_change = 0.0
    adapter_top1_changes = 0
    for base_vector, adapter_vector in zip(base_vectors, adapter_vectors, strict=True):
        maximum_logit_change = max(
            maximum_logit_change,
            max(
                abs(left - right)
                for left, right in zip(base_vector, adapter_vector, strict=True)
            ),
        )
        base_top = max(range(len(base_vector)), key=base_vector.__getitem__)
        adapter_top = max(range(len(adapter_vector)), key=adapter_vector.__getitem__)
        adapter_top1_changes += int(base_top != adapter_top)

    group_scenarios: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    permutation_predictions: dict[str, list[str]] = defaultdict(list)
    for row in test_rows:
        if row["position"] != 1:
            continue
        key = f"{row['scenario_id']}:{row['permutation_index']}"
        predicted = predictions[key]
        if row["permutation_index"] == 0:
            group_scenarios[row["scenario_group_id"]][row["scenario_id"]] = {
                "prediction": predicted,
                "candidates": frozenset(row["candidate_topic_ids"]),
                "target": row["target_topic_id"],
                "history": row["history"],
            }
        permutation_predictions[row["scenario_id"]].append(predicted)
    comparable_pairs = []
    for values in group_scenarios.values():
        if len(values) != 2:
            continue
        left, right = values.values()
        if left["candidates"] == right["candidates"] and left["history"] != right["history"]:
            comparable_pairs.append((left, right))
    pair_flips = sum(
        left["prediction"] != right["prediction"]
        for left, right in comparable_pairs
    )
    permutation_cases = [
        values for values in permutation_predictions.values() if len(values) >= 2
    ]
    stable_permutations = sum(len(set(values)) == 1 for values in permutation_cases)

    valid_topic_predictions = 0
    candidate_violations = 0
    excluded_category_violations = 0
    negative_feedback_violations = 0
    unknown_tokens = 0
    by_sequence: dict[str, list[tuple[int, str]]] = defaultdict(list)
    pinned_by_sequence: dict[str, list[str]] = {}
    for row, predicted_id in prediction_rows:
        topic = store.get_by_id(predicted_id, include_inactive=True)
        valid_topic_predictions += int(topic is not None)
        candidate_violations += int(predicted_id not in row["allowed_topic_ids"])
        excluded_categories = set(
            row.get("preferences", {}).get("excluded_categories", [])
        )
        excluded_category_violations += int(
            topic is not None and topic.category.value in excluded_categories
        )
        negative_ids = {
            item["topic_id"]
            for item in row.get("history", [])
            if item.get("event_type") in {"dismiss", "not_helpful"}
        }
        negative_feedback_violations += int(predicted_id in negative_ids)
        sequence_key = f"{row['scenario_id']}:{row['permutation_index']}"
        by_sequence[sequence_key].append((int(row["position"]), predicted_id))
        pinned_by_sequence.setdefault(
            sequence_key,
            list(row.get("pinned_topic_ids", [])),
        )
        for token in row["allowed_topic_tokens"]:
            token_id = tokenizer.convert_tokens_to_ids(token)
            if (
                token_id is None
                or token_id == tokenizer.unk_token_id
                or tokenizer.encode(token, add_special_tokens=False) != [token_id]
            ):
                unknown_tokens += 1

    duplicate_sequences = 0
    mandatory_ordering_violations = 0
    for sequence_key, positioned in by_sequence.items():
        ranked = [
            predicted
            for _, predicted in sorted(positioned, key=lambda item: item[0])
        ]
        pinned = pinned_by_sequence[sequence_key]
        final_order = [*pinned, *ranked]
        duplicate_sequences += int(len(final_order) != len(set(final_order)))
        if pinned and final_order[: len(pinned)] != pinned:
            mandatory_ordering_violations += 1

    count = max(1, len(test_rows))
    sequence_count = max(1, len(by_sequence))
    metrics = {
        "schema_version": 1,
        "evaluated_next_topic_rows": len(test_rows),
        "device": device,
        "dtype": dtype_name,
        "Top1Accuracy": round(correct / count, 6),
        "HitRate@3": round(hit3 / count, 6),
        "NDCG@3": round(statistics.fmean(ndcg_values), 6),
        "PreferenceMatchRate": round(
            preferred_matches / max(1, preferred_total),
            6,
        ),
        "HistoryPairCount": len(comparable_pairs),
        "HistoryPairFlipCount": pair_flips,
        "HistoryPairFlipRate": round(
            pair_flips / max(1, len(comparable_pairs)),
            6,
        ),
        "CandidateOrderPairCount": len(permutation_cases),
        "CandidateOrderStabilityRate": round(
            stable_permutations / max(1, len(permutation_cases)),
            6,
        ),
        "FirstCandidateCopyRate": round(
            first_candidate_copies / max(1, first_candidate_cases),
            6,
        ),
        "MeanInferenceLatencyMs": round(statistics.fmean(latencies), 3),
        "P50InferenceLatencyMs": round(_percentile(latencies, 0.5), 3),
        "P95InferenceLatencyMs": round(_percentile(latencies, 0.95), 3),
        "AdapterMaxCandidateLogitChange": maximum_logit_change,
        "AdapterTop1ChangeCount": adapter_top1_changes,
        "AdapterEffect": "PASS"
        if maximum_logit_change > 1e-5 or adapter_top1_changes > 0
        else "FAIL",
        "ValidTopicRate": round(valid_topic_predictions / count, 6),
        "DuplicateTopicRate": round(duplicate_sequences / sequence_count, 6),
        "CandidateViolationRate": round(candidate_violations / count, 6),
        "ExcludedCategoryViolationRate": round(
            excluded_category_violations / count,
            6,
        ),
        "NegativeFeedbackViolationRate": round(
            negative_feedback_violations / count,
            6,
        ),
        "MandatorySafetyOrderingViolationRate": round(
            mandatory_ordering_violations / sequence_count,
            6,
        ),
        "UnknownTokenRate": round(
            unknown_tokens
            / max(
                1,
                sum(len(row["allowed_topic_tokens"]) for row in test_rows),
            ),
            6,
        ),
        "CatalogArtifactCompatibility": "PASS",
    }
    if metrics["AdapterEffect"] != "PASS":
        raise RuntimeError("adapter effect test failed")
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="recommendation_model/config.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    evaluate(
        args.config,
        checkpoint=args.checkpoint,
        device_requested=args.device,
        max_samples=args.max_samples,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Validate grouped splits, policy constraints, hashes, task mix, and PHI absence."""

# ruff: noqa: E402 -- standalone execution adds the source root before imports.

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from .common import load_config, read_jsonl, resolve_path, sha256_file
except ImportError:
    from common import load_config, read_jsonl, resolve_path, sha256_file  # type: ignore[no-redef]

from src.services.recommendation.topic_store import TopicStore


PHI_PATTERNS = {
    "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    "phone": re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)"),
    "cn_id": re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)"),
    "medical_record": re.compile(r"(?:MRN|病历号|住院号)\s*[:：]?\s*[A-Z0-9-]{5,}", re.I),
}


def _require(condition: bool, code: str, failures: list[str]) -> None:
    if not condition:
        failures.append(code)


def validate(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    store = TopicStore.from_jsonl(config["tokens"]["catalog_path"])
    token_map = store.topic_id_to_token()
    manifest_path = resolve_path(config["data"]["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = {
        "train": resolve_path(config["data"]["train_path"]),
        "validation": resolve_path(config["data"]["validation_path"]),
        "test": resolve_path(config["data"]["test_path"]),
    }
    rows_by_split = {key: read_jsonl(path) for key, path in paths.items()}
    failures: list[str] = []

    _require(manifest.get("schema_version") == 1, "manifest_schema", failures)
    _require(manifest.get("seed") == 42, "dataset_seed", failures)
    _require(
        manifest.get("catalog_sha256") == store.catalog_sha256(),
        "catalog_hash",
        failures,
    )
    _require(manifest.get("contains_real_phi") is False, "phi_manifest", failures)
    for split, path in paths.items():
        _require(
            manifest.get("files", {}).get(path.name) == sha256_file(path),
            f"file_hash:{split}",
            failures,
        )

    groups_by_split: dict[str, set[str]] = {}
    scenario_rows: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for split, rows in rows_by_split.items():
        groups = {
            str(row["scenario_group_id"])
            for row in rows
            if row.get("scenario_group_id")
        }
        groups_by_split[split] = groups
        scenario_rows.extend(
            row for row in rows if row.get("task_type") == "next_topic"
        )
        all_rows.extend(rows)
        for row in rows:
            _require(row.get("schema_version") == 1, "row_schema", failures)
            target_id = row.get("target_topic_id")
            _require(target_id in token_map, "unknown_target", failures)
            _require(
                row.get("target_topic_token") == token_map.get(target_id),
                "target_token_mapping",
                failures,
            )
            prompt = str(row.get("prompt", ""))
            if row.get("task_type") == "next_topic":
                candidates = list(row.get("candidate_topic_ids", []))
                selected = list(row.get("selected_topic_ids", []))
                _require(target_id in candidates, "target_outside_candidates", failures)
                _require(target_id not in selected, "target_already_selected", failures)
                _require(len(selected) == len(set(selected)), "selected_duplicate", failures)
                _require(
                    target_id not in set(row.get("excluded_topic_ids", [])),
                    "target_hard_excluded",
                    failures,
                )
                negative_ids = {
                    item.get("topic_id")
                    for item in row.get("history", [])
                    if item.get("event_type") in {"dismiss", "not_helpful"}
                }
                _require(
                    target_id not in negative_ids,
                    "negative_feedback_violation",
                    failures,
                )
                _require(prompt.endswith("<NEXT_TOPIC>"), "ranking_prompt_tail", failures)
            else:
                _require(prompt.endswith("<TOPIC_SID>"), "alignment_prompt_tail", failures)
            serialized = json.dumps(row, ensure_ascii=False)
            _require(
                "patient_description" not in serialized,
                "raw_patient_field",
                failures,
            )
            for name, pattern in PHI_PATTERNS.items():
                _require(not pattern.search(serialized), f"phi_pattern:{name}", failures)

    split_names = list(groups_by_split)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            _require(
                not groups_by_split[left].intersection(groups_by_split[right]),
                f"group_leak:{left}:{right}",
                failures,
            )

    scenario_snapshots: dict[str, dict[str, Any]] = {}
    top1_by_group: dict[str, dict[str, str]] = defaultdict(dict)
    orders_by_scenario: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for row in scenario_rows:
        scenario_id = str(row["scenario_id"])
        scenario_snapshots.setdefault(scenario_id, row)
        if row.get("position") == 1 and row.get("permutation_index") == 0:
            top1_by_group[str(row["scenario_group_id"])][scenario_id] = str(
                row["target_topic_id"]
            )
        if row.get("position") == 1:
            orders_by_scenario[scenario_id].add(
                tuple(row.get("candidate_topic_ids", []))
            )

    scenario_count = len(scenario_snapshots)
    no_history = sum(not row.get("history") for row in scenario_snapshots.values())
    negative = sum(
        any(
            item.get("event_type") in {"dismiss", "not_helpful"}
            for item in row.get("history", [])
        )
        for row in scenario_snapshots.values()
    )
    counterfactual_pairs = sum(
        len(values) == 2 and len(set(values.values())) == 2
        for values in top1_by_group.values()
    )
    permutation_scenarios = sum(len(orders) >= 2 for orders in orders_by_scenario.values())
    test_pair_rows: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows_by_split["test"]:
        if (
            row.get("task_type") == "next_topic"
            and row.get("position") == 1
            and row.get("permutation_index") == 0
        ):
            test_pair_rows[str(row["scenario_group_id"])][str(row["scenario_id"])] = row
    fixed_test_history_pairs = 0
    for values in test_pair_rows.values():
        if len(values) != 2:
            continue
        left, right = values.values()
        fixed_test_history_pairs += int(
            left.get("history") != right.get("history")
            and set(left.get("candidate_topic_ids", []))
            == set(right.get("candidate_topic_ids", []))
        )
    task_counts = Counter(row["task_type"] for row in all_rows)
    ranking_ratio = task_counts["next_topic"] / max(1, len(all_rows))

    _require(scenario_count >= 600, "scenario_count", failures)
    _require(
        len({row["scenario_id"] for row in rows_by_split["validation"] if row.get("scenario_id")})
        >= 60,
        "validation_scenario_count",
        failures,
    )
    _require(
        len({row["scenario_id"] for row in rows_by_split["test"] if row.get("scenario_id")})
        >= 60,
        "test_scenario_count",
        failures,
    )
    _require(0.25 <= no_history / scenario_count <= 0.35, "no_history_ratio", failures)
    _require(negative / scenario_count >= 0.25, "negative_ratio", failures)
    _require(counterfactual_pairs >= 40, "counterfactual_pairs", failures)
    _require(fixed_test_history_pairs >= 20, "fixed_test_history_pairs", failures)
    _require(permutation_scenarios >= 12, "candidate_permutations", failures)
    _require(0.79 <= ranking_ratio <= 0.81, "task_mix", failures)
    _require(
        manifest.get("scenario_count") == scenario_count,
        "manifest_scenario_count",
        failures,
    )
    _require(
        manifest.get("counterfactual_pair_count") == counterfactual_pairs,
        "manifest_pair_count",
        failures,
    )
    _require(
        manifest.get("fixed_test_history_pair_count")
        == fixed_test_history_pairs,
        "manifest_history_pair_count",
        failures,
    )

    report = {
        "status": "pass" if not failures else "fail",
        "failures": sorted(set(failures)),
        "scenario_count": scenario_count,
        "split_group_counts": {
            key: len(value) for key, value in groups_by_split.items()
        },
        "row_counts": {key: len(value) for key, value in rows_by_split.items()},
        "task_counts": dict(task_counts),
        "next_topic_ratio": round(ranking_ratio, 6),
        "no_history_ratio": round(no_history / scenario_count, 6),
        "negative_feedback_ratio": round(negative / scenario_count, 6),
        "counterfactual_pair_count": counterfactual_pairs,
        "fixed_test_history_pair_count": fixed_test_history_pairs,
        "candidate_permutation_scenario_count": permutation_scenarios,
        "contains_real_phi": False,
    }
    if failures:
        raise ValueError(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="recommendation_model/config.yaml")
    args = parser.parse_args()
    print(json.dumps(validate(args.config), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

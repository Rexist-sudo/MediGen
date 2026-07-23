"""Build deterministic grouped synthetic SFT data without external model calls."""

# ruff: noqa: E402 -- standalone execution adds the source root before imports.

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from .common import (
        canonical_sha256,
        load_config,
        repository_sha,
        resolve_path,
        sha256_file,
        write_jsonl,
    )
except ImportError:
    from common import (  # type: ignore[no-redef]
        canonical_sha256,
        load_config,
        repository_sha,
        resolve_path,
        sha256_file,
        write_jsonl,
    )

from src.models.recommendation import (
    RecommendationContext,
    UserHistoryContext,
    UserPreferenceContext,
)
from src.services.recommendation.candidate_policy import CandidatePolicy
from src.services.recommendation.history_normalizer import HistoryNormalizer
from src.services.recommendation.minionerec_prompt import MiniOneRecPromptBuilder
from src.services.recommendation.ranker_protocol import RankerInput
from src.services.recommendation.rule_fallback_ranker import RuleFallbackRanker
from src.services.recommendation.topic_store import TopicStore


TEMPLATE_ROOT = Path(__file__).resolve().parent / "templates"


def _yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError(f"invalid template: {path.name}")
    return value


def _stable_seed(*parts: object) -> int:
    value = "|".join(str(item) for item in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def _timestamp(day: int) -> str:
    return f"2026-01-{day:02d}T00:00:00+00:00"


def _history_for(
    family: dict[str, Any],
    *,
    pair_index: int,
    variant: str,
) -> list[dict[str, str]]:
    primary = str(family["primary_topic_id"])
    external = str(family["positive_history_topic_id"])
    if variant == "a":
        if pair_index % 5 < 3:
            return []
        return [
            {
                "topic_id": primary,
                "event_type": "view",
                "occurred_at": _timestamp(1),
            }
        ]
    if pair_index % 2 == 0:
        return [
            {
                "topic_id": primary,
                "event_type": "dismiss",
                "occurred_at": _timestamp(2),
            }
        ]
    return [
        {
            "topic_id": primary,
            "event_type": "view",
            "occurred_at": _timestamp(1),
        },
        {
            "topic_id": external,
            "event_type": "helpful",
            "occurred_at": _timestamp(2),
        },
    ]


def _preference_for(
    patterns: list[dict[str, Any]],
    *,
    family_index: int,
    pair_index: int,
) -> dict[str, Any]:
    value = patterns[(family_index * 7 + pair_index) % len(patterns)]
    return json.loads(json.dumps(value, ensure_ascii=False))


def _scenario_source(pair_index: int, variant: str) -> str:
    if variant == "b":
        return "counterfactual_template"
    return ("manual_template", "preference_template", "rule_teacher")[
        pair_index % 3
    ]


def build_base_scenarios(
    *,
    store: TopicStore,
    seed: int,
) -> list[dict[str, Any]]:
    clinical = _yaml(TEMPLATE_ROOT / "clinical_scenarios.yaml")
    preferences = _yaml(TEMPLATE_ROOT / "preference_patterns.yaml")["patterns"]
    _yaml(TEMPLATE_ROOT / "history_patterns.yaml")
    policy = CandidatePolicy(store)
    normalizer = HistoryNormalizer(store, 20)
    fallback = RuleFallbackRanker(store)
    scenarios: list[dict[str, Any]] = []

    for family_index, family in enumerate(clinical["families"]):
        context = RecommendationContext(
            diagnosis_codes=family["diagnosis_codes"],
            diagnosis_terms=family["diagnosis_terms"],
            recommended_tests=family["recommended_tests"],
            medication_names=family["medication_names"],
            demo_safe=True,
        )
        for pair_index in range(50):
            group_id = f"{family['id']}_pair_{pair_index:03d}"
            preference_payload = _preference_for(
                preferences,
                family_index=family_index,
                pair_index=pair_index,
            )
            preference = UserPreferenceContext.model_validate(preference_payload)
            for variant in ("a", "b"):
                history_payload = _history_for(
                    family,
                    pair_index=pair_index,
                    variant=variant,
                )
                history_context = UserHistoryContext.model_validate(
                    {"interactions": history_payload}
                )
                history = normalizer.normalize(history_context)
                policy_result = policy.apply(
                    context=context,
                    preferences=preference,
                    history=history,
                    recalled_topics=store.list_all(),
                    all_active_topics=store.list_active(),
                    top_k=3,
                    max_candidates=20,
                )
                candidates = policy_result.rankable_topics
                if not candidates:
                    raise RuntimeError(f"scenario has no candidates: {group_id}")
                candidate_ids = {topic.topic_id for topic in candidates}
                primary = str(family["primary_topic_id"])
                secondary = str(family["secondary_topic_id"])
                desired = (
                    [primary, secondary]
                    if variant == "a"
                    else [secondary, primary]
                )
                desired = [item for item in desired if item in candidate_ids]
                teacher = fallback.rank(
                    RankerInput(
                        context=context,
                        preferences=preference,
                        history=history,
                        candidates=candidates,
                        already_selected_topic_ids=tuple(
                            item.topic_id for item in policy_result.pinned_topics
                        ),
                        top_k=min(3, len(candidates)),
                    )
                )
                target_ids = list(dict.fromkeys([*desired, *teacher.topic_ids]))[
                    : min(3, len(candidates))
                ]
                if not target_ids:
                    raise RuntimeError(f"scenario has no targets: {group_id}")
                scenarios.append(
                    {
                        "schema_version": 1,
                        "scenario_id": f"{group_id}_{variant}",
                        "scenario_group_id": group_id,
                        "family": family["id"],
                        "task_type": "next_topic",
                        "clinical_context": context.model_dump(mode="json"),
                        "preferences": preference.model_dump(mode="json"),
                        "history": history_payload,
                        "candidate_topic_ids": [
                            item.topic_id for item in candidates
                        ],
                        "pinned_topic_ids": [
                            item.topic_id for item in policy_result.pinned_topics
                        ],
                        "excluded_topic_ids": list(
                            policy_result.excluded_topic_ids
                        ),
                        "target_topic_ids": target_ids,
                        "target_source": _scenario_source(pair_index, variant),
                        "permutation_variants": 2 if pair_index < 2 else 1,
                        "seed": _stable_seed(seed, group_id, variant),
                    }
                )
    return scenarios


def split_scenarios(
    scenarios: list[dict[str, Any]],
    *,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for scenario in scenarios:
        by_group[scenario["scenario_group_id"]].append(scenario)
    group_ids = sorted(by_group)
    random.Random(seed).shuffle(group_ids)
    validation_count = int(len(group_ids) * 0.1)
    test_count = len(group_ids) - int(len(group_ids) * 0.9)

    def is_fixed_history_pair(group_id: str) -> bool:
        values = by_group[group_id]
        return (
            len(values) == 2
            and values[0]["history"] != values[1]["history"]
            and set(values[0]["candidate_topic_ids"])
            == set(values[1]["candidate_topic_ids"])
        )

    compatible = [group for group in group_ids if is_fixed_history_pair(group)]
    required_history_pairs = 20
    if len(compatible) < required_history_pairs or test_count < required_history_pairs:
        raise RuntimeError("insufficient fixed history pairs for the test split")
    test_groups = compatible[:required_history_pairs]
    test_group_set = set(test_groups)
    for group in group_ids:
        if len(test_groups) >= test_count:
            break
        if group not in test_group_set:
            test_groups.append(group)
            test_group_set.add(group)
    remaining = [group for group in group_ids if group not in test_group_set]
    validation_groups = remaining[:validation_count]
    train_groups = remaining[validation_count:]
    split_groups = {
        "train": train_groups,
        "validation": validation_groups,
        "test": test_groups,
    }
    return {
        split: [scenario for group in groups for scenario in by_group[group]]
        for split, groups in split_groups.items()
    }


def _ordered_candidates(
    store: TopicStore,
    scenario: dict[str, Any],
    *,
    permutation_index: int,
) -> list:
    topics = [
        store.get_by_id(topic_id)
        for topic_id in scenario["candidate_topic_ids"]
    ]
    if any(topic is None for topic in topics):
        raise RuntimeError("candidate is absent from catalog")
    topics = [topic for topic in topics if topic is not None]
    random.Random(
        _stable_seed(scenario["seed"], 11 if permutation_index == 0 else 29)
    ).shuffle(topics)
    if permutation_index and len(topics) > 1:
        topics = topics[1:] + topics[:1]
    return topics


def expand_scenarios(
    scenarios: list[dict[str, Any]],
    *,
    store: TopicStore,
) -> list[dict[str, Any]]:
    prompt_builder = MiniOneRecPromptBuilder()
    token_map = store.topic_id_to_token()
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        context = RecommendationContext.model_validate(
            scenario["clinical_context"]
        )
        preferences = UserPreferenceContext.model_validate(
            scenario["preferences"]
        )
        history_context = UserHistoryContext.model_validate(
            {"interactions": scenario["history"]}
        )
        history = HistoryNormalizer(store, 20).normalize(history_context)
        for permutation_index in range(scenario["permutation_variants"]):
            candidates = _ordered_candidates(
                store,
                scenario,
                permutation_index=permutation_index,
            )
            selected_ids: list[str] = []
            selected_tokens: list[str] = []
            for position, target_id in enumerate(
                scenario["target_topic_ids"],
                start=1,
            ):
                ranker_input = RankerInput(
                    context=context,
                    preferences=preferences,
                    history=history,
                    candidates=tuple(candidates),
                    already_selected_topic_ids=tuple(
                        scenario["pinned_topic_ids"]
                    ),
                    top_k=len(scenario["target_topic_ids"]),
                )
                prompt = prompt_builder.build(
                    ranker_input=ranker_input,
                    selected_topic_tokens=selected_tokens,
                    preserve_candidate_order=True,
                    history_topic_tokens=token_map,
                )
                rows.append(
                    {
                        "schema_version": 1,
                        "sample_id": (
                            f"{scenario['scenario_id']}:p{position}:"
                            f"perm{permutation_index}"
                        ),
                        "scenario_id": scenario["scenario_id"],
                        "scenario_group_id": scenario["scenario_group_id"],
                        "family": scenario["family"],
                        "task_type": "next_topic",
                        "position": position,
                        "prompt": prompt,
                        "target_topic_id": target_id,
                        "target_topic_token": token_map[target_id],
                        "candidate_topic_ids": [
                            item.topic_id for item in candidates
                        ],
                        "selected_topic_ids": list(selected_ids),
                        "pinned_topic_ids": scenario["pinned_topic_ids"],
                        "excluded_topic_ids": scenario["excluded_topic_ids"],
                        "preferences": scenario["preferences"],
                        "history": scenario["history"],
                        "target_source": scenario["target_source"],
                        "permutation_index": permutation_index,
                    }
                )
                selected_ids.append(target_id)
                selected_tokens.append(token_map[target_id])
    return rows


def add_alignment_rows(
    rows: list[dict[str, Any]],
    *,
    store: TopicStore,
    split: str,
) -> list[dict[str, Any]]:
    desired = (len(rows) + 3) // 4
    topics = store.list_all()
    alignment: list[dict[str, Any]] = []
    for index in range(desired):
        topic = topics[index % len(topics)]
        alignment.append(
            {
                "schema_version": 1,
                "sample_id": f"alignment:{split}:{index:04d}:{topic.topic_id}",
                "scenario_id": None,
                "scenario_group_id": None,
                "family": "catalog_alignment",
                "task_type": "topic_to_sid",
                "position": 1,
                "prompt": MiniOneRecPromptBuilder.build_topic_alignment_prompt(
                    topic
                ),
                "target_topic_id": topic.topic_id,
                "target_topic_token": topic.topic_token,
                "candidate_topic_ids": [],
                "selected_topic_ids": [],
                "pinned_topic_ids": [],
                "excluded_topic_ids": [],
                "preferences": {},
                "history": [],
                "target_source": "catalog_alignment",
                "permutation_index": 0,
            }
        )
    return [*rows, *alignment]


def _created_at(manifest_path: Path) -> str:
    if manifest_path.exists():
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            value = previous.get("created_at")
            if isinstance(value, str):
                return value
        except (OSError, json.JSONDecodeError):
            pass
    return datetime.now(timezone.utc).isoformat()


def build_dataset(config_path: str | Path, *, seed: int) -> dict[str, Any]:
    config = load_config(config_path)
    store = TopicStore.from_jsonl(config["tokens"]["catalog_path"])
    scenarios = build_base_scenarios(store=store, seed=seed)
    splits = split_scenarios(scenarios, seed=seed)
    output_paths = {
        "train": resolve_path(config["data"]["train_path"]),
        "validation": resolve_path(config["data"]["validation_path"]),
        "test": resolve_path(config["data"]["test_path"]),
    }
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    for split, split_scenarios_value in splits.items():
        rows = expand_scenarios(split_scenarios_value, store=store)
        rows = add_alignment_rows(rows, store=store, split=split)
        random.Random(_stable_seed(seed, split)).shuffle(rows)
        rows_by_split[split] = rows
        write_jsonl(output_paths[split], rows)

    file_hashes = {
        path.name: sha256_file(path) for path in output_paths.values()
    }
    scenario_sources = Counter(item["target_source"] for item in scenarios)
    row_sources = Counter(
        row["target_source"]
        for rows in rows_by_split.values()
        for row in rows
    )
    task_counts = Counter(
        row["task_type"] for rows in rows_by_split.values() for row in rows
    )
    no_history = sum(not item["history"] for item in scenarios)
    negative = sum(
        any(
            event["event_type"] in {"dismiss", "not_helpful"}
            for event in item["history"]
        )
        for item in scenarios
    )
    groups = defaultdict(list)
    for scenario in scenarios:
        groups[scenario["scenario_group_id"]].append(scenario)
    counterfactual_pairs = sum(
        len(items) == 2
        and items[0]["target_topic_ids"][0]
        != items[1]["target_topic_ids"][0]
        for items in groups.values()
    )
    test_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for scenario in splits["test"]:
        test_groups[scenario["scenario_group_id"]].append(scenario)
    fixed_test_history_pairs = sum(
        len(items) == 2
        and items[0]["history"] != items[1]["history"]
        and set(items[0]["candidate_topic_ids"])
        == set(items[1]["candidate_topic_ids"])
        for items in test_groups.values()
    )
    manifest_path = resolve_path(config["data"]["manifest_path"])
    manifest = {
        "schema_version": 1,
        "seed": seed,
        "catalog_sha256": store.catalog_sha256(),
        "topic_token_map_sha256": sha256_file(
            resolve_path(config["tokens"]["token_map_path"])
        ),
        "generator_git_sha": repository_sha(),
        "scenario_count": len(scenarios),
        "split_scenario_counts": {
            key: len(value) for key, value in splits.items()
        },
        "train_rows": len(rows_by_split["train"]),
        "validation_rows": len(rows_by_split["validation"]),
        "test_rows": len(rows_by_split["test"]),
        "task_counts": dict(sorted(task_counts.items())),
        "scenario_source_counts": dict(sorted(scenario_sources.items())),
        "source_counts": dict(sorted(row_sources.items())),
        "counterfactual_pair_count": counterfactual_pairs,
        "fixed_test_history_pair_count": fixed_test_history_pairs,
        "no_history_scenario_count": no_history,
        "no_history_scenario_ratio": round(no_history / len(scenarios), 6),
        "negative_feedback_scenario_count": negative,
        "negative_feedback_scenario_ratio": round(
            negative / len(scenarios),
            6,
        ),
        "candidate_permutation_scenario_count": sum(
            item["permutation_variants"] > 1 for item in scenarios
        ),
        "contains_real_phi": False,
        "created_at": _created_at(manifest_path),
        "files": file_hashes,
        "dataset_sha256": canonical_sha256(file_hashes),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="recommendation_model/config.yaml",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = build_dataset(args.config, seed=args.seed)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

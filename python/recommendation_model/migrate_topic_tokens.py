"""Apply or check the fixed 15-topic Direct-SID migration table."""

# ruff: noqa: E402 -- standalone execution adds the source root before imports.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from .common import sha256_file
except ImportError:
    from common import sha256_file  # type: ignore[no-redef]

from src.services.recommendation.topic_store import TopicStore


FIXED_MAPPING = {
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--catalog",
        default="data/recommendation/knowledge_topics.jsonl",
        type=Path,
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    catalog = args.catalog if args.catalog.is_absolute() else ROOT / args.catalog
    store = TopicStore.from_jsonl(catalog)
    if store.topic_id_to_token() != dict(sorted(FIXED_MAPPING.items())):
        raise ValueError("catalog does not match the fixed Direct-SID mapping")
    result = {
        "status": "pass",
        "topic_count": len(store.list_all()),
        "catalog_sha256": store.catalog_sha256(),
        "token_map_file_sha256": sha256_file(
            catalog.with_name("topic_token_map.json")
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

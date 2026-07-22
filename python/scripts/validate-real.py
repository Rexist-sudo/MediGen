"""Run the browser quick-fill case catalog through the real local stack."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx


ROOT = Path(__file__).resolve().parents[1]
CASE_PATH = ROOT / "src" / "web" / "validation-cases.json"
DEFAULT_OUTPUT = ROOT / ".runtime" / "real-validation-last.json"
FORBIDDEN_VISIBLE_TEXT = (
    "MVP",
    "暂无",
    "未使用",
    "待接入",
    "不会",
)


def _as_dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: object) -> list[Any]:
    return value if isinstance(value, list) else []


def _contains_value(value: object) -> bool:
    if isinstance(value, list):
        return bool(value)
    if isinstance(value, dict):
        return any(item is not None and item != "" for item in value.values())
    return value is not None and value != ""


def _validate_response(case: dict[str, Any], response: dict[str, Any]) -> list[str]:
    failures: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    expected = _as_dict(case.get("expected"))
    patient = _as_dict(response.get("patient_info"))
    diagnosis = _as_dict(response.get("diagnosis"))
    treatment = _as_dict(response.get("treatment_plan"))
    coding = _as_dict(response.get("coding_result"))
    audit = _as_dict(response.get("audit_result"))
    education = _as_dict(response.get("education_recommendations"))
    trace = _as_dict(response.get("integration_trace"))
    timeline = _as_dict(response.get("processing_timeline"))

    require(response.get("analysis_status") == "completed", "analysis_status is not completed")
    for section in ("patient_info", "diagnosis", "treatment_plan", "coding_result", "audit_result"):
        require(isinstance(response.get(section), dict), f"{section} is missing")

    for field in ("chief_complaint", "symptoms", "vital_signs", "lab_results"):
        require(_contains_value(patient.get(field)), f"patient_info.{field} is empty")
    for field in _as_list(expected.get("required_extracted_fields")):
        require(_contains_value(patient.get(str(field))), f"patient_info.{field} is empty")

    primary_code = _as_dict(coding.get("primary_icd10")).get("code")
    require(primary_code == expected.get("primary_icd10"), f"primary ICD-10 is {primary_code!r}")
    require(_as_dict(coding.get("local_validation")).get("primary_code_matched") is True, "local ICD-10 validation did not match")
    minimum_confidence = float(expected.get("minimum_confidence", 0.0))
    diagnosis_confidence = float(_as_dict(diagnosis.get("primary_diagnosis")).get("confidence", 0.0))
    coding_confidence = float(coding.get("coding_confidence", 0.0))
    require(
        diagnosis_confidence >= minimum_confidence,
        f"diagnosis confidence {diagnosis_confidence:.2f} is below {minimum_confidence:.2f}",
    )
    require(
        coding_confidence >= minimum_confidence,
        f"coding confidence {coding_confidence:.2f} is below {minimum_confidence:.2f}",
    )
    if expected.get("expected_drg"):
        require(
            _as_dict(coding.get("drg_group")).get("drg_code") == expected.get("expected_drg"),
            "expected DRG enrichment is missing",
        )

    graph = _as_dict(diagnosis.get("knowledge_graph"))
    evidence = _as_list(graph.get("evidence"))
    graph_diseases = {str(_as_dict(item).get("disease")) for item in evidence}
    require(graph.get("provider") == "Neo4j", "diagnosis graph provider is not Neo4j")
    require(expected.get("graph_disease") in graph_diseases, "expected graph disease evidence is missing")
    require(bool(evidence), "Neo4j evidence is empty")

    require(bool(_as_list(treatment.get("non_drug_treatments"))), "non-drug treatment output is empty")
    require(bool(_as_list(treatment.get("lifestyle_recommendations"))), "lifestyle output is empty")
    require(bool(treatment.get("follow_up_plan")), "follow-up plan is empty")

    expected_pair = expected.get("interaction_pair")
    if isinstance(expected_pair, list) and len(expected_pair) == 2:
        interactions = _as_list(treatment.get("drug_interactions"))
        normalized_pairs = {
            frozenset(
                {
                    str(_as_dict(item).get("drug_a", "")).casefold(),
                    str(_as_dict(item).get("drug_b", "")).casefold(),
                }
            )
            for item in interactions
        }
        require(
            frozenset(str(item).casefold() for item in expected_pair) in normalized_pairs,
            "expected medication interaction is missing",
        )

    if expected.get("allergy_drug"):
        warnings = " ".join(str(item) for item in _as_list(treatment.get("warnings")))
        require("过敏史" in warnings, "current medication allergy warning is missing")
        require(str(expected["allergy_drug"]).casefold() in warnings.casefold(), "allergy drug is absent from warning")

    require(audit.get("demo_safe") is True, "identifier audit did not pass")
    checks = _as_list(audit.get("compliance_checks"))
    require(len(checks) >= 2 and all(_as_dict(item).get("passed") for item in checks), "audit checks are incomplete")

    recommendations = _as_list(education.get("recommendations"))
    require(education.get("candidate_source") == "neo4j", "education candidates did not come from Neo4j")
    require(education.get("strategy_used") == "rule_v1_deepseek", "DeepSeek education generation was not used")
    require(len(recommendations) == 3, "education card count is not 3")
    for item in recommendations:
        card = _as_dict(item)
        require(card.get("content_source") == "deepseek_generated", f"card {card.get('topic_id')} is using fallback text")
        require(card.get("content_depth") == "standard", f"card {card.get('topic_id')} depth is not standard")
        require(len(str(card.get("summary", ""))) >= 100, f"card {card.get('topic_id')} content is too short")

    providers = {
        "privacy_scan": "Presidio + local rules",
        "knowledge_graph": "Neo4j",
        "cache_and_rate_limit": "Redis",
        "persistence": "PostgreSQL",
        "interoperability": "HAPI FHIR",
    }
    for key, provider in providers.items():
        require(_as_dict(trace.get(key)).get("provider") == provider, f"{provider} trace is missing")
    persistence = _as_dict(trace.get("persistence"))
    require(persistence.get("session_id") == response.get("session_id"), "PostgreSQL session identifier mismatch")
    require(int(persistence.get("clinical_sessions", 0)) >= 1, "PostgreSQL session count is empty")
    interoperability = _as_dict(trace.get("interoperability"))
    resource_types = set(_as_list(interoperability.get("resource_types")))
    require({"Patient", "Condition"}.issubset(resource_types), "FHIR core resources are incomplete")
    require(int(interoperability.get("resource_count", 0)) >= 2, "FHIR resource count is too small")

    total_seconds = float(timeline.get("total_seconds", 0.0))
    require(total_seconds > 0, "total processing time is missing")
    stage_timings = _as_dict(timeline.get("stages"))
    for stage in ("intake", "diagnosis", "treatment", "coding", "audit", "recommendations"):
        elapsed = float(_as_dict(stage_timings.get(stage)).get("elapsed_seconds", 0.0))
        require(elapsed > 0, f"{stage} processing time is missing")
    supporting_timings = _as_dict(timeline.get("supporting_steps"))
    for step in ("interoperability", "persistence"):
        elapsed = float(_as_dict(supporting_timings.get(step)).get("elapsed_seconds", 0.0))
        require(elapsed > 0, f"{step} processing time is missing")

    visible_text = json.dumps(response, ensure_ascii=False)
    for phrase in FORBIDDEN_VISIBLE_TEXT:
        require(phrase not in visible_text, f"visible response contains forbidden wording: {phrase}")
    require(re.search(r"不是.{0,40}而是", visible_text) is None, "visible response contains contrastive explanatory wording")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--case", default="all")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    cases = json.loads(CASE_PATH.read_text(encoding="utf-8"))
    by_id = {item["id"]: item for item in cases}
    if args.case != "all":
        if args.case not in by_id:
            parser.error(f"unknown case: {args.case}")
        cases = [by_id[args.case]]

    base_url = args.base_url.rstrip("/")
    records: list[dict[str, Any]] = []
    any_failures = False
    with httpx.Client(timeout=420) as client:
        ready = client.get(f"{base_url}/ready")
        ready.raise_for_status()
        ready_payload = ready.json()
        if ready_payload.get("status") != "ready":
            raise RuntimeError(f"service is not ready: {ready_payload}")

        for case in cases:
            started = datetime.now(timezone.utc)
            request_payload = {
                "patient_description": case["description"],
                "include_recommendations": True,
                "recommendation_top_k": 3,
                "user_preferences": {
                    "preferred_categories": [
                        "disease_basics",
                        "test_explanation",
                        "medication_safety",
                        "warning_signs",
                    ],
                    "preferred_depth": "standard",
                },
            }
            response = client.post(
                f"{base_url}/api/v1/clinical/analyze",
                json=request_payload,
            )
            body = response.json()
            failures = []
            if response.status_code == 200:
                failures = _validate_response(case, body)
            else:
                failures = [f"HTTP {response.status_code}: {body}"]
            any_failures = any_failures or bool(failures)
            elapsed = (datetime.now(timezone.utc) - started).total_seconds()
            records.append(
                {
                    "case_id": case["id"],
                    "readme_case_id": case["readme_case_id"],
                    "recorded_at_utc": started.isoformat(),
                    "elapsed_seconds": round(elapsed, 2),
                    "http_status": response.status_code,
                    "passed": not failures,
                    "failures": failures,
                    "request": request_payload,
                    "response": body,
                }
            )
            status = "PASS" if not failures else "FAIL"
            print(f"[{status}] {case['id']} ({elapsed:.1f}s)", flush=True)
            for failure in failures:
                print(f"  - {failure}", flush=True)

    output = args.output
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Validation transcript: {output}")
    return 1 if any_failures else 0


if __name__ == "__main__":
    sys.exit(main())

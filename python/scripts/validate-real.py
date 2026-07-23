"""Run the browser quick-fill case catalog through the real local stack."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx


ROOT = Path(__file__).resolve().parents[1]
CASE_PATH = ROOT / "src" / "web" / "validation-cases.json"
TOPIC_PATH = ROOT / "data" / "recommendation" / "knowledge_topics.jsonl"
DEFAULT_OUTPUT = ROOT / ".runtime" / "real-validation-last.json"
DIABETES_DESCRIPTION = (
    "一名 56 岁成人合成病例，近 3 个月口渴、夜间多尿并有多次空腹血糖升高记录；"
    "既往有高血压，无手术史，不吸烟，无已知药物过敏。当前服用 lisinopril。"
    "生命体征稳定，查体无急性异常。检查结果：空腹血糖 9.8 mmol/L，HbA1c 8.2%，"
    "肾功能正常。需要评估 2 型糖尿病及长期血糖趋势。"
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


def _validate_response(
    case: dict[str, Any],
    response: dict[str, Any],
    *,
    expected_ranking: str,
    expected_content: str,
    expected_fallback_reason: str | None,
    expected_model_version: str | None,
    catalog_topic_ids: set[str],
) -> list[str]:
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
    require(
        education.get("ranking_strategy_used") == expected_ranking,
        f"ranking strategy is {education.get('ranking_strategy_used')!r}",
    )
    content_strategy = education.get("content_strategy_used")
    if expected_content == "auto":
        require(
            content_strategy in {"deepseek_generated", "catalog_fallback"},
            f"content strategy is {content_strategy!r}",
        )
    else:
        require(
            content_strategy == expected_content,
            f"content strategy is {content_strategy!r}",
        )
    require(
        education.get("fallback_reason") == expected_fallback_reason,
        f"fallback reason is {education.get('fallback_reason')!r}",
    )
    if expected_ranking == "mini_onerec_mvp":
        require(education.get("model_ready") is True, "Mini-OneRec is not ready")
        require(
            education.get("model_version") == expected_model_version,
            "response model version differs from readiness",
        )
        require(
            float(education.get("ranker_inference_ms") or 0) > 0,
            "ranker inference time is missing",
        )
    require(int(education.get("candidate_count", 0)) >= 1, "candidate count is empty")
    require(len(recommendations) == 3, "education card count is not 3")
    for item in recommendations:
        card = _as_dict(item)
        require(card.get("topic_id") in catalog_topic_ids, "unknown recommendation topic")
        require(card.get("content_source") == content_strategy, f"card {card.get('topic_id')} content source differs")
        require(card.get("content_depth") == "standard", f"card {card.get('topic_id')} depth is not standard")
        minimum_summary = 100 if content_strategy == "deepseek_generated" else 20
        require(len(str(card.get("summary", ""))) >= minimum_summary, f"card {card.get('topic_id')} content is too short")

    mandatory_topic = {
        "stemi_interaction": "myocardial_infarction_warning_signs",
        "heart_failure": "heart_failure_warning_signs",
    }.get(case.get("id"))
    if mandatory_topic:
        require(
            bool(recommendations)
            and _as_dict(recommendations[0]).get("topic_id") == mandatory_topic,
            f"mandatory safety topic {mandatory_topic} is not first",
        )

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
    ranker_trace = _as_dict(trace.get("recommendation_ranker"))
    require(
        ranker_trace.get("used_strategy") == expected_ranking,
        "ranker trace strategy differs from response",
    )
    require("prompt" not in json.dumps(ranker_trace).casefold(), "ranker trace exposes prompt data")

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

    return failures


def _validate_model_response(
    response: dict[str, Any],
    *,
    model_version: str,
    catalog_topic_ids: set[str],
) -> list[str]:
    failures: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    education = _as_dict(response.get("education_recommendations"))
    recommendations = _as_list(education.get("recommendations"))
    require(response.get("analysis_status") == "completed", "analysis did not complete")
    require(
        education.get("ranking_strategy_used") == "mini_onerec_mvp",
        "request did not use Mini-OneRec",
    )
    require(education.get("model_ready") is True, "model_ready is false")
    require(education.get("model_version") == model_version, "model version mismatch")
    require(education.get("fallback_reason") is None, "request used fallback")
    content_strategy = education.get("content_strategy_used")
    require(
        content_strategy in {"deepseek_generated", "catalog_fallback"},
        "unsupported content strategy",
    )
    require(int(education.get("candidate_count", 0)) >= 1, "candidate count is empty")
    require(1 <= len(recommendations) <= 3, "recommendation count is outside 1-3")
    require(
        all(
            _as_dict(item).get("topic_id") in catalog_topic_ids
            for item in recommendations
        ),
        "request returned an unknown topic",
    )
    require(
        all(
            _as_dict(item).get("content_source") == content_strategy
            for item in recommendations
        ),
        "card content source differs from the response strategy",
    )
    trace = _as_dict(
        _as_dict(response.get("integration_trace")).get("recommendation_ranker")
    )
    require("prompt" not in json.dumps(trace).casefold(), "trace exposes prompt data")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--case", default="all")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--expected-ranking",
        choices=["mini_onerec_mvp", "rule_v1_fallback"],
        default="mini_onerec_mvp",
    )
    parser.add_argument(
        "--expected-content",
        choices=["auto", "deepseek_generated", "catalog_fallback"],
        default="deepseek_generated",
    )
    parser.add_argument("--expected-fallback-reason", default="")
    parser.add_argument("--model-scenarios", action="store_true")
    parser.add_argument("--require-fallback-disabled", action="store_true")
    parser.add_argument("--allow-model-not-ready", action="store_true")
    args = parser.parse_args()

    cases = json.loads(CASE_PATH.read_text(encoding="utf-8"))
    catalog_topic_ids = {
        json.loads(line)["topic_id"]
        for line in TOPIC_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    by_id = {item["id"]: item for item in cases}
    if args.case == "none":
        if not args.model_scenarios:
            parser.error("--case none requires --model-scenarios")
        cases = []
    elif args.case != "all":
        if args.case not in by_id:
            parser.error(f"unknown case: {args.case}")
        cases = [by_id[args.case]]

    base_url = args.base_url.rstrip("/")
    records: list[dict[str, Any]] = []
    any_failures = False
    expected_fallback_reason = args.expected_fallback_reason or None
    with httpx.Client(timeout=420) as client:
        ready = client.get(f"{base_url}/ready")
        ready.raise_for_status()
        ready_payload = ready.json()
        if (
            ready_payload.get("status") != "ready"
            and not args.allow_model_not_ready
        ):
            raise RuntimeError(f"service is not ready: {ready_payload}")
        dependencies = _as_dict(ready_payload.get("dependencies"))
        if not dependencies or not all(dependencies.values()):
            raise RuntimeError(f"runtime dependency is unavailable: {dependencies}")
        model_status = _as_dict(ready_payload.get("recommendation_model"))
        model_version = model_status.get("model_version")
        if args.expected_ranking == "mini_onerec_mvp":
            if not model_status.get("artifact_valid") or not model_status.get("loaded"):
                raise RuntimeError(f"Mini-OneRec is not loaded: {model_status}")
            if not model_version:
                raise RuntimeError("readiness has no model version")
        if args.require_fallback_disabled and model_status.get("fallback_available"):
            raise RuntimeError("fallback is enabled in the primary-proof profile")

        def submit(
            *,
            scenario_id: str,
            request_payload: dict[str, Any],
            validator,
            readme_case_id: str | None = None,
        ) -> dict[str, Any]:
            nonlocal any_failures
            started = datetime.now(timezone.utc)
            response = client.post(
                f"{base_url}/api/v1/clinical/analyze",
                json=request_payload,
            )
            try:
                body = response.json()
            except ValueError:
                body = {"raw_response": response.text[:1000]}
            failures = (
                validator(body)
                if response.status_code == 200
                else [f"HTTP {response.status_code}: {body}"]
            )
            elapsed = (datetime.now(timezone.utc) - started).total_seconds()
            record = {
                "case_id": scenario_id,
                "readme_case_id": readme_case_id,
                "recorded_at_utc": started.isoformat(),
                "elapsed_seconds": round(elapsed, 2),
                "http_status": response.status_code,
                "passed": not failures,
                "failures": failures,
                "request": request_payload,
                "response": body,
            }
            records.append(record)
            any_failures = any_failures or bool(failures)
            status = "PASS" if not failures else "FAIL"
            print(f"[{status}] {scenario_id} ({elapsed:.1f}s)", flush=True)
            for failure in failures:
                print(f"  - {failure}", flush=True)
            return record

        for case in cases:
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
            submit(
                scenario_id=case["id"],
                readme_case_id=case["readme_case_id"],
                request_payload=request_payload,
                validator=lambda body, case=case: _validate_response(
                    case,
                    body,
                    expected_ranking=args.expected_ranking,
                    expected_content=args.expected_content,
                    expected_fallback_reason=expected_fallback_reason,
                    expected_model_version=model_version,
                    catalog_topic_ids=catalog_topic_ids,
                ),
            )

        if args.model_scenarios:
            if args.expected_ranking != "mini_onerec_mvp":
                parser.error("--model-scenarios requires the Mini-OneRec profile")
            base_payload = {
                "patient_description": DIABETES_DESCRIPTION,
                "include_recommendations": True,
                "recommendation_top_k": 3,
                "user_preferences": {
                    "preferred_categories": [
                        "disease_basics",
                        "test_explanation",
                        "follow_up_education",
                    ],
                    "preferred_depth": "standard",
                    "preferred_format": "bullet_points",
                    "max_reading_minutes": 5,
                },
            }

            def model_validator(body):
                return _validate_model_response(
                    body,
                    model_version=str(model_version),
                    catalog_topic_ids=catalog_topic_ids,
                )

            cold = submit(
                scenario_id="model_cold_start",
                request_payload=base_payload,
                validator=model_validator,
            )
            history_payload = json.loads(json.dumps(base_payload))
            history_payload["user_history_context"] = {
                "interactions": [
                    {
                        "topic_id": "diabetes_basics",
                        "event_type": "view",
                        "occurred_at": "2026-01-01T00:00:00Z",
                    },
                    {
                        "topic_id": "chest_xray_explanation",
                        "event_type": "helpful",
                        "occurred_at": "2026-01-02T00:00:00Z",
                    },
                ]
            }
            history = submit(
                scenario_id="model_history_pair",
                request_payload=history_payload,
                validator=model_validator,
            )
            negative_payload = json.loads(json.dumps(base_payload))
            negative_payload["user_history_context"] = {
                "interactions": [
                    {
                        "topic_id": "hba1c_test_explanation",
                        "event_type": "dismiss",
                        "occurred_at": "2026-01-03T00:00:00Z",
                    }
                ]
            }
            negative = submit(
                scenario_id="model_negative_feedback",
                request_payload=negative_payload,
                validator=model_validator,
            )

            cold_education = _as_dict(
                _as_dict(cold["response"]).get("education_recommendations")
            )
            history_education = _as_dict(
                _as_dict(history["response"]).get("education_recommendations")
            )
            negative_education = _as_dict(
                _as_dict(negative["response"]).get("education_recommendations")
            )
            pair_failures: list[str] = []
            if cold_education.get("history_used") is not False:
                pair_failures.append("cold-start request reported history usage")
            if int(cold_education.get("valid_history_count", -1)) != 0:
                pair_failures.append("cold-start history count is not zero")
            if history_education.get("history_used") is not True:
                pair_failures.append("history request did not use history")
            if int(history_education.get("valid_history_count", 0)) != 2:
                pair_failures.append("history request did not retain two events")
            cold_ids = [
                _as_dict(item).get("topic_id")
                for item in _as_list(cold_education.get("recommendations"))
            ]
            history_ids = [
                _as_dict(item).get("topic_id")
                for item in _as_list(history_education.get("recommendations"))
            ]
            if not cold_ids or not history_ids or cold_ids[0] == history_ids[0]:
                pair_failures.append("fixed history pair did not change top-1")
            negative_ids = {
                _as_dict(item).get("topic_id")
                for item in _as_list(negative_education.get("recommendations"))
            }
            if "hba1c_test_explanation" in negative_ids:
                pair_failures.append("dismissed topic remained in recommendations")
            if pair_failures:
                any_failures = True
                history["failures"].extend(pair_failures)
                history["passed"] = False
                print("[FAIL] model scenario cross-checks", flush=True)
                for failure in pair_failures:
                    print(f"  - {failure}", flush=True)
            else:
                print("[PASS] model scenario cross-checks", flush=True)

    output = args.output
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    transcript = {
        "schema_version": 2,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "readiness": ready_payload,
        "expected_profile": {
            "ranking": args.expected_ranking,
            "content": args.expected_content,
            "fallback_reason": expected_fallback_reason,
        },
        "records": records,
        "passed": not any_failures,
    }
    output.write_text(
        json.dumps(transcript, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Validation transcript: {output}")
    return 1 if any_failures else 0


if __name__ == "__main__":
    sys.exit(main())

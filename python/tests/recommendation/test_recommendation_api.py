from __future__ import annotations

from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api import routes
from src.models.recommendation import EducationRecommendationResult


def _clinical_result() -> dict:
    return {
        "patient_info": {"chief_complaint": "fatigue"},
        "diagnosis": {
            "primary_diagnosis": {"disease_name": "Type 2 diabetes"},
            "recommended_tests": ["HbA1c"],
            "knowledge_graph": {"cache_status": "miss", "evidence_count": 1},
        },
        "treatment_plan": {"medications": []},
        "coding_result": {"primary_icd10": {"code": "E11.9"}},
        "audit_result": {"demo_safe": True},
        "information_gaps": [],
        "errors": [],
        "warnings": [],
        "stage_timings_seconds": {
            "intake": 0.1,
            "diagnosis": 0.1,
            "treatment": 0.1,
            "coding": 0.1,
            "audit": 0.1,
        },
    }


class _Pipeline:
    def invoke(self, _payload):
        return _clinical_result()


class _RecommendationService:
    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.calls = []

    def recommend_after_analysis(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("injected ranker failure")
        return EducationRecommendationResult(
            recommendation_status="ok",
            strategy_used="mini_onerec_mvp",
            ranking_strategy_used="mini_onerec_mvp",
            content_strategy_used="catalog_fallback",
            model_version="model-test-v1",
            model_ready=True,
            ranker_inference_ms=7.25,
            candidate_source="local_catalog",
            candidate_cache_status="offline",
            content_cache_status="fallback",
            history_used=True,
            valid_history_count=1,
            candidate_count=3,
        )


def _settings(*, backend: str = "fixture"):
    return SimpleNamespace(
        prototype_reject_obvious_phi=True,
        llm_backend=backend,
        deepseek_configured=backend == "deepseek",
        infrastructure_required=False,
        recommendation_ranker="minionerec",
    )


def _client(monkeypatch, service, *, backend: str = "fixture") -> TestClient:
    monkeypatch.setattr(routes, "get_settings", lambda: _settings(backend=backend))
    monkeypatch.setattr(routes, "find_obvious_identifiers", lambda _text: [])
    monkeypatch.setattr(routes, "get_pipeline", lambda: _Pipeline())
    monkeypatch.setattr(routes, "get_recommendation_service", lambda: service)
    app = FastAPI()
    app.include_router(routes.router, prefix="/api/v1")
    return TestClient(app)


def _request_payload(**updates) -> dict:
    payload = {
        "patient_description": "Synthetic adult case with fatigue and elevated glucose.",
        "include_recommendations": True,
        "recommendation_top_k": 3,
        "user_preferences": {"preferred_categories": ["test_explanation"]},
        "user_history_context": {
            "interactions": [
                {
                    "topic_id": "diabetes_basics",
                    "event_type": "helpful",
                    "occurred_at": "2026-01-01T00:00:00Z",
                }
            ]
        },
    }
    payload.update(updates)
    return payload


def test_api_appends_model_fields_and_keeps_trace_free_of_prompt(monkeypatch) -> None:
    service = _RecommendationService()
    response = _client(monkeypatch, service).post(
        "/api/v1/clinical/analyze",
        json=_request_payload(),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["analysis_status"] == "completed"
    education = body["education_recommendations"]
    assert education["strategy_used"] == "mini_onerec_mvp"
    assert education["ranking_strategy_used"] == "mini_onerec_mvp"
    assert education["content_strategy_used"] == "catalog_fallback"
    assert education["model_version"] == "model-test-v1"
    assert education["model_ready"] is True
    trace = body["integration_trace"]["recommendation_ranker"]
    assert trace["used_strategy"] == "mini_onerec_mvp"
    assert "prompt" not in str(trace).casefold()
    assert service.calls[0]["top_k"] == 3


def test_include_recommendations_false_does_not_resolve_model_service(monkeypatch) -> None:
    def unexpected_service():
        raise AssertionError("recommendation service was resolved")

    monkeypatch.setattr(routes, "get_settings", lambda: _settings())
    monkeypatch.setattr(routes, "find_obvious_identifiers", lambda _text: [])
    monkeypatch.setattr(routes, "get_pipeline", lambda: _Pipeline())
    monkeypatch.setattr(routes, "get_recommendation_service", unexpected_service)
    app = FastAPI()
    app.include_router(routes.router, prefix="/api/v1")
    response = TestClient(app).post(
        "/api/v1/clinical/analyze",
        json=_request_payload(include_recommendations=False),
    )
    assert response.status_code == 200
    education = response.json()["education_recommendations"]
    assert education["recommendation_status"] == "disabled"
    assert education["ranking_strategy_used"] == "none"


def test_unexpected_recommendation_failure_keeps_clinical_response(monkeypatch) -> None:
    response = _client(
        monkeypatch,
        _RecommendationService(fail=True),
    ).post("/api/v1/clinical/analyze", json=_request_payload())
    assert response.status_code == 200
    body = response.json()
    assert body["analysis_status"] == "completed"
    assert body["diagnosis"]["primary_diagnosis"]["disease_name"]
    education = body["education_recommendations"]
    assert education["recommendation_status"] == "degraded"
    assert education["warnings"] == ["recommendation_unavailable"]


def test_deepseek_path_persists_extended_recommendation_json(monkeypatch) -> None:
    service = _RecommendationService()

    class FHIR:
        def export_analysis(self, **_kwargs):
            return {
                "provider": "HAPI FHIR",
                "resource_count": 2,
                "resource_types": ["Patient", "Condition"],
            }

    class Database:
        saved = None

        def save_analysis(self, **kwargs):
            self.saved = kwargs
            return str(kwargs["session_id"])

        def counts(self):
            return {
                "provider": "PostgreSQL",
                "clinical_sessions": 1,
                "audit_records": 1,
            }

    class Graph:
        def stats(self):
            return {"provider": "Neo4j", "nodes": 1, "relationships": 1}

    class Redis:
        def info_summary(self):
            return {"provider": "Redis"}

    database = Database()
    monkeypatch.setattr(routes, "get_fhir_service", lambda: FHIR())
    monkeypatch.setattr(routes, "get_database_service", lambda: database)
    monkeypatch.setattr(routes, "get_graphrag_service", lambda: Graph())
    monkeypatch.setattr(routes, "get_redis_service", lambda: Redis())
    response = _client(monkeypatch, service, backend="deepseek").post(
        "/api/v1/clinical/analyze",
        json=_request_payload(),
    )
    assert response.status_code == 200
    saved = database.saved["recommendation_result"]
    assert saved["model_version"] == "model-test-v1"
    assert saved["ranking_strategy_used"] == "mini_onerec_mvp"
    assert saved["ranker_inference_ms"] == 7.25
    assert response.json()["fhir_export"]["provider"] == "HAPI FHIR"


def test_api_rejects_top_k_outside_one_to_three(monkeypatch) -> None:
    client = _client(monkeypatch, _RecommendationService())
    for invalid in (0, 4):
        response = client.post(
            "/api/v1/clinical/analyze",
            json=_request_payload(recommendation_top_k=invalid),
        )
        assert response.status_code == 422

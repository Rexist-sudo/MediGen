"""Contract and integration tests for the finite MediGen MVP."""

from __future__ import annotations

import importlib
import json

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel, SecretStr

from src.config.settings import Settings, get_settings
from src.graph.clinical_pipeline import build_clinical_pipeline, get_pipeline
from src.services.deepseek_client import (
    DeepSeekConfigurationError,
    DeepSeekJSONClient,
    DeepSeekOutputError,
    DeepSeekRequestError,
    get_deepseek_client,
    get_json_client,
)
from src.services.phi_guard import find_obvious_identifiers
from src.services.recommendation import get_recommendation_service
from src.services.recommendation.ranker import RuleRecommendationRanker
from src.services.recommendation.service import RecommendationService


class TinyOutput(BaseModel):
    value: int


def _settings(**updates) -> Settings:
    values = {
        "deepseek_api_key": SecretStr("test-key"),
        "deepseek_max_retries": 0,
    }
    values.update(updates)
    return Settings(_env_file=None, **values)


def _clear_runtime_caches() -> None:
    get_settings.cache_clear()
    get_deepseek_client.cache_clear()
    get_json_client.cache_clear()
    get_pipeline.cache_clear()
    get_recommendation_service.cache_clear()


@pytest.fixture
def fixture_api(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "fixture")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    _clear_runtime_caches()
    import src.api.main as main_module

    main_module = importlib.reload(main_module)
    with TestClient(main_module.app) as client:
        yield client
    _clear_runtime_caches()


def test_deepseek_defaults_and_payload_contract() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        captured["_authorization"] = request.headers.get("authorization")
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": '{"value": 7}'},
                    }
                ]
            },
        )

    http_client = httpx.Client(
        base_url="https://api.deepseek.com",
        transport=httpx.MockTransport(handler),
    )
    client = DeepSeekJSONClient(_settings(), http_client=http_client)
    result = client.invoke_json(
        task_name="contract",
        system_prompt="Return JSON matching {\"value\": 1}",
        user_prompt="Return JSON now.",
        response_model=TinyOutput,
    )

    assert result.value == 7
    assert captured["model"] == "deepseek-v4-pro"
    assert captured["_authorization"] == "Bearer test-key"
    assert captured["thinking"] == {"type": "disabled"}
    assert captured["response_format"] == {"type": "json_object"}
    assert captured["stream"] is False
    assert captured["temperature"] == 0
    assert captured["max_tokens"] == 2048


def test_deepseek_missing_key_fails_before_network() -> None:
    with pytest.raises(DeepSeekConfigurationError):
        DeepSeekJSONClient(Settings(_env_file=None, deepseek_api_key=None))


def test_deepseek_retries_invalid_output_once() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        content = "not-json" if calls == 1 else '{"value": 9}'
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"finish_reason": "stop", "message": {"content": content}}
                ]
            },
        )

    http_client = httpx.Client(
        base_url="https://api.deepseek.com",
        transport=httpx.MockTransport(handler),
    )
    client = DeepSeekJSONClient(
        _settings(deepseek_max_retries=1),
        http_client=http_client,
        sleeper=lambda _seconds: None,
    )
    assert client.invoke_json(
        task_name="retry",
        system_prompt="Return JSON matching {\"value\": 1}",
        user_prompt="JSON only",
        response_model=TinyOutput,
    ).value == 9
    assert calls == 2


def test_deepseek_retries_any_5xx_status() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(501, json={"ignored": True})
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": '{"value": 11}'},
                    }
                ]
            },
        )

    client = DeepSeekJSONClient(
        _settings(deepseek_max_retries=1),
        http_client=httpx.Client(
            base_url="https://api.deepseek.com",
            transport=httpx.MockTransport(handler),
        ),
        sleeper=lambda _seconds: None,
    )
    assert client.invoke_json(
        task_name="server_error",
        system_prompt="Return JSON matching {\"value\": 1}",
        user_prompt="JSON only",
        response_model=TinyOutput,
    ).value == 11
    assert calls == 2


def test_deepseek_auth_and_request_errors_are_not_retried() -> None:
    for status, expected_error in (
        (401, DeepSeekConfigurationError),
        (400, DeepSeekRequestError),
    ):
        calls = 0

        def handler(_request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(status, json={"sensitive": "must not escape"})

        client = DeepSeekJSONClient(
            _settings(deepseek_max_retries=2),
            http_client=httpx.Client(
                base_url="https://api.deepseek.com",
                transport=httpx.MockTransport(handler),
            ),
            sleeper=lambda _seconds: None,
        )
        with pytest.raises(expected_error) as captured_error:
            client.invoke_json(
                task_name="non_retryable",
                system_prompt="Return JSON matching {\"value\": 1}",
                user_prompt="JSON only",
                response_model=TinyOutput,
            )
        assert calls == 1
        assert "sensitive" not in str(captured_error.value)


def test_deepseek_truncation_becomes_output_error() -> None:
    http_client = httpx.Client(
        base_url="https://api.deepseek.com",
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "finish_reason": "length",
                            "message": {"content": '{"value":'},
                        }
                    ]
                },
            )
        ),
    )
    client = DeepSeekJSONClient(_settings(), http_client=http_client)
    with pytest.raises(DeepSeekOutputError):
        client.invoke_json(
            task_name="truncated",
            system_prompt="Return JSON matching {\"value\": 1}",
            user_prompt="JSON only",
            response_model=TinyOutput,
        )


def test_pipeline_normal_path_is_finite_and_ordered() -> None:
    calls: list[str] = []

    def intake(_state):
        calls.append("intake")
        return {"patient_info": {"chief_complaint": "synthetic"}}

    def diagnosis(_state):
        calls.append("diagnosis")
        return {
            "diagnosis": {"primary_diagnosis": {"disease_name": "demo"}},
            "needs_more_info": False,
        }

    def treatment(_state):
        calls.append("treatment")
        return {"treatment_plan": {"diagnosis_addressed": "demo"}}

    def coding(_state):
        calls.append("coding")
        return {"coding_result": {"primary_icd10": {"code": "R69"}}}

    def audit(_state):
        calls.append("audit")
        return {"audit_result": {"prototype_only": True}}

    result = build_clinical_pipeline(
        intake_node=intake,
        diagnosis_node=diagnosis,
        treatment_node=treatment,
        coding_node=coding,
        audit_node=audit,
    ).invoke({"raw_input": "synthetic input"})

    assert calls == ["intake", "diagnosis", "treatment", "coding", "audit"]
    assert result["audit_result"]["prototype_only"] is True


def test_pipeline_information_gap_goes_directly_to_audit() -> None:
    calls: list[str] = []

    def intake(_state):
        calls.append("intake")
        return {"patient_info": {"chief_complaint": "vague"}}

    def diagnosis(_state):
        calls.append("diagnosis")
        return {
            "diagnosis": None,
            "needs_more_info": True,
            "information_gaps": ["more_context_required"],
        }

    def should_not_run(_state):
        raise AssertionError("treatment/coding must not run")

    def audit(_state):
        calls.append("audit")
        return {"audit_result": {"prototype_only": True}}

    result = build_clinical_pipeline(
        intake_node=intake,
        diagnosis_node=diagnosis,
        treatment_node=should_not_run,
        coding_node=should_not_run,
        audit_node=audit,
    ).invoke({"raw_input": "vague synthetic input"})

    assert calls == ["intake", "diagnosis", "audit"]
    assert result["needs_more_info"] is True
    assert result.get("treatment_plan") is None
    assert result.get("coding_result") is None


def test_pipeline_coding_continues_after_treatment_output_failure() -> None:
    calls: list[str] = []

    def intake(_state):
        calls.append("intake")
        return {"patient_info": {"chief_complaint": "synthetic"}}

    def diagnosis(_state):
        calls.append("diagnosis")
        return {
            "diagnosis": {"primary_diagnosis": {"disease_name": "demo"}},
            "needs_more_info": False,
        }

    def treatment(state):
        calls.append("treatment")
        return {
            "treatment_plan": None,
            "errors": state.errors + ["Treatment failed: DeepSeekOutputError"],
        }

    def coding(_state):
        calls.append("coding")
        return {"coding_result": {"primary_icd10": {"code": "R69"}}}

    def audit(_state):
        calls.append("audit")
        return {"audit_result": {"prototype_only": True}}

    result = build_clinical_pipeline(
        intake_node=intake,
        diagnosis_node=diagnosis,
        treatment_node=treatment,
        coding_node=coding,
        audit_node=audit,
    ).invoke({"raw_input": "synthetic input"})

    assert calls == ["intake", "diagnosis", "treatment", "coding", "audit"]
    assert result["treatment_plan"] is None
    assert result["coding_result"] is not None
    assert result["errors"] == ["Treatment failed: DeepSeekOutputError"]


def test_pipeline_invocations_do_not_share_state() -> None:
    def intake(state):
        return {"patient_info": {"source": state.raw_input}}

    def diagnosis(_state):
        return {"diagnosis": None, "needs_more_info": True}

    def unused(_state):
        raise AssertionError("unreachable")

    def audit(_state):
        return {"audit_result": {"prototype_only": True}}

    pipeline = build_clinical_pipeline(
        intake_node=intake,
        diagnosis_node=diagnosis,
        treatment_node=unused,
        coding_node=unused,
        audit_node=audit,
    )
    first = pipeline.invoke({"raw_input": "first synthetic case"})
    second = pipeline.invoke({"raw_input": "second synthetic case"})
    assert first["patient_info"]["source"] == "first synthetic case"
    assert second["patient_info"]["source"] == "second synthetic case"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Contact demo.person@example.com for details", "email"),
        ("Call 555-123-4567 for details", "phone"),
        ("Synthetic SSN 123-45-6789 appears here", "ssn"),
        ("Host address 192.168.1.20 appears here", "ip_address"),
        ("Patient ID: DEMO-12345 appears here", "medical_record_id"),
    ],
)
def test_phi_guard_detects_obvious_identifier_types(text: str, expected: str) -> None:
    assert expected in find_obvious_identifiers(text)


def test_fixture_api_runs_completed_full_chain(fixture_api: TestClient) -> None:
    response = fixture_api.post(
        "/api/v1/clinical/analyze",
        json={
            "patient_description": (
                "56-year-old adult with increased thirst and fatigue. Prior high "
                "glucose readings. A clinician suggested checking HbA1c."
            ),
            "recommendation_top_k": 3,
            "user_preferences": {
                "preferred_categories": ["test_explanation"],
                "preferred_depth": "beginner",
                "preferred_format": "bullet_points",
                "max_reading_minutes": 3,
            },
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["analysis_status"] == "completed"
    assert body["llm_backend"] == "fixture"
    assert all(
        body[field] is not None
        for field in (
            "patient_info",
            "diagnosis",
            "treatment_plan",
            "coding_result",
            "audit_result",
        )
    )
    assert body["audit_result"]["prototype_only"] is True
    assert body["audit_result"]["hipaa_compliant"] is False
    education = body["education_recommendations"]
    assert education["recommendation_status"] == "ok"
    assert education["strategy_used"] == "rule_v1"
    assert 1 <= len(education["recommendations"]) <= 3


def test_fixture_api_information_gap_has_no_loop(fixture_api: TestClient) -> None:
    response = fixture_api.post(
        "/api/v1/clinical/analyze",
        json={
            "patient_description": (
                "Adult reports one vague unspecified concern with no other context."
            )
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["analysis_status"] == "needs_more_info"
    assert body["treatment_plan"] is None
    assert body["coding_result"] is None
    assert body["audit_result"] is not None
    assert body["information_gaps"]


def test_api_rejects_phi_before_pipeline(fixture_api: TestClient) -> None:
    response = fixture_api.post(
        "/api/v1/clinical/analyze",
        json={
            "patient_description": (
                "Synthetic adult with fatigue; contact person@example.com for records."
            )
        },
    )
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "prototype_phi_not_allowed"
    assert response.json()["detail"]["detected_types"] == ["email"]


def test_fixture_health_readiness_and_disabled_recommendations(
    fixture_api: TestClient,
) -> None:
    health = fixture_api.get("/health")
    ready = fixture_api.get("/ready")
    disabled = fixture_api.post(
        "/api/v1/clinical/analyze",
        json={
            "patient_description": (
                "56-year-old adult with thirst and high glucose in a synthetic demo."
            ),
            "include_recommendations": False,
        },
    )

    assert health.status_code == 200
    assert health.json()["prototype_only"] is True
    assert ready.json()["status"] == "ready"
    assert ready.json()["deepseek_configured"] is False
    assert ready.json()["recommendation_store_loaded"] is True
    education = disabled.json()["education_recommendations"]
    assert education["recommendation_status"] == "disabled"
    assert education["strategy_used"] == "none"


def test_recommendation_failure_does_not_fail_clinical_api(
    fixture_api: TestClient,
    monkeypatch,
    tmp_path,
) -> None:
    unavailable_service = RecommendationService(
        topic_path=str(tmp_path / "missing-topics.jsonl"),
        ranker=RuleRecommendationRanker(),
        enabled=True,
    )
    monkeypatch.setattr(
        "src.api.routes.get_recommendation_service",
        lambda: unavailable_service,
    )
    response = fixture_api.post(
        "/api/v1/clinical/analyze",
        json={
            "patient_description": (
                "56-year-old adult with thirst and high glucose in a synthetic demo."
            )
        },
    )
    body = response.json()
    assert response.status_code == 200
    assert body["analysis_status"] == "completed"
    assert body["diagnosis"] is not None
    assert body["education_recommendations"]["recommendation_status"] == "degraded"
    assert body["education_recommendations"]["warnings"] == [
        "recommendation_unavailable"
    ]


def test_default_deepseek_without_key_returns_sanitized_503(monkeypatch) -> None:
    monkeypatch.setenv("LLM_BACKEND", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    _clear_runtime_caches()
    import src.api.main as main_module

    main_module = importlib.reload(main_module)
    with TestClient(main_module.app) as client:
        response = client.post(
            "/api/v1/clinical/analyze",
            json={
                "patient_description": (
                    "Synthetic adult with increased thirst and high glucose readings."
                )
            },
        )
    assert response.status_code == 503
    assert response.json() == {"detail": {"code": "llm_not_configured"}}
    _clear_runtime_caches()

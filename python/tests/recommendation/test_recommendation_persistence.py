from __future__ import annotations

import json
from uuid import uuid4

import pytest
from sqlalchemy import delete, select

from src.models.recommendation import (
    EducationRecommendationResult,
    KnowledgeRecommendation,
    TopicCategory,
)


def _result() -> EducationRecommendationResult:
    return EducationRecommendationResult(
        recommendation_status="degraded",
        strategy_used="rule_v1_fallback",
        ranking_strategy_used="rule_v1_fallback",
        content_strategy_used="catalog_fallback",
        model_version="minionerec-mvp-direct-sid-v1",
        model_ready=False,
        fallback_reason="artifact_incompatible",
        ranker_inference_ms=4.125,
        candidate_source="local_catalog",
        candidate_cache_status="offline",
        content_cache_status="fallback",
        history_used=True,
        valid_history_count=2,
        candidate_count=3,
        recommendations=[
            KnowledgeRecommendation(
                rank=1,
                topic_id="diabetes_basics",
                title="2 型糖尿病基础知识",
                category=TopicCategory.DISEASE_BASICS,
                reason="与结构化诊断相关。",
                summary="用于持久化回归测试的固定目录摘要，内容仅描述合成数据与推荐字段的 JSON 序列化行为。",
                source_label="MediGen reviewed catalog",
                safety_note="教育信息需由专业人员结合实际情况复核。",
            )
        ],
        warnings=["ranker_fallback"],
    )


def test_recommendation_payload_is_jsonb_safe_and_legacy_payload_still_loads() -> None:
    payload = _result().model_dump(mode="json")
    encoded = json.dumps(payload, ensure_ascii=False)
    decoded = json.loads(encoded)
    assert decoded["recommendations"][0]["category"] == "disease_basics"
    assert decoded["fallback_reason"] == "artifact_incompatible"
    assert decoded["ranker_inference_ms"] == 4.125

    legacy = EducationRecommendationResult.model_validate(
        {
            "recommendation_status": "ok",
            "strategy_used": "rule_v1_deepseek",
            "recommendations": [],
        }
    )
    assert legacy.strategy_used == "rule_v1_deepseek"
    assert legacy.ranking_strategy_used == "none"
    assert legacy.model_version is None


@pytest.mark.integration
def test_postgresql_round_trip_preserves_model_metadata_and_cards() -> None:
    from src.services.database import DatabaseService, clinical_sessions

    database = DatabaseService()
    database.initialize()
    session_id = uuid4()
    payload = _result().model_dump(mode="json")
    clinical = {
        "patient_info": {"chief_complaint": "synthetic fatigue"},
        "diagnosis": {"primary_diagnosis": {"disease_name": "Type 2 diabetes"}},
        "treatment_plan": {"medications": []},
        "coding_result": {"primary_icd10": {"code": "E11.9"}},
        "audit_result": {"demo_safe": True, "audit_trail": []},
        "errors": [],
    }
    try:
        saved_id = database.save_analysis(
            session_id=session_id,
            raw_input="synthetic persistence test case",
            clinical_result=clinical,
            recommendation_result=payload,
            fhir_export={"provider": "test"},
            analysis_status="completed",
            llm_backend="fixture",
            client_ip="127.0.0.1",
        )
        assert saved_id == str(session_id)
        with database.engine.connect() as connection:
            stored = connection.execute(
                select(clinical_sessions.c.recommendation_result).where(
                    clinical_sessions.c.id == session_id
                )
            ).scalar_one()
        assert stored["model_version"] == payload["model_version"]
        assert stored["fallback_reason"] == "artifact_incompatible"
        assert stored["ranker_inference_ms"] == 4.125
        assert stored["recommendations"][0]["category"] == "disease_basics"
        assert stored["recommendations"][0]["topic_id"] == "diabetes_basics"
    finally:
        with database.engine.begin() as connection:
            connection.execute(
                delete(clinical_sessions).where(clinical_sessions.c.id == session_id)
            )
        database.close()

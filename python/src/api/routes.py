"""HTTP routes for the MediGen workflow and local clinical-data services."""

from __future__ import annotations

from time import perf_counter
from typing import Literal
from uuid import uuid4

import structlog
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ..config.settings import get_settings
from ..graph.clinical_pipeline import get_pipeline
from ..models.recommendation import (
    EducationRecommendationResult,
    UserHistoryContext,
    UserPreferenceContext,
)
from ..services.deepseek_client import (
    DeepSeekConfigurationError,
    DeepSeekRequestError,
    DeepSeekUpstreamError,
)
from ..services.database import get_database_service
from ..services.drug_interaction import check_interactions
from ..services.icd10_service import get_drg_group, lookup_icd10, search_icd10_by_text
from ..services.fhir_service import get_fhir_service
from ..services.graphrag_service import get_graphrag_service
from ..services.phi_guard import find_obvious_identifiers
from ..services.recommendation import get_recommendation_service
from ..services.redis_service import get_redis_service
from ..services.runtime_services import runtime_status

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["Clinical Analysis"])


class AnalyzeRequest(BaseModel):
    patient_description: str = Field(
        ...,
        min_length=10,
        max_length=12000,
        description="Synthetic, de-identified patient narrative only",
        examples=[
            "56-year-old adult with increased thirst and fatigue. Prior high "
            "glucose readings. A clinician suggested checking HbA1c."
        ],
    )
    include_recommendations: bool = True
    recommendation_top_k: int = Field(default=3, ge=1, le=3)
    user_preferences: UserPreferenceContext | None = None
    user_history_context: UserHistoryContext | None = None


class AnalyzeResponse(BaseModel):
    analysis_status: Literal["completed", "needs_more_info", "partial"]
    llm_backend: Literal["deepseek", "fixture"]
    patient_info: dict | None = None
    diagnosis: dict | None = None
    treatment_plan: dict | None = None
    coding_result: dict | None = None
    audit_result: dict | None = None
    information_gaps: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    education_recommendations: EducationRecommendationResult
    session_id: str | None = None
    fhir_export: dict | None = None
    integration_trace: dict = Field(default_factory=dict)
    processing_timeline: dict = Field(default_factory=dict)


class ICD10SearchRequest(BaseModel):
    query: str = Field(..., min_length=2, description="Search text for demo ICD-10 data")


class DDICheckRequest(BaseModel):
    new_drugs: list[str] = Field(..., min_length=1)
    current_drugs: list[str] = Field(default_factory=list)


@router.post("/clinical/analyze", response_model=AnalyzeResponse)
def analyze_patient(req: AnalyzeRequest, request: Request) -> AnalyzeResponse:
    """Run clinical analysis, graph retrieval, export, and persistence."""

    request_started = perf_counter()
    settings = get_settings()
    identifiers = find_obvious_identifiers(req.patient_description)
    if settings.prototype_reject_obvious_phi:
        if identifiers:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "prototype_phi_not_allowed",
                    "message": (
                        "This workspace accepts synthetic, de-identified data only."
                    ),
                    "detected_types": identifiers,
                },
            )

    client_ip = request.client.host if request.client else "local"
    rate_limit_remaining: int | None = None
    if settings.llm_backend == "deepseek" and not settings.deepseek_configured:
        raise HTTPException(
            status_code=503,
            detail={"code": "llm_not_configured"},
        )
    if settings.llm_backend == "deepseek" and settings.infrastructure_required:
        dependencies = runtime_status()
        unavailable = sorted(
            name for name, ready in dependencies.items() if not ready
        )
        if unavailable:
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "infrastructure_unavailable",
                    "services": unavailable,
                },
            )
        allowed, rate_limit_remaining = get_redis_service().allow_request(client_ip)
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail={"code": "rate_limit_exceeded"},
            )

    try:
        result = get_pipeline().invoke(
            {"raw_input": req.patient_description},
        )
    except DeepSeekConfigurationError as exc:
        raise HTTPException(
            status_code=503,
            detail={"code": "llm_not_configured"},
        ) from exc
    except DeepSeekRequestError as exc:
        raise HTTPException(
            status_code=502,
            detail={"code": "llm_request_rejected"},
        ) from exc
    except DeepSeekUpstreamError as exc:
        raise HTTPException(
            status_code=502,
            detail={"code": "llm_upstream_unavailable"},
        ) from exc
    except Exception as exc:
        logger.error("pipeline.unhandled", error_type=type(exc).__name__)
        raise HTTPException(
            status_code=500,
            detail={"code": "pipeline_internal_error"},
        ) from exc

    if not isinstance(result, dict):
        logger.error("pipeline.invalid_result", result_type=type(result).__name__)
        raise HTTPException(
            status_code=500,
            detail={"code": "pipeline_internal_error"},
        )

    recommendation_started = perf_counter()
    if req.include_recommendations:
        education = get_recommendation_service().recommend_after_analysis(
            clinical_result=result,
            user_preferences=req.user_preferences,
            user_history_context=req.user_history_context,
            top_k=req.recommendation_top_k,
        )
    else:
        education = EducationRecommendationResult(
            recommendation_status="disabled",
            strategy_used="none",
        )
    recommendation_elapsed = round(perf_counter() - recommendation_started, 3)

    if result.get("needs_more_info"):
        analysis_status = "needs_more_info"
    elif all(
        result.get(field) is not None
        for field in (
            "patient_info",
            "diagnosis",
            "treatment_plan",
            "coding_result",
            "audit_result",
        )
    ):
        analysis_status = "completed"
    else:
        analysis_status = "partial"

    session_uuid = uuid4()
    fhir_export: dict | None = None
    integration_trace: dict = {}
    supporting_timings: dict[str, float] = {}
    if settings.llm_backend == "deepseek":
        patient_info = result.get("patient_info")
        if not isinstance(patient_info, dict):
            raise HTTPException(
                status_code=503,
                detail={"code": "fhir_export_requires_patient_info"},
            )
        try:
            fhir_started = perf_counter()
            fhir_export = get_fhir_service().export_analysis(
                session_id=session_uuid,
                patient_info=patient_info,
                diagnosis=result.get("diagnosis"),
                treatment_plan=result.get("treatment_plan"),
            )
            supporting_timings["interoperability"] = round(
                perf_counter() - fhir_started,
                3,
            )
            persistence_started = perf_counter()
            session_id = get_database_service().save_analysis(
                session_id=session_uuid,
                raw_input=req.patient_description,
                clinical_result=result,
                recommendation_result=education.model_dump(mode="json"),
                fhir_export=fhir_export,
                analysis_status=analysis_status,
                llm_backend=settings.llm_backend,
                client_ip=client_ip,
            )
            supporting_timings["persistence"] = round(
                perf_counter() - persistence_started,
                3,
            )
        except RuntimeError as exc:
            logger.error("integration.persist_or_export_failed", error_type=type(exc).__name__)
            raise HTTPException(
                status_code=503,
                detail={"code": "integration_write_failed"},
            ) from exc

        diagnosis_graph = (
            result.get("diagnosis", {}).get("knowledge_graph", {})
            if isinstance(result.get("diagnosis"), dict)
            else {}
        )
        integration_trace = {
            "privacy_scan": {
                "provider": "Presidio + local rules",
                "result": "clear",
                "detected_categories": identifiers,
            },
            "knowledge_graph": {
                **get_graphrag_service().stats(),
                "query_cache": diagnosis_graph.get("cache_status", "miss"),
                "evidence_count": diagnosis_graph.get("evidence_count", 0),
                "recommendation_candidates": education.candidate_source,
            },
            "cache_and_rate_limit": {
                **get_redis_service().info_summary(),
                "rate_limit_remaining": rate_limit_remaining,
                "recommendation_cache": education.content_cache_status,
            },
            "persistence": {
                **get_database_service().counts(),
                "session_id": session_id,
            },
            "interoperability": fhir_export,
        }
    else:
        session_id = str(session_uuid)

    stage_timings = {
        key: float(value)
        for key, value in result.get("stage_timings_seconds", {}).items()
        if isinstance(value, int | float)
    }
    stage_timings["recommendations"] = recommendation_elapsed
    processing_timeline = {
        "total_seconds": round(perf_counter() - request_started, 3),
        "stages": {
            key: {"elapsed_seconds": value}
            for key, value in stage_timings.items()
        },
        "supporting_steps": {
            key: {"elapsed_seconds": value}
            for key, value in supporting_timings.items()
        },
    }

    return AnalyzeResponse(
        analysis_status=analysis_status,
        llm_backend=settings.llm_backend,
        patient_info=result.get("patient_info"),
        diagnosis=result.get("diagnosis"),
        treatment_plan=result.get("treatment_plan"),
        coding_result=result.get("coding_result"),
        audit_result=result.get("audit_result"),
        information_gaps=result.get("information_gaps", []),
        errors=result.get("errors", []),
        warnings=result.get("warnings", []),
        education_recommendations=education,
        session_id=session_id,
        fhir_export=fhir_export,
        integration_trace=integration_trace,
        processing_timeline=processing_timeline,
    )


@router.post("/clinical/icd10/search")
def search_icd10(req: ICD10SearchRequest) -> dict:
    """Search the small local ICD-10 demonstration table."""

    results = search_icd10_by_text(req.query)
    return {"query": req.query, "results": results, "count": len(results)}


@router.get("/clinical/icd10/{code}")
def get_icd10(code: str) -> dict:
    """Look up a code in the small local demonstration table."""

    result = lookup_icd10(code)
    if not result:
        raise HTTPException(status_code=404, detail="ICD-10 code not found")
    return {"icd10": result, "drg_group": get_drg_group(code)}


@router.post("/clinical/ddi/check")
def check_ddi(req: DDICheckRequest) -> dict:
    """Check the small local drug-interaction demonstration table."""

    interactions = check_interactions(req.new_drugs, req.current_drugs)
    return {
        "new_drugs": req.new_drugs,
        "current_drugs": req.current_drugs,
        "interactions": interactions,
        "interaction_count": len(interactions),
        "has_major_interaction": any(
            item["severity"] in ("major", "contraindicated")
            for item in interactions
        ),
    }

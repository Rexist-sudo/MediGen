"""HTTP routes for the finite MediGen MVP workflow and local demo services."""

from __future__ import annotations

from typing import Literal

import structlog
from fastapi import APIRouter, HTTPException
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
from ..services.drug_interaction import check_interactions
from ..services.icd10_service import get_drg_group, lookup_icd10, search_icd10_by_text
from ..services.phi_guard import find_obvious_identifiers
from ..services.recommendation import get_recommendation_service

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["Clinical Prototype"])


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


class ICD10SearchRequest(BaseModel):
    query: str = Field(..., min_length=2, description="Search text for demo ICD-10 data")


class DDICheckRequest(BaseModel):
    new_drugs: list[str] = Field(..., min_length=1)
    current_drugs: list[str] = Field(default_factory=list)


@router.post("/clinical/analyze", response_model=AnalyzeResponse)
def analyze_patient(req: AnalyzeRequest) -> AnalyzeResponse:
    """Run the five finite nodes, then the isolated local recommendation step."""

    settings = get_settings()
    if settings.prototype_reject_obvious_phi:
        identifiers = find_obvious_identifiers(req.patient_description)
        if identifiers:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "prototype_phi_not_allowed",
                    "message": (
                        "This MVP accepts synthetic, de-identified demo data only."
                    ),
                    "detected_types": identifiers,
                },
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

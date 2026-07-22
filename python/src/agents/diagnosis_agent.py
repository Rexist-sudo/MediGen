"""Diagnosis node with an explicit information-gaps outcome."""

from __future__ import annotations

import json

import structlog

from ..models.diagnosis import DiagnosisLLMOutput
from ..services.deepseek_client import DeepSeekOutputError, get_json_client

logger = structlog.get_logger(__name__)

DIAGNOSIS_SYSTEM_PROMPT = """Analyze synthetic structured demo data only.
Return ONLY one valid JSON object matching this shape:
{
  "primary_diagnosis": {"disease_name": "example", "icd10_hint": "R69", "confidence": 0.5, "evidence": [], "reasoning": "prototype-only reasoning"},
  "differential_list": [],
  "recommended_tests": [],
  "clinical_notes": "prototype output; not for medical use",
  "knowledge_sources": [],
  "needs_more_info": false,
  "information_gaps": []
}
If information is insufficient, set primary_diagnosis to null,
needs_more_info to true, and list concise information_gaps. Never invent missing
patient facts. The output must be JSON with no markdown."""


def diagnosis_agent(state) -> dict:
    """Create a diagnosis payload or a finite information-gaps result."""

    logger.info("diagnosis.start")
    if not state.patient_info:
        return {
            "diagnosis": None,
            "needs_more_info": True,
            "information_gaps": state.information_gaps
            + ["structured_patient_information_required"],
            "current_agent": "diagnosis",
            "errors": state.errors + ["Diagnosis skipped: patient_info_unavailable"],
        }

    patient_summary = json.dumps(
        state.patient_info,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    try:
        output = get_json_client().invoke_json(
            task_name="diagnosis",
            system_prompt=DIAGNOSIS_SYSTEM_PROMPT,
            user_prompt=(
                "Structured synthetic patient JSON:\n"
                f"{patient_summary}\n"
                "Return the requested diagnosis JSON only."
            ),
            response_model=DiagnosisLLMOutput,
        )
    except DeepSeekOutputError as exc:
        logger.warning("diagnosis.output_unavailable", error_type=type(exc).__name__)
        return {
            "diagnosis": None,
            "needs_more_info": True,
            "information_gaps": state.information_gaps
            + ["diagnosis_structured_output_unavailable"],
            "current_agent": "diagnosis",
            "errors": state.errors + ["Diagnosis failed: DeepSeekOutputError"],
        }

    diagnosis_payload = output.model_dump(
        mode="json",
        exclude={"needs_more_info", "information_gaps"},
    )
    logger.info(
        "diagnosis.success",
        needs_more_info=output.needs_more_info,
    )
    return {
        "diagnosis": diagnosis_payload,
        "needs_more_info": output.needs_more_info,
        "information_gaps": state.information_gaps + output.information_gaps,
        "current_agent": "diagnosis",
    }

"""Treatment-plan node for a synthetic software demonstration."""

from __future__ import annotations

import json

import structlog

from ..models.treatment import TreatmentPlan
from ..services.deepseek_client import DeepSeekOutputError, get_json_client

logger = structlog.get_logger(__name__)

TREATMENT_SYSTEM_PROMPT = """Generate a prototype treatment-plan JSON object for
synthetic software-architecture demonstration only. It is not a prescription,
has not been clinically validated, and must not claim otherwise. Return ONLY JSON:
{
  "diagnosis_addressed": "example",
  "medications": [],
  "drug_interactions": [],
  "non_drug_treatments": [],
  "lifestyle_recommendations": [],
  "follow_up_plan": "",
  "warnings": ["Not for medical use."],
  "evidence_references": []
}
If medications are included, retain the exact schema required by the response
contract. The output must be valid JSON without markdown."""


def treatment_agent(state) -> dict:
    """Generate a validated plan while preserving earlier results on failure."""

    logger.info("treatment.start")
    if not state.diagnosis:
        return {
            "treatment_plan": None,
            "current_agent": "treatment",
            "errors": state.errors + ["Treatment skipped: diagnosis_unavailable"],
        }

    context = json.dumps(
        {"patient_info": state.patient_info, "diagnosis": state.diagnosis},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    try:
        plan = get_json_client().invoke_json(
            task_name="treatment",
            system_prompt=TREATMENT_SYSTEM_PROMPT,
            user_prompt=(
                f"Synthetic clinical context JSON:\n{context}\n"
                "Return the requested treatment-plan JSON only."
            ),
            response_model=TreatmentPlan,
        )
    except DeepSeekOutputError as exc:
        logger.warning("treatment.output_unavailable", error_type=type(exc).__name__)
        return {
            "treatment_plan": None,
            "current_agent": "treatment",
            "errors": state.errors + ["Treatment failed: DeepSeekOutputError"],
        }

    logger.info("treatment.success")
    return {
        "treatment_plan": plan.model_dump(mode="json"),
        "current_agent": "treatment",
    }

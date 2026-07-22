"""Intake node for synthetic, de-identified prototype narratives."""

from __future__ import annotations

import structlog

from ..models.patient import PatientInfo
from ..services.deepseek_client import DeepSeekOutputError, get_json_client

logger = structlog.get_logger(__name__)

INTAKE_SYSTEM_PROMPT = """You extract structured information from synthetic demo narratives.
Return ONLY a valid JSON object matching this example:
{
  "patient_id": null,
  "name": "Unknown",
  "age": 56,
  "gender": "unknown",
  "chief_complaint": "fatigue",
  "symptoms": [{"name": "fatigue", "duration_days": null, "severity": "moderate", "description": null}],
  "medical_history": [],
  "family_history": [],
  "allergies": [],
  "current_medications": [],
  "vital_signs": null,
  "lab_results": []
}
Use null for an unprovided age; never estimate it. Always use "Unknown" for name
and null for patient_id. Use empty lists or null for missing fields. Do not extract
direct identifiers. The output must be JSON and contain no markdown."""


def intake_agent(state) -> dict:
    """Parse the raw narrative into the validated ``PatientInfo`` contract."""

    raw = state.raw_input or ""
    logger.info("intake.start", input_length=len(raw))
    if not raw.strip():
        return {
            "patient_info": None,
            "current_agent": "intake",
            "information_gaps": state.information_gaps + ["patient_narrative_required"],
            "errors": state.errors + ["Intake failed: missing_input"],
        }

    try:
        patient = get_json_client().invoke_json(
            task_name="intake",
            system_prompt=INTAKE_SYSTEM_PROMPT,
            user_prompt=f"Patient narrative:\n\n{raw}\n\nReturn JSON only.",
            response_model=PatientInfo,
        )
    except DeepSeekOutputError as exc:
        logger.warning("intake.output_unavailable", error_type=type(exc).__name__)
        return {
            "patient_info": None,
            "current_agent": "intake",
            "information_gaps": state.information_gaps
            + ["intake_structured_output_unavailable"],
            "errors": state.errors + ["Intake failed: DeepSeekOutputError"],
        }

    logger.info("intake.success")
    return {
        "patient_info": patient.model_dump(mode="json"),
        "current_agent": "intake",
    }

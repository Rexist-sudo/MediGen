"""Prototype ICD-10/DRG coding node."""

from __future__ import annotations

import json

import structlog

from ..models.treatment import CodingResult
from ..services.deepseek_client import DeepSeekOutputError, get_json_client

logger = structlog.get_logger(__name__)

CODING_SYSTEM_PROMPT = """Return a prototype medical-coding JSON object for
synthetic software demonstration only. Do not claim validation against a complete
coding database. Return ONLY valid JSON matching this example:
{
  "primary_icd10": {"code": "R69", "description": "example", "confidence": 0.5, "category": "prototype"},
  "secondary_icd10_codes": [],
  "drg_group": null,
  "coding_notes": "prototype-only suggestion",
  "coding_confidence": 0.5
}
The output must contain JSON and no markdown."""


def coding_agent(state) -> dict:
    """Generate a validated prototype coding result."""

    logger.info("coding.start")
    if not state.diagnosis:
        return {
            "coding_result": None,
            "current_agent": "coding",
            "errors": state.errors + ["Coding skipped: diagnosis_unavailable"],
        }

    context = json.dumps(
        {
            "diagnosis": state.diagnosis,
            "treatment_plan": state.treatment_plan,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    try:
        coding = get_json_client().invoke_json(
            task_name="coding",
            system_prompt=CODING_SYSTEM_PROMPT,
            user_prompt=(
                f"Synthetic clinical coding context JSON:\n{context}\n"
                "Return the requested coding JSON only."
            ),
            response_model=CodingResult,
        )
    except DeepSeekOutputError as exc:
        logger.warning("coding.output_unavailable", error_type=type(exc).__name__)
        return {
            "coding_result": None,
            "current_agent": "coding",
            "errors": state.errors + ["Coding failed: DeepSeekOutputError"],
        }

    logger.info("coding.success")
    return {
        "coding_result": coding.model_dump(mode="json"),
        "current_agent": "coding",
    }

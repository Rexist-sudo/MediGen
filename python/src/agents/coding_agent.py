"""ICD-10 candidate coding with local catalog and DRG enrichment."""

from __future__ import annotations

import json

import structlog

from ..models.treatment import CodingResult
from ..services.deepseek_client import DeepSeekOutputError, get_json_client
from ..services.icd10_service import get_drg_group, lookup_icd10

logger = structlog.get_logger(__name__)

CODING_SYSTEM_PROMPT = """Return a medical-coding candidate JSON object for
synthetic, de-identified input. Use the most specific supported ICD-10-CM code
justified by the supplied diagnosis, while keeping uncertain items explicit.
Return ONLY valid JSON matching this example:
{
  "primary_icd10": {"code": "R69", "description": "example", "confidence": 0.5, "category": "Symptoms and signs"},
  "secondary_icd10_codes": [],
  "drg_group": null,
  "coding_notes": "concise rationale for professional coding review",
  "coding_confidence": 0.5
}
Write coding notes and model-supplied descriptions in clear Simplified Chinese,
while retaining official codes and standard abbreviations. The output must
contain JSON and no markdown."""


def coding_agent(state) -> dict:
    """Generate and enrich a validated coding result."""

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

    payload = coding.model_dump(mode="json")
    primary = payload["primary_icd10"]
    primary_lookup = lookup_icd10(primary["code"])
    if primary_lookup:
        primary["description"] = primary_lookup["description"]
        primary["category"] = primary_lookup["category"]
    matched_secondary = 0
    for secondary in payload["secondary_icd10_codes"]:
        local = lookup_icd10(secondary["code"])
        if local:
            secondary["description"] = local["description"]
            secondary["category"] = local["category"]
            matched_secondary += 1
    drg = get_drg_group(primary["code"])
    if drg:
        payload["drg_group"] = drg
    payload["local_validation"] = {
        "catalog": "MediGen local ICD-10-CM subset",
        "primary_code_matched": bool(primary_lookup),
        "secondary_codes_matched": matched_secondary,
        "secondary_code_count": len(payload["secondary_icd10_codes"]),
        "drg_prefix_matched": bool(drg),
    }

    logger.info(
        "coding.success",
        primary_code_matched=bool(primary_lookup),
        drg_prefix_matched=bool(drg),
    )
    return {
        "coding_result": payload,
        "current_agent": "coding",
    }

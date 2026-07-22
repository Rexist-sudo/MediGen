"""Treatment-plan node with deterministic medication safety checks."""

from __future__ import annotations

import json

import structlog

from ..models.treatment import DrugInteraction, TreatmentPlan
from ..services.deepseek_client import DeepSeekOutputError, get_json_client
from ..services.drug_interaction import (
    check_allergy_contraindication,
    check_interactions,
)

logger = structlog.get_logger(__name__)

TREATMENT_SYSTEM_PROMPT = """Generate a structured treatment-plan candidate for
synthetic, de-identified clinical data. The result requires professional review
and must never claim to be a prescription. Return ONLY JSON:
{
  "diagnosis_addressed": "example",
  "medications": [
    {
      "drug_name": "example medicine",
      "generic_name": "generic name or empty string",
      "dosage": "example dose",
      "route": "oral",
      "frequency": "example frequency",
      "duration": "example duration",
      "contraindications": [],
      "side_effects": []
    }
  ],
  "drug_interactions": [
    {
      "drug_a": "medicine A",
      "drug_b": "medicine B",
      "severity": "none",
      "description": "interaction description",
      "recommendation": "risk-management note"
    }
  ],
  "non_drug_treatments": [],
  "lifestyle_recommendations": [],
  "follow_up_plan": "",
  "warnings": ["Professional review required before clinical use."],
  "evidence_references": []
}
Use exactly these medication and interaction field names. The output must be
valid JSON without markdown. Write all narrative fields in clear Simplified
Chinese; keep generic medication names in their standard English form when
that improves unambiguous matching."""


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

    current_drugs = [
        str(item.get("name", ""))
        for item in (state.patient_info or {}).get("current_medications", [])
        if isinstance(item, dict) and item.get("name")
    ]
    new_drugs = [
        medication.generic_name or medication.drug_name
        for medication in plan.medications
    ]
    deterministic = [
        *check_interactions(new_drugs, current_drugs),
        *check_interactions(current_drugs, []),
    ]
    merged_interactions = list(plan.drug_interactions)
    pair_indexes = {
        frozenset((item.drug_a.casefold(), item.drug_b.casefold())): index
        for index, item in enumerate(merged_interactions)
    }
    for item in deterministic:
        pair = frozenset((item["drug_a"].casefold(), item["drug_b"].casefold()))
        validated = DrugInteraction.model_validate(item)
        if pair in pair_indexes:
            merged_interactions[pair_indexes[pair]] = validated
        else:
            pair_indexes[pair] = len(merged_interactions)
            merged_interactions.append(validated)

    allergy_names = [
        str(item.get("substance", ""))
        for item in (state.patient_info or {}).get("allergies", [])
        if isinstance(item, dict) and item.get("substance")
    ]
    warnings = list(plan.warnings)
    for current_drug in current_drugs:
        current_check = check_allergy_contraindication(current_drug, allergy_names)
        if current_check:
            warnings.append(current_check["recommendation"])
    medications = []
    for medication in plan.medications:
        check = check_allergy_contraindication(
            medication.generic_name or medication.drug_name,
            allergy_names,
        )
        if check:
            medication = medication.model_copy(
                update={
                    "contraindications": list(
                        dict.fromkeys(
                            [*medication.contraindications, check["recommendation"]]
                        )
                    )
                }
            )
            warnings.append(check["recommendation"])
        medications.append(medication)
    plan = plan.model_copy(
        update={
            "medications": medications,
            "drug_interactions": merged_interactions,
            "warnings": list(dict.fromkeys(warnings)),
            "evidence_references": list(
                dict.fromkeys(
                    [
                        *plan.evidence_references,
                        *(["本地药物安全规则"] if deterministic else []),
                    ]
                )
            ),
        }
    )

    logger.info("treatment.success", local_interaction_count=len(deterministic))
    return {
        "treatment_plan": plan.model_dump(mode="json"),
        "current_agent": "treatment",
    }

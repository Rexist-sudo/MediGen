"""Diagnosis node with an explicit information-gaps outcome."""

from __future__ import annotations

import json

import structlog

from ..models.diagnosis import DiagnosisLLMOutput
from ..services.deepseek_client import DeepSeekOutputError, get_json_client
from ..services.graphrag_service import get_graphrag_service

logger = structlog.get_logger(__name__)

DIAGNOSIS_SYSTEM_PROMPT = """Analyze synthetic, de-identified structured data.
Return ONLY one valid JSON object matching this shape:
{
  "primary_diagnosis": {"disease_name": "example", "icd10_hint": "R69", "confidence": 0.5, "evidence": [], "reasoning": "concise evidence-based reasoning"},
  "differential_list": [],
  "recommended_tests": [],
  "clinical_notes": "concise clinical synthesis for professional review",
  "knowledge_sources": [],
  "needs_more_info": false,
  "information_gaps": []
}
If information is insufficient, set primary_diagnosis to null,
needs_more_info to true, and list concise information_gaps. Never invent missing
patient facts or claim a supplied examination or diagnostic study is absent.
Write disease names, evidence, reasoning, notes, tests, and information gaps in
clear Simplified Chinese while retaining standard clinical abbreviations and
ICD-10 codes. The output must be JSON with no markdown."""


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
    symptom_names = [
        item.get("name", "")
        for item in state.patient_info.get("symptoms", [])
        if isinstance(item, dict) and item.get("name")
    ]
    graph_trace = get_graphrag_service().find_diseases_with_trace(symptom_names)
    graph_records = graph_trace["records"]
    graph_context = json.dumps(
        graph_records[:6],
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
                "Neo4j graph retrieval results from the curated local medical graph:\n"
                f"{graph_context}\n"
                "Use graph results as bounded supporting evidence. Reconcile them "
                "with the patient facts and do not force a graph candidate when the "
                "clinical evidence conflicts. Include 'Neo4j local medical graph' in "
                "knowledge_sources when graph evidence contributes.\n"
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
    sources = list(diagnosis_payload.get("knowledge_sources", []))
    if graph_records and "Neo4j local medical graph" not in sources:
        sources.append("Neo4j local medical graph")
    diagnosis_payload["knowledge_sources"] = sources
    diagnosis_payload["knowledge_graph"] = {
        "provider": "Neo4j",
        "cache_status": graph_trace["cache_status"],
        "evidence_count": len(graph_records),
        "evidence": graph_records[:6],
    }
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

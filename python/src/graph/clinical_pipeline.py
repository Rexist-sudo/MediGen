"""Stateless, finite LangGraph workflow for the MediGen MVP."""

from __future__ import annotations

from functools import lru_cache
from time import perf_counter
from typing import Callable

from langgraph.graph import END, StateGraph

from ..agents.audit_agent import audit_agent
from ..agents.coding_agent import coding_agent
from ..agents.diagnosis_agent import diagnosis_agent
from ..agents.intake_agent import intake_agent
from ..agents.treatment_agent import treatment_agent
from .state import ClinicalState

Node = Callable[[ClinicalState], dict]


def _timed_node(stage: str, node: Node) -> Node:
    """Record wall-clock time while preserving the finite graph state."""

    def invoke(state: ClinicalState) -> dict:
        started = perf_counter()
        output = node(state)
        timings = dict(state.stage_timings_seconds)
        timings[stage] = round(perf_counter() - started, 3)
        return {**output, "stage_timings_seconds": timings}

    return invoke


def _route_after_diagnosis(state: ClinicalState) -> str:
    if state.needs_more_info or not state.diagnosis:
        return "audit"
    return "treatment"


def build_clinical_pipeline(
    *,
    intake_node: Node = intake_agent,
    diagnosis_node: Node = diagnosis_agent,
    treatment_node: Node = treatment_agent,
    coding_node: Node = coding_agent,
    audit_node: Node = audit_agent,
):
    """Compile a graph with no checkpoint memory and no backward edge."""

    workflow = StateGraph(ClinicalState)
    workflow.add_node("intake", _timed_node("intake", intake_node))
    workflow.add_node("diagnosis", _timed_node("diagnosis", diagnosis_node))
    workflow.add_node("treatment", _timed_node("treatment", treatment_node))
    workflow.add_node("coding", _timed_node("coding", coding_node))
    workflow.add_node("audit", _timed_node("audit", audit_node))

    workflow.set_entry_point("intake")
    workflow.add_edge("intake", "diagnosis")
    workflow.add_conditional_edges(
        "diagnosis",
        _route_after_diagnosis,
        {"audit": "audit", "treatment": "treatment"},
    )
    workflow.add_edge("treatment", "coding")
    workflow.add_edge("coding", "audit")
    workflow.add_edge("audit", END)
    return workflow.compile()


@lru_cache(maxsize=1)
def get_pipeline():
    return build_clinical_pipeline()

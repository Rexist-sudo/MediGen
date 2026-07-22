"""Finite per-request state for the clinical prototype pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ClinicalState(BaseModel):
    raw_input: str = Field(default="", description="Synthetic patient narrative")
    patient_info: dict | None = None
    diagnosis: dict | None = None
    needs_more_info: bool = False
    information_gaps: list[str] = Field(default_factory=list)
    treatment_plan: dict | None = None
    coding_result: dict | None = None
    audit_result: dict | None = None
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    current_agent: str = ""

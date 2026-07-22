"""Structured synthetic patient data used by the MVP workflow."""

from __future__ import annotations

from datetime import date
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"
    UNKNOWN = "unknown"


class Severity(str, Enum):
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


class Symptom(BaseModel):
    name: str
    duration_days: int | None = Field(default=None, ge=0)
    severity: Severity = Severity.MODERATE
    description: str | None = None


class Allergy(BaseModel):
    substance: str
    reaction: str | None = None
    severity: Severity = Severity.MODERATE


class Medication(BaseModel):
    name: str
    dosage: str | None = None
    frequency: str | None = None
    start_date: date | None = None


class VitalSigns(BaseModel):
    temperature: float | None = Field(None, description="Body temperature in Celsius")
    heart_rate: int | None = Field(None, description="Beats per minute")
    blood_pressure_systolic: int | None = None
    blood_pressure_diastolic: int | None = None
    respiratory_rate: int | None = None
    oxygen_saturation: float | None = None


class LabResult(BaseModel):
    test_name: str
    value: str
    unit: str | None = None
    reference_range: str | None = None
    is_abnormal: bool = False


class PatientInfo(BaseModel):
    """Core patient information for intake processing."""

    patient_id: str | None = None
    name: str = "Unknown"
    age: int | None = Field(default=None, ge=0, le=130)
    gender: Gender = Gender.UNKNOWN
    chief_complaint: str
    symptoms: list[Symptom] = Field(default_factory=list)
    medical_history: list[str] = Field(default_factory=list)
    family_history: list[str] = Field(default_factory=list)
    allergies: list[Allergy] = Field(default_factory=list)
    current_medications: list[Medication] = Field(default_factory=list)
    vital_signs: VitalSigns | None = None
    lab_results: list[LabResult] = Field(default_factory=list)

    @field_validator("name", mode="before")
    @classmethod
    def force_anonymous_name(cls, _value: object) -> str:
        """Never retain a direct name from an external model response."""

        return "Unknown"

    @field_validator("patient_id", mode="before")
    @classmethod
    def discard_patient_identifier(cls, _value: object) -> None:
        """The stateless prototype does not collect patient identifiers."""

        return None

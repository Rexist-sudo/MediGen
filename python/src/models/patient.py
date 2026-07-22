"""Structured synthetic patient data used by the MVP workflow."""

from __future__ import annotations

from datetime import date
from enum import Enum

from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator


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


def _normalize_optional_severity(value: object) -> Severity | None:
    if value is None or isinstance(value, Severity):
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold().replace("_", " ").replace("-", " ")
        if normalized in {"mild", "moderate", "severe", "critical"}:
            return Severity(normalized)
    # Severity is optional. Discard provider-specific labels instead of
    # inventing a supported clinical grade.
    return None


class Symptom(BaseModel):
    name: str
    duration_days: int | None = Field(default=None, ge=0)
    severity: Severity | None = None
    description: str | None = None

    @field_validator("duration_days", mode="before")
    @classmethod
    def normalize_duration_days(cls, value: object) -> object:
        if value is None or isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str) and value.strip().isdigit():
            return int(value.strip())
        return None

    @field_validator("severity", mode="before")
    @classmethod
    def normalize_severity(cls, value: object) -> Severity | None:
        return _normalize_optional_severity(value)


class Allergy(BaseModel):
    substance: str
    reaction: str | None = None
    severity: Severity | None = None

    @field_validator("severity", mode="before")
    @classmethod
    def normalize_severity(cls, value: object) -> Severity | None:
        return _normalize_optional_severity(value)


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
    test_name: str = Field(
        validation_alias=AliasChoices("test_name", "test"),
    )
    value: str | None = Field(
        default=None,
        validation_alias=AliasChoices("value", "result"),
    )
    unit: str | None = None
    reference_range: str | None = None
    is_abnormal: bool = False

    @field_validator("value", mode="before")
    @classmethod
    def normalize_lab_value(cls, value: object) -> str | None:
        if value is None or isinstance(value, str):
            return value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)
        return None

    @field_validator("is_abnormal", mode="before")
    @classmethod
    def normalize_abnormal_flag(cls, value: object) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return value.strip().casefold() in {"true", "yes", "abnormal", "1"}
        return bool(value)


class DiagnosticStudy(BaseModel):
    study_name: str = Field(
        validation_alias=AliasChoices("study_name", "name", "test_name", "study"),
    )
    result: str = Field(
        validation_alias=AliasChoices("result", "finding", "findings"),
    )
    is_abnormal: bool = False

    @field_validator("is_abnormal", mode="before")
    @classmethod
    def normalize_abnormal_flag(cls, value: object) -> bool:
        return LabResult.normalize_abnormal_flag(value)


class PatientInfo(BaseModel):
    """Core patient information for intake processing."""

    patient_id: str | None = None
    name: str = "Unknown"
    age: int | None = Field(default=None, ge=0, le=130)
    gender: Gender = Gender.UNKNOWN
    chief_complaint: str
    symptoms: list[Symptom] = Field(default_factory=list)
    medical_history: list[str] = Field(default_factory=list)
    surgical_history: list[str] = Field(default_factory=list)
    family_history: list[str] = Field(default_factory=list)
    social_history: list[str] = Field(default_factory=list)
    allergies: list[Allergy] = Field(default_factory=list)
    current_medications: list[Medication] = Field(default_factory=list)
    vital_signs: VitalSigns | None = None
    physical_exam: list[str] = Field(default_factory=list)
    lab_results: list[LabResult] = Field(default_factory=list)
    diagnostic_studies: list[DiagnosticStudy] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_nullable_collections(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        for field_name in (
            "symptoms",
            "medical_history",
            "surgical_history",
            "family_history",
            "social_history",
            "allergies",
            "current_medications",
            "physical_exam",
            "lab_results",
            "diagnostic_studies",
        ):
            if normalized.get(field_name) is None:
                normalized[field_name] = []
        return normalized

    @field_validator("gender", mode="before")
    @classmethod
    def normalize_gender(cls, value: object) -> Gender:
        if isinstance(value, Gender):
            return value
        if isinstance(value, str):
            normalized = value.strip().casefold()
            aliases = {
                "male": Gender.MALE,
                "man": Gender.MALE,
                "female": Gender.FEMALE,
                "woman": Gender.FEMALE,
                "other": Gender.OTHER,
            }
            if normalized in aliases:
                return aliases[normalized]
        return Gender.UNKNOWN

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

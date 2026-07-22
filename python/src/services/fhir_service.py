"""FHIR R4 export validated with fhir.resources and written to HAPI FHIR."""

from __future__ import annotations

from datetime import date, datetime, timezone
from functools import lru_cache
from typing import Any
from uuid import UUID, uuid4

import httpx
import structlog
from fhir.resources.R4B.bundle import Bundle

from ..config.settings import get_settings

logger = structlog.get_logger(__name__)


def patient_to_fhir(patient_info: dict[str, Any], patient_id: str = "") -> dict[str, Any]:
    """Convert a de-identified patient structure to a FHIR Patient resource."""

    gender = str(patient_info.get("gender", "unknown"))
    if gender not in {"male", "female", "other", "unknown"}:
        gender = "unknown"
    age = patient_info.get("age")
    resource: dict[str, Any] = {
        "resourceType": "Patient",
        "id": patient_id or f"synthetic-{uuid4().hex}",
        "gender": gender,
        "meta": {
            "tag": [
                {
                    "system": "https://medigen.local/fhir/tags",
                    "code": "synthetic-deidentified",
                    "display": "Synthetic or de-identified workspace data",
                }
            ]
        },
    }
    if isinstance(age, int) and 0 <= age <= 130:
        resource["birthDate"] = f"{date.today().year - age}-01-01"
    return resource


def diagnosis_to_fhir_condition(
    diagnosis: dict[str, Any],
    *,
    patient_reference: str,
    condition_id: str,
) -> dict[str, Any] | None:
    primary = diagnosis.get("primary_diagnosis")
    if not isinstance(primary, dict) or not primary.get("disease_name"):
        return None
    coding: list[dict[str, str]] = []
    if primary.get("icd10_hint"):
        coding.append(
            {
                "system": "http://hl7.org/fhir/sid/icd-10-cm",
                "code": str(primary["icd10_hint"]),
                "display": str(primary["disease_name"]),
            }
        )
    return {
        "resourceType": "Condition",
        "id": condition_id,
        "clinicalStatus": {
            "coding": [
                {
                    "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                    "code": "active",
                }
            ]
        },
        "verificationStatus": {
            "coding": [
                {
                    "system": "http://terminology.hl7.org/CodeSystem/condition-ver-status",
                    "code": "provisional",
                }
            ]
        },
        "subject": {"reference": patient_reference},
        "code": {"coding": coding, "text": str(primary["disease_name"])},
        "note": ([{"text": str(primary["reasoning"])}] if primary.get("reasoning") else []),
    }


def medication_to_fhir(
    medication: dict[str, Any],
    *,
    patient_reference: str,
    medication_request_id: str,
) -> dict[str, Any]:
    display = str(
        medication.get("generic_name")
        or medication.get("drug_name")
        or "Medication proposal"
    )
    dosage_parts = [
        str(medication.get(field, "")).strip()
        for field in ("dosage", "route", "frequency", "duration")
    ]
    dosage_text = " · ".join(part for part in dosage_parts if part)
    return {
        "resourceType": "MedicationRequest",
        "id": medication_request_id,
        "status": "active",
        "intent": "proposal",
        "subject": {"reference": patient_reference},
        "medicationCodeableConcept": {"text": display},
        "dosageInstruction": ([{"text": dosage_text}] if dosage_text else []),
    }


def build_transaction_bundle(
    *,
    session_id: UUID,
    patient_info: dict[str, Any],
    diagnosis: dict[str, Any] | None,
    treatment_plan: dict[str, Any] | None,
) -> dict[str, Any]:
    bundle_id = f"analysis-{session_id.hex}"
    patient_id = f"patient-{session_id.hex}"
    patient_full_url = f"urn:uuid:{uuid4()}"
    entries: list[dict[str, Any]] = [
        {
            "fullUrl": patient_full_url,
            "resource": patient_to_fhir(patient_info, patient_id),
            "request": {"method": "POST", "url": "Patient"},
        }
    ]
    condition = diagnosis_to_fhir_condition(
        diagnosis or {},
        patient_reference=patient_full_url,
        condition_id=f"condition-{session_id.hex}",
    )
    if condition:
        entries.append(
            {
                "fullUrl": f"urn:uuid:{uuid4()}",
                "resource": condition,
                "request": {"method": "POST", "url": "Condition"},
            }
        )
    for index, medication in enumerate((treatment_plan or {}).get("medications", []), start=1):
        if not isinstance(medication, dict):
            continue
        entries.append(
            {
                "fullUrl": f"urn:uuid:{uuid4()}",
                "resource": medication_to_fhir(
                    medication,
                    patient_reference=patient_full_url,
                    medication_request_id=f"medication-{index}-{session_id.hex}",
                ),
                "request": {"method": "POST", "url": "MedicationRequest"},
            }
        )
    raw_bundle = {
        "resourceType": "Bundle",
        "id": bundle_id,
        "type": "transaction",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "entry": entries,
    }
    validated = Bundle.model_validate(raw_bundle)
    return validated.model_dump(mode="json", by_alias=True, exclude_none=True)


class FHIRService:
    def __init__(self) -> None:
        settings = get_settings()
        self.base_url = settings.fhir_server_url.rstrip("/")
        self.client = httpx.Client(timeout=settings.fhir_timeout_seconds)

    def is_ready(self) -> bool:
        try:
            response = self.client.get(f"{self.base_url}/metadata")
            return (
                response.status_code == 200
                and response.json().get("resourceType") == "CapabilityStatement"
            )
        except (httpx.HTTPError, ValueError):
            return False

    def export_analysis(
        self,
        *,
        session_id: UUID,
        patient_info: dict[str, Any],
        diagnosis: dict[str, Any] | None,
        treatment_plan: dict[str, Any] | None,
    ) -> dict[str, Any]:
        bundle = build_transaction_bundle(
            session_id=session_id,
            patient_info=patient_info,
            diagnosis=diagnosis,
            treatment_plan=treatment_plan,
        )
        try:
            response = self.client.post(
                self.base_url,
                json=bundle,
                headers={"Accept": "application/fhir+json", "Content-Type": "application/fhir+json"},
            )
            response.raise_for_status()
            response_bundle = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            logger.error("fhir.export_failed", error_type=type(exc).__name__)
            raise RuntimeError("FHIR export failed") from exc

        locations = [
            entry.get("response", {}).get("location", "")
            for entry in response_bundle.get("entry", [])
            if isinstance(entry, dict) and entry.get("response", {}).get("location")
        ]
        resource_types = [
            entry.get("resource", {}).get("resourceType", "")
            for entry in bundle.get("entry", [])
            if isinstance(entry, dict)
        ]
        result = {
            "provider": "HAPI FHIR",
            "standard": "FHIR R4",
            "bundle_id": bundle["id"],
            "bundle_type": response_bundle.get("type", "transaction-response"),
            "resource_count": len(resource_types),
            "resource_types": resource_types,
            "resource_locations": locations,
            "validation": "fhir.resources R4B model + HAPI FHIR transaction",
        }
        logger.info(
            "fhir.export_success",
            bundle_id=result["bundle_id"],
            resource_count=result["resource_count"],
        )
        return result

    def close(self) -> None:
        self.client.close()


@lru_cache(maxsize=1)
def get_fhir_service() -> FHIRService:
    return FHIRService()


async def push_to_fhir_server(resource: dict[str, Any]) -> dict[str, Any] | None:
    """Backward-compatible single-resource helper backed by the configured server."""

    settings = get_settings()
    url = f"{settings.fhir_server_url.rstrip('/')}/{resource.get('resourceType', '')}"
    try:
        async with httpx.AsyncClient(timeout=settings.fhir_timeout_seconds) as client:
            response = await client.post(url, json=resource)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as exc:
        logger.warning("fhir.push_failed", error_type=type(exc).__name__)
        return None

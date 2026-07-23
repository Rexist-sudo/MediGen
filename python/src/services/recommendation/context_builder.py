"""Build a bounded recommendation context from structured clinical output."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

from ...models.recommendation import RecommendationContext


class RecommendationContextBuilder:
    """Extract only controlled structured fields; raw narratives are ignored."""

    MAX_VALUE_LENGTH = 128
    LIMITS = {
        "diagnosis_codes": 10,
        "diagnosis_terms": 20,
        "recommended_tests": 20,
        "medication_names": 20,
    }

    @staticmethod
    def _as_dict(value: object) -> dict:
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _dict_items(value: object) -> list[dict]:
        if not isinstance(value, list):
            return []
        return [item for item in value if isinstance(item, dict)]

    @classmethod
    def _clean_values(
        cls,
        values: Iterable[object],
        *,
        limit: int,
        code: bool = False,
    ) -> list[str]:
        cleaned_values: list[str] = []
        seen: set[str] = set()
        for value in values:
            if not isinstance(value, str):
                continue
            normalized = unicodedata.normalize("NFKC", value)
            normalized = re.sub(r"[\x00-\x1f\x7f|=<>]", " ", normalized)
            normalized = " ".join(normalized.strip().split())
            if code:
                normalized = normalized.replace(" ", "").upper()
            normalized = normalized[: cls.MAX_VALUE_LENGTH]
            key = normalized.casefold()
            if normalized and key not in seen:
                cleaned_values.append(normalized)
                seen.add(key)
            if len(cleaned_values) >= limit:
                break
        return cleaned_values

    def build(self, clinical_result: dict) -> RecommendationContext:
        if not isinstance(clinical_result, dict):
            clinical_result = {}
        diagnosis = self._as_dict(clinical_result.get("diagnosis"))
        primary = self._as_dict(diagnosis.get("primary_diagnosis"))
        differential = self._dict_items(diagnosis.get("differential_list"))
        coding = self._as_dict(clinical_result.get("coding_result"))
        primary_code = self._as_dict(coding.get("primary_icd10"))
        secondary_codes = self._dict_items(coding.get("secondary_icd10_codes"))
        treatment = self._as_dict(clinical_result.get("treatment_plan"))
        medications = self._dict_items(treatment.get("medications"))
        audit = self._as_dict(clinical_result.get("audit_result"))

        return RecommendationContext(
            diagnosis_terms=self._clean_values(
                [primary.get("disease_name")]
                + [item.get("disease_name") for item in differential],
                limit=self.LIMITS["diagnosis_terms"],
            ),
            diagnosis_codes=self._clean_values(
                [primary_code.get("code")]
                + [item.get("code") for item in secondary_codes],
                limit=self.LIMITS["diagnosis_codes"],
                code=True,
            ),
            recommended_tests=self._clean_values(
                diagnosis.get("recommended_tests", [])
                if isinstance(diagnosis.get("recommended_tests"), list)
                else [],
                limit=self.LIMITS["recommended_tests"],
            ),
            medication_names=self._clean_values(
                [
                    name
                    for medication in medications
                    for name in (
                        medication.get("generic_name"),
                        medication.get("drug_name"),
                    )
                ],
                limit=self.LIMITS["medication_names"],
            ),
            demo_safe=bool(audit.get("demo_safe", False)),
        )


def build_recommendation_context(clinical_result: dict) -> RecommendationContext:
    """Compatibility function retained for existing imports."""

    return RecommendationContextBuilder().build(clinical_result)


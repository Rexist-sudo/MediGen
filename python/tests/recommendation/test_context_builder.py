from __future__ import annotations

from src.services.recommendation.context_builder import RecommendationContextBuilder


def test_extracts_normalizes_bounds_and_ignores_raw_text() -> None:
    result = {
        "patient_description": "must never enter the recommendation context",
        "diagnosis": {
            "primary_diagnosis": {"disease_name": "  Type 2  Diabetes  "},
            "differential_list": [
                {"disease_name": "糖尿病"},
                {"disease_name": "Type 2 Diabetes"},
            ],
            "recommended_tests": [" HbA1c ", "HbA1c", "x" * 300],
        },
        "coding_result": {
            "primary_icd10": {"code": " e11.9 "},
            "secondary_icd10_codes": [{"code": " i50.9 "}],
        },
        "treatment_plan": {
            "medications": [
                {"generic_name": " metformin ", "drug_name": "Metformin"}
            ]
        },
        "audit_result": {"demo_safe": True},
    }

    context = RecommendationContextBuilder().build(result)

    assert context.diagnosis_codes == ["E11.9", "I50.9"]
    assert context.diagnosis_terms == ["Type 2 Diabetes", "糖尿病"]
    assert context.medication_names == ["metformin"]
    assert all(len(item) <= 128 for item in context.recommended_tests)
    assert "patient_description" not in context.model_dump()
    assert context.demo_safe is True


def test_missing_or_malformed_fields_are_safe() -> None:
    context = RecommendationContextBuilder().build({"diagnosis": "bad"})
    assert context.diagnosis_terms == []
    assert context.demo_safe is False


def test_all_lists_and_values_are_bounded_and_delimiters_are_removed() -> None:
    result = {
        "diagnosis": {
            "primary_diagnosis": {"disease_name": "dx|=<>\n" + "x" * 300},
            "differential_list": [
                {"disease_name": f"diagnosis-{index}"} for index in range(40)
            ],
            "recommended_tests": [f"test-{index}" for index in range(40)],
        },
        "coding_result": {
            "primary_icd10": {"code": " e 11.9 "},
            "secondary_icd10_codes": [
                {"code": f"z {index:02d}"} for index in range(20)
            ],
        },
        "treatment_plan": {
            "medications": [
                {"generic_name": f"medicine-{index}"} for index in range(30)
            ]
        },
        "audit_result": {"demo_safe": True},
    }
    context = RecommendationContextBuilder().build(result)
    assert len(context.diagnosis_terms) == 20
    assert len(context.diagnosis_codes) == 10
    assert len(context.recommended_tests) == 20
    assert len(context.medication_names) == 20
    assert context.diagnosis_codes[0] == "E11.9"
    serialized = str(context.model_dump())
    assert all(character not in serialized for character in "|=<>")
    assert max(map(len, context.diagnosis_terms)) <= 128

"""Central JSON client for DeepSeek and an explicit local fixture backend."""

from __future__ import annotations

import json
import re
import time
from functools import lru_cache
from typing import Callable, Protocol, TypeVar

import httpx
import structlog
from pydantic import BaseModel, ValidationError

from ..config.settings import Settings, get_settings

logger = structlog.get_logger(__name__)
T = TypeVar("T", bound=BaseModel)


class DeepSeekConfigurationError(RuntimeError):
    """DeepSeek credentials or authentication are unavailable."""


class DeepSeekRequestError(RuntimeError):
    """DeepSeek rejected a non-retryable request."""


class DeepSeekUpstreamError(RuntimeError):
    """DeepSeek or the network remained unavailable after bounded retries."""


class DeepSeekOutputError(RuntimeError):
    """DeepSeek returned unusable structured output after bounded retries."""


class JSONClient(Protocol):
    def invoke_json(
        self,
        *,
        task_name: str,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
        max_tokens: int | None = None,
    ) -> T: ...


def _clean_json_text(value: str) -> str:
    text = value.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _safe_validation_summary(exc: ValidationError) -> list[dict[str, str]]:
    """Return schema diagnostics without logging values or model output text."""

    summary: list[dict[str, str]] = []
    for item in exc.errors(include_input=False, include_url=False)[:8]:
        location = ".".join(str(part) for part in item.get("loc", ())) or "root"
        summary.append(
            {
                "location": location,
                "type": str(item.get("type", "validation_error")),
            }
        )
    return summary


def _repair_instruction(
    response_model: type[BaseModel],
    issues: list[dict[str, str]],
) -> str:
    issue_text = ", ".join(
        f"{item['location']} ({item['type']})" for item in issues
    ) or "root (invalid_json)"
    schema = json.dumps(
        response_model.model_json_schema(),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return (
        "Return a new corrected JSON object only. A prior attempt did not match "
        f"the contract at: {issue_text}. Preserve only facts present in the "
        "original input; use null or an empty list for genuinely missing optional "
        f"data. Required JSON Schema: {schema}"
    )


class DeepSeekJSONClient:
    """Synchronous DeepSeek client with validation and bounded retries."""

    RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}

    def __init__(
        self,
        settings: Settings,
        *,
        http_client: httpx.Client | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ):
        key = (
            settings.deepseek_api_key.get_secret_value()
            if settings.deepseek_api_key
            else ""
        ).strip()
        if not key:
            raise DeepSeekConfigurationError("DEEPSEEK_API_KEY is not configured")

        self.settings = settings
        self._sleeper = sleeper
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        if http_client is not None:
            http_client.headers.update(headers)
            self.client = http_client
        else:
            self.client = httpx.Client(
                base_url=settings.deepseek_base_url.rstrip("/"),
                headers=headers,
                timeout=httpx.Timeout(settings.deepseek_timeout_seconds),
            )

    def invoke_json(
        self,
        *,
        task_name: str,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
        max_tokens: int | None = None,
    ) -> T:
        base_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        payload = {
            "model": self.settings.deepseek_model,
            "messages": base_messages,
            "stream": False,
            "thinking": {"type": "disabled"},
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "max_tokens": max_tokens or self.settings.deepseek_max_tokens,
        }

        attempts = self.settings.deepseek_max_retries + 1
        last_error: Exception | None = None

        for attempt in range(attempts):
            started = time.monotonic()
            try:
                response = self.client.post("/chat/completions", json=payload)

                if response.status_code in {401, 403}:
                    raise DeepSeekConfigurationError(
                        f"DeepSeek authentication failed: {response.status_code}"
                    )
                if (
                    response.status_code in self.RETRYABLE_STATUS
                    or 500 <= response.status_code < 600
                ):
                    raise DeepSeekUpstreamError(
                        f"retryable DeepSeek status: {response.status_code}"
                    )
                if response.status_code >= 400:
                    raise DeepSeekRequestError(
                        f"DeepSeek request rejected: {response.status_code}"
                    )

                body = response.json()
                choice = body["choices"][0]
                if choice.get("finish_reason") == "length":
                    raise DeepSeekOutputError("DeepSeek JSON was truncated")

                content = choice.get("message", {}).get("content") or ""
                if not isinstance(content, str) or not content.strip():
                    raise DeepSeekOutputError("DeepSeek returned empty content")

                try:
                    parsed = json.loads(_clean_json_text(content))
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "deepseek.invalid_json",
                        task=task_name,
                        content_length=len(content),
                        error_position=exc.pos,
                    )
                    if attempt + 1 < attempts:
                        payload["messages"] = base_messages + [
                            {
                                "role": "user",
                                "content": _repair_instruction(response_model, []),
                            }
                        ]
                    raise

                try:
                    result = response_model.model_validate(parsed)
                except ValidationError as exc:
                    issues = _safe_validation_summary(exc)
                    logger.warning(
                        "deepseek.schema_invalid",
                        task=task_name,
                        error_count=exc.error_count(),
                        errors=issues,
                    )
                    if attempt + 1 < attempts:
                        payload["messages"] = base_messages + [
                            {
                                "role": "user",
                                "content": _repair_instruction(
                                    response_model,
                                    issues,
                                ),
                            }
                        ]
                    raise
                logger.info(
                    "deepseek.success",
                    task=task_name,
                    elapsed_ms=round((time.monotonic() - started) * 1000),
                    attempt=attempt + 1,
                )
                return result

            except (DeepSeekConfigurationError, DeepSeekRequestError):
                raise
            except (
                httpx.TimeoutException,
                httpx.TransportError,
                DeepSeekUpstreamError,
            ) as exc:
                last_error = DeepSeekUpstreamError(type(exc).__name__)
            except (
                json.JSONDecodeError,
                KeyError,
                IndexError,
                TypeError,
                AttributeError,
                ValidationError,
                DeepSeekOutputError,
            ) as exc:
                last_error = DeepSeekOutputError(type(exc).__name__)

            logger.warning(
                "deepseek.retry_or_fail",
                task=task_name,
                error_type=type(last_error).__name__,
                attempt=attempt + 1,
            )
            if attempt + 1 < attempts:
                self._sleeper(0.25 * (attempt + 1))

        if isinstance(last_error, (DeepSeekUpstreamError, DeepSeekOutputError)):
            raise last_error
        raise DeepSeekOutputError("DeepSeek failed without a classified error")


class FixtureJSONClient:
    """Deterministic synthetic responses for local wiring demonstrations.

    This backend performs no model or network call. It intentionally provides
    only enough fixed behavior to validate application integration.
    """

    def invoke_json(
        self,
        *,
        task_name: str,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
        max_tokens: int | None = None,
    ) -> T:
        del system_prompt, max_tokens
        builders = {
            "intake": self._intake,
            "diagnosis": self._diagnosis,
            "treatment": self._treatment,
            "coding": self._coding,
        }
        try:
            payload = builders[task_name](user_prompt)
            result = response_model.model_validate(payload)
        except KeyError as exc:
            raise DeepSeekOutputError("Unsupported fixture task") from exc
        except ValidationError as exc:
            raise DeepSeekOutputError("Invalid fixture output") from exc

        logger.info("fixture.success", task=task_name)
        return result

    @staticmethod
    def _intake(user_prompt: str) -> dict:
        raw = user_prompt.split("Patient narrative:", 1)[-1].strip()
        normalized = raw.casefold()
        age_match = re.search(
            r"\b(\d{1,3})(?:\s*[- ]?year(?:s)?(?:-old)?|\s*岁)",
            normalized,
        )
        age = int(age_match.group(1)) if age_match else None

        if re.search(r"\b(female|woman)\b|女性", normalized):
            gender = "female"
        elif re.search(r"\b(male|man)\b|男性", normalized):
            gender = "male"
        else:
            gender = "unknown"

        symptom_seeds = [
            ("increased thirst", "increased thirst"),
            ("thirst", "increased thirst"),
            ("口渴", "口渴"),
            ("fatigue", "fatigue"),
            ("乏力", "乏力"),
            ("fever", "fever"),
            ("发热", "发热"),
            ("cough", "cough"),
            ("咳嗽", "咳嗽"),
            ("chest pain", "chest pain"),
            ("胸痛", "胸痛"),
            ("abdominal pain", "abdominal pain"),
            ("腹痛", "腹痛"),
            ("shortness of breath", "shortness of breath"),
            ("呼吸困难", "呼吸困难"),
            ("edema", "edema"),
            ("水肿", "水肿"),
        ]
        symptom_names: list[str] = []
        for needle, label in symptom_seeds:
            if needle in normalized and label not in symptom_names:
                symptom_names.append(label)

        if not symptom_names:
            symptom_names = ["unspecified concern"]

        return {
            "patient_id": None,
            "name": "Unknown",
            "age": age,
            "gender": gender,
            "chief_complaint": ", ".join(symptom_names[:3]),
            "symptoms": [
                {
                    "name": name,
                    "duration_days": None,
                    "severity": "moderate",
                    "description": None,
                }
                for name in symptom_names
            ],
            "medical_history": [],
            "family_history": [],
            "allergies": [],
            "current_medications": [],
            "vital_signs": None,
            "lab_results": [],
        }

    @staticmethod
    def _diagnosis(user_prompt: str) -> dict:
        text = user_prompt.casefold()
        profiles = [
            (
                ("glucose", "hba1c", "increased thirst", "口渴", "糖尿病"),
                "Type 2 diabetes mellitus",
                "E11.9",
                ["HbA1c", "fasting glucose"],
            ),
            (
                ("fever", "cough", "infiltrate", "肺炎", "发热", "咳嗽"),
                "Pneumonia",
                "J18.9",
                ["chest X-ray", "oxygen saturation"],
            ),
            (
                ("troponin", "ecg", "myocardial", "心肌梗死"),
                "Myocardial infarction",
                "I21.9",
                ["ECG", "troponin"],
            ),
            (
                ("tsh", "free t4", "hypothyroid", "甲状腺功能减退"),
                "Hypothyroidism",
                "E03.9",
                ["TSH", "free T4"],
            ),
            (
                ("heart failure", "edema", "水肿", "心力衰竭"),
                "Heart failure",
                "I50.9",
                ["clinical assessment", "cardiac imaging"],
            ),
            (
                ("right lower abdominal", "appendicitis", "右下腹", "阑尾炎"),
                "Appendicitis",
                "K35.80",
                ["abdominal examination", "abdominal imaging"],
            ),
        ]
        for seeds, disease, code, tests in profiles:
            if any(seed in text for seed in seeds):
                return {
                    "primary_diagnosis": {
                        "disease_name": disease,
                        "icd10_hint": code,
                        "confidence": 0.72,
                        "evidence": ["synthetic fixture keyword match"],
                        "reasoning": "Deterministic fixture output; not clinical reasoning.",
                    },
                    "differential_list": [
                        {
                            "disease_name": "Alternative condition",
                            "icd10_hint": "",
                            "confidence": 0.2,
                            "evidence": ["fixture placeholder"],
                            "reasoning": "Fixture demonstration only.",
                        }
                    ],
                    "recommended_tests": tests,
                    "clinical_notes": "Fixture output only; not for medical use.",
                    "knowledge_sources": ["fixture_demo_only"],
                    "needs_more_info": False,
                    "information_gaps": [],
                }

        return {
            "primary_diagnosis": None,
            "differential_list": [],
            "recommended_tests": [],
            "clinical_notes": "The fixture has no matching synthetic scenario.",
            "knowledge_sources": ["fixture_demo_only"],
            "needs_more_info": True,
            "information_gaps": ["additional_synthetic_clinical_context_required"],
        }

    @staticmethod
    def _treatment(user_prompt: str) -> dict:
        disease = FixtureJSONClient._profile(user_prompt)[0]
        return {
            "diagnosis_addressed": disease,
            "medications": [],
            "drug_interactions": [],
            "non_drug_treatments": [
                "Seek evaluation from a qualified clinician; fixture provides no treatment."
            ],
            "lifestyle_recommendations": [],
            "follow_up_plan": "Fixture output only; follow a qualified clinician's advice.",
            "warnings": ["Not a prescription or clinically validated plan."],
            "evidence_references": ["fixture_demo_only"],
        }

    @staticmethod
    def _coding(user_prompt: str) -> dict:
        disease, code = FixtureJSONClient._profile(user_prompt)
        return {
            "primary_icd10": {
                "code": code,
                "description": f"Prototype code suggestion for {disease}",
                "confidence": 0.7,
                "category": "fixture",
            },
            "secondary_icd10_codes": [],
            "drg_group": None,
            "coding_notes": "Fixture output only; not validated medical coding.",
            "coding_confidence": 0.7,
        }

    @staticmethod
    def _profile(value: str) -> tuple[str, str]:
        text = value.casefold()
        profiles = [
            (("e11.9", "diabetes"), "Type 2 diabetes mellitus", "E11.9"),
            (("j18.9", "pneumonia"), "Pneumonia", "J18.9"),
            (("i21.9", "myocardial"), "Myocardial infarction", "I21.9"),
            (("e03.9", "hypothyroid"), "Hypothyroidism", "E03.9"),
            (("i50.9", "heart failure"), "Heart failure", "I50.9"),
            (("k35.80", "appendicitis"), "Appendicitis", "K35.80"),
        ]
        for seeds, disease, code in profiles:
            if any(seed in text for seed in seeds):
                return disease, code
        return "Unspecified synthetic condition", "R69"


@lru_cache
def get_deepseek_client() -> DeepSeekJSONClient:
    return DeepSeekJSONClient(get_settings())


@lru_cache
def get_json_client() -> JSONClient:
    settings = get_settings()
    if settings.llm_backend == "fixture":
        return FixtureJSONClient()
    return get_deepseek_client()

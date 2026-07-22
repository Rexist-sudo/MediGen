"""Local identifier scan and ephemeral pipeline audit node."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import structlog

from ..models.treatment import AuditRecord, AuditResult, ComplianceCheck
from ..services.phi_guard import find_obvious_identifiers

logger = structlog.get_logger(__name__)


def _record(action: str, detail: str, outcome: str = "success") -> AuditRecord:
    return AuditRecord(
        timestamp=datetime.now(timezone.utc).isoformat(),
        action=action,
        resource_type="pipeline_output",
        detail=detail,
        outcome=outcome,
    )


def _build_audit_result(state) -> AuditResult:
    sections = {
        name: getattr(state, name)
        for name in ("patient_info", "diagnosis", "treatment_plan", "coding_result")
        if getattr(state, name, None) is not None
    }
    serialized = json.dumps(sections, ensure_ascii=False, default=str)
    found = find_obvious_identifiers(serialized)
    demo_safe = not found

    checks = [
        ComplianceCheck(
            check_name="presidio_and_rule_identifier_scan",
            passed=demo_safe,
            detail=(
                "Presidio and local rules found no direct identifier category."
                if demo_safe
                else f"Detected {len(found)} direct identifier category or categories."
            ),
        ),
        ComplianceCheck(
            check_name="structured_section_coverage",
            passed=bool(sections),
            detail=f"Reviewed {len(sections)} structured pipeline sections.",
        ),
    ]
    recommendations: list[str] = []
    if found:
        recommendations.append("direct_identifier_review_required")

    return AuditResult(
        prototype_only=True,
        demo_safe=demo_safe,
        hipaa_compliant=False,
        compliance_checks=checks,
        phi_fields_found=found,
        audit_trail=[
            _record(
                "presidio_identifier_scan",
                f"Scanned {len(sections)} generated section(s).",
                "success" if demo_safe else "needs_review",
            )
        ],
        recommendations=recommendations,
        overall_risk_level=(
            "identifier_scan_clear"
            if demo_safe
            else "direct_identifiers_detected"
        ),
    )


def audit_agent(state) -> dict:
    """Audit locally and fail closed without destroying earlier results."""

    logger.info("audit.start")
    try:
        result = _build_audit_result(state)
    except Exception as exc:
        logger.warning("audit.unavailable", error_type=type(exc).__name__)
        fallback = AuditResult(
            prototype_only=True,
            demo_safe=False,
            hipaa_compliant=False,
            recommendations=["audit_service_error"],
            overall_risk_level="review_required",
        )
        return {
            "audit_result": fallback.model_dump(mode="json"),
            "current_agent": "audit",
            "errors": state.errors + [f"Audit failed: {type(exc).__name__}"],
        }

    logger.info("audit.success", demo_safe=result.demo_safe)
    return {
        "audit_result": result.model_dump(mode="json"),
        "current_agent": "audit",
    }

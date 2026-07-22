"""Local, deliberately limited prototype audit node."""

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
            check_name="limited_obvious_identifier_scan",
            passed=demo_safe,
            detail=(
                "No identifier matched the limited prototype rules."
                if demo_safe
                else f"Detected {len(found)} obvious identifier type(s)."
            ),
        ),
        ComplianceCheck(
            check_name="hipaa_compliance_assessment",
            passed=False,
            detail="Not implemented; this prototype cannot establish HIPAA compliance.",
        ),
        ComplianceCheck(
            check_name="production_security_controls",
            passed=False,
            detail="Encryption, RBAC, retention, and breach workflows are out of scope.",
        ),
    ]
    recommendations = ["prototype_audit_only_not_a_compliance_assessment"]
    if found:
        recommendations.insert(0, "obvious_identifier_detected_in_generated_output")

    return AuditResult(
        prototype_only=True,
        demo_safe=demo_safe,
        hipaa_compliant=False,
        compliance_checks=checks,
        phi_fields_found=found,
        audit_trail=[
            _record(
                "limited_identifier_scan",
                f"Scanned {len(sections)} generated section(s).",
                "success" if demo_safe else "needs_review",
            )
        ],
        recommendations=recommendations,
        overall_risk_level=(
            "limited_scan_no_obvious_identifiers"
            if demo_safe
            else "obvious_identifiers_detected"
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
            recommendations=["prototype_audit_unavailable"],
            overall_risk_level="unknown",
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

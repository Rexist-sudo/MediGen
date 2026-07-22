"""PostgreSQL persistence for clinical sessions and append-only audit records."""

from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from typing import Any
from uuid import UUID

import structlog
from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    func,
    insert,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from ..config.settings import get_settings

logger = structlog.get_logger(__name__)
metadata = MetaData()

audit_logs = Table(
    "audit_logs",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("timestamp", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("user_id", String(128), nullable=False, server_default="system"),
    Column("action", String(64), nullable=False),
    Column("resource_type", String(64), nullable=False),
    Column("resource_id", String(256), server_default=""),
    Column("detail", Text, server_default=""),
    Column("outcome", String(32), server_default="success"),
    Column("ip_address", String(45), server_default=""),
)

clinical_sessions = Table(
    "clinical_sessions",
    metadata,
    Column("id", PGUUID(as_uuid=True), primary_key=True),
    Column("thread_id", String(128), nullable=False),
    Column("raw_input", Text),
    Column("patient_info", JSONB),
    Column("diagnosis", JSONB),
    Column("treatment_plan", JSONB),
    Column("coding_result", JSONB),
    Column("audit_result", JSONB),
    Column("recommendation_result", JSONB),
    Column("knowledge_context", JSONB),
    Column("fhir_export", JSONB),
    Column("analysis_status", String(32)),
    Column("llm_backend", String(32)),
    Column("errors", JSONB, server_default=text("'[]'::jsonb")),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
)


class DatabaseService:
    def __init__(self) -> None:
        settings = get_settings()
        self.engine: Engine = create_engine(
            settings.sqlalchemy_database_uri,
            pool_pre_ping=True,
            pool_size=3,
            max_overflow=2,
            connect_args={"connect_timeout": 3},
        )
        self._initialized = False

    def initialize(self) -> None:
        if self._initialized:
            return
        metadata.create_all(self.engine)
        with self.engine.begin() as connection:
            for statement in (
                "ALTER TABLE clinical_sessions ADD COLUMN IF NOT EXISTS recommendation_result JSONB",
                "ALTER TABLE clinical_sessions ADD COLUMN IF NOT EXISTS knowledge_context JSONB",
                "ALTER TABLE clinical_sessions ADD COLUMN IF NOT EXISTS fhir_export JSONB",
                "ALTER TABLE clinical_sessions ADD COLUMN IF NOT EXISTS analysis_status VARCHAR(32)",
                "ALTER TABLE clinical_sessions ADD COLUMN IF NOT EXISTS llm_backend VARCHAR(32)",
                """
                CREATE OR REPLACE FUNCTION reject_audit_log_mutation()
                RETURNS trigger AS $$
                BEGIN
                    RAISE EXCEPTION 'audit_logs is append-only';
                END;
                $$ LANGUAGE plpgsql
                """,
                "DROP TRIGGER IF EXISTS audit_logs_append_only ON audit_logs",
                """
                CREATE TRIGGER audit_logs_append_only
                BEFORE UPDATE OR DELETE ON audit_logs
                FOR EACH ROW EXECUTE FUNCTION reject_audit_log_mutation()
                """,
            ):
                connection.execute(text(statement))
            connection.execute(text("SELECT 1"))
        self._initialized = True

    def is_ready(self) -> bool:
        try:
            self.initialize()
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            return True
        except SQLAlchemyError as exc:
            logger.warning("postgres.readiness_failed", error_type=type(exc).__name__)
            return False

    def save_analysis(
        self,
        *,
        session_id: UUID,
        raw_input: str,
        clinical_result: dict[str, Any],
        recommendation_result: dict[str, Any],
        fhir_export: dict[str, Any],
        analysis_status: str,
        llm_backend: str,
        client_ip: str,
    ) -> str:
        knowledge_context = (
            clinical_result.get("diagnosis", {}).get("knowledge_graph")
            if isinstance(clinical_result.get("diagnosis"), dict)
            else None
        )
        audit_result = clinical_result.get("audit_result") or {}
        audit_trail = audit_result.get("audit_trail", []) if isinstance(audit_result, dict) else []
        try:
            with self.engine.begin() as connection:
                connection.execute(
                    insert(clinical_sessions).values(
                        id=session_id,
                        thread_id=str(session_id),
                        raw_input=raw_input,
                        patient_info=clinical_result.get("patient_info"),
                        diagnosis=clinical_result.get("diagnosis"),
                        treatment_plan=clinical_result.get("treatment_plan"),
                        coding_result=clinical_result.get("coding_result"),
                        audit_result=clinical_result.get("audit_result"),
                        recommendation_result=recommendation_result,
                        knowledge_context=knowledge_context,
                        fhir_export=fhir_export,
                        analysis_status=analysis_status,
                        llm_backend=llm_backend,
                        errors=clinical_result.get("errors", []),
                    )
                )
                records: list[dict[str, Any]] = []
                for item in audit_trail:
                    if not isinstance(item, dict):
                        continue
                    timestamp = item.get("timestamp")
                    try:
                        parsed_timestamp = datetime.fromisoformat(timestamp) if timestamp else None
                    except (TypeError, ValueError):
                        parsed_timestamp = None
                    records.append(
                        {
                            "timestamp": parsed_timestamp or func.now(),
                            "user_id": "system",
                            "action": str(item.get("action", "pipeline_audit")),
                            "resource_type": str(item.get("resource_type", "pipeline_output")),
                            "resource_id": str(session_id),
                            "detail": str(item.get("detail", "")),
                            "outcome": str(item.get("outcome", "success")),
                            "ip_address": client_ip,
                        }
                    )
                records.append(
                    {
                        "timestamp": datetime.now(timezone.utc),
                        "user_id": "system",
                        "action": "analysis_persisted",
                        "resource_type": "clinical_session",
                        "resource_id": str(session_id),
                        "detail": "Structured analysis and FHIR export metadata persisted.",
                        "outcome": "success",
                        "ip_address": client_ip,
                    }
                )
                connection.execute(insert(audit_logs), records)
        except SQLAlchemyError as exc:
            logger.error("postgres.persist_failed", error_type=type(exc).__name__)
            raise RuntimeError("clinical session persistence failed") from exc
        logger.info("postgres.analysis_saved", session_id=str(session_id))
        return str(session_id)

    def counts(self) -> dict[str, int | str]:
        try:
            with self.engine.connect() as connection:
                sessions = connection.scalar(select(func.count()).select_from(clinical_sessions))
                audits = connection.scalar(select(func.count()).select_from(audit_logs))
            return {
                "provider": "PostgreSQL",
                "clinical_sessions": int(sessions or 0),
                "audit_records": int(audits or 0),
            }
        except SQLAlchemyError as exc:
            raise RuntimeError("postgres unavailable") from exc

    def close(self) -> None:
        self.engine.dispose()
        self._initialized = False


@lru_cache(maxsize=1)
def get_database_service() -> DatabaseService:
    return DatabaseService()

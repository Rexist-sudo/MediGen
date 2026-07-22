"""Lifecycle and readiness checks for the local runtime dependencies."""

from __future__ import annotations

import structlog

from ..config.settings import get_settings
from .database import get_database_service
from .fhir_service import get_fhir_service
from .graphrag_service import get_graphrag_service
from .phi_guard import presidio_is_ready
from .redis_service import get_redis_service

logger = structlog.get_logger(__name__)


def initialize_runtime_services() -> dict[str, bool]:
    settings = get_settings()
    if settings.llm_backend == "fixture":
        return {
            "postgresql": True,
            "neo4j": True,
            "redis": True,
            "fhir": True,
            "presidio": True,
        }
    statuses: dict[str, bool] = {}
    try:
        graph = get_graphrag_service()
        graph.initialize()
        statuses["neo4j"] = graph.is_ready()
    except Exception as exc:
        logger.warning("runtime.neo4j_initialization_failed", error_type=type(exc).__name__)
        statuses["neo4j"] = False
    statuses["redis"] = get_redis_service().is_ready()
    statuses["postgresql"] = get_database_service().is_ready()
    statuses["fhir"] = get_fhir_service().is_ready()
    statuses["presidio"] = presidio_is_ready()
    return statuses


def runtime_status() -> dict[str, bool]:
    settings = get_settings()
    if settings.llm_backend == "fixture":
        return {
            "postgresql": True,
            "neo4j": True,
            "redis": True,
            "fhir": True,
            "presidio": True,
        }
    return {
        "postgresql": get_database_service().is_ready(),
        "neo4j": get_graphrag_service().is_ready(),
        "redis": get_redis_service().is_ready(),
        "fhir": get_fhir_service().is_ready(),
        "presidio": presidio_is_ready(),
    }


def close_runtime_services() -> None:
    settings = get_settings()
    if settings.llm_backend == "fixture":
        return
    get_fhir_service().close()
    get_graphrag_service().close()
    get_redis_service().close()
    get_database_service().close()

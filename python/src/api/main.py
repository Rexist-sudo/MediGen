"""FastAPI entry point for the MediGen clinical analysis workspace."""

from contextlib import asynccontextmanager

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ..config.settings import get_settings
from ..services.recommendation import get_recommendation_service
from ..services.runtime_services import (
    close_runtime_services,
    initialize_runtime_services,
    runtime_status,
)
from .routes import router

settings = get_settings()
web_directory = Path(__file__).resolve().parents[1] / "web"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    initialize_runtime_services()
    yield
    close_runtime_services()

app = FastAPI(
    title="MediGen Clinical Analysis Service",
    description=(
        "Structured clinical analysis workflow for synthetic, de-identified "
        "cases. All results require review by qualified professionals."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)
app.include_router(router, prefix="/api/v1")
app.mount("/static", StaticFiles(directory=web_directory), name="static")


@app.get("/", include_in_schema=False)
def web_console() -> FileResponse:
    """Serve the same-origin local visualization for the prototype workflow."""

    return FileResponse(
        web_directory / "index.html",
        media_type="text/html; charset=utf-8",
    )


@app.get("/health")
def health_check() -> dict:
    """Process liveness only; no external model call is performed."""

    return {
        "status": "healthy",
        "service": "medigen-mvp",
        "version": "0.1.0",
        "prototype_only": True,
    }


@app.get("/ready")
def readiness_check() -> dict:
    """Report local configuration readiness without calling DeepSeek."""

    store_loaded = (
        not settings.recommendation_enabled
        or get_recommendation_service().is_store_ready()
    )
    llm_ready = settings.llm_backend == "fixture" or settings.deepseek_configured
    dependencies = runtime_status()
    infrastructure_ready = all(dependencies.values())
    return {
        "status": (
            "ready"
            if llm_ready and store_loaded and infrastructure_ready
            else "not_ready"
        ),
        "llm_backend": settings.llm_backend,
        "deepseek_configured": settings.deepseek_configured,
        "recommendation_store_loaded": store_loaded,
        "dependencies": dependencies,
        "prototype_only": True,
    }

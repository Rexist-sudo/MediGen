"""FastAPI entry point for the MediGen software prototype."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..config.settings import get_settings
from ..services.recommendation import get_recommendation_service
from .routes import router

settings = get_settings()

app = FastAPI(
    title="MediGen MVP Prototype",
    description=(
        "Prototype multi-agent clinical workflow for software architecture "
        "demonstration. Synthetic, de-identified inputs only. Not for medical use."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)
app.include_router(router, prefix="/api/v1")


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
    return {
        "status": "ready" if llm_ready and store_loaded else "not_ready",
        "llm_backend": settings.llm_backend,
        "deepseek_configured": settings.deepseek_configured,
        "recommendation_store_loaded": store_loaded,
        "prototype_only": True,
    }

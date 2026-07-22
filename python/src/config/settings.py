"""Application configuration loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the local MVP and optional demo services."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # DeepSeek is the default runtime. ``fixture`` is an explicit local-only
    # demonstration backend and must never be represented as a model call.
    llm_backend: Literal["deepseek", "fixture"] = "deepseek"
    deepseek_api_key: SecretStr | None = None
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-pro"
    deepseek_timeout_seconds: float = Field(default=90.0, gt=0, le=300)
    deepseek_max_retries: int = Field(default=1, ge=0, le=2)
    deepseek_max_tokens: int = Field(default=2048, ge=256, le=8192)

    # Recommendation MVP
    recommendation_enabled: bool = True
    recommendation_top_k: int = Field(default=3, ge=1, le=3)
    recommendation_topic_path: str = (
        "./data/recommendation/knowledge_topics.jsonl"
    )
    max_history_interactions: int = Field(default=20, ge=0, le=100)

    # Prototype input guard
    prototype_reject_obvious_phi: bool = True

    # PostgreSQL
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "clinical_decision"
    postgres_user: str = "postgres"
    postgres_password: str = ""

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379

    # FHIR
    fhir_server_url: str = "http://localhost:8080/fhir"

    # App
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"
    cors_origins: str = "http://localhost:3000,http://localhost:5173"

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def cors_origin_list(self) -> list[str]:
        return [
            item.strip()
            for item in self.cors_origins.split(",")
            if item.strip()
        ]

    @property
    def deepseek_configured(self) -> bool:
        if self.deepseek_api_key is None:
            return False
        return bool(self.deepseek_api_key.get_secret_value().strip())


@lru_cache
def get_settings() -> Settings:
    return Settings()

"""Redis-backed cache and fixed-window request limiter."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import redis
import structlog

from ..config.settings import get_settings

logger = structlog.get_logger(__name__)


class RedisService:
    def __init__(self) -> None:
        settings = get_settings()
        self.default_ttl = settings.redis_cache_ttl_seconds
        self.rate_limit = settings.rate_limit_requests_per_minute
        self.client = redis.Redis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
            health_check_interval=30,
        )

    def is_ready(self) -> bool:
        try:
            return bool(self.client.ping())
        except redis.RedisError as exc:
            logger.warning("redis.readiness_failed", error_type=type(exc).__name__)
            return False

    def get_json(self, key: str) -> Any | None:
        try:
            value = self.client.get(key)
            return json.loads(value) if value else None
        except (redis.RedisError, json.JSONDecodeError) as exc:
            logger.warning("redis.cache_read_failed", error_type=type(exc).__name__)
            return None

    def set_json(self, key: str, value: Any, *, ttl: int | None = None) -> bool:
        try:
            return bool(
                self.client.setex(
                    key,
                    ttl or self.default_ttl,
                    json.dumps(value, ensure_ascii=False, separators=(",", ":")),
                )
            )
        except (redis.RedisError, TypeError, ValueError) as exc:
            logger.warning("redis.cache_write_failed", error_type=type(exc).__name__)
            return False

    def allow_request(self, client_key: str) -> tuple[bool, int]:
        """Apply a one-minute fixed window and return allowance plus remaining quota."""

        key = f"medigen:rate:{client_key}"
        try:
            with self.client.pipeline(transaction=True) as pipe:
                pipe.incr(key)
                pipe.expire(key, 60, nx=True)
                count, _ = pipe.execute()
        except redis.RedisError as exc:
            logger.warning("redis.rate_limit_failed", error_type=type(exc).__name__)
            raise RuntimeError("redis rate limiter unavailable") from exc
        remaining = max(0, self.rate_limit - int(count))
        return int(count) <= self.rate_limit, remaining

    def info_summary(self) -> dict[str, Any]:
        try:
            info = self.client.info(section="server")
            return {
                "provider": "Redis",
                "version": info.get("redis_version", ""),
                "database_keys": self.client.dbsize(),
            }
        except redis.RedisError as exc:
            raise RuntimeError("redis unavailable") from exc

    def close(self) -> None:
        self.client.close()


@lru_cache(maxsize=1)
def get_redis_service() -> RedisService:
    return RedisService()

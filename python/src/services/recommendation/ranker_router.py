"""Select Mini-OneRec as primary and map controlled failures to rule fallback."""

from __future__ import annotations

from dataclasses import replace

from .ranker_protocol import (
    RankerInput,
    RankerResult,
    RecommendationRankerError,
)


FALLBACK_REASONS = {
    "model_disabled",
    "artifact_missing",
    "artifact_incompatible",
    "model_load_failed",
    "model_not_ready",
    "unsafe_context",
    "no_rankable_candidates",
    "inference_failed",
    "cuda_oom",
    "invalid_model_output",
    "concurrency_busy",
}


class RankerRouter:
    def __init__(self, *, primary, fallback, settings, model_loader):
        self._primary = primary
        self._fallback = fallback
        self._settings = settings
        self._model_loader = model_loader

    def _fallback_result(
        self,
        ranker_input: RankerInput,
        *,
        reason: str,
    ) -> RankerResult:
        if not self._settings.recommendation_rule_fallback_enabled:
            raise RecommendationRankerError(reason)
        result = self._fallback.rank(ranker_input)
        return replace(
            result,
            fallback_reason=(
                reason if reason in FALLBACK_REASONS else "inference_failed"
            ),
            warnings=tuple(dict.fromkeys((*result.warnings, "ranker_fallback"))),
        )

    def rank(self, ranker_input: RankerInput) -> RankerResult:
        strategy = self._settings.recommendation_ranker
        if strategy == "rule_v1" or not self._settings.minionerec_enabled:
            return self._fallback_result(ranker_input, reason="model_disabled")
        if not ranker_input.context.demo_safe:
            return self._fallback_result(ranker_input, reason="unsafe_context")
        if not ranker_input.candidates:
            return self._fallback_result(
                ranker_input,
                reason="no_rankable_candidates",
            )
        try:
            return self._primary.rank(ranker_input)
        except RecommendationRankerError as exc:
            reason = exc.code if exc.code in FALLBACK_REASONS else "inference_failed"
            return self._fallback_result(ranker_input, reason=reason)
        except Exception:
            return self._fallback_result(ranker_input, reason="inference_failed")

    def rank_fallback(
        self,
        ranker_input: RankerInput,
        *,
        reason: str,
    ) -> RankerResult:
        return self._fallback_result(ranker_input, reason=reason)

    def readiness(self, *, load: bool = False):
        return self._model_loader.readiness(load=load)


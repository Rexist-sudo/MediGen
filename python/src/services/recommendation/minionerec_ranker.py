"""History-aware iterative Top-K ranking with exact candidate-token scoring."""

from __future__ import annotations

import hashlib
from time import perf_counter

from .minionerec_decoder import DirectCandidateTokenScorer
from .minionerec_prompt import MiniOneRecPromptBuilder
from .ranker_protocol import (
    InvalidModelOutputError,
    ModelInferenceError,
    ModelNotReadyError,
    RankerInput,
    RankerResult,
    RecommendationRankerError,
)


class MiniOneRecRanker:
    STRATEGY = "mini_onerec_mvp"

    def __init__(
        self,
        *,
        model_loader,
        topic_store,
        inference_semaphore,
        semaphore_wait_seconds: float,
        max_input_tokens: int,
        decoder_factory=None,
    ):
        self._model_loader = model_loader
        self._topic_store = topic_store
        self._inference_semaphore = inference_semaphore
        self._semaphore_wait_seconds = semaphore_wait_seconds
        self._max_input_tokens = max_input_tokens
        self._decoder_factory = decoder_factory

    def rank(self, ranker_input: RankerInput) -> RankerResult:
        if not ranker_input.context.demo_safe:
            raise ModelNotReadyError("unsafe_context")
        if not ranker_input.candidates or ranker_input.top_k <= 0:
            raise ModelNotReadyError("no_rankable_candidates")
        acquired = self._inference_semaphore.acquire(
            timeout=self._semaphore_wait_seconds
        )
        if not acquired:
            raise ModelNotReadyError("concurrency_busy")

        started = perf_counter()
        try:
            loaded = self._model_loader.load()
            prompt_builder = MiniOneRecPromptBuilder(
                tokenizer=loaded.tokenizer,
                max_input_tokens=min(
                    self._max_input_tokens,
                    loaded.manifest.max_input_tokens,
                ),
            )
            scorer = (
                self._decoder_factory(loaded)
                if self._decoder_factory is not None
                else DirectCandidateTokenScorer(
                    loaded.model,
                    loaded.tokenizer,
                    loaded.device,
                    min(
                        self._max_input_tokens,
                        loaded.manifest.max_input_tokens,
                    ),
                )
            )
            remaining = {
                topic.topic_token: topic.topic_id
                for topic in ranker_input.candidates
            }
            selected_topic_ids: list[str] = []
            selected_topic_tokens: list[str] = []
            prompt_hashes: list[str] = []
            prompt_token_counts: list[int] = []
            for _ in range(min(ranker_input.top_k, len(remaining))):
                prompt = prompt_builder.build(
                    ranker_input=ranker_input,
                    selected_topic_tokens=selected_topic_tokens,
                    history_topic_tokens=self._topic_store.topic_id_to_token(),
                )
                prompt_hashes.append(
                    hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                )
                prompt_token_counts.append(prompt_builder.token_count(prompt))
                selected_token, _ = scorer.select_one(
                    prompt=prompt,
                    allowed_topic_tokens=sorted(remaining),
                )
                if selected_token not in remaining:
                    raise InvalidModelOutputError()
                selected_topic_ids.append(remaining.pop(selected_token))
                selected_topic_tokens.append(selected_token)

            candidate_ids = {topic.topic_id for topic in ranker_input.candidates}
            if len(selected_topic_ids) != len(set(selected_topic_ids)):
                raise InvalidModelOutputError()
            if not set(selected_topic_ids).issubset(candidate_ids):
                raise InvalidModelOutputError()
            if len(selected_topic_ids) > ranker_input.top_k:
                raise InvalidModelOutputError()
            return RankerResult(
                topic_ids=tuple(selected_topic_ids),
                strategy_used=self.STRATEGY,
                model_version=loaded.manifest.model_version,
                inference_ms=round((perf_counter() - started) * 1000, 3),
                diagnostics={
                    "prompt_sha256": tuple(prompt_hashes),
                    "prompt_token_count": tuple(prompt_token_counts),
                },
            )
        except RecommendationRankerError:
            raise
        except Exception as exc:
            if type(exc).__name__ == "OutOfMemoryError":
                self._model_loader.mark_failure("cuda_oom")
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                raise ModelInferenceError("cuda_oom") from exc
            raise ModelInferenceError() from exc
        finally:
            self._inference_semaphore.release()


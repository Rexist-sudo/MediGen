"""Exact next-token scoring restricted to the approved candidate IDs."""

from __future__ import annotations

from .ranker_protocol import ModelInferenceError


class DirectCandidateTokenScorer:
    def __init__(self, model, tokenizer, device: str, max_input_tokens: int):
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._max_input_tokens = max_input_tokens

    def select_one(
        self,
        *,
        prompt: str,
        allowed_topic_tokens: list[str],
    ) -> tuple[str, float]:
        if not allowed_topic_tokens:
            raise ModelInferenceError("empty_allowed_tokens")
        if not prompt.endswith("<NEXT_TOPIC>"):
            raise ModelInferenceError("prompt_tail_missing")
        token_ids: list[int] = []
        for token in allowed_topic_tokens:
            token_id = self._tokenizer.convert_tokens_to_ids(token)
            encoded_token = self._tokenizer.encode(
                token,
                add_special_tokens=False,
            )
            if token_id is None or token_id == self._tokenizer.unk_token_id:
                raise ModelInferenceError("unknown_candidate_token")
            if encoded_token != [token_id]:
                raise ModelInferenceError("candidate_token_not_single_id")
            token_ids.append(token_id)

        encoded = self._tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        if int(encoded["input_ids"].shape[-1]) > self._max_input_tokens:
            raise ModelInferenceError("prompt_too_long")

        import torch

        encoded = {key: value.to(self._device) for key, value in encoded.items()}
        with torch.inference_mode():
            outputs = self._model(**encoded, use_cache=False)
        next_logits = outputs.logits[0, -1, :]
        candidate_ids = torch.tensor(
            token_ids,
            dtype=torch.long,
            device=next_logits.device,
        )
        candidate_logits = next_logits.index_select(0, candidate_ids)
        selected_offset = int(torch.argmax(candidate_logits).item())
        return (
            allowed_topic_tokens[selected_offset],
            float(candidate_logits[selected_offset].item()),
        )


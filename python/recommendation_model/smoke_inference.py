"""Offline base-load and real adapter inference smoke tests for CPU or CUDA."""

# ruff: noqa: E402 -- standalone execution adds the source root before imports.

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from .common import load_config, resolve_path
except ImportError:
    from common import load_config, resolve_path  # type: ignore[no-redef]

from src.models.recommendation import (
    RecommendationContext,
    TopicInteraction,
    UserHistoryContext,
    UserPreferenceContext,
)
from src.services.recommendation.history_normalizer import HistoryNormalizer
from src.services.recommendation.minionerec_decoder import DirectCandidateTokenScorer
from src.services.recommendation.minionerec_prompt import MiniOneRecPromptBuilder
from src.services.recommendation.model_manifest import (
    ModelArtifactManifest,
    validate_artifact_layout,
)
from src.services.recommendation.ranker_protocol import RankerInput
from src.services.recommendation.topic_store import TopicStore


def _device_dtype(torch, requested: str):
    device = (
        "cuda"
        if requested == "auto" and torch.cuda.is_available()
        else "cpu"
        if requested == "auto"
        else requested
    )
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return device, torch.bfloat16, "bfloat16"
    if device == "cuda":
        return device, torch.float16, "float16"
    return device, torch.float32, "float32"


def _load_base(config, device_requested: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_path = resolve_path(config["model"]["local_base_model_path"])
    device, dtype, dtype_name = _device_dtype(torch, device_requested)
    tokenizer = AutoTokenizer.from_pretrained(
        base_path,
        local_files_only=True,
        trust_remote_code=False,
    )
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=dtype,
        device_map=None,
    )
    model.eval().to(device)
    encoded = tokenizer(
        "offline model load smoke",
        return_tensors="pt",
        add_special_tokens=False,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        output = model(**encoded, use_cache=False)
    if output.logits.ndim != 3:
        raise RuntimeError("base model forward shape is invalid")
    return {
        "status": "pass",
        "mode": "base-load",
        "device": device,
        "dtype": dtype_name,
        "load_and_forward_ms": round((time.perf_counter() - started) * 1000, 3),
        "vocabulary_size": int(output.logits.shape[-1]),
    }


def _rank_case(
    *,
    model,
    tokenizer,
    device: str,
    manifest: ModelArtifactManifest,
    store: TopicStore,
    history_context: UserHistoryContext | None,
) -> tuple[list[str], float]:
    context = RecommendationContext(
        diagnosis_codes=["E11.9"],
        diagnosis_terms=["type 2 diabetes"],
        recommended_tests=["HbA1c"],
        demo_safe=True,
    )
    preferences = UserPreferenceContext(
        preferred_categories=["test_explanation"],
        preferred_depth="beginner",
        preferred_format="bullet_points",
        max_reading_minutes=3,
    )
    candidate_ids = [
        "diabetes_basics",
        "hba1c_test_explanation",
        "follow_up_checklist",
    ]
    candidates = tuple(store.get_by_id(item) for item in candidate_ids)
    if any(item is None for item in candidates):
        raise RuntimeError("smoke candidate is absent")
    candidates = tuple(item for item in candidates if item is not None)
    history = HistoryNormalizer(store, manifest.max_history).normalize(
        history_context
    )
    ranker_input = RankerInput(
        context=context,
        preferences=preferences,
        history=history,
        candidates=candidates,
        already_selected_topic_ids=(),
        top_k=3,
    )
    prompt_builder = MiniOneRecPromptBuilder(
        tokenizer=tokenizer,
        max_input_tokens=manifest.max_input_tokens,
    )
    scorer = DirectCandidateTokenScorer(
        model,
        tokenizer,
        device,
        manifest.max_input_tokens,
    )
    remaining = {item.topic_token: item.topic_id for item in candidates}
    selected_tokens: list[str] = []
    selected_ids: list[str] = []
    started = time.perf_counter()
    while remaining:
        prompt = prompt_builder.build(
            ranker_input=ranker_input,
            selected_topic_tokens=selected_tokens,
            history_topic_tokens=store.topic_id_to_token(),
        )
        token, _ = scorer.select_one(
            prompt=prompt,
            allowed_topic_tokens=sorted(remaining),
        )
        selected_tokens.append(token)
        selected_ids.append(remaining.pop(token))
    return selected_ids, (time.perf_counter() - started) * 1000


def _artifact_smoke(config, artifact: Path, device_requested: str) -> dict:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    store = TopicStore.from_jsonl(config["tokens"]["catalog_path"])
    base_path = resolve_path(config["model"]["local_base_model_path"])
    manifest = validate_artifact_layout(
        artifact_path=artifact,
        base_model_path=base_path,
        topic_store=store,
        expected_model_version=config["artifact"]["model_version"],
        max_input_tokens=int(config["model"]["max_input_tokens"]),
        max_history=20,
        max_candidates=20,
    )
    device, dtype, dtype_name = _device_dtype(torch, device_requested)
    tokenizer = AutoTokenizer.from_pretrained(
        artifact / "tokenizer",
        local_files_only=True,
        trust_remote_code=False,
    )
    store.validate_tokenizer(tokenizer)
    load_started = time.perf_counter()
    base = AutoModelForCausalLM.from_pretrained(
        base_path,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=dtype,
        device_map=None,
    )
    base.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(
        base,
        artifact / "adapter",
        is_trainable=False,
    )
    model.eval().to(device)
    load_ms = (time.perf_counter() - load_started) * 1000
    cold_ids, cold_ms = _rank_case(
        model=model,
        tokenizer=tokenizer,
        device=device,
        manifest=manifest,
        store=store,
        history_context=None,
    )
    history_ids, history_ms = _rank_case(
        model=model,
        tokenizer=tokenizer,
        device=device,
        manifest=manifest,
        store=store,
        history_context=UserHistoryContext(
            interactions=[
                TopicInteraction(
                    topic_id="chest_xray_explanation",
                    event_type="helpful",
                )
            ]
        ),
    )
    valid_ids = set(store.topic_id_to_token())
    if not set(cold_ids + history_ids).issubset(valid_ids):
        raise RuntimeError("smoke inference returned an unknown topic")
    report = {
        "status": "pass",
        "mode": "artifact",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "model_version": manifest.model_version,
        "device": device,
        "dtype": dtype_name,
        "offline": True,
        "load_ms": round(load_ms, 3),
        "cold_start": {
            "topic_ids": cold_ids,
            "inference_ms": round(cold_ms, 3),
        },
        "history_case": {
            "topic_ids": history_ids,
            "inference_ms": round(history_ms, 3),
        },
    }
    (artifact / f"smoke_report_{device}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="recommendation_model/config.yaml")
    parser.add_argument("--mode", choices=["base-load", "artifact"], default="artifact")
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    config = load_config(args.config)
    if args.mode == "base-load":
        report = _load_base(config, args.device)
    else:
        artifact = (
            resolve_path(args.artifact)
            if args.artifact
            else resolve_path(config["artifact"]["output_dir"])
        )
        report = _artifact_smoke(config, artifact, args.device)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

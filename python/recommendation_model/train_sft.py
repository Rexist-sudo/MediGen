"""LoRA SFT for candidate-constrained next-topic and topic-alignment tasks."""

# ruff: noqa: E402 -- standalone execution adds the source root before imports.

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from .common import load_config, read_jsonl, resolve_path
except ImportError:
    from common import load_config, read_jsonl, resolve_path  # type: ignore[no-redef]

from src.services.recommendation.ranker_protocol import CONTROL_TOKENS
from src.services.recommendation.topic_store import TopicStore


class SFTDataset:
    def __init__(self, rows, tokenizer, max_input_tokens: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_input_tokens = max_input_tokens

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        row = self.rows[index]
        prompt_ids = self.tokenizer.encode(
            row["prompt"],
            add_special_tokens=False,
        )
        target_id = self.tokenizer.convert_tokens_to_ids(
            row["target_topic_token"]
        )
        if target_id == self.tokenizer.unk_token_id:
            raise ValueError("target topic token is unknown")
        if self.tokenizer.encode(
            row["target_topic_token"],
            add_special_tokens=False,
        ) != [target_id]:
            raise ValueError("target topic token is not a single ID")
        target_ids = [target_id]
        if self.tokenizer.eos_token_id is not None:
            target_ids.append(self.tokenizer.eos_token_id)
        if len(prompt_ids) + len(target_ids) > self.max_input_tokens:
            raise ValueError("training sample exceeds max_input_tokens")
        return {
            "input_ids": [*prompt_ids, *target_ids],
            "attention_mask": [1] * (len(prompt_ids) + len(target_ids)),
            "labels": [-100] * len(prompt_ids) + target_ids,
        }


@dataclass
class CausalTopicCollator:
    pad_token_id: int

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        max_length = max(len(item["input_ids"]) for item in features)
        input_ids: list[list[int]] = []
        attention_masks: list[list[int]] = []
        labels: list[list[int]] = []
        for item in features:
            padding = max_length - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.pad_token_id] * padding)
            attention_masks.append(item["attention_mask"] + [0] * padding)
            labels.append(item["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _precision(config: dict[str, Any]) -> tuple[object, bool, bool, str]:
    import torch

    if torch.cuda.is_available() and config["precision"]["prefer_bf16"]:
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16, True, False, "bfloat16"
    if torch.cuda.is_available() and config["precision"]["allow_fp16"]:
        return torch.float16, False, True, "float16"
    return torch.float32, False, False, "float32"


def train(
    config_path: str | Path,
    *,
    max_steps_override: int | None = None,
    resume_from_checkpoint: str | None = None,
    output_dir_override: str | Path | None = None,
) -> dict[str, Any]:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    config = load_config(config_path)
    seed = int(config["seed"])
    _seed_everything(seed)

    import peft
    import torch
    import transformers
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    store = TopicStore.from_jsonl(config["tokens"]["catalog_path"])
    data_manifest_path = resolve_path(config["data"]["manifest_path"])
    data_manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
    if data_manifest["catalog_sha256"] != store.catalog_sha256():
        raise ValueError("dataset and catalog hashes differ")
    base_path = resolve_path(config["model"]["local_base_model_path"])
    base_manifest = json.loads(
        (base_path / "base_model_manifest.json").read_text(encoding="utf-8")
    )
    if (
        base_manifest["resolved_revision"]
        != config["model"]["base_model_revision"]
    ):
        raise ValueError("base model revision differs from training config")

    tokenizer = AutoTokenizer.from_pretrained(
        base_path,
        local_files_only=True,
        trust_remote_code=False,
    )
    special_tokens = [
        *[topic.topic_token for topic in store.list_all()],
        *CONTROL_TOKENS,
    ]
    tokenizer.add_special_tokens(
        {"additional_special_tokens": list(dict.fromkeys(special_tokens))}
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    store.validate_tokenizer(tokenizer)

    torch_dtype, bf16, fp16, dtype_name = _precision(config)
    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch_dtype,
        device_map=None,
    )
    model.resize_token_embeddings(len(tokenizer))
    model.config.use_cache = False
    lora = config["lora"]
    lora_config = LoraConfig(
        r=int(lora["rank"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),
        target_modules=list(lora["target_modules"]),
        modules_to_save=list(lora["modules_to_save"]),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    if config["training"]["gradient_checkpointing"]:
        model.enable_input_require_grads()
    trainable = [
        (name, parameter.numel())
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    trainable_count = sum(count for _, count in trainable)
    lora_count = sum(count for name, count in trainable if "lora_" in name)
    embedding_saved = any(
        "embed_tokens" in name or "lm_head" in name for name, _ in trainable
    )
    if trainable_count <= 0 or lora_count <= 0 or not embedding_saved:
        raise RuntimeError("trainable LoRA and token-output parameters are required")
    print(
        json.dumps(
            {
                "trainable_parameters": trainable_count,
                "lora_parameters": lora_count,
                "embedding_or_lm_head_trainable": embedding_saved,
                "dtype": dtype_name,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    train_rows = read_jsonl(resolve_path(config["data"]["train_path"]))
    validation_rows = read_jsonl(
        resolve_path(config["data"]["validation_path"])
    )
    max_input_tokens = int(config["model"]["max_input_tokens"])
    train_dataset = SFTDataset(train_rows, tokenizer, max_input_tokens)
    validation_dataset = SFTDataset(
        validation_rows,
        tokenizer,
        max_input_tokens,
    )
    collator = CausalTopicCollator(tokenizer.pad_token_id)
    smoke_batch = collator([train_dataset[0], train_dataset[1]])
    if smoke_batch["input_ids"].shape != smoke_batch["labels"].shape:
        raise RuntimeError("input and label shapes differ")
    if int((smoke_batch["labels"] != -100).sum()) < 2:
        raise RuntimeError("label masking removed every target")

    training = config["training"]
    output_dir = (
        resolve_path(output_dir_override)
        if output_dir_override is not None
        else resolve_path(training["output_dir"])
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    max_steps = (
        int(max_steps_override)
        if max_steps_override is not None
        else int(training.get("max_steps", -1))
    )
    arguments = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=float(training["num_train_epochs"]),
        max_steps=max_steps,
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        warmup_ratio=float(training["warmup_ratio"]),
        per_device_train_batch_size=int(training["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(training["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(
            training["gradient_accumulation_steps"]
        ),
        gradient_checkpointing=bool(training["gradient_checkpointing"]),
        logging_steps=int(training["logging_steps"]),
        eval_strategy=str(training["eval_strategy"]),
        save_strategy=str(training["save_strategy"]),
        save_total_limit=int(training["save_total_limit"]),
        load_best_model_at_end=bool(training["load_best_model_at_end"]),
        metric_for_best_model=str(training["metric_for_best_model"]),
        greater_is_better=bool(training["greater_is_better"]),
        bf16=bf16,
        fp16=fp16,
        report_to=[],
        label_names=["labels"],
        remove_unused_columns=False,
        dataloader_num_workers=0,
        seed=seed,
        data_seed=seed,
        save_safetensors=True,
        optim="adamw_torch",
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=collator,
    )
    started = time.perf_counter()
    train_result = trainer.train(
        resume_from_checkpoint=resume_from_checkpoint or None
    )
    elapsed = time.perf_counter() - started
    final_adapter = output_dir / "final_adapter"
    trainer.save_model(str(final_adapter))
    tokenizer.save_pretrained(output_dir / "tokenizer")
    resolved_config = json.loads(json.dumps(config))
    resolved_config["model"]["resolved_base_model_revision"] = base_manifest[
        "resolved_revision"
    ]
    (output_dir / "resolved_config.json").write_text(
        json.dumps(resolved_config, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    summary = {
        "schema_version": 1,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed, 3),
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "dataset_sha256": data_manifest["dataset_sha256"],
        "base_model_revision": base_manifest["resolved_revision"],
        "trainable_parameters": trainable_count,
        "lora_parameters": lora_count,
        "embedding_or_lm_head_trainable": embedding_saved,
        "device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        ),
        "dtype": dtype_name,
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "peft_version": peft.__version__,
        "train_metrics": train_result.metrics,
        "final_adapter": str(final_adapter.relative_to(ROOT)),
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="recommendation_model/config.yaml")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    train(
        args.config,
        max_steps_override=args.max_steps,
        resume_from_checkpoint=args.resume_from_checkpoint,
        output_dir_override=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

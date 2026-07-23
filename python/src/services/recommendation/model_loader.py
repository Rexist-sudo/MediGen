"""Thread-safe offline loader for the local Direct-SID LoRA artifact."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .model_manifest import (
    ModelArtifactManifest,
    validate_artifact_layout,
)
from .ranker_protocol import (
    ModelArtifactError,
    ModelArtifactMissingError,
    ModelLoadError,
    ModelNotReadyError,
)
from .topic_store import TopicStore


def _resolve_from_python_root(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (Path(__file__).resolve().parents[3] / path).resolve()


@dataclass(frozen=True)
class ModelReadiness:
    configured: bool
    artifact_valid: bool
    loaded: bool
    device: str | None
    dtype: str | None
    model_version: str | None
    last_failure_code: str | None

    def model_dump(self) -> dict[str, object]:
        return {
            "configured": self.configured,
            "artifact_valid": self.artifact_valid,
            "loaded": self.loaded,
            "device": self.device,
            "dtype": self.dtype,
            "model_version": self.model_version,
            "last_failure_code": self.last_failure_code,
        }


@dataclass(frozen=True)
class LoadedModel:
    model: object
    tokenizer: object
    device: str
    dtype: str
    manifest: ModelArtifactManifest


class MiniOneRecModelLoader:
    def __init__(self, *, settings, topic_store: TopicStore):
        self._settings = settings
        self._topic_store = topic_store
        self._lock = threading.Lock()
        self._loaded: LoadedModel | None = None
        self._manifest: ModelArtifactManifest | None = None
        self._artifact_valid = False
        self._validated = False
        self._last_failure_code: str | None = None
        self._last_failure_time: float | None = None

    @property
    def artifact_path(self) -> Path:
        return _resolve_from_python_root(self._settings.minionerec_artifact_path)

    @property
    def base_model_path(self) -> Path:
        return _resolve_from_python_root(self._settings.minionerec_base_model_path)

    def _in_cooldown(self) -> bool:
        if self._last_failure_time is None:
            return False
        elapsed = time.monotonic() - self._last_failure_time
        return elapsed < self._settings.minionerec_retry_cooldown_seconds

    def mark_failure(self, code: str) -> None:
        self._last_failure_code = code
        self._last_failure_time = time.monotonic()

    def validate_artifact(self, *, force: bool = False) -> ModelArtifactManifest:
        if not self._settings.minionerec_enabled:
            raise ModelNotReadyError("model_disabled")
        with self._lock:
            if self._validated and not force:
                if self._manifest is not None and self._artifact_valid:
                    return self._manifest
                if self._last_failure_code == "artifact_missing":
                    raise ModelArtifactMissingError()
                raise ModelArtifactError()
            try:
                manifest = validate_artifact_layout(
                    artifact_path=self.artifact_path,
                    base_model_path=self.base_model_path,
                    topic_store=self._topic_store,
                    expected_model_version=self._settings.minionerec_model_version,
                    max_input_tokens=self._settings.minionerec_max_input_tokens,
                    max_history=self._settings.minionerec_max_history,
                    max_candidates=self._settings.minionerec_max_candidates,
                )
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(
                    self.artifact_path / "tokenizer",
                    local_files_only=True,
                    trust_remote_code=False,
                )
                self._topic_store.validate_tokenizer(tokenizer)
                self._manifest = manifest
                self._artifact_valid = True
                self._validated = True
                self._last_failure_code = None
                return manifest
            except ModelArtifactMissingError:
                self._validated = True
                self._artifact_valid = False
                self.mark_failure("artifact_missing")
                raise
            except ModelArtifactError:
                self._validated = True
                self._artifact_valid = False
                self.mark_failure("artifact_incompatible")
                raise
            except Exception as exc:
                self._validated = True
                self._artifact_valid = False
                self.mark_failure("model_load_failed")
                raise ModelLoadError() from exc

    def _resolve_device_and_dtype(self, torch) -> tuple[str, object, str]:
        configured_device = self._settings.minionerec_device
        if configured_device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif self._settings.minionerec_allow_cpu:
                device = "cpu"
            else:
                raise ModelLoadError()
        else:
            device = configured_device
        if device == "cuda" and not torch.cuda.is_available():
            raise ModelLoadError()
        if device == "cpu" and not self._settings.minionerec_allow_cpu:
            raise ModelLoadError()

        configured_dtype = self._settings.minionerec_dtype
        if configured_dtype == "auto":
            if device == "cuda" and torch.cuda.is_bf16_supported():
                dtype_name = "bfloat16"
            elif device == "cuda":
                dtype_name = "float16"
            else:
                dtype_name = "float32"
        else:
            dtype_name = configured_dtype
        if device == "cpu" and dtype_name == "float16":
            dtype_name = "float32"
        return device, getattr(torch, dtype_name), dtype_name

    def load(self) -> LoadedModel:
        if self._loaded is not None:
            return self._loaded
        if self._in_cooldown() and self._last_failure_code not in {
            None,
            "artifact_missing",
            "artifact_incompatible",
        }:
            raise ModelNotReadyError()
        with self._lock:
            if self._loaded is not None:
                return self._loaded
        # Validation has its own lock and is intentionally done before the
        # exclusive load section to keep the critical section straightforward.
        manifest = self.validate_artifact()
        with self._lock:
            if self._loaded is not None:
                return self._loaded
            started = time.perf_counter()
            try:
                import torch
                from peft import PeftModel
                from transformers import AutoModelForCausalLM, AutoTokenizer

                device, torch_dtype, dtype_name = self._resolve_device_and_dtype(
                    torch
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    self.artifact_path / "tokenizer",
                    local_files_only=True,
                    trust_remote_code=False,
                )
                self._topic_store.validate_tokenizer(tokenizer)
                base_model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_path,
                    local_files_only=True,
                    trust_remote_code=False,
                    torch_dtype=torch_dtype,
                    device_map=None,
                )
                base_model.resize_token_embeddings(len(tokenizer))
                model = PeftModel.from_pretrained(
                    base_model,
                    self.artifact_path / "adapter",
                    is_trainable=False,
                )
                model.eval()
                model.to(device)
                torch.set_grad_enabled(False)
                smoke = tokenizer(
                    "<REC>\nHISTORY=<NO_HISTORY>\nNEXT=\n</REC>\n<NEXT_TOPIC>",
                    return_tensors="pt",
                    add_special_tokens=False,
                )
                smoke = {key: value.to(device) for key, value in smoke.items()}
                with torch.inference_mode():
                    output = model(**smoke, use_cache=False)
                if output.logits.ndim != 3:
                    raise ModelLoadError()
                self._loaded = LoadedModel(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    dtype=dtype_name,
                    manifest=manifest,
                )
                self._last_failure_code = None
                self._last_failure_time = None
                _ = time.perf_counter() - started
                return self._loaded
            except ModelLoadError:
                self.mark_failure("model_load_failed")
                raise
            except Exception as exc:
                if type(exc).__name__ == "OutOfMemoryError":
                    try:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                    self.mark_failure("cuda_oom")
                    raise ModelLoadError("cuda_oom") from exc
                self.mark_failure("model_load_failed")
                raise ModelLoadError() from exc

    def readiness(self, *, load: bool = False) -> ModelReadiness:
        configured = bool(self._settings.minionerec_enabled)
        if configured:
            try:
                self.validate_artifact()
                if load:
                    self.load()
            except Exception:
                pass
        loaded = self._loaded
        return ModelReadiness(
            configured=configured,
            artifact_valid=self._artifact_valid,
            loaded=loaded is not None,
            device=loaded.device if loaded else None,
            dtype=loaded.dtype if loaded else None,
            model_version=(
                loaded.manifest.model_version
                if loaded
                else self._manifest.model_version
                if self._manifest
                else None
            ),
            last_failure_code=self._last_failure_code,
        )


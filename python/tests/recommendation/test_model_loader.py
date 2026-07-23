from __future__ import annotations

import contextlib
import sys
import threading
from types import SimpleNamespace

import pytest

from src.services.recommendation.model_loader import MiniOneRecModelLoader
from src.services.recommendation.ranker_protocol import (
    ModelLoadError,
    ModelNotReadyError,
)


class _Tensor:
    def to(self, _device):
        return self


class _Tokenizer:
    calls = []
    unk_token_id = 0

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        cls.calls.append((path, kwargs))
        return cls()

    def convert_tokens_to_ids(self, token):
        return abs(hash(token)) % 10000 + 1

    def encode(self, token, *, add_special_tokens=False):
        return [self.convert_tokens_to_ids(token)]

    def __call__(self, _text, **_kwargs):
        return {"input_ids": _Tensor(), "attention_mask": _Tensor()}

    def __len__(self):
        return 160000


class _BaseModel:
    calls = []

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        cls.calls.append((path, kwargs))
        return cls()

    def resize_token_embeddings(self, _size):
        return None


class _LoadedModel:
    def eval(self):
        return self

    def to(self, _device):
        return self

    def __call__(self, **_kwargs):
        return SimpleNamespace(logits=SimpleNamespace(ndim=3))


class _PeftModel:
    calls = 0
    fail = False

    @classmethod
    def from_pretrained(cls, _base, _path, **_kwargs):
        cls.calls += 1
        if cls.fail:
            raise RuntimeError("injected load error")
        return _LoadedModel()


def _install_fake_model_modules(monkeypatch) -> None:
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: False,
            is_bf16_supported=lambda: False,
            empty_cache=lambda: None,
        ),
        float32=object(),
        float16=object(),
        bfloat16=object(),
        inference_mode=contextlib.nullcontext,
        set_grad_enabled=lambda _enabled: None,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoModelForCausalLM=_BaseModel,
            AutoTokenizer=_Tokenizer,
        ),
    )
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(PeftModel=_PeftModel))


def _settings(tmp_path):
    return SimpleNamespace(
        minionerec_enabled=True,
        minionerec_artifact_path=str(tmp_path / "artifact"),
        minionerec_base_model_path=str(tmp_path / "base"),
        minionerec_model_version="v1",
        minionerec_max_input_tokens=1024,
        minionerec_max_history=20,
        minionerec_max_candidates=20,
        minionerec_retry_cooldown_seconds=60,
        minionerec_device="cpu",
        minionerec_allow_cpu=True,
        minionerec_dtype="auto",
    )


def _manifest():
    return SimpleNamespace(model_version="v1", max_input_tokens=1024)


def test_loader_uses_offline_flags_and_loads_once_across_threads(
    monkeypatch,
    tmp_path,
    topic_store,
) -> None:
    _Tokenizer.calls = []
    _BaseModel.calls = []
    _PeftModel.calls = 0
    _PeftModel.fail = False
    _install_fake_model_modules(monkeypatch)
    loader = MiniOneRecModelLoader(
        settings=_settings(tmp_path),
        topic_store=topic_store,
    )
    monkeypatch.setattr(loader, "validate_artifact", lambda: _manifest())

    results = []
    failures = []

    def load() -> None:
        try:
            results.append(loader.load())
        except Exception as exc:  # pragma: no cover - asserted below
            failures.append(exc)

    threads = [threading.Thread(target=load) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert not failures
    assert len(results) == 6
    assert len({id(result) for result in results}) == 1
    assert _PeftModel.calls == 1
    assert _Tokenizer.calls[0][1]["local_files_only"] is True
    assert _Tokenizer.calls[0][1]["trust_remote_code"] is False
    assert _BaseModel.calls[0][1]["local_files_only"] is True
    assert _BaseModel.calls[0][1]["trust_remote_code"] is False


def test_load_failure_enters_cooldown(monkeypatch, tmp_path, topic_store) -> None:
    _PeftModel.calls = 0
    _PeftModel.fail = True
    _install_fake_model_modules(monkeypatch)
    loader = MiniOneRecModelLoader(
        settings=_settings(tmp_path),
        topic_store=topic_store,
    )
    monkeypatch.setattr(loader, "validate_artifact", lambda: _manifest())

    with pytest.raises(ModelLoadError, match="model_load_failed"):
        loader.load()
    with pytest.raises(ModelNotReadyError, match="model_not_ready"):
        loader.load()
    assert _PeftModel.calls == 1

"""Resolve, download, hash, and offline-verify the pinned Qwen base snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

try:
    from .common import resolve_path, sha256_file
except ImportError:
    from common import resolve_path, sha256_file  # type: ignore[no-redef]


MANIFEST_NAME = "base_model_manifest.json"


def inventory(path: Path) -> dict[str, dict[str, object]]:
    return {
        item.relative_to(path).as_posix(): {
            "sha256": sha256_file(item),
            "size": item.stat().st_size,
        }
        for item in sorted(path.rglob("*"))
        if item.is_file()
        and item.name != MANIFEST_NAME
        and ".cache" not in item.relative_to(path).parts
    }


def verify(path: Path) -> dict:
    manifest_path = path / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError("base model manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual = inventory(path)
    expected = manifest.get("files")
    if actual != expected:
        raise ValueError("base model file hashes do not match the manifest")
    if not any(name.endswith(".safetensors") for name in actual):
        raise ValueError("base model weights are missing")
    return manifest


def offline_load_smoke(path: Path) -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoConfig, AutoTokenizer

    AutoConfig.from_pretrained(
        path,
        local_files_only=True,
        trust_remote_code=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        path,
        local_files_only=True,
        trust_remote_code=False,
    )
    encoded = tokenizer("offline base snapshot", add_special_tokens=False)
    if not encoded.input_ids:
        raise ValueError("base tokenizer smoke produced no tokens")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--output",
        default="artifacts/base-models/qwen2.5-0.5b",
        type=Path,
    )
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--load-smoke", action="store_true")
    args = parser.parse_args()
    output = resolve_path(args.output)

    if args.verify_only:
        manifest = verify(output)
    else:
        from huggingface_hub import HfApi, snapshot_download

        info = HfApi().model_info(args.model_id, revision=args.revision)
        resolved_revision = info.sha
        if not resolved_revision:
            raise RuntimeError("model revision could not be resolved")
        output.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=args.model_id,
            revision=resolved_revision,
            local_dir=output,
            ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
        )
        files = inventory(output)
        manifest = {
            "schema_version": 1,
            "model_id": args.model_id,
            "requested_revision": args.revision,
            "resolved_revision": resolved_revision,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "files": files,
            "snapshot_sha256": hashlib.sha256(
                json.dumps(
                    files,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        }
        (output / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        verify(output)

    if args.load_smoke:
        offline_load_smoke(output)
    print(
        json.dumps(
            {
                "status": "pass",
                "model_id": manifest["model_id"],
                "resolved_revision": manifest["resolved_revision"],
                "file_count": len(manifest["files"]),
                "snapshot_sha256": manifest["snapshot_sha256"],
                "offline_config_tokenizer_smoke": bool(args.load_smoke),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

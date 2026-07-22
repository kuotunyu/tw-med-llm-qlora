"""Deterministic, content-safe sharding for the approved Phase 4 evaluation."""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PUBLIC_FORBIDDEN_KEYS = frozenset({"question", "choices", "prompt", "raw_output"})
SHARD_MEMBERS = frozenset({"manifest.json", "public.jsonl", "private.jsonl"})


def canonical_json(value: Any) -> str:
    """Serialize a value deterministically for fingerprints and JSONL records."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_bytes(value: bytes) -> str:
    """Return a lowercase SHA-256 digest."""

    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    """Hash a file in bounded chunks."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluation_request_id(*, suite: str, example_id: str, option_seed: int | None) -> str:
    """Build the model-independent ID used to align paired formal results."""

    if not suite or not example_id:
        raise ValueError("suite and example_id must not be empty")
    payload = {"suite": suite, "example_id": example_id, "option_seed": option_seed}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:24]


@dataclass(frozen=True)
class ResultShardPlan:
    """One immutable chunk of the approved full-evaluation request graph."""

    suite: str
    model: str
    shard_index: int
    request_ids: tuple[str, ...]
    contract_fingerprint: str

    def __post_init__(self) -> None:
        if not self.suite or not self.model or not self.contract_fingerprint:
            raise ValueError("shard identifiers must not be empty")
        if self.shard_index < 0:
            raise ValueError("shard_index must be non-negative")
        if not self.request_ids or len(set(self.request_ids)) != len(self.request_ids):
            raise ValueError("request_ids must be non-empty and unique")

    @property
    def filename(self) -> str:
        safe_suite = self.suite.replace("/", "-")
        safe_model = self.model.replace("/", "-")
        return f"{safe_model}--{safe_suite}--{self.shard_index:04d}.zip"

    @property
    def fingerprint(self) -> str:
        return sha256_bytes(canonical_json(asdict(self)).encode("utf-8"))

    def as_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "request_ids": list(self.request_ids),
            "fingerprint": self.fingerprint,
        }


def plan_result_shards(
    *,
    suite: str,
    model: str,
    request_ids: Sequence[str],
    shard_size: int,
    contract_fingerprint: str,
) -> list[ResultShardPlan]:
    """Split an ordered request graph without changing order or adding requests."""

    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    if not request_ids or len(set(request_ids)) != len(request_ids):
        raise ValueError("request_ids must be non-empty and unique")
    return [
        ResultShardPlan(
            suite=suite,
            model=model,
            shard_index=index // shard_size,
            request_ids=tuple(request_ids[index : index + shard_size]),
            contract_fingerprint=contract_fingerprint,
        )
        for index in range(0, len(request_ids), shard_size)
    ]


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return ("".join(f"{canonical_json(dict(row))}\n" for row in rows)).encode("utf-8")


def _read_jsonl(payload: bytes, *, member: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(payload.decode("utf-8").splitlines(), start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid {member} line {line_number}") from error
        if not isinstance(row, dict):
            raise ValueError(f"{member} line {line_number} must be an object")
        rows.append(row)
    return rows


def _validate_rows(
    plan: ResultShardPlan,
    public_rows: Sequence[Mapping[str, Any]],
    private_rows: Sequence[Mapping[str, Any]],
) -> None:
    expected = list(plan.request_ids)
    for label, rows in (("public", public_rows), ("private", private_rows)):
        actual = [row.get("request_id") for row in rows]
        if actual != expected:
            raise ValueError(f"{label} request IDs do not match the shard plan")
    for row in public_rows:
        leaked = PUBLIC_FORBIDDEN_KEYS.intersection(row)
        if leaked:
            raise ValueError(f"public result contains private keys: {sorted(leaked)}")
        if row.get("model") != plan.model or row.get("suite") != plan.suite:
            raise ValueError("public result metadata does not match the shard plan")
    for public, private in zip(public_rows, private_rows, strict=True):
        if private.get("model") != plan.model or private.get("suite") != plan.suite:
            raise ValueError("private result metadata does not match the shard plan")
        if public.get("raw_output_sha256") != sha256_bytes(
            str(private.get("raw_output", "")).encode("utf-8")
        ):
            raise ValueError("private raw output does not match its public digest")


def write_result_shard(
    path: Path,
    *,
    plan: ResultShardPlan,
    public_rows: Sequence[Mapping[str, Any]],
    private_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Atomically write one ZIP containing aligned public/private result JSONL."""

    _validate_rows(plan, public_rows, private_rows)
    public_payload = _jsonl_bytes(public_rows)
    private_payload = _jsonl_bytes(private_rows)
    manifest = {
        "schema_version": 1,
        "plan": plan.as_dict(),
        "rows": len(public_rows),
        "members": {
            "public.jsonl": {"sha256": sha256_bytes(public_payload), "bytes": len(public_payload)},
            "private.jsonl": {
                "sha256": sha256_bytes(private_payload),
                "bytes": len(private_payload),
            },
        },
    }
    manifest_payload = (canonical_json(manifest) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.partial")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", manifest_payload)
        archive.writestr("public.jsonl", public_payload)
        archive.writestr("private.jsonl", private_payload)
    os.replace(temporary, path)
    return {"path": str(path), "sha256": file_sha256(path), "bytes": path.stat().st_size}


def read_verified_result_shard(
    path: Path,
    *,
    expected_plan: ResultShardPlan,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Read a completed shard only after its plan, members, hashes, and rows agree."""

    if not path.is_file():
        raise FileNotFoundError(path)
    with zipfile.ZipFile(path) as archive:
        if set(archive.namelist()) != SHARD_MEMBERS:
            raise ValueError(f"unexpected shard members: {archive.namelist()}")
        manifest_payload = archive.read("manifest.json")
        public_payload = archive.read("public.jsonl")
        private_payload = archive.read("private.jsonl")
    manifest = json.loads(manifest_payload)
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported result shard schema")
    plan_payload = manifest.get("plan")
    if plan_payload != expected_plan.as_dict():
        raise ValueError("result shard plan mismatch")
    if manifest.get("rows") != len(expected_plan.request_ids):
        raise ValueError("result shard row count mismatch")
    for member, payload in (("public.jsonl", public_payload), ("private.jsonl", private_payload)):
        metadata = manifest.get("members", {}).get(member, {})
        if metadata.get("sha256") != sha256_bytes(payload) or metadata.get("bytes") != len(payload):
            raise ValueError(f"result shard member integrity failed: {member}")
    public_rows = _read_jsonl(public_payload, member="public.jsonl")
    private_rows = _read_jsonl(private_payload, member="private.jsonl")
    _validate_rows(expected_plan, public_rows, private_rows)
    return public_rows, private_rows, manifest


def atomic_copy_verified(source: Path, destination: Path) -> dict[str, Any]:
    """Copy a shard to its stable Drive path and verify before atomic promotion."""

    import shutil

    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.partial")
    shutil.copy2(source, temporary)
    source_hash = file_sha256(source)
    if file_sha256(temporary) != source_hash:
        raise RuntimeError("copied result shard SHA-256 mismatch")
    os.replace(temporary, destination)
    return {
        "path": str(destination),
        "sha256": source_hash,
        "bytes": destination.stat().st_size,
    }

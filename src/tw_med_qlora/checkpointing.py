"""Integrity-checked checkpoint archives for interruptible Colab training."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import zipfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

CHECKPOINT_PATTERN = re.compile(r"checkpoint-(\d+)")
REQUIRED_CHECKPOINT_FILES = {
    "optimizer.pt",
    "rng_state.pth",
    "scheduler.pt",
    "trainer_state.json",
    "training_args.bin",
}
ADAPTER_WEIGHT_FILES = {"adapter_model.bin", "adapter_model.safetensors"}


class CheckpointArchiveError(RuntimeError):
    """Raised when a checkpoint archive fails an integrity or scope check."""


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def experiment_fingerprint(payload: Mapping[str, Any]) -> str:
    """Build a stable short identifier from the training contract."""

    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]


def _checkpoint_step(checkpoint_dir: Path) -> int:
    match = CHECKPOINT_PATTERN.fullmatch(checkpoint_dir.name)
    if match is None:
        raise CheckpointArchiveError(
            f"checkpoint directory must be named checkpoint-N: {checkpoint_dir.name}"
        )
    return int(match.group(1))


def _checkpoint_files(checkpoint_dir: Path) -> list[Path]:
    if not checkpoint_dir.is_dir():
        raise CheckpointArchiveError(f"checkpoint directory does not exist: {checkpoint_dir}")
    files = sorted(path for path in checkpoint_dir.rglob("*") if path.is_file())
    if not files:
        raise CheckpointArchiveError("checkpoint directory is empty")
    if any(path.is_symlink() for path in files):
        raise CheckpointArchiveError("checkpoint must not contain symbolic links")
    names = {path.name for path in files}
    missing = sorted(REQUIRED_CHECKPOINT_FILES - names)
    if missing:
        raise CheckpointArchiveError(f"checkpoint is missing resumable state: {missing}")
    if not names.intersection(ADAPTER_WEIGHT_FILES):
        raise CheckpointArchiveError("checkpoint is missing adapter weights")
    return files


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def _archive_to_local_zip(checkpoint_dir: Path, destination: Path) -> list[str]:
    files = _checkpoint_files(checkpoint_dir)
    members: list[str] = []
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_STORED) as archive:
        for source in files:
            member = source.relative_to(checkpoint_dir).as_posix()
            archive.write(source, member)
            members.append(member)
    return members


def _remove_old_archives(drive_checkpoint_dir: Path, *, keep: int) -> None:
    records: list[tuple[int, Path, dict[str, Any]]] = []
    for metadata_path in drive_checkpoint_dir.glob("checkpoint-*.json"):
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            step = int(payload["global_step"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        records.append((step, metadata_path, payload))
    for _, metadata_path, payload in sorted(records, reverse=True)[keep:]:
        archive_name = payload.get("archive")
        if isinstance(archive_name, str):
            archive_path = drive_checkpoint_dir / archive_name
            if archive_path.is_file():
                archive_path.unlink()
        metadata_path.unlink(missing_ok=True)


def archive_checkpoint(
    *,
    checkpoint_dir: Path,
    drive_checkpoint_dir: Path,
    fingerprint: str,
    keep: int = 2,
) -> dict[str, Any]:
    """Package locally, verify after copy, then atomically publish to Drive."""

    if keep < 1:
        raise ValueError("keep must be at least 1")
    if not re.fullmatch(r"[0-9a-f]{20}", fingerprint):
        raise ValueError("fingerprint must be 20 lowercase hexadecimal characters")
    step = _checkpoint_step(checkpoint_dir)
    drive_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    archive_name = f"checkpoint-{step}.zip"
    destination = drive_checkpoint_dir / archive_name

    with tempfile.TemporaryDirectory(prefix="tw-med-checkpoint-") as temp_dir:
        local_archive = Path(temp_dir) / archive_name
        members = _archive_to_local_zip(checkpoint_dir, local_archive)
        expected_sha256 = sha256_file(local_archive)
        partial = destination.with_name(f".{archive_name}.partial")
        shutil.copy2(local_archive, partial)
        copied_sha256 = sha256_file(partial)
        if copied_sha256 != expected_sha256:
            partial.unlink(missing_ok=True)
            raise CheckpointArchiveError("Drive checkpoint copy failed SHA-256 verification")
        os.replace(partial, destination)

    record = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "experiment_fingerprint": fingerprint,
        "checkpoint": checkpoint_dir.name,
        "global_step": step,
        "archive": archive_name,
        "archive_sha256": sha256_file(destination),
        "archive_bytes": destination.stat().st_size,
        "members": members,
    }
    metadata_path = drive_checkpoint_dir / f"checkpoint-{step}.json"
    _atomic_write_json(metadata_path, record)
    _atomic_write_json(drive_checkpoint_dir / "latest.json", record)
    _remove_old_archives(drive_checkpoint_dir, keep=keep)
    return record


def _safe_extract(archive_path: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            member_path = PurePosixPath(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise CheckpointArchiveError(f"unsafe checkpoint archive member: {member.filename}")
        archive.extractall(destination)


def restore_latest_checkpoint(
    *,
    drive_checkpoint_dir: Path,
    local_output_dir: Path,
    fingerprint: str,
) -> Path | None:
    """Verify and restore the latest complete Drive checkpoint to local storage."""

    latest_path = drive_checkpoint_dir / "latest.json"
    if not latest_path.is_file():
        return None
    try:
        record = json.loads(latest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise CheckpointArchiveError("latest checkpoint metadata is invalid JSON") from error
    if record.get("experiment_fingerprint") != fingerprint:
        raise CheckpointArchiveError("checkpoint experiment fingerprint mismatch")

    checkpoint_name = str(record.get("checkpoint", ""))
    if CHECKPOINT_PATTERN.fullmatch(checkpoint_name) is None:
        raise CheckpointArchiveError("latest checkpoint name is invalid")
    archive_name = str(record.get("archive", ""))
    if archive_name != f"{checkpoint_name}.zip":
        raise CheckpointArchiveError("latest checkpoint archive name mismatch")
    archive_path = drive_checkpoint_dir / archive_name
    if not archive_path.is_file():
        raise CheckpointArchiveError("latest checkpoint archive is missing")
    if archive_path.stat().st_size != int(record.get("archive_bytes", -1)):
        raise CheckpointArchiveError("latest checkpoint archive size mismatch")
    if sha256_file(archive_path) != record.get("archive_sha256"):
        raise CheckpointArchiveError("latest checkpoint archive SHA-256 mismatch")

    local_output_dir.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".restore-", dir=local_output_dir))
    target = local_output_dir / checkpoint_name
    try:
        _safe_extract(archive_path, staging)
        _checkpoint_files(staging)
        restored_members = sorted(
            path.relative_to(staging).as_posix()
            for path in staging.rglob("*")
            if path.is_file()
        )
        if restored_members != sorted(record.get("members", [])):
            raise CheckpointArchiveError("restored checkpoint member list mismatch")
        if target.exists():
            shutil.rmtree(target)
        os.replace(staging, target)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return target

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tw_med_qlora.checkpointing import (
    CheckpointArchiveError,
    archive_checkpoint,
    experiment_fingerprint,
    restore_latest_checkpoint,
)


def _fake_checkpoint(root: Path, step: int) -> Path:
    checkpoint = root / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    for name in (
        "adapter_model.safetensors",
        "optimizer.pt",
        "rng_state.pth",
        "scheduler.pt",
        "trainer_state.json",
        "training_args.bin",
    ):
        (checkpoint / name).write_bytes(f"{step}:{name}".encode())
    return checkpoint


def test_checkpoint_round_trip_and_retention(tmp_path: Path) -> None:
    fingerprint = experiment_fingerprint({"model": "fixed", "seed": 3407})
    source = tmp_path / "source"
    drive = tmp_path / "drive"
    restored = tmp_path / "restored"

    for step in (100, 200, 300):
        record = archive_checkpoint(
            checkpoint_dir=_fake_checkpoint(source, step),
            drive_checkpoint_dir=drive,
            fingerprint=fingerprint,
            keep=2,
        )

    assert record["global_step"] == 300
    assert not (drive / "checkpoint-100.zip").exists()
    assert not (drive / "checkpoint-100.json").exists()
    assert (drive / "checkpoint-200.zip").is_file()
    assert (drive / "checkpoint-300.zip").is_file()

    checkpoint = restore_latest_checkpoint(
        drive_checkpoint_dir=drive,
        local_output_dir=restored,
        fingerprint=fingerprint,
    )

    assert checkpoint == restored / "checkpoint-300"
    assert (checkpoint / "optimizer.pt").read_bytes() == b"300:optimizer.pt"


def test_restore_returns_none_without_checkpoint(tmp_path: Path) -> None:
    restored = restore_latest_checkpoint(
        drive_checkpoint_dir=tmp_path / "drive",
        local_output_dir=tmp_path / "local",
        fingerprint="a" * 20,
    )

    assert restored is None


def test_restore_rejects_corrupt_archive(tmp_path: Path) -> None:
    fingerprint = "b" * 20
    drive = tmp_path / "drive"
    archive_checkpoint(
        checkpoint_dir=_fake_checkpoint(tmp_path / "source", 100),
        drive_checkpoint_dir=drive,
        fingerprint=fingerprint,
    )
    with (drive / "checkpoint-100.zip").open("ab") as archive:
        archive.write(b"corrupt")

    with pytest.raises(CheckpointArchiveError, match="size mismatch"):
        restore_latest_checkpoint(
            drive_checkpoint_dir=drive,
            local_output_dir=tmp_path / "local",
            fingerprint=fingerprint,
        )


def test_restore_rejects_wrong_experiment(tmp_path: Path) -> None:
    drive = tmp_path / "drive"
    archive_checkpoint(
        checkpoint_dir=_fake_checkpoint(tmp_path / "source", 100),
        drive_checkpoint_dir=drive,
        fingerprint="c" * 20,
    )

    with pytest.raises(CheckpointArchiveError, match="fingerprint mismatch"):
        restore_latest_checkpoint(
            drive_checkpoint_dir=drive,
            local_output_dir=tmp_path / "local",
            fingerprint="d" * 20,
        )


def test_archive_requires_optimizer_scheduler_and_rng(tmp_path: Path) -> None:
    checkpoint = _fake_checkpoint(tmp_path / "source", 100)
    (checkpoint / "optimizer.pt").unlink()

    with pytest.raises(CheckpointArchiveError, match="resumable state"):
        archive_checkpoint(
            checkpoint_dir=checkpoint,
            drive_checkpoint_dir=tmp_path / "drive",
            fingerprint="e" * 20,
        )


def test_latest_metadata_is_valid_json(tmp_path: Path) -> None:
    drive = tmp_path / "drive"
    archive_checkpoint(
        checkpoint_dir=_fake_checkpoint(tmp_path / "source", 100),
        drive_checkpoint_dir=drive,
        fingerprint="f" * 20,
    )

    latest = json.loads((drive / "latest.json").read_text(encoding="utf-8"))
    assert latest["checkpoint"] == "checkpoint-100"
    assert latest["archive_sha256"]

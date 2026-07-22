from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tw_med_qlora.phase3_evidence import Phase3EvidenceValidationError
from tw_med_qlora.phase3_full_evidence import validate_phase3_full_evidence

ROOT = Path(__file__).parents[1]
CALIBRATION_MANIFEST = ROOT / "reports" / "phase3" / "20260721T171557Z-run-manifest.json"
CALIBRATION_VALIDATION = ROOT / "reports" / "phase3" / "20260721T171557Z-validation.json"
CONFIG = ROOT / "configs" / "project.toml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_full_artifacts(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    run_id = "20260723T000000Z"
    manifest = json.loads(CALIBRATION_MANIFEST.read_text(encoding="utf-8"))
    calibration = json.loads(CALIBRATION_VALIDATION.read_text(encoding="utf-8"))
    projection = calibration["full_run_projection"]
    manifest.update(
        {
            "run_id": run_id,
            "run_mode": "full",
            "full_training_enabled": True,
            "full_training_approval_verified": True,
            "full_training_approval": {
                "approved_at": "2026-07-22",
                "approved_buffered_compute_units": projection[
                    "projected_compute_units_with_20pct_buffer"
                ],
                "compute_units_per_hour_at_approval": 5.3,
                "compute_units_balance_at_approval": 436.2,
            },
        }
    )
    manifest["training"]["masking_audit"].update(
        {
            "train_rows": 11248,
            "validation_rows": 1409,
            "resume_checkpoint": None,
        }
    )
    manifest["training"]["metrics"] = {
        "mode": "full",
        "global_step": 703,
        "completed_steps_this_session": 703,
        "selected_adapter_step": 700,
        "wall_seconds": 14500.0,
        "seconds_per_step_this_session": 20.63,
        "training_loop_without_checkpoint_seconds": 14415.0,
        "seconds_per_step_excluding_checkpoint": 20.51,
        "checkpoint_sync_wall_seconds": 63.0,
        "checkpoint_cycle_wall_seconds": 85.0,
        "checkpoint_seconds_per_save": 12.14,
        "final_eval_wall_seconds": 430.0,
        "final_eval_rows": 1409,
        "training_loss": 0.31,
        "logged_losses": [0.72, 0.51, 0.29],
        "logged_eval_losses": [0.72, 0.69, 0.67, 0.65, 0.63, 0.61, 0.58],
        "final_eval_metrics": {
            "eval_loss": 0.58,
            "epoch": 0.995,
            "step": 700,
            "source": "scheduled_full_validation",
            "selected_checkpoint": "checkpoint-700",
        },
        "post_train_eval_attempt": {
            "status": "rejected_non_finite",
            "attempted_step": 703,
            "eval_loss": "NaN",
        },
        "peak_allocated_gib": 12.0,
        "peak_reserved_gib": 14.0,
        "oom": False,
        "all_losses_finite": True,
    }
    base_record = manifest["checkpoint_audit"]["archives_written_this_session"][0]
    records = []
    for step in [*range(100, 701, 100), 703]:
        record = dict(base_record)
        record.update(
            {
                "checkpoint": f"checkpoint-{step}",
                "global_step": step,
                "archive": f"checkpoint-{step}.zip",
            }
        )
        records.append(record)
    manifest["checkpoint_audit"].update(
        {
            "resumed_from": None,
            "archives_written_this_session": records,
            "restore_test_passed": True,
            "restored_checkpoint": "checkpoint-703",
            "restored_global_step": 703,
            "selected_adapter_checkpoint": "checkpoint-700",
            "selected_adapter_global_step": 700,
            "evaluation_selection": {
                "status": "recovered_from_non_finite_post_train_evaluation",
                "selected_adapter_global_step": 700,
            },
        }
    )
    manifest["cost_estimate"].update(
        {
            "compute_units_per_hour_user_input": 5.3,
            "current_compute_units_user_input": 436.2,
            "projected_hours": projection["hours"],
            "projected_compute_units": projection["projected_compute_units"],
            "projected_compute_units_with_20pct_buffer": projection[
                "projected_compute_units_with_20pct_buffer"
            ],
            "actual_session_hours": 4.15,
            "actual_session_compute_units": 21.995,
        }
    )

    trainer_log = tmp_path / f"{run_id}-trainer_log.csv"
    trainer_log.write_text(
        "step,loss,eval_loss\n1,0.72,\n100,,0.72\n703,0.29,0.58\n",
        encoding="utf-8",
    )
    curves = tmp_path / f"{run_id}-training_curves.png"
    curves.write_bytes(b"\x89PNG\r\n\x1a\nsynthetic-test-image")
    drive_evidence = {}
    evidence_sizes = {
        "trainer_log.csv": trainer_log.stat().st_size,
        "training_curves.png": curves.stat().st_size,
        "trainer_log.json": 10,
        "MODEL_CARD_DRAFT.md": 10,
        "pip-freeze.txt": 10,
    }
    evidence_hashes = {
        "trainer_log.csv": _sha256(trainer_log),
        "training_curves.png": _sha256(curves),
        "trainer_log.json": "a" * 64,
        "MODEL_CARD_DRAFT.md": "b" * 64,
        "pip-freeze.txt": "c" * 64,
    }
    for name in evidence_sizes:
        drive_evidence[name] = {
            "path": f"/content/drive/{run_id}-{name}",
            "sha256": evidence_hashes[name],
            "bytes": evidence_sizes[name],
        }
    receipt = {
        "phase": 3,
        "run_mode": "full",
        "experiment_fingerprint": manifest["experiment_fingerprint"],
        "drive_archive": f"/content/drive/{run_id}-phase3-full.zip",
        "drive_manifest": f"/content/drive/{run_id}-run-manifest.json",
        "archive_sha256": "d" * 64,
        "archive_bytes": 123456,
        "drive_evidence": drive_evidence,
    }
    manifest_path = tmp_path / f"{run_id}-run-manifest.json"
    receipt_path = tmp_path / f"{run_id}-receipt.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    return manifest_path, receipt_path, trainer_log, curves


def _validate(paths: tuple[Path, Path, Path, Path]) -> dict[str, object]:
    manifest, receipt, trainer_log, curves = paths
    return validate_phase3_full_evidence(
        manifest_path=manifest,
        receipt_path=receipt,
        trainer_log_path=trainer_log,
        training_curves_path=curves,
        calibration_validation_path=CALIBRATION_VALIDATION,
        config_path=CONFIG,
    )


def test_phase3_full_evidence_passes_synthetic_contract(tmp_path: Path) -> None:
    summary = _validate(_write_full_artifacts(tmp_path))

    assert summary["status"] == "pass"
    assert summary["hardware"]["profile"] == "primary_40g"
    assert summary["training"]["steps"] == 703
    assert summary["training"]["selected_adapter_step"] == 700
    assert summary["training"]["validation_examples"] == 1409
    assert summary["checkpoint"]["restored_global_step"] == 703
    assert summary["checkpoint"]["selected_adapter_step"] == 700
    assert summary["cost"]["approved_projected_compute_units"] == pytest.approx(26.4874030128)


@pytest.mark.parametrize(
    ("target_file", "mutation", "message"),
    [
        ("manifest", ("full_training_enabled", False), "full training was not enabled"),
        (
            "manifest",
            ("full_training_approval_verified", False),
            "approval was not verified",
        ),
        ("manifest", ("data.audit.test_used_for_training", True), "test was used"),
        ("manifest", ("training.metrics.global_step", 702), "step count mismatch"),
        (
            "manifest",
            ("checkpoint_audit.selected_adapter_global_step", 600),
            "selected adapter checkpoint step mismatch",
        ),
        ("receipt", ("experiment_fingerprint", "wrong"), "fingerprint mismatch"),
    ],
)
def test_phase3_full_evidence_rejects_failed_gate(
    tmp_path: Path,
    target_file: str,
    mutation: tuple[str, object],
    message: str,
) -> None:
    paths = _write_full_artifacts(tmp_path)
    artifact_path = paths[0] if target_file == "manifest" else paths[1]
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    path, value = mutation
    target = artifact
    parts = path.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(Phase3EvidenceValidationError, match=message):
        _validate(paths)


def test_phase3_full_evidence_rejects_modified_curve(tmp_path: Path) -> None:
    paths = _write_full_artifacts(tmp_path)
    paths[3].write_bytes(b"not-the-recorded-curve")

    with pytest.raises(Phase3EvidenceValidationError, match="training_curves.png size mismatch"):
        _validate(paths)


def test_phase3_full_evidence_requires_non_finite_fallback_audit(tmp_path: Path) -> None:
    paths = _write_full_artifacts(tmp_path)
    manifest = json.loads(paths[0].read_text(encoding="utf-8"))
    manifest["checkpoint_audit"].pop("evaluation_selection")
    manifest["training"]["metrics"].pop("post_train_eval_attempt")
    paths[0].write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        Phase3EvidenceValidationError,
        match="lacks a non-finite evaluation audit",
    ):
        _validate(paths)


def test_phase3_full_evidence_rejects_wrong_selected_validation_source(
    tmp_path: Path,
) -> None:
    paths = _write_full_artifacts(tmp_path)
    manifest = json.loads(paths[0].read_text(encoding="utf-8"))
    manifest["training"]["metrics"]["final_eval_metrics"]["source"] = "post_train"
    paths[0].write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(Phase3EvidenceValidationError, match="validation source mismatch"):
        _validate(paths)

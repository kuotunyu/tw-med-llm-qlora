"""Validate completed Phase 3 full-training evidence."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

from tw_med_qlora.config import load_project_config, select_training_profile
from tw_med_qlora.phase3_evidence import (
    _assert_no_private_payload,
    _load_object,
    _require,
    _sha256,
)


def _valid_sha256(value: object) -> bool:
    rendered = str(value)
    return len(rendered) == 64 and all(char in "0123456789abcdef" for char in rendered)


def _require_local_evidence(
    *,
    path: Path,
    evidence_name: str,
    drive_evidence: dict[str, Any],
) -> None:
    _require(path.is_file(), f"missing local evidence: {path.name}")
    record = drive_evidence.get(evidence_name)
    _require(isinstance(record, dict), f"receipt is missing {evidence_name}")
    _require(int(record["bytes"]) == path.stat().st_size, f"{evidence_name} size mismatch")
    _require(str(record["sha256"]) == _sha256(path), f"{evidence_name} SHA-256 mismatch")


def validate_phase3_full_evidence(
    *,
    manifest_path: Path,
    receipt_path: Path,
    trainer_log_path: Path,
    training_curves_path: Path,
    calibration_validation_path: Path,
    config_path: Path,
) -> dict[str, Any]:
    """Validate a completed full run against its approved A100 calibration."""

    manifest = _load_object(manifest_path)
    receipt = _load_object(receipt_path)
    calibration = _load_object(calibration_validation_path)
    _assert_no_private_payload(manifest)
    _assert_no_private_payload(receipt)
    _assert_no_private_payload(calibration)
    config = load_project_config(config_path)

    run_id = str(manifest.get("run_id", ""))
    _require(run_id != "", "manifest run_id is required")
    _require(manifest_path.name == f"{run_id}-run-manifest.json", "manifest filename mismatch")
    _require(receipt_path.name == f"{run_id}-receipt.json", "receipt filename mismatch")
    _require(manifest.get("schema_version") == 1, "unsupported manifest schema")
    _require(manifest.get("phase") == 3, "manifest is not Phase 3")
    _require(manifest.get("run_mode") == "full", "run is not full training")
    _require(manifest.get("full_training_enabled") is True, "full training was not enabled")
    _require(
        manifest.get("full_training_approval_verified") is True,
        "full training approval was not verified",
    )
    _require(calibration.get("status") == "pass", "calibration evidence did not pass")
    _require(calibration.get("run_mode") == "calibration", "calibration mode mismatch")

    fingerprint = str(manifest.get("experiment_fingerprint", ""))
    _require(len(fingerprint) == 20, "experiment fingerprint is invalid")
    _require(receipt.get("phase") == 3, "receipt is not Phase 3")
    _require(receipt.get("run_mode") == "full", "receipt mode mismatch")
    _require(receipt.get("experiment_fingerprint") == fingerprint, "fingerprint mismatch")

    gpu = manifest["gpu"]
    model = manifest["model"]
    selected = select_training_profile(
        config,
        total_vram_gib=float(gpu["total_vram_gib"]),
        bf16_supported=bool(gpu["bf16_supported"]),
        compute_capability=tuple(int(item) for item in gpu["compute_capability"]),
    )
    calibrated_hardware = calibration["hardware"]
    _require("A100" in str(gpu["name"]).upper(), "full run did not use A100")
    _require(selected.hardware_profile == "primary_40g", "full run profile is not primary_40g")
    _require(
        selected.hardware_profile == calibrated_hardware["profile"],
        "full run profile differs from calibration",
    )
    _require(model["hardware_profile"] == selected.hardware_profile, "hardware profile mismatch")
    _require(model["model_id"] == selected.model_id, "model ID mismatch")
    _require(model["revision"] == selected.revision, "model revision mismatch")
    _require(model["load_in_4bit"] is True, "base model was not loaded in 4-bit")
    _require(model["adapter_base_verified"] is True, "adapter base was not verified")
    snapshot = model["snapshot_audit"]
    _require(snapshot["complete"] is True, "base snapshot is incomplete")
    _require(snapshot["revision"] == selected.revision, "snapshot revision mismatch")
    _require(snapshot["indexed_shards"] == snapshot["remote_weight_files"], "shard audit failed")

    data_config = config.raw["data"]["medqa"]
    training_config = config.raw["training"]
    expected_rows = {
        "train": int(data_config["expected_train_rows"]),
        "validation": int(data_config["expected_validation_rows"]),
        "test": int(data_config["expected_test_rows"]),
    }
    data = manifest["data"]
    audit = data["audit"]
    _require(data["dataset_id"] == data_config["dataset_id"], "dataset ID mismatch")
    _require(data["revision"] == data_config["revision"], "dataset revision mismatch")
    _require(audit["clean_rows"] == expected_rows, "clean split counts mismatch")
    _require(audit["test_used_for_training"] is False, "test was used for training")
    _require(
        audit["trainer_referenced_splits"] == ["train", "validation"],
        "trainer split isolation failed",
    )

    training = manifest["training"]
    contract = manifest["training_contract"]
    _require(training["batch_size"] == selected.batch_size, "batch size mismatch")
    _require(
        training["gradient_accumulation_steps"] == selected.gradient_accumulation_steps,
        "gradient accumulation mismatch",
    )
    _require(training["max_sequence_length"] == selected.max_sequence_length, "sequence mismatch")
    _require(training["seed"] == config.seed, "training seed mismatch")
    _require(training["lora_rank"] == int(training_config["lora_rank"]), "LoRA rank mismatch")
    _require(training["lora_alpha"] == int(training_config["lora_alpha"]), "LoRA alpha mismatch")
    _require(training["response_only_loss"] is True, "response-only loss was disabled")
    _require(contract["hardware_profile"] == selected.hardware_profile, "contract profile mismatch")
    _require(contract["model_id"] == selected.model_id, "contract model mismatch")
    _require(contract["model_revision"] == selected.revision, "contract revision mismatch")
    _require(contract["dataset_id"] == data_config["dataset_id"], "contract dataset mismatch")
    _require(
        contract["dataset_revision"] == data_config["revision"],
        "contract data revision mismatch",
    )
    _require(
        contract["effective_batch_size"] == config.effective_batch_size,
        "contract batch mismatch",
    )

    masking = training["masking_audit"]
    _require(masking["vision_collator"] is False, "vision collator was used for text data")
    _require(masking["test_passed_to_trainer"] is False, "test reached the trainer")
    _require(masking["train_rows"] == expected_rows["train"], "full train rows mismatch")
    _require(
        masking["validation_rows"] == expected_rows["validation"],
        "full validation rows mismatch",
    )
    _require(masking["masked_prompt_tokens"] > 0, "prompt tokens were not masked")
    _require(masking["response_loss_tokens"] > 0, "no response tokens contributed to loss")

    epochs = float(training_config["num_train_epochs"])
    full_steps = math.ceil(expected_rows["train"] * epochs / config.effective_batch_size)
    save_steps = int(training_config["save_steps"])
    latest_scheduled_checkpoint_step = full_steps // save_steps * save_steps
    expected_eval_events = full_steps // int(training_config["eval_steps"])
    metrics = training["metrics"]
    losses = [float(loss) for loss in metrics["logged_losses"]]
    eval_losses = [float(loss) for loss in metrics["logged_eval_losses"]]
    _require(metrics["mode"] == "full", "training metrics mode mismatch")
    _require(metrics["global_step"] == full_steps, "full training step count mismatch")
    selected_adapter_step = int(
        metrics.get("selected_adapter_step", checkpoint_step)
        if (checkpoint_step := manifest["checkpoint_audit"].get("selected_adapter_global_step"))
        is not None
        else metrics["global_step"]
    )
    _require(
        selected_adapter_step in {latest_scheduled_checkpoint_step, full_steps},
        "selected adapter step is neither the final nor last scheduled checkpoint",
    )
    _require(0 < metrics["completed_steps_this_session"] <= full_steps, "invalid session steps")
    _require(len(losses) >= 2, "full training has insufficient logged losses")
    _require(len(eval_losses) >= expected_eval_events, "validation events are incomplete")
    _require(metrics["oom"] is False, "full training reported OOM")
    _require(metrics["all_losses_finite"] is True, "full training reported non-finite loss")
    _require(all(math.isfinite(loss) for loss in losses), "logged losses contain NaN or Inf")
    _require(all(math.isfinite(loss) for loss in eval_losses), "eval losses contain NaN or Inf")
    _require(
        metrics["final_eval_rows"] == expected_rows["validation"],
        "final validation row count mismatch",
    )
    _require(math.isfinite(float(metrics["training_loss"])), "training loss is non-finite")
    _require(
        math.isfinite(float(metrics["final_eval_metrics"]["eval_loss"])),
        "final eval loss is non-finite",
    )
    _require(
        float(metrics["peak_reserved_gib"]) < float(gpu["total_vram_gib"]),
        "peak VRAM exceeds GPU",
    )

    checkpoint = manifest["checkpoint_audit"]
    _require(checkpoint["restore_test_passed"] is True, "checkpoint restore test failed")
    restored_step = int(checkpoint["restored_global_step"])
    _require(
        restored_step in {selected_adapter_step, full_steps},
        "restored checkpoint is unrelated to the selected or completed checkpoint",
    )
    _require(
        checkpoint.get("selected_adapter_checkpoint") == f"checkpoint-{selected_adapter_step}",
        "selected adapter checkpoint name mismatch",
    )
    _require(
        int(checkpoint.get("selected_adapter_global_step", -1)) == selected_adapter_step,
        "selected adapter checkpoint step mismatch",
    )
    _require(checkpoint["retention_limit"] == 2, "checkpoint retention mismatch")
    records = checkpoint["archives_written_this_session"]
    resumed = checkpoint["resumed_from"] is not None
    expected_checkpoint_steps = list(
        range(save_steps, latest_scheduled_checkpoint_step + 1, save_steps)
    )
    if full_steps != latest_scheduled_checkpoint_step:
        expected_checkpoint_steps.append(full_steps)
    if not resumed:
        _require(
            [int(record["global_step"]) for record in records] == expected_checkpoint_steps,
            "fresh run checkpoint sequence mismatch",
        )
    required_members = {
        "adapter_config.json",
        "adapter_model.safetensors",
        "optimizer.pt",
        "rng_state.pth",
        "scheduler.pt",
        "trainer_state.json",
        "training_args.bin",
    }
    for record in records:
        _require(record["experiment_fingerprint"] == fingerprint, "checkpoint fingerprint mismatch")
        _require(
            record["global_step"] % save_steps == 0 or record["global_step"] == full_steps,
            "checkpoint step mismatch",
        )
        _require(required_members <= set(record["members"]), "checkpoint state is incomplete")
        _require(_valid_sha256(record["archive_sha256"]), "checkpoint SHA-256 is invalid")
        _require(int(record["archive_bytes"]) > 0, "checkpoint archive is empty")

    reload_check = manifest["reload_check"]
    _require(reload_check["adapter_reloaded"] is True, "adapter reload failed")
    _require(reload_check["strict_parse"] is True, "validation output was not strictly parsed")
    _require(reload_check["probe_split"] == "validation", "reload probe is not validation")
    _require(
        reload_check["published_base_model_id"] == selected.model_id,
        "published base mismatch",
    )
    _require(reload_check["adapter_parameters"] > 0, "adapter has no parameters")
    _require(reload_check["trainable_adapter_parameters"] == 0, "reloaded adapter is trainable")

    selection = checkpoint.get("evaluation_selection") or metrics.get("evaluation_selection")
    if selected_adapter_step == latest_scheduled_checkpoint_step:
        selection_status = selection.get("status") if isinstance(selection, dict) else None
        fallback_statuses = {
            "recovered_from_non_finite_post_train_evaluation",
            "rejected_non_finite",
        }
        post_train_attempt = metrics.get("post_train_eval_attempt", {})
        _require(
            selection_status in fallback_statuses
            or post_train_attempt.get("status") == "rejected_non_finite"
            or post_train_attempt.get("eval_loss") == "NaN",
            "scheduled checkpoint selection lacks a non-finite evaluation audit",
        )
        _require(
            int(metrics["final_eval_metrics"].get("step", -1)) == latest_scheduled_checkpoint_step,
            "selected validation step mismatch",
        )
        _require(
            metrics["final_eval_metrics"].get("source") == "scheduled_full_validation",
            "selected validation source mismatch",
        )

    approved_projection = calibration["full_run_projection"]
    cost = manifest["cost_estimate"]
    approval = manifest["full_training_approval"]
    _require(str(approval["approved_at"]) != "", "approval date is missing")
    _require(
        math.isclose(
            float(cost["compute_units_per_hour_user_input"]),
            float(approved_projection["compute_units_per_hour_user_observed"]),
        ),
        "approved CU rate mismatch",
    )
    _require(
        math.isclose(float(cost["projected_hours"]), float(approved_projection["hours"])),
        "approved hour projection mismatch",
    )
    _require(
        math.isclose(
            float(cost["projected_compute_units"]),
            float(approved_projection["projected_compute_units"]),
        ),
        "approved CU projection mismatch",
    )
    _require(
        math.isclose(
            float(approval["approved_buffered_compute_units"]),
            float(approved_projection["projected_compute_units_with_20pct_buffer"]),
        ),
        "approved CU buffer mismatch",
    )

    _require(_valid_sha256(receipt["archive_sha256"]), "delivery SHA-256 is invalid")
    _require(int(receipt["archive_bytes"]) > 0, "delivery archive is empty")
    _require(run_id in str(receipt["drive_archive"]), "Drive archive run ID mismatch")
    _require(run_id in str(receipt["drive_manifest"]), "Drive manifest run ID mismatch")
    drive_evidence = receipt.get("drive_evidence")
    _require(isinstance(drive_evidence, dict), "receipt has no separate evidence files")
    expected_evidence = {
        "trainer_log.csv",
        "trainer_log.json",
        "training_curves.png",
        "MODEL_CARD_DRAFT.md",
        "pip-freeze.txt",
    }
    _require(expected_evidence <= set(drive_evidence), "Drive evidence set is incomplete")
    for evidence_name in expected_evidence:
        record = drive_evidence[evidence_name]
        _require(_valid_sha256(record["sha256"]), f"{evidence_name} SHA-256 is invalid")
        _require(int(record["bytes"]) > 0, f"{evidence_name} is empty")
        _require(run_id in str(record["path"]), f"{evidence_name} run ID mismatch")

    _require_local_evidence(
        path=trainer_log_path,
        evidence_name="trainer_log.csv",
        drive_evidence=drive_evidence,
    )
    _require_local_evidence(
        path=training_curves_path,
        evidence_name="training_curves.png",
        drive_evidence=drive_evidence,
    )
    with trainer_log_path.open(encoding="utf-8", newline="") as log_file:
        log_rows = list(csv.DictReader(log_file))
    _require(len(log_rows) > 0, "trainer log has no rows")
    _require(any(row.get("loss") for row in log_rows), "trainer log has no training loss")
    _require(any(row.get("eval_loss") for row in log_rows), "trainer log has no eval loss")
    _require(
        training_curves_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"),
        "training curve is not PNG",
    )

    return {
        "schema_version": 1,
        "phase": 3,
        "run_mode": "full",
        "run_id": run_id,
        "status": "pass",
        "source_artifacts": {
            "manifest": manifest_path.name,
            "manifest_sha256": _sha256(manifest_path),
            "receipt": receipt_path.name,
            "receipt_sha256": _sha256(receipt_path),
            "trainer_log": trainer_log_path.name,
            "trainer_log_sha256": _sha256(trainer_log_path),
            "training_curves": training_curves_path.name,
            "training_curves_sha256": _sha256(training_curves_path),
            "drive_archive_sha256": receipt["archive_sha256"],
        },
        "validated_invariants": [
            "full_training_approval_verified",
            "same_a100_40g_profile_as_calibration",
            "pinned_complete_base_snapshot",
            "full_train_and_validation_counts",
            "test_excluded_from_trainer",
            "response_only_masking",
            "703_training_steps_with_finite_logged_training_loss",
            "finite_full_validation_for_selected_adapter_checkpoint",
            "non_finite_step_703_validation_rejected_when_applicable",
            "checkpoint_integrity_and_final_checkpoint_restore",
            "adapter_saved_reloaded_and_strictly_parsed",
            "trainer_log_and_curve_hashes_match_receipt",
            "no_private_question_or_secret_fields",
        ],
        "hardware": {
            "gpu": gpu["name"],
            "profile": selected.hardware_profile,
            "precision": gpu["precision"],
            "total_vram_gib": float(gpu["total_vram_gib"]),
        },
        "training": {
            "train_examples": expected_rows["train"],
            "validation_examples": expected_rows["validation"],
            "steps": full_steps,
            "selected_adapter_step": selected_adapter_step,
            "completed_steps_this_session": int(metrics["completed_steps_this_session"]),
            "resumed": resumed,
            "training_loss": float(metrics["training_loss"]),
            "final_eval_loss": float(metrics["final_eval_metrics"]["eval_loss"]),
            "first_logged_loss": losses[0],
            "last_logged_loss": losses[-1],
            "peak_allocated_gib": float(metrics["peak_allocated_gib"]),
            "peak_reserved_gib": float(metrics["peak_reserved_gib"]),
            "training_wall_seconds": float(metrics["wall_seconds"]),
            "final_eval_wall_seconds": float(metrics["final_eval_wall_seconds"]),
        },
        "checkpoint": {
            "restored_global_step": restored_step,
            "selected_adapter_step": selected_adapter_step,
            "archives_written_this_session": len(records),
            "retention_limit": int(checkpoint["retention_limit"]),
        },
        "cost": {
            "approved_projected_hours": float(cost["projected_hours"]),
            "approved_projected_compute_units": float(cost["projected_compute_units"]),
            "approved_buffered_compute_units": float(approval["approved_buffered_compute_units"]),
            "measured_training_and_final_eval_hours": float(cost["actual_session_hours"]),
            "measured_training_and_final_eval_compute_units": float(
                cost["actual_session_compute_units"]
            ),
            "monetary_estimate": cost["estimated_cost"],
        },
        "reload_probe": {
            "split": reload_check["probe_split"],
            "strict_parse": True,
            "correct": reload_check["prediction"] == reload_check["gold"],
            "note": "single reload probe; not an accuracy evaluation",
        },
    }

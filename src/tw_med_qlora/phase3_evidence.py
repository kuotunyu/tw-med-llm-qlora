"""Validate Phase 3 A100 calibration evidence and correct its CU projection."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from tw_med_qlora.config import load_project_config, select_training_profile


class Phase3EvidenceValidationError(ValueError):
    """Raised when a Phase 3 calibration artifact violates an invariant."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Phase3EvidenceValidationError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as artifact:
        value = json.load(artifact)
    _require(isinstance(value, dict), f"{path.name} must contain a JSON object")
    return value


def _assert_no_private_payload(value: Any, *, path: str = "root") -> None:
    forbidden_keys = {
        "choices",
        "discord_webhook_url",
        "google_api_key",
        "hf_token",
        "openai_api_key",
        "prompt",
        "question",
        "raw_output",
        "wandb_api_key",
    }
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).casefold()
            _require(normalized not in forbidden_keys, f"private field found at {path}.{key}")
            _assert_no_private_payload(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_no_private_payload(item, path=f"{path}[{index}]")


def validate_phase3_calibration_evidence(
    *,
    manifest_path: Path,
    receipt_path: Path,
    config_path: Path,
    observed_compute_units_per_hour: float,
    observed_current_compute_units: float,
) -> dict[str, Any]:
    """Validate an A100 calibration and return a deterministic public-safe summary."""

    _require(observed_compute_units_per_hour > 0, "observed CU per hour must be positive")
    _require(observed_current_compute_units >= 0, "observed current CU cannot be negative")

    manifest = _load_object(manifest_path)
    receipt = _load_object(receipt_path)
    _assert_no_private_payload(manifest)
    _assert_no_private_payload(receipt)
    config = load_project_config(config_path)

    run_id = str(manifest.get("run_id", ""))
    _require(run_id != "", "manifest run_id is required")
    _require(manifest_path.name == f"{run_id}-run-manifest.json", "manifest filename mismatch")
    _require(receipt_path.name == f"{run_id}-receipt.json", "receipt filename mismatch")
    _require(manifest.get("schema_version") == 1, "unsupported manifest schema")
    _require(manifest.get("phase") == 3, "manifest is not Phase 3")
    _require(manifest.get("run_mode") == "calibration", "run is not calibration")
    _require(manifest.get("full_training_enabled") is False, "full training was enabled")

    fingerprint = str(manifest.get("experiment_fingerprint", ""))
    _require(len(fingerprint) == 20, "experiment fingerprint is invalid")
    _require(receipt.get("phase") == 3, "receipt is not Phase 3")
    _require(receipt.get("run_mode") == "calibration", "receipt mode mismatch")
    _require(receipt.get("experiment_fingerprint") == fingerprint, "fingerprint mismatch")

    gpu = manifest["gpu"]
    model = manifest["model"]
    selected = select_training_profile(
        config,
        total_vram_gib=float(gpu["total_vram_gib"]),
        bf16_supported=bool(gpu["bf16_supported"]),
        compute_capability=tuple(int(item) for item in gpu["compute_capability"]),
    )
    _require(selected.hardware_profile in {"primary_40g", "primary_80g"}, "GPU is not premium")
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
    data = manifest["data"]
    audit = data["audit"]
    _require(data["dataset_id"] == data_config["dataset_id"], "dataset ID mismatch")
    _require(data["revision"] == data_config["revision"], "dataset revision mismatch")
    expected_rows = {
        "train": int(data_config["expected_train_rows"]),
        "validation": int(data_config["expected_validation_rows"]),
        "test": int(data_config["expected_test_rows"]),
    }
    _require(audit["clean_rows"] == expected_rows, "clean split counts mismatch")
    _require(audit["test_used_for_training"] is False, "test was used for training")
    _require(
        audit["trainer_referenced_splits"] == ["train", "validation"],
        "trainer split isolation failed",
    )
    _require(
        audit["smoke_rows"] == int(config.raw["training"]["smoke_examples"]),
        "calibration train rows mismatch",
    )
    _require(audit["calibration_validation_rows"] == 100, "calibration validation rows mismatch")

    training = manifest["training"]
    training_config = config.raw["training"]
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

    contract = manifest["training_contract"]
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
    _require(masking["train_rows"] == 100, "masking train rows mismatch")
    _require(masking["validation_rows"] == 100, "masking validation rows mismatch")
    _require(masking["masked_prompt_tokens"] > 0, "prompt tokens were not masked")
    _require(masking["response_loss_tokens"] > 0, "no response tokens contributed to loss")
    _require(masking["validation_masked_prompt_tokens"] > 0, "validation prompt not masked")
    _require(masking["validation_response_loss_tokens"] > 0, "validation response not trained")

    metrics = training["metrics"]
    expected_steps = int(training_config["smoke_steps"])
    losses = [float(loss) for loss in metrics["logged_losses"]]
    eval_losses = [float(loss) for loss in metrics["logged_eval_losses"]]
    _require(metrics["global_step"] == expected_steps, "calibration step count mismatch")
    _require(metrics["completed_steps_this_session"] == expected_steps, "session step mismatch")
    _require(len(losses) == expected_steps, "logged loss count mismatch")
    _require(metrics["oom"] is False, "calibration reported OOM")
    _require(metrics["all_losses_finite"] is True, "calibration reported non-finite loss")
    _require(all(math.isfinite(loss) for loss in losses), "logged losses contain NaN or Inf")
    _require(eval_losses and all(math.isfinite(loss) for loss in eval_losses), "invalid eval loss")
    _require(metrics["final_eval_rows"] == 100, "final eval row count mismatch")
    _require(float(metrics["seconds_per_step_excluding_checkpoint"]) > 0, "invalid step timing")
    _require(float(metrics["checkpoint_seconds_per_save"]) > 0, "invalid checkpoint timing")
    _require(float(metrics["final_eval_wall_seconds"]) > 0, "invalid evaluation timing")
    _require(
        float(metrics["peak_reserved_gib"]) < float(gpu["total_vram_gib"]),
        "peak VRAM exceeds GPU",
    )

    checkpoint = manifest["checkpoint_audit"]
    _require(checkpoint["restore_test_passed"] is True, "checkpoint restore test failed")
    _require(checkpoint["restored_global_step"] == expected_steps, "restored step mismatch")
    _require(checkpoint["retention_limit"] == 2, "checkpoint retention mismatch")
    archives = checkpoint["archives_written_this_session"]
    _require(len(archives) == 1, "calibration must write exactly one checkpoint")
    archive = archives[0]
    _require(archive["experiment_fingerprint"] == fingerprint, "checkpoint fingerprint mismatch")
    _require(archive["global_step"] == expected_steps, "checkpoint global step mismatch")
    required_members = {
        "adapter_config.json",
        "adapter_model.safetensors",
        "optimizer.pt",
        "rng_state.pth",
        "scheduler.pt",
        "trainer_state.json",
        "training_args.bin",
    }
    _require(required_members <= set(archive["members"]), "checkpoint state is incomplete")
    _require(int(archive["archive_bytes"]) > 0, "checkpoint archive is empty")

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

    recorded_cost = manifest["cost_estimate"]
    step_seconds = float(metrics["seconds_per_step_excluding_checkpoint"])
    checkpoint_seconds = float(metrics["checkpoint_seconds_per_save"])
    full_eval_seconds = (
        float(metrics["final_eval_wall_seconds"]) * expected_rows["validation"] / 100
    )
    checkpoint_events = int(recorded_cost["full_steps"]) // 100
    eval_events = checkpoint_events + 1
    projected_seconds = (
        int(recorded_cost["full_steps"]) * step_seconds
        + checkpoint_events * checkpoint_seconds
        + eval_events * full_eval_seconds
    )
    _require(
        math.isclose(float(recorded_cost["calibrated_seconds_per_step"]), step_seconds),
        "recorded step timing mismatch",
    )
    _require(
        math.isclose(
            float(recorded_cost["calibrated_checkpoint_seconds_per_save"]), checkpoint_seconds
        ),
        "recorded checkpoint timing mismatch",
    )
    _require(
        math.isclose(float(recorded_cost["calibrated_full_eval_seconds"]), full_eval_seconds),
        "recorded eval timing mismatch",
    )
    _require(
        math.isclose(float(recorded_cost["projected_seconds"]), projected_seconds),
        "recorded full projection mismatch",
    )

    archive_sha256 = str(receipt["archive_sha256"])
    _require(
        len(archive_sha256) == 64 and all(char in "0123456789abcdef" for char in archive_sha256),
        "archive SHA-256 is invalid",
    )
    _require(int(receipt["archive_bytes"]) > 0, "delivery archive is empty")
    _require(run_id in str(receipt["drive_archive"]), "Drive archive run ID mismatch")
    _require(run_id in str(receipt["drive_manifest"]), "Drive manifest run ID mismatch")

    projected_hours = projected_seconds / 3600
    projected_cu = projected_hours * observed_compute_units_per_hour
    buffered_cu = projected_cu * 1.2
    actual_session_cu = (
        float(recorded_cost["actual_session_hours"]) * observed_compute_units_per_hour
    )
    return {
        "schema_version": 1,
        "phase": 3,
        "run_mode": "calibration",
        "run_id": run_id,
        "status": "pass",
        "source_artifacts": {
            "manifest": manifest_path.name,
            "manifest_sha256": _sha256(manifest_path),
            "receipt": receipt_path.name,
            "receipt_sha256": _sha256(receipt_path),
            "drive_archive_sha256": archive_sha256,
            "drive_archive_bytes": int(receipt["archive_bytes"]),
            "checkpoint_archive_sha256": archive["archive_sha256"],
            "checkpoint_archive_bytes": int(archive["archive_bytes"]),
        },
        "validated_invariants": [
            "calibration_mode_and_full_training_disabled",
            "approved_a100_profile",
            "pinned_complete_base_snapshot",
            "fixed_data_revision_and_counts",
            "test_excluded_from_trainer",
            "response_only_masking_for_train_and_validation",
            "finite_train_and_eval_loss_without_oom",
            "checkpoint_contains_optimizer_scheduler_rng_and_trainer_state",
            "checkpoint_integrity_restore_passed",
            "adapter_saved_reloaded_and_strictly_parsed",
            "no_private_question_or_secret_fields",
        ],
        "hardware": {
            "gpu": gpu["name"],
            "profile": selected.hardware_profile,
            "precision": gpu["precision"],
            "total_vram_gib": float(gpu["total_vram_gib"]),
        },
        "calibration": {
            "train_examples": int(audit["smoke_rows"]),
            "validation_examples": int(audit["calibration_validation_rows"]),
            "steps": expected_steps,
            "seconds_per_step_excluding_checkpoint": step_seconds,
            "checkpoint_seconds_per_save": checkpoint_seconds,
            "validation_100_rows_seconds": float(metrics["final_eval_wall_seconds"]),
            "training_loss": float(metrics["training_loss"]),
            "eval_loss": eval_losses[-1],
            "first_logged_loss": losses[0],
            "last_logged_loss": losses[-1],
            "peak_allocated_gib": float(metrics["peak_allocated_gib"]),
            "peak_reserved_gib": float(metrics["peak_reserved_gib"]),
        },
        "checkpoint": {
            "fingerprint": fingerprint,
            "restore_test_passed": True,
            "restored_global_step": int(checkpoint["restored_global_step"]),
            "retention_limit": int(checkpoint["retention_limit"]),
        },
        "reload_probe": {
            "split": reload_check["probe_split"],
            "strict_parse": True,
            "correct": reload_check["prediction"] == reload_check["gold"],
            "note": "single reload probe; not an accuracy evaluation",
        },
        "billing_correction": {
            "manifest_compute_units_per_hour": float(
                recorded_cost["compute_units_per_hour_user_input"]
            ),
            "observed_compute_units_per_hour": observed_compute_units_per_hour,
            "manifest_current_compute_units": float(
                recorded_cost["current_compute_units_user_input"]
            ),
            "observed_current_compute_units": observed_current_compute_units,
            "reason": "Notebook contained example values; corrected from the Colab resource panel.",
        },
        "full_run_projection": {
            "train_examples": expected_rows["train"],
            "validation_examples": expected_rows["validation"],
            "steps": int(recorded_cost["full_steps"]),
            "checkpoint_events": checkpoint_events,
            "eval_events_including_final": eval_events,
            "hours": projected_hours,
            "compute_units_per_hour_user_observed": observed_compute_units_per_hour,
            "projected_compute_units": projected_cu,
            "projected_compute_units_with_20pct_buffer": buffered_cu,
            "current_compute_units_user_observed": observed_current_compute_units,
            "projected_remaining_compute_units": observed_current_compute_units - projected_cu,
            "projected_remaining_after_20pct_buffer": observed_current_compute_units - buffered_cu,
            "share_of_current_balance_percent": (
                projected_cu / observed_current_compute_units * 100
                if observed_current_compute_units > 0
                else None
            ),
            "actual_calibration_session_compute_units_corrected": actual_session_cu,
            "monetary_estimate": None,
            "monetary_estimate_note": "No per-compute-unit billing price was supplied.",
        },
    }

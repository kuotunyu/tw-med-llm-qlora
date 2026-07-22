"""Validate Phase 2 smoke-test evidence and derive a reproducible cost summary."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from tw_med_qlora.config import load_project_config, select_training_profile
from tw_med_qlora.cost import estimate_training_cost


class EvidenceValidationError(ValueError):
    """Raised when a smoke-test artifact violates a Phase 2 invariant."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceValidationError(message)


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


def validate_phase2_evidence(
    *,
    manifest_path: Path,
    receipt_path: Path,
    config_path: Path,
    compute_units_per_hour: float,
    current_compute_units: float,
) -> dict[str, Any]:
    """Validate a smoke run and return a deterministic, public-safe summary."""

    _require(compute_units_per_hour > 0, "compute_units_per_hour must be positive")
    _require(current_compute_units >= 0, "current_compute_units cannot be negative")

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
    _require(manifest.get("phase") == 2, "manifest is not Phase 2")
    _require(manifest.get("full_training_enabled") is False, "full training was enabled")

    gpu = manifest["gpu"]
    model = manifest["model"]
    selected = select_training_profile(
        config,
        total_vram_gib=float(gpu["total_vram_gib"]),
        bf16_supported=bool(gpu["bf16_supported"]),
        compute_capability=tuple(int(item) for item in gpu["compute_capability"]),
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
    _require(audit["trainer_referenced_splits"] == ["train"], "trainer split isolation failed")
    _require(
        audit["smoke_rows"] == int(config.raw["training"]["smoke_examples"]),
        "smoke rows mismatch",
    )

    training = manifest["training"]
    training_config = config.raw["training"]
    _require(training["batch_size"] == selected.batch_size, "batch size mismatch")
    _require(
        training["gradient_accumulation_steps"] == selected.gradient_accumulation_steps,
        "gradient accumulation mismatch",
    )
    _require(
        training["max_sequence_length"] == selected.max_sequence_length,
        "sequence length mismatch",
    )
    _require(training["seed"] == config.seed, "training seed mismatch")
    _require(training["lora_rank"] == int(training_config["lora_rank"]), "LoRA rank mismatch")
    _require(training["lora_alpha"] == int(training_config["lora_alpha"]), "LoRA alpha mismatch")
    _require(training["response_only_loss"] is True, "response-only loss was disabled")
    masking = training["masking_audit"]
    _require(masking["vision_collator"] is False, "vision collator was used for text data")
    _require(masking["test_passed_to_trainer"] is False, "test reached the collator")
    _require(masking["masked_prompt_tokens"] > 0, "prompt tokens were not masked")
    _require(masking["response_loss_tokens"] > 0, "no response tokens contributed to loss")

    smoke = training["smoke_metrics"]
    expected_steps = int(training_config["smoke_steps"])
    losses = [float(loss) for loss in smoke["logged_losses"]]
    _require(smoke["steps"] == expected_steps, "smoke step count mismatch")
    _require(len(losses) == expected_steps, "logged loss count mismatch")
    _require(smoke["oom"] is False, "smoke run reported OOM")
    _require(smoke["all_losses_finite"] is True, "smoke run reported non-finite loss")
    _require(all(math.isfinite(loss) for loss in losses), "logged losses contain NaN or Inf")
    _require(
        float(smoke["peak_reserved_gib"]) < float(gpu["total_vram_gib"]),
        "peak VRAM exceeds GPU",
    )

    reload_check = manifest["reload_check"]
    _require(reload_check["adapter_reloaded"] is True, "adapter reload failed")
    _require(reload_check["strict_parse"] is True, "validation output was not strictly parsed")
    _require(reload_check["probe_split"] == "validation", "reload probe did not use validation")
    _require(
        reload_check["published_base_model_id"] == selected.model_id,
        "published base mismatch",
    )
    _require(reload_check["adapter_parameters"] > 0, "adapter has no parameters")
    _require(reload_check["trainable_adapter_parameters"] == 0, "reloaded adapter is trainable")

    effective_batch_size = training["batch_size"] * training["gradient_accumulation_steps"]
    projection = estimate_training_cost(
        smoke_wall_seconds=float(smoke["wall_seconds"]),
        smoke_steps=int(smoke["steps"]),
        full_train_examples=expected_rows["train"],
        effective_batch_size=effective_batch_size,
        epochs=float(training_config["num_train_epochs"]),
        compute_units_per_hour=compute_units_per_hour,
    )
    recorded_cost = manifest["cost_estimate"]
    _require(recorded_cost["full_steps"] == projection.full_steps, "recorded full steps mismatch")
    _require(
        math.isclose(float(recorded_cost["projected_hours"]), projection.projected_hours),
        "recorded projected hours mismatch",
    )

    archive_sha256 = str(receipt["archive_sha256"])
    _require(
        len(archive_sha256) == 64 and all(char in "0123456789abcdef" for char in archive_sha256),
        "archive SHA-256 is invalid",
    )
    _require(int(receipt["archive_bytes"]) > 0, "archive is empty")
    _require(run_id in str(receipt["drive_archive"]), "Drive archive run ID mismatch")
    _require(run_id in str(receipt["drive_manifest"]), "Drive manifest run ID mismatch")

    projected_compute_units = float(projection.compute_units)
    projected_remaining = current_compute_units - projected_compute_units
    return {
        "schema_version": 1,
        "phase": 2,
        "run_id": run_id,
        "status": "pass",
        "source_artifacts": {
            "manifest": manifest_path.name,
            "manifest_sha256": _sha256(manifest_path),
            "receipt": receipt_path.name,
            "receipt_sha256": _sha256(receipt_path),
            "drive_archive_sha256": archive_sha256,
            "drive_archive_bytes": int(receipt["archive_bytes"]),
        },
        "validated_invariants": [
            "full_training_disabled",
            "approved_gpu_profile",
            "pinned_complete_base_snapshot",
            "fixed_data_revision_and_counts",
            "test_excluded_from_trainer",
            "response_only_masking",
            "finite_loss_without_oom",
            "adapter_saved_and_reloaded",
            "strict_validation_parse",
            "no_private_question_or_secret_fields",
        ],
        "hardware": {
            "gpu": gpu["name"],
            "profile": selected.hardware_profile,
            "precision": gpu["precision"],
            "total_vram_gib": float(gpu["total_vram_gib"]),
        },
        "smoke": {
            "examples": int(audit["smoke_rows"]),
            "steps": int(smoke["steps"]),
            "wall_seconds": float(smoke["wall_seconds"]),
            "seconds_per_step": float(smoke["seconds_per_step"]),
            "training_loss": float(smoke["training_loss"]),
            "first_logged_loss": losses[0],
            "last_logged_loss": losses[-1],
            "peak_allocated_gib": float(smoke["peak_allocated_gib"]),
            "peak_reserved_gib": float(smoke["peak_reserved_gib"]),
        },
        "reload_probe": {
            "split": reload_check["probe_split"],
            "strict_parse": True,
            "correct": reload_check["prediction"] == reload_check["gold"],
            "note": "single reload probe; not an accuracy evaluation",
        },
        "full_run_projection": {
            "examples": expected_rows["train"],
            "steps": projection.full_steps,
            "hours": projection.projected_hours,
            "compute_units_per_hour_user_observed": compute_units_per_hour,
            "projected_compute_units": projected_compute_units,
            "current_compute_units_user_observed": current_compute_units,
            "projected_remaining_compute_units": projected_remaining,
            "share_of_current_balance_percent": (
                projected_compute_units / current_compute_units * 100
                if current_compute_units > 0
                else None
            ),
            "monetary_estimate": None,
            "monetary_estimate_note": "No per-compute-unit billing price was supplied.",
        },
    }

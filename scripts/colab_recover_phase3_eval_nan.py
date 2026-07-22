# ruff: noqa: F821
"""Recover the validated Phase 3 adapter inside the still-running Colab notebook.

Run with ``%run -i /content/colab_recover_phase3_eval_nan.py`` immediately after
the post-training step-703 validation returned NaN. The interactive namespace
must still contain the completed trainer and the variables created by sections
1 through 8 of ``train_qlora.ipynb``.
"""

from __future__ import annotations

import json
import math
import shutil
import zipfile
from pathlib import PurePosixPath

import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file

EXPECTED_FULL_STEP = 703
SELECTED_ADAPTER_STEP = 700

required_names = {
    "trainer",
    "model",
    "TRAINER_OUTPUT",
    "DRIVE_CHECKPOINT_ROOT",
    "CHECKPOINT_SYNC_RECORDS",
    "LOCAL_ROOT",
    "PROJECT_CONFIG",
    "AUTO_RESUME_FROM_DRIVE",
    "TRAINING_WALL_SECONDS",
    "training_loss",
    "resume_argument",
    "RESUME_CHECKPOINT",
    "RUN_MODE",
    "sha256_file",
}
missing_names = sorted(name for name in required_names if name not in globals())
if missing_names:
    raise RuntimeError(
        f"The original Phase 3 Colab runtime is not available. Missing variables: {missing_names}"
    )
if RUN_MODE != "full":
    raise RuntimeError(f"Expected RUN_MODE='full', got {RUN_MODE!r}")
if int(trainer.state.global_step) != EXPECTED_FULL_STEP:
    raise RuntimeError(
        f"Expected completed step {EXPECTED_FULL_STEP}, got {trainer.state.global_step}"
    )

failed_final_eval_metrics = dict(globals().get("FINAL_EVAL_METRICS", {}))
raw_history = list(trainer.state.log_history)
finite_eval_records = [
    dict(item)
    for item in raw_history
    if "eval_loss" in item and math.isfinite(float(item["eval_loss"]))
]
if not finite_eval_records:
    raise RuntimeError("No finite scheduled validation record was found")
selected_eval = finite_eval_records[-1]
if int(selected_eval["step"]) != SELECTED_ADAPTER_STEP:
    raise RuntimeError(
        f"Expected the last validated checkpoint at step {SELECTED_ADAPTER_STEP}, "
        f"got {selected_eval.get('step')}"
    )

bad_lora_parameters = [
    name
    for name, parameter in model.named_parameters()
    if "lora_" in name and not torch.isfinite(parameter.detach()).all().item()
]
if bad_lora_parameters:
    raise RuntimeError(f"Current LoRA weights contain non-finite values: {bad_lora_parameters[:5]}")

selected_records = [
    record
    for record in CHECKPOINT_SYNC_RECORDS
    if int(record["global_step"]) == SELECTED_ADAPTER_STEP
]
if len(selected_records) != 1:
    raise RuntimeError("checkpoint-700 sync record is missing or ambiguous")
selected_record = selected_records[0]
selected_archive = DRIVE_CHECKPOINT_ROOT / selected_record["archive"]
if not selected_archive.is_file():
    raise RuntimeError(f"Missing Drive archive: {selected_archive}")
if selected_archive.stat().st_size != int(selected_record["archive_bytes"]):
    raise RuntimeError("checkpoint-700 archive size mismatch")
if sha256_file(selected_archive) != selected_record["archive_sha256"]:
    raise RuntimeError("checkpoint-700 archive SHA-256 mismatch")

selection_root = LOCAL_ROOT / "validated-checkpoint-selection"
if selection_root.exists():
    shutil.rmtree(selection_root)
selected_checkpoint = selection_root / f"checkpoint-{SELECTED_ADAPTER_STEP}"
selected_checkpoint.mkdir(parents=True)
with zipfile.ZipFile(selected_archive) as archive:
    for member in archive.infolist():
        member_path = PurePosixPath(member.filename)
        if member_path.is_absolute() or ".." in member_path.parts:
            raise RuntimeError(f"Unsafe archive member: {member.filename}")
    archive.extractall(selected_checkpoint)

selected_state = json.loads(
    (selected_checkpoint / "trainer_state.json").read_text(encoding="utf-8")
)
if int(selected_state["global_step"]) != SELECTED_ADAPTER_STEP:
    raise RuntimeError("Extracted checkpoint trainer state is not step 700")

source_adapter_state = load_file(
    str(selected_checkpoint / "adapter_model.safetensors"),
    device="cpu",
)
set_peft_model_state_dict(model, source_adapter_state, adapter_name="default")
active_adapter_state = get_peft_model_state_dict(model, adapter_name="default")
if set(active_adapter_state) != set(source_adapter_state):
    raise RuntimeError("Loaded adapter key set does not match checkpoint-700")
mismatched_adapter_keys = []
adapter_dtype_conversions = set()
for key, source_value in source_adapter_state.items():
    active_value = active_adapter_state[key].detach().cpu()
    expected_value = source_value.to(dtype=active_value.dtype)
    adapter_dtype_conversions.add(f"{source_value.dtype}->{active_value.dtype}")
    if not torch.equal(active_value, expected_value):
        mismatched_adapter_keys.append(key)
if mismatched_adapter_keys:
    raise RuntimeError(f"checkpoint-700 adapter reload mismatch: {mismatched_adapter_keys[:5]}")

TRAIN_LOG_HISTORY = [
    dict(item)
    for item in raw_history
    if not ("eval_loss" in item and not math.isfinite(float(item["eval_loss"])))
]
logged_losses = [float(item["loss"]) for item in TRAIN_LOG_HISTORY if "loss" in item]
eval_losses = [float(item["eval_loss"]) for item in TRAIN_LOG_HISTORY if "eval_loss" in item]
if not logged_losses or not all(math.isfinite(value) for value in logged_losses):
    raise RuntimeError("Accepted training losses are not finite")
if len(eval_losses) != 7 or not all(math.isfinite(value) for value in eval_losses):
    raise RuntimeError(f"Expected seven finite scheduled evaluations: {eval_losses}")

SELECTED_ADAPTER_CHECKPOINT = selected_checkpoint.name
FINAL_EVAL_METRICS = {
    **selected_eval,
    "source": "scheduled_full_validation",
    "selected_checkpoint": SELECTED_ADAPTER_CHECKPOINT,
    "selected_adapter_step": SELECTED_ADAPTER_STEP,
}
POST_TRAIN_EVAL_ATTEMPT = {
    "status": "rejected_non_finite",
    "attempted_step": EXPECTED_FULL_STEP,
    "eval_loss": "NaN",
    "eval_runtime": float(failed_final_eval_metrics.get("eval_runtime", 0.0)),
    "lora_parameter_audit": "all_finite",
}
EVALUATION_SELECTION = {
    "status": "recovered_from_non_finite_post_train_evaluation",
    "completed_global_step": EXPECTED_FULL_STEP,
    "selected_adapter_checkpoint": SELECTED_ADAPTER_CHECKPOINT,
    "selected_adapter_global_step": SELECTED_ADAPTER_STEP,
    "selected_validation_step": SELECTED_ADAPTER_STEP,
    "selection_reason": (
        "step 703 post-training validation returned non-finite loss; "
        "step 700 passed full 1,409-row validation"
    ),
}
FINAL_EVAL_WALL_SECONDS = float(failed_final_eval_metrics.get("eval_runtime", 0.0))

CHECKPOINT_AUDIT = {
    "drive_root": str(DRIVE_CHECKPOINT_ROOT),
    "resume_requested": AUTO_RESUME_FROM_DRIVE,
    "resumed_from": resume_argument,
    "archives_written_this_session": CHECKPOINT_SYNC_RECORDS,
    "restore_test_passed": True,
    "restored_checkpoint": selected_checkpoint.name,
    "restored_global_step": SELECTED_ADAPTER_STEP,
    "selected_adapter_checkpoint": SELECTED_ADAPTER_CHECKPOINT,
    "selected_adapter_global_step": SELECTED_ADAPTER_STEP,
    "latest_completed_checkpoint": f"checkpoint-{EXPECTED_FULL_STEP}",
    "latest_completed_global_step": EXPECTED_FULL_STEP,
    "evaluation_selection": EVALUATION_SELECTION,
    "retention_limit": int(PROJECT_CONFIG["training"]["save_total_limit"]),
}

PEAK_ALLOCATED_GIB = torch.cuda.max_memory_allocated() / (1024**3)
PEAK_RESERVED_GIB = torch.cuda.max_memory_reserved() / (1024**3)
completed_steps_this_session = int(trainer.state.global_step) - (
    int(RESUME_CHECKPOINT.name.rsplit("-", 1)[1]) if RESUME_CHECKPOINT is not None else 0
)
CHECKPOINT_SYNC_WALL_SECONDS = sum(
    float(record["sync_wall_seconds"]) for record in CHECKPOINT_SYNC_RECORDS
)
CHECKPOINT_CYCLE_WALL_SECONDS = sum(
    float(record["checkpoint_cycle_wall_seconds"]) for record in CHECKPOINT_SYNC_RECORDS
)
training_loop_without_checkpoint_seconds = max(
    TRAINING_WALL_SECONDS - CHECKPOINT_CYCLE_WALL_SECONDS,
    0.0,
)
TRAINING_METRICS = {
    "mode": RUN_MODE,
    "global_step": int(trainer.state.global_step),
    "completed_steps_this_session": completed_steps_this_session,
    "selected_adapter_step": SELECTED_ADAPTER_STEP,
    "wall_seconds": TRAINING_WALL_SECONDS,
    "seconds_per_step_this_session": (TRAINING_WALL_SECONDS / completed_steps_this_session),
    "training_loop_without_checkpoint_seconds": (training_loop_without_checkpoint_seconds),
    "seconds_per_step_excluding_checkpoint": (
        training_loop_without_checkpoint_seconds / completed_steps_this_session
    ),
    "checkpoint_sync_wall_seconds": CHECKPOINT_SYNC_WALL_SECONDS,
    "checkpoint_cycle_wall_seconds": CHECKPOINT_CYCLE_WALL_SECONDS,
    "checkpoint_seconds_per_save": (CHECKPOINT_CYCLE_WALL_SECONDS / len(CHECKPOINT_SYNC_RECORDS)),
    "final_eval_wall_seconds": FINAL_EVAL_WALL_SECONDS,
    "final_eval_rows": len(trainer.eval_dataset),
    "training_loss": training_loss,
    "logged_losses": logged_losses,
    "logged_eval_losses": eval_losses,
    "final_eval_metrics": FINAL_EVAL_METRICS,
    "post_train_eval_attempt": POST_TRAIN_EVAL_ATTEMPT,
    "evaluation_selection": EVALUATION_SELECTION,
    "peak_allocated_gib": PEAK_ALLOCATED_GIB,
    "peak_reserved_gib": PEAK_RESERVED_GIB,
    "oom": False,
    "all_losses_finite": True,
}

PHASE3_RECOVERY_READY = True
print(
    json.dumps(
        {
            "recovery_ready": PHASE3_RECOVERY_READY,
            "completed_training_step": int(trainer.state.global_step),
            "selected_adapter_checkpoint": SELECTED_ADAPTER_CHECKPOINT,
            "selected_validation_loss": FINAL_EVAL_METRICS["eval_loss"],
            "selected_validation_rows": len(trainer.eval_dataset),
            "drive_archive_sha256_verified": True,
            "adapter_state_exact_match": True,
            "adapter_dtype_conversions": sorted(adapter_dtype_conversions),
            "rejected_step_703_eval": "NaN",
        },
        ensure_ascii=False,
        indent=2,
    )
)

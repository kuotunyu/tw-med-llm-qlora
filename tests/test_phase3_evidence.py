from __future__ import annotations

import json
from pathlib import Path

import pytest

from tw_med_qlora.phase3_evidence import (
    Phase3EvidenceValidationError,
    validate_phase3_calibration_evidence,
)

ROOT = Path(__file__).parents[1]
RUN_DIR = ROOT / "reports" / "phase3"
MANIFEST = RUN_DIR / "20260721T171557Z-run-manifest.json"
RECEIPT = RUN_DIR / "20260721T171557Z-receipt.json"
VALIDATION = RUN_DIR / "20260721T171557Z-validation.json"
CONFIG = ROOT / "configs" / "project.toml"


def test_phase3_evidence_passes_and_corrects_resource_panel_values() -> None:
    summary = validate_phase3_calibration_evidence(
        manifest_path=MANIFEST,
        receipt_path=RECEIPT,
        config_path=CONFIG,
        observed_compute_units_per_hour=5.3,
        observed_current_compute_units=436.2,
    )

    assert summary["status"] == "pass"
    assert summary["hardware"]["profile"] == "primary_40g"
    assert summary["calibration"]["steps"] == 10
    assert summary["checkpoint"]["restore_test_passed"] is True
    assert summary["full_run_projection"]["hours"] == pytest.approx(4.99762320997)
    assert summary["full_run_projection"]["projected_compute_units"] == pytest.approx(
        26.4874030128
    )
    assert summary["full_run_projection"][
        "projected_compute_units_with_20pct_buffer"
    ] == pytest.approx(31.7848836154)
    assert summary["reload_probe"]["correct"] is False
    assert summary == json.loads(VALIDATION.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("target_file", "mutation", "message"),
    [
        ("manifest", ("full_training_enabled", True), "full training was enabled"),
        ("manifest", ("data.audit.test_used_for_training", True), "test was used"),
        ("manifest", ("training.metrics.oom", True), "reported OOM"),
        (
            "manifest",
            ("checkpoint_audit.restore_test_passed", False),
            "checkpoint restore test failed",
        ),
        (
            "manifest",
            ("checkpoint_audit.archives_written_this_session.0.members", []),
            "checkpoint state is incomplete",
        ),
        ("receipt", ("experiment_fingerprint", "wrong"), "fingerprint mismatch"),
    ],
)
def test_phase3_evidence_rejects_failed_gate(
    tmp_path: Path,
    target_file: str,
    mutation: tuple[str, object],
    message: str,
) -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    target = manifest if target_file == "manifest" else receipt
    path, value = mutation
    parts = path.split(".")
    for part in parts[:-1]:
        target = target[int(part)] if part.isdigit() else target[part]
    last = parts[-1]
    if last.isdigit():
        target[int(last)] = value
    else:
        target[last] = value

    manifest_path = tmp_path / MANIFEST.name
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    receipt_path = tmp_path / RECEIPT.name
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(Phase3EvidenceValidationError, match=message):
        validate_phase3_calibration_evidence(
            manifest_path=manifest_path,
            receipt_path=receipt_path,
            config_path=CONFIG,
            observed_compute_units_per_hour=5.3,
            observed_current_compute_units=436.2,
        )

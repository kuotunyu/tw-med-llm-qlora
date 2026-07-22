from __future__ import annotations

import json
from pathlib import Path

import pytest

from tw_med_qlora.phase2_evidence import (
    EvidenceValidationError,
    validate_phase2_evidence,
)

ROOT = Path(__file__).parents[1]
RUN_DIR = ROOT / "reports" / "phase2"
MANIFEST = RUN_DIR / "20260721T160727Z-run-manifest.json"
RECEIPT = RUN_DIR / "20260721T160727Z-receipt.json"
VALIDATION = RUN_DIR / "20260721T160727Z-validation.json"
CONFIG = ROOT / "configs" / "project.toml"


def test_phase2_evidence_passes_and_recomputes_user_observed_cu() -> None:
    summary = validate_phase2_evidence(
        manifest_path=MANIFEST,
        receipt_path=RECEIPT,
        config_path=CONFIG,
        compute_units_per_hour=1.54,
        current_compute_units=437.16,
    )

    assert summary["status"] == "pass"
    assert summary["hardware"]["profile"] == "primary_24g"
    assert summary["smoke"]["steps"] == 10
    assert summary["full_run_projection"]["hours"] == pytest.approx(13.3646885297)
    assert summary["full_run_projection"]["projected_compute_units"] == pytest.approx(
        20.5816203358
    )
    assert summary["reload_probe"]["correct"] is False
    assert summary == json.loads(VALIDATION.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (("full_training_enabled", True), "full training was enabled"),
        (("data.audit.test_used_for_training", True), "test was used for training"),
        (("training.smoke_metrics.oom", True), "reported OOM"),
        (("training.response_only_loss", False), "response-only loss was disabled"),
    ],
)
def test_phase2_evidence_rejects_failed_gate(
    tmp_path: Path,
    mutation: tuple[str, object],
    message: str,
) -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    path, value = mutation
    target = manifest
    parts = path.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value

    manifest_path = tmp_path / MANIFEST.name
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    receipt_path = tmp_path / RECEIPT.name
    receipt_path.write_bytes(RECEIPT.read_bytes())

    with pytest.raises(EvidenceValidationError, match=message):
        validate_phase2_evidence(
            manifest_path=manifest_path,
            receipt_path=receipt_path,
            config_path=CONFIG,
            compute_units_per_hour=1.54,
            current_compute_units=437.16,
        )

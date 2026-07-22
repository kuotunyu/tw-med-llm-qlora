from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from tw_med_qlora.phase4_evidence import (
    Phase4EvidenceValidationError,
    validate_phase4_calibration_evidence,
)

ROOT = Path(__file__).parents[1]
SOURCE_DIR = ROOT / "reports" / "phase4" / "calibration"
CONFIG = ROOT / "configs" / "project.toml"
RUN_ID = "20260722T052039Z"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fixture(
    tmp_path: Path,
    *,
    unlock_full: bool = False,
    include_test_member: bool = False,
    reviewed_protocol: bool = False,
    reviewed_gate_passed: bool = True,
    reviewed_parse_failure_count: int = 0,
) -> tuple[Path, Path, Path, Path]:
    manifest = json.loads(
        (SOURCE_DIR / f"{RUN_ID}-run-manifest.json").read_text(encoding="utf-8")
    )
    receipt = json.loads(
        (SOURCE_DIR / f"{RUN_ID}-receipt.json").read_text(encoding="utf-8")
    )
    summary_source = SOURCE_DIR / f"{RUN_ID}-calibration-summary.json"
    summary = json.loads(summary_source.read_text(encoding="utf-8"))

    manifest["full_evaluation_unlocked"] = unlock_full
    receipt["full_evaluation_unlocked"] = unlock_full
    archive_path = tmp_path / f"{RUN_ID}-phase4-calibration-private.zip"
    with zipfile.ZipFile(archive_path, mode="w") as archive:
        model_outputs = {
            "original-instruct": [rf"\boxed{{{'ABCD'[index % 4]}}}" for index in range(20)],
            "localized-base": (
                [
                    (
                        "reasoning without a final answer"
                        if index < reviewed_parse_failure_count
                        else "ABCD"[index % 4]
                    )
                    for index in range(20)
                ]
                if reviewed_protocol
                else ["reasoning was truncated" for _ in range(20)]
            ),
            "localized-medical-adapter": ["ABCD"[index % 4] for index in range(20)],
        }
        for model_name, outputs in model_outputs.items():
            rows = [
                json.dumps(
                    {
                        "example_id": f"id-{index:02d}",
                        "model": model_name,
                        "subject": f"subject-{index % 13}",
                        "gold": "ABCD"[index % 4],
                        "raw_output": output,
                    },
                    ensure_ascii=False,
                )
                for index, output in enumerate(outputs)
            ]
            archive.writestr(f"private/{model_name}-raw.jsonl", "\n".join(rows) + "\n")
        if include_test_member:
            archive.writestr("private/forbidden_test.csv", "private")

    archive_hash = _sha256(archive_path)
    archive_bytes = archive_path.stat().st_size
    if reviewed_protocol:
        for metrics in summary["models"].values():
            metrics.update(
                {
                    "parsed": 20,
                    "parse_failures": 0,
                    "correct": 20,
                    "accuracy": 1.0,
                    "parse_rate": 1.0,
                    "completion_tokens_total": 20,
                    "max_token_limit_hits": 0,
                }
            )
        summary["generation_contract"] = {
            "parser": "standalone_A-D_or_exactly_one_simple_boxed_A-D",
            "scorer": "exact_match",
            "max_tokens": 256,
            "minimum_parse_rate": 0.8,
        }
        if reviewed_parse_failure_count:
            base = summary["models"]["localized-base"]
            parsed = 20 - reviewed_parse_failure_count
            base.update(
                {
                    "parsed": parsed,
                    "parse_failures": reviewed_parse_failure_count,
                    "correct": parsed,
                    "accuracy": parsed / 20,
                    "parse_rate": parsed / 20,
                }
            )
        parse_rate_failures = (
            {"localized-base": (20 - reviewed_parse_failure_count) / 20}
            if reviewed_parse_failure_count > 4
            else {}
        )
        producer_gate_passed = reviewed_gate_passed and not parse_rate_failures
        summary["generation_gate"] = {
            "passed": producer_gate_passed,
            "parse_rate_failures": parse_rate_failures,
            "max_token_limit_failures": (
                {} if reviewed_gate_passed else {"localized-base": 1}
            ),
            "failure_action": (
                None
                if producer_gate_passed
                else "Do not unlock full evaluation; review archived evidence."
            ),
        }
        if not reviewed_gate_passed:
            summary["models"]["localized-base"]["max_token_limit_hits"] = 1
        manifest["calibration_summary"] = summary
    manifest["private_archive"] = {"sha256": archive_hash, "bytes": archive_bytes}
    receipt["archive_sha256"] = archive_hash
    receipt["archive_bytes"] = archive_bytes

    manifest_path = tmp_path / f"{RUN_ID}-run-manifest.json"
    receipt_path = tmp_path / f"{RUN_ID}-receipt.json"
    summary_path = tmp_path / f"{RUN_ID}-calibration-summary.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    return manifest_path, receipt_path, summary_path, archive_path


def _validate(paths: tuple[Path, Path, Path, Path]) -> dict[str, object]:
    manifest, receipt, summary, archive = paths
    return validate_phase4_calibration_evidence(
        manifest_path=manifest,
        receipt_path=receipt,
        calibration_summary_path=summary,
        private_archive_path=archive,
        config_path=CONFIG,
    )


def test_phase4_evidence_integrity_passes_but_requires_recalibration(tmp_path: Path) -> None:
    result = _validate(_write_fixture(tmp_path))

    assert result["status"] == "recalibration_required"
    assert result["evidence_integrity"] == "pass"
    assert result["cost_projection_is_provisional"] is True
    assert result["test_isolation"] == {
        "manifest_test_files_loaded": 0,
        "private_archive_test_members": 0,
        "full_evaluation_unlocked": False,
    }
    audit = result["calibration"]["format_audit"]
    assert audit["localized-medical-adapter"]["reparsed"] == 20
    assert audit["localized-base"]["reparsed"] == 0


def test_phase4_evidence_rejects_unlocked_full_evaluation(tmp_path: Path) -> None:
    with pytest.raises(Phase4EvidenceValidationError, match="full evaluation was unlocked"):
        _validate(_write_fixture(tmp_path, unlock_full=True))


def test_phase4_reviewed_protocol_can_pass(tmp_path: Path) -> None:
    result = _validate(_write_fixture(tmp_path, reviewed_protocol=True))

    assert result["status"] == "pass"
    assert result["cost_projection_is_provisional"] is False
    assert result["diagnosis"]["all_public_counts_reproduced_from_private_outputs"] is True


def test_phase4_reviewed_gate_failure_preserves_integrity_result(tmp_path: Path) -> None:
    result = _validate(
        _write_fixture(
            tmp_path,
            reviewed_protocol=True,
            reviewed_gate_passed=False,
        )
    )

    assert result["status"] == "pass_after_protocol_review"
    assert result["evidence_integrity"] == "pass"
    assert result["cost_projection_is_provisional"] is False
    assert result["diagnosis"]["producer_generation_gate_passed"] is False
    assert result["diagnosis"]["reviewed_generation_gate_passed"] is True
    assert result["diagnosis"]["observed_max_token_limit_hits"] == {
        "localized-base": 1
    }


def test_phase4_review_does_not_override_parse_rate_failure(tmp_path: Path) -> None:
    result = _validate(
        _write_fixture(
            tmp_path,
            reviewed_protocol=True,
            reviewed_parse_failure_count=5,
        )
    )

    assert result["status"] == "recalibration_required"
    assert result["diagnosis"]["reviewed_generation_gate_passed"] is False


def test_phase4_evidence_rejects_test_member_in_private_archive(tmp_path: Path) -> None:
    with pytest.raises(Phase4EvidenceValidationError, match="contains test data"):
        _validate(_write_fixture(tmp_path, include_test_member=True))

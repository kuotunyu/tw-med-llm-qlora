"""Validate Phase 4 calibration evidence without publishing private prompts or outputs."""

from __future__ import annotations

import hashlib
import json
import math
import re
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from tw_med_qlora.config import load_project_config
from tw_med_qlora.evaluation import parse_mcq_answer


class Phase4EvidenceValidationError(ValueError):
    """Raised when a Phase 4 calibration artifact violates an invariant."""


_RUN_ID = re.compile(r"\A(\d{8}T\d{6}Z)-run-manifest\.json\Z")
_MODEL_FILES = {
    "original-instruct": "private/original-instruct-raw.jsonl",
    "localized-base": "private/localized-base-raw.jsonl",
    "localized-medical-adapter": "private/localized-medical-adapter-raw.jsonl",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Phase4EvidenceValidationError(message)


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


def _assert_public_safe(value: Any, *, location: str = "root") -> None:
    forbidden = {
        "choices",
        "hf_token",
        "prompt",
        "question",
        "raw_output",
    }
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).casefold()
            _require(normalized not in forbidden, f"private field found at {location}.{key}")
            _assert_public_safe(item, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_public_safe(item, location=f"{location}[{index}]")


def _validate_summary(summary: dict[str, Any], *, expected_rows: int) -> None:
    _require(summary.get("split") == "validation", "summary is not validation-only")
    _require(summary.get("unique_questions") == expected_rows, "summary row count mismatch")
    models = summary.get("models")
    _require(isinstance(models, dict), "summary models are missing")
    _require(set(models) == set(_MODEL_FILES), "summary model set mismatch")
    for model_name, metrics in models.items():
        total = metrics.get("total")
        parsed = metrics.get("parsed")
        failures = metrics.get("parse_failures")
        correct = metrics.get("correct")
        _require(total == expected_rows, f"{model_name} total mismatch")
        _require(parsed + failures == total, f"{model_name} parse counts mismatch")
        _require(0 <= correct <= parsed <= total, f"{model_name} score counts are invalid")
        _require(
            math.isclose(float(metrics["accuracy"]), correct / total),
            f"{model_name} accuracy mismatch",
        )
        _require(
            math.isclose(float(metrics["parse_rate"]), parsed / total),
            f"{model_name} parse rate mismatch",
        )


def _read_private_records(
    archive: zipfile.ZipFile,
    *,
    member: str,
    model_name: str,
    expected_rows: int,
) -> list[dict[str, Any]]:
    try:
        payload = archive.read(member).decode("utf-8")
    except KeyError as error:
        raise Phase4EvidenceValidationError(f"private archive is missing {member}") from error
    records = [json.loads(line) for line in payload.splitlines() if line.strip()]
    _require(len(records) == expected_rows, f"{model_name} private row count mismatch")
    for record in records:
        _require(record.get("model") == model_name, f"{model_name} record label mismatch")
        _require(record.get("gold") in {"A", "B", "C", "D"}, "invalid private gold")
        _require(isinstance(record.get("example_id"), str), "private example ID is invalid")
        _require(isinstance(record.get("raw_output"), str), "private raw output is invalid")
    ids = [record["example_id"] for record in records]
    _require(len(set(ids)) == expected_rows, f"{model_name} private IDs are not unique")
    return records


def validate_phase4_calibration_evidence(
    *,
    manifest_path: Path,
    receipt_path: Path,
    calibration_summary_path: Path,
    private_archive_path: Path,
    config_path: Path,
) -> dict[str, Any]:
    """Validate evidence integrity and diagnose the first calibration parser contract."""

    manifest = _load_object(manifest_path)
    receipt = _load_object(receipt_path)
    calibration_summary = _load_object(calibration_summary_path)
    for value in (manifest, receipt, calibration_summary):
        _assert_public_safe(value)

    name_match = _RUN_ID.fullmatch(manifest_path.name)
    _require(name_match is not None, "manifest filename is invalid")
    run_id = name_match.group(1)
    _require(receipt_path.name == f"{run_id}-receipt.json", "receipt filename mismatch")
    _require(
        calibration_summary_path.name == f"{run_id}-calibration-summary.json",
        "calibration summary filename mismatch",
    )
    _require(
        private_archive_path.name == f"{run_id}-phase4-calibration-private.zip",
        "private archive filename mismatch",
    )

    config = load_project_config(config_path).raw
    evaluation = config["evaluation"]
    primary = config["models"]["primary"]
    tmmlu = config["data"]["tmmluplus"]
    expected_rows = int(evaluation["calibration_examples"])

    _require(manifest.get("schema_version") == 1, "unsupported manifest schema")
    _require(manifest.get("phase") == 4, "manifest is not Phase 4")
    _require(manifest.get("run_mode") == "calibration", "manifest is not calibration")
    _require(manifest.get("full_evaluation_unlocked") is False, "full evaluation was unlocked")
    _require(manifest.get("test_files_loaded") == 0, "test files were loaded")
    _require(receipt.get("phase") == 4, "receipt is not Phase 4")
    _require(receipt.get("run_mode") == "calibration", "receipt mode mismatch")
    _require(receipt.get("full_evaluation_unlocked") is False, "receipt unlocked full eval")

    hardware = manifest["hardware"]
    _require("A100" in str(hardware["gpu_name"]).upper(), "calibration GPU is not A100")
    _require(float(hardware["gpu_vram_gib"]) >= 38, "A100 VRAM is below reviewed profile")
    _require(hardware["bf16_supported"] is True, "BF16 support was not detected")
    _require(hardware["cuda_version"] == "12.9", "CUDA runtime mismatch")

    models = manifest["models"]
    _require(models["original"]["id"] == primary["baseline_id"], "baseline ID mismatch")
    _require(
        models["original"]["revision"] == primary["baseline_revision"],
        "baseline revision mismatch",
    )
    _require(models["localized_base"]["id"] == primary["model_id"], "base ID mismatch")
    _require(
        models["localized_base"]["revision"] == primary["revision"],
        "base revision mismatch",
    )
    adapter = models["adapter"]
    expected_adapter = evaluation["phase3_adapter"]
    _require(
        adapter["archive_sha256"] == expected_adapter["archive_sha256"],
        "adapter archive hash mismatch",
    )
    _require(
        adapter["archive_bytes"] == expected_adapter["archive_bytes"],
        "adapter size mismatch",
    )
    _require(adapter["base_model_id"] == primary["model_id"], "adapter base mismatch")

    data = manifest["data"]
    _require(data["dataset_id"] == tmmlu["dataset_id"], "TMMLU+ dataset ID mismatch")
    _require(data["revision"] == tmmlu["revision"], "TMMLU+ revision mismatch")
    _require(data["split"] == "validation", "calibration accessed a non-validation split")
    _require(data["unique_questions"] == expected_rows, "manifest question count mismatch")
    _require(
        data["subject_count"]
        == len(evaluation["medical_subjects"] + evaluation["control_subjects"]),
        "subject count mismatch",
    )

    dependencies = manifest["dependencies"]
    _require(dependencies["vllm"]["installed_version"] == "0.25.1+cu129", "vLLM mismatch")
    _require(dependencies["twinkle-eval"]["installed_version"] == "2.8.0", "Twinkle mismatch")
    _require(
        dependencies["native_cuda_preflight"]["cuda_available"] is True,
        "CUDA preflight failed",
    )
    _require(
        dependencies["native_cuda_preflight"]["torch_cuda"] == "12.9",
        "native CUDA preflight mismatch",
    )

    _require(manifest["calibration_summary"] == calibration_summary, "summary payload mismatch")
    _validate_summary(calibration_summary, expected_rows=expected_rows)

    workload = manifest["workload"]
    expected_workload = evaluation["workload"]
    _require(workload["total"] == expected_workload["expected_total_requests"], "workload mismatch")
    cost = manifest["cost_estimate"]
    _require(cost["measured_requests"] == expected_rows * 3, "measured request count mismatch")
    _require(float(cost["compute_units_per_hour_user_input"]) > 0, "CU rate is invalid")
    _require(float(cost["projected_hours"]) > 0, "projected hours are invalid")

    archive_hash = _sha256(private_archive_path)
    archive_bytes = private_archive_path.stat().st_size
    _require(
        manifest["private_archive"]["sha256"] == archive_hash,
        "private archive hash mismatch",
    )
    _require(
        manifest["private_archive"]["bytes"] == archive_bytes,
        "private archive size mismatch",
    )
    _require(receipt["archive_sha256"] == archive_hash, "private archive hash mismatch")
    _require(receipt["archive_bytes"] == archive_bytes, "private archive size mismatch")

    private_records: dict[str, list[dict[str, Any]]] = {}
    with zipfile.ZipFile(private_archive_path) as archive:
        for info in archive.infolist():
            path = PurePosixPath(info.filename)
            _require(not path.is_absolute(), "private archive contains an absolute path")
            _require(".." not in path.parts, "private archive contains path traversal")
            _require("test" not in info.filename.casefold(), "private archive contains test data")
        for model_name, member in _MODEL_FILES.items():
            private_records[model_name] = _read_private_records(
                archive,
                member=member,
                model_name=model_name,
                expected_rows=expected_rows,
            )

    id_sets = [{record["example_id"] for record in rows} for rows in private_records.values()]
    _require(all(ids == id_sets[0] for ids in id_sets[1:]), "private model question sets differ")

    format_audit: dict[str, dict[str, Any]] = {}
    for model_name, records in private_records.items():
        parsed = [parse_mcq_answer(record["raw_output"]) for record in records]
        reparsed_correct = sum(
            prediction == record["gold"]
            for prediction, record in zip(parsed, records, strict=True)
        )
        output_lengths = [len(record["raw_output"]) for record in records]
        exact_letters = sum(
            re.fullmatch(r"\s*[A-D]\s*[。.]?\s*", record["raw_output"]) is not None
            for record in records
        )
        valid_box_counts = [
            len(re.findall(r"\\boxed\s*\{\s*[A-D]\s*\}", record["raw_output"]))
            for record in records
        ]
        format_audit[model_name] = {
            "rows": len(records),
            "reparsed": sum(value is not None for value in parsed),
            "reparsed_correct": reparsed_correct,
            "exact_standalone_answers": exact_letters,
            "responses_with_one_valid_box": sum(count == 1 for count in valid_box_counts),
            "responses_with_multiple_valid_boxes": sum(count > 1 for count in valid_box_counts),
            "minimum_characters": min(output_lengths),
            "maximum_characters": max(output_lengths),
            "mean_characters": sum(output_lengths) / len(output_lengths),
        }

    reported_models = calibration_summary["models"]
    reported_adapter_parsed = reported_models["localized-medical-adapter"]["parsed"]
    reparsed_adapter = format_audit["localized-medical-adapter"]["reparsed"]
    generation_contract = calibration_summary.get("generation_contract")
    if generation_contract is None:
        _require(reported_adapter_parsed == 0, "unexpected legacy adapter parse count")
        _require(
            reparsed_adapter == expected_rows,
            "adapter direct-answer diagnosis did not reproduce",
        )
        _require(
            format_audit["localized-base"]["reparsed"] == 0,
            "unexpected localized-base parse result in legacy calibration",
        )
        status = "recalibration_required"
        diagnosis = {
            "legacy_extractor": manifest["twinkle_eval_contract"]["extractor"],
            "adapter_reported_parsed": reported_adapter_parsed,
            "adapter_reparsed_with_reviewed_contract": reparsed_adapter,
            "localized_base_reparsed": format_audit["localized-base"]["reparsed"],
            "conclusion": (
                "The adapter emitted valid standalone A-D answers, but the box-only "
                "extractor rejected them. Localized-base outputs reached the generation "
                "cap before a unique answer. Accuracy values from this run are invalid."
            ),
        }
    else:
        expected_generation = evaluation["generation"]
        _require(
            generation_contract["parser"]
            == "standalone_A-D_or_exactly_one_simple_boxed_A-D",
            "reviewed parser contract mismatch",
        )
        _require(
            generation_contract["max_tokens"] == expected_generation["max_tokens"],
            "generation token limit mismatch",
        )
        _require(
            generation_contract["minimum_parse_rate"]
            == expected_generation["minimum_calibration_parse_rate"],
            "minimum parse rate mismatch",
        )
        for model_name, audit in format_audit.items():
            reported = reported_models[model_name]
            _require(reported["parsed"] == audit["reparsed"], f"{model_name} parser mismatch")
            _require(
                reported["correct"] == audit["reparsed_correct"],
                f"{model_name} correctness mismatch",
            )
        expected_parse_failures = {
            model_name: reported["parse_rate"]
            for model_name, reported in reported_models.items()
            if reported["parse_rate"]
            < expected_generation["minimum_calibration_parse_rate"]
        }
        expected_token_failures = {
            model_name: reported["max_token_limit_hits"]
            for model_name, reported in reported_models.items()
            if reported["max_token_limit_hits"] > 0
        }
        producer_token_limits_fatal = bool(
            generation_contract.get("token_limit_hits_fail_calibration", True)
        )
        _require(
            generation_contract.get("token_limit_hits_count_as_incorrect", True) is True,
            "token-limit outputs were not counted as incorrect",
        )
        expected_producer_gate_passed = not (
            expected_parse_failures
            or (producer_token_limits_fatal and expected_token_failures)
        )
        gate = calibration_summary.get("generation_gate")
        _require(isinstance(gate, dict), "generation gate is missing")
        _require(
            gate.get("passed") is expected_producer_gate_passed,
            "generation gate status mismatch",
        )
        _require(
            gate.get("parse_rate_failures") == expected_parse_failures,
            "generation gate parse failures mismatch",
        )
        if "observed_max_token_limit_hits" in gate:
            _require(
                gate["observed_max_token_limit_hits"] == expected_token_failures,
                "observed token-limit hits mismatch",
            )
        _require(
            gate.get("max_token_limit_failures")
            == (expected_token_failures if producer_token_limits_fatal else {}),
            "generation gate token failures mismatch",
        )
        reviewed_token_limits_fatal = bool(
            expected_generation["token_limit_hits_fail_calibration"]
        )
        reviewed_gate_passed = not (
            expected_parse_failures
            or (reviewed_token_limits_fatal and expected_token_failures)
        )
        if expected_producer_gate_passed:
            status = "pass"
        elif reviewed_gate_passed:
            status = "pass_after_protocol_review"
        else:
            status = "recalibration_required"
        diagnosis = {
            "parser_contract": generation_contract["parser"],
            "all_public_counts_reproduced_from_private_outputs": True,
            "producer_generation_gate_passed": expected_producer_gate_passed,
            "reviewed_generation_gate_passed": reviewed_gate_passed,
            "parse_rate_failures": expected_parse_failures,
            "observed_max_token_limit_hits": expected_token_failures,
            "reviewed_token_limit_policy": (
                "fatal"
                if reviewed_token_limits_fatal
                else "count_as_incorrect_and_report"
            ),
            "conclusion": (
                "The reviewed calibration generation and parsing contract passed."
                if status == "pass"
                else (
                    "The producer gate failed only because token-limit hits were fatal; "
                    "the reviewed policy counts those outputs as incorrect and reports them."
                    if status == "pass_after_protocol_review"
                    else "Evidence integrity passed, but the reviewed generation gate failed."
                )
            ),
        }

    return {
        "status": status,
        "evidence_integrity": "pass",
        "phase": 4,
        "run_mode": "calibration",
        "run_id": run_id,
        "test_isolation": {
            "manifest_test_files_loaded": 0,
            "private_archive_test_members": 0,
            "full_evaluation_unlocked": False,
        },
        "hardware": {
            "gpu_name": hardware["gpu_name"],
            "gpu_vram_gib": hardware["gpu_vram_gib"],
            "torch_version": hardware["torch_version"],
            "cuda_version": hardware["cuda_version"],
            "vllm_version": dependencies["vllm"]["installed_version"],
        },
        "calibration": {
            "split": "validation",
            "unique_questions": expected_rows,
            "requests": expected_rows * 3,
            "format_audit": format_audit,
        },
        "diagnosis": diagnosis,
        "cost_projection_is_provisional": status == "recalibration_required",
        "artifact_sha256": {
            "manifest": _sha256(manifest_path),
            "receipt": _sha256(receipt_path),
            "calibration_summary": _sha256(calibration_summary_path),
            "private_archive": archive_hash,
        },
    }

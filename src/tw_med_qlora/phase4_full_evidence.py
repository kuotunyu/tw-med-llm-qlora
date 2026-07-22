"""Validate downloaded Phase 4 full-evaluation evidence without exposing content."""

from __future__ import annotations

import json
import math
import re
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

from tw_med_qlora.evaluation import PredictionRecord, accuracy_summary, subject_accuracy
from tw_med_qlora.phase4_full import PUBLIC_FORBIDDEN_KEYS, file_sha256

_SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")
_MODEL_LABELS = {
    "original-instruct",
    "localized-base",
    "localized-medical-adapter",
}


def _read_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    payload = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path.name}")
    return payload


def _safe_members(archive: zipfile.ZipFile) -> list[str]:
    names = archive.namelist()
    for name in names:
        member = PurePosixPath(name.replace("\\", "/"))
        if member.is_absolute() or ".." in member.parts:
            raise ValueError(f"unsafe ZIP member: {name}")
    return [name for name in names if not name.endswith("/")]


def _json_from_zip(archive: zipfile.ZipFile, name: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    return json.loads(archive.read(name).decode("utf-8"), parse_constant=reject_constant)


def _public_rows(archive: zipfile.ZipFile) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        archive.read("public/public-predictions.jsonl").decode("utf-8").splitlines(), start=1
    ):
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"public prediction line {line_number} must be an object")
        leaked = PUBLIC_FORBIDDEN_KEYS.intersection(row)
        if leaked:
            raise ValueError(f"public prediction contains private keys: {sorted(leaked)}")
        rows.append(row)
    return rows


def _as_prediction(row: dict[str, Any]) -> PredictionRecord:
    return PredictionRecord(
        example_id=row["example_id"],
        model=row["model"],
        source=row["source"],
        subject=row["subject"],
        gold=row["gold"],
        prediction=row["prediction"],
        raw_output_sha256=row["raw_output_sha256"],
        latency_seconds=float(row["latency_seconds"]),
        prompt_tokens=row["prompt_tokens"],
        completion_tokens=row["completion_tokens"],
    )


def validate_phase4_full_evidence(
    *,
    manifest_path: Path,
    receipt_path: Path,
    public_archive_path: Path,
    private_cases_archive_path: Path,
    expected_requests: int = 28_758,
) -> dict[str, Any]:
    """Recompute the content-safe invariants for one formal Phase 4 delivery."""

    if expected_requests <= 0:
        raise ValueError("expected_requests must be positive")
    manifest = _read_json(manifest_path)
    receipt = _read_json(receipt_path)
    errors: list[str] = []

    if manifest.get("phase") != 4 or manifest.get("run_mode") != "full":
        errors.append("manifest is not a Phase 4 full run")
    if manifest.get("full_evaluation_unlocked") is not True:
        errors.append("full evaluation was not unlocked")
    approval = manifest.get("user_approval", {})
    if approval.get("approved_requests") != expected_requests:
        errors.append("approved request count mismatch")
    fingerprint = manifest.get("contract_fingerprint")
    if not isinstance(fingerprint, str) or not _SHA256.fullmatch(fingerprint):
        errors.append("invalid contract fingerprint")
    if receipt.get("contract_fingerprint") != fingerprint:
        errors.append("receipt contract fingerprint mismatch")
    if receipt.get("completed_requests") != expected_requests:
        errors.append("receipt completed request count mismatch")
    if manifest.get("resumption", {}).get("completed_requests") != expected_requests:
        errors.append("manifest completed request count mismatch")

    for label, path, metadata_key, receipt_key in (
        (
            "public",
            public_archive_path,
            "public_archive",
            "public_archive_sha256",
        ),
        (
            "private_cases",
            private_cases_archive_path,
            "private_cases_archive",
            "private_cases_archive_sha256",
        ),
    ):
        if not path.is_file():
            errors.append(f"{label} archive is missing")
            continue
        actual_hash = file_sha256(path)
        metadata = manifest.get(metadata_key, {})
        if metadata.get("sha256") != actual_hash or receipt.get(receipt_key) != actual_hash:
            errors.append(f"{label} archive SHA-256 mismatch")
        if metadata.get("bytes") != path.stat().st_size:
            errors.append(f"{label} archive size mismatch")

    if errors:
        return {"status": "failed", "errors": errors, "checks": {}}

    with zipfile.ZipFile(public_archive_path) as archive:
        names = set(_safe_members(archive))
        expected_names = {
            "public/phase4-results.json",
            "public/medqa-summary.json",
            "public/tmmlu-summary.json",
            "public/stability-summary.json",
            "public/public-predictions.jsonl",
        }
        if names != expected_names:
            errors.append(f"unexpected public archive members: {sorted(names)}")
            return {"status": "failed", "errors": errors, "checks": {}}
        results = _json_from_zip(archive, "public/phase4-results.json")
        medqa_summary = _json_from_zip(archive, "public/medqa-summary.json")
        tmmlu_summary = _json_from_zip(archive, "public/tmmlu-summary.json")
        stability_summary = _json_from_zip(archive, "public/stability-summary.json")
        rows = _public_rows(archive)

    if len(rows) != expected_requests or results.get("generation_requests") != expected_requests:
        errors.append("public prediction count mismatch")
    if results.get("contract_fingerprint") != fingerprint:
        errors.append("public results contract fingerprint mismatch")
    if results.get("medqa") != medqa_summary:
        errors.append("embedded MedQA summary differs from standalone file")
    if results.get("tmmluplus") != tmmlu_summary:
        errors.append("embedded TMMLU+ summary differs from standalone file")
    if results.get("stability") != stability_summary:
        errors.append("embedded stability summary differs from standalone file")

    pairs = [(row.get("model"), row.get("request_id")) for row in rows]
    if len(set(pairs)) != len(pairs):
        errors.append("duplicate model/request result")
    if {row.get("model") for row in rows} != _MODEL_LABELS:
        errors.append("formal result model set mismatch")
    for row in rows:
        if row.get("parsed") != (row.get("prediction") is not None):
            errors.append("stored parse flag mismatch")
            break
        if row.get("correct") != (row.get("prediction") == row.get("gold")):
            errors.append("stored correctness flag mismatch")
            break
        if row.get("max_token_limit_hit") and row.get("prediction") is not None:
            errors.append("token-limit output was not forced to parse failure")
            break
        latency = row.get("latency_seconds")
        if not isinstance(latency, (int, float)) or not math.isfinite(latency) or latency < 0:
            errors.append("invalid public latency")
            break

    suite_counts = Counter((row.get("model"), row.get("suite")) for row in rows)
    expected_suite_counts = {
        ("original-instruct", "medqa-full"): 1413,
        ("original-instruct", "tmmlu-full"): 5573,
        ("localized-base", "medqa-full"): 1413,
        ("localized-base", "tmmlu-full"): 5573,
        ("localized-medical-adapter", "medqa-full"): 1413,
        ("localized-medical-adapter", "tmmlu-full"): 5573,
    }
    if expected_requests == 28_758:
        for model in ("localized-base", "localized-medical-adapter"):
            for seed in (3407, 3408, 3409):
                expected_suite_counts[(model, f"tmmlu-stability-{seed}")] = 1300
        if dict(suite_counts) != expected_suite_counts:
            errors.append("formal suite request counts mismatch")

    for model in _MODEL_LABELS:
        for suite, summary in (
            ("medqa-full", medqa_summary.get("models", {}).get(model)),
            ("tmmlu-full", tmmlu_summary.get("models", {}).get(model, {}).get("overall")),
        ):
            model_rows = [
                row
                for row in rows
                if row.get("model") == model and row.get("suite") == suite
            ]
            if not model_rows or summary is None:
                errors.append(f"missing summary rows: {model}/{suite}")
                continue
            recomputed = accuracy_summary(_as_prediction(row) for row in model_rows)
            for key, value in recomputed.items():
                if summary.get(key) != value:
                    errors.append(f"summary mismatch: {model}/{suite}/{key}")
                    break
            if suite == "tmmlu-full":
                by_subject = subject_accuracy(_as_prediction(row) for row in model_rows)
                if tmmlu_summary["models"][model].get("by_subject") != by_subject:
                    errors.append(f"subject summary mismatch: {model}")

    with zipfile.ZipFile(private_cases_archive_path) as archive:
        names = _safe_members(archive)
        if names != ["medqa-representative-cases-private.json"]:
            errors.append(f"unexpected private cases members: {names}")
            private_cases = []
        else:
            private_cases = _json_from_zip(
                archive, "medqa-representative-cases-private.json"
            )
    if not isinstance(private_cases, list) or len(private_cases) != 10:
        errors.append("private representative case count must be 10")

    checks = {
        "request_count": len(rows),
        "unique_model_request_pairs": len(set(pairs)),
        "public_archive_sha256": file_sha256(public_archive_path),
        "private_cases_archive_sha256": file_sha256(private_cases_archive_path),
        "public_content_keys_blocked": sorted(PUBLIC_FORBIDDEN_KEYS),
        "private_representative_cases": len(private_cases),
        "suite_counts": {
            f"{model}/{suite}": count
            for (model, suite), count in sorted(suite_counts.items())
        },
    }
    return {"status": "passed" if not errors else "failed", "errors": errors, "checks": checks}

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from tw_med_qlora.evaluation import PredictionRecord, accuracy_summary, subject_accuracy
from tw_med_qlora.phase4_full_evidence import validate_phase4_full_evidence


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _evidence(tmp_path: Path, *, leak_question: bool = False):
    models = ["original-instruct", "localized-base", "localized-medical-adapter"]
    rows = []
    medqa_models = {}
    tmmlu_models = {}
    for model_index, model in enumerate(models):
        for suite in ("medqa-full", "tmmlu-full"):
            raw = "A"
            row = {
                "request_id": hashlib.sha256(f"{suite}:{model}".encode()).hexdigest()[:24],
                "example_id": f"{suite}-q",
                "suite": suite,
                "option_seed": None if suite == "medqa-full" else 3407,
                "model": model,
                "source": "dataset",
                "subject": "medqa_total" if suite == "medqa-full" else "subject",
                "gold": "A",
                "prediction": "A" if model_index != 1 else None,
                "parsed": model_index != 1,
                "correct": model_index != 1,
                "raw_output_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                "latency_seconds": 0.1,
                "prompt_tokens": 10,
                "completion_tokens": 1,
                "finish_reason": "stop",
                "max_token_limit_hit": False,
            }
            if leak_question:
                row["question"] = "private"
            rows.append(row)

        medqa_record = _record(rows[-2])
        tmmlu_record = _record(rows[-1])
        medqa_models[model] = accuracy_summary([medqa_record])
        tmmlu_models[model] = {
            "overall": accuracy_summary([tmmlu_record]),
            "by_subject": subject_accuracy([tmmlu_record]),
        }
    medqa = {"models": medqa_models}
    tmmlu = {"models": tmmlu_models}
    stability = {"models": {}}
    results = {
        "contract_fingerprint": "a" * 64,
        "generation_requests": 6,
        "medqa": medqa,
        "tmmluplus": tmmlu,
        "stability": stability,
    }

    public_archive = tmp_path / "public.zip"
    with zipfile.ZipFile(public_archive, "w") as archive:
        archive.writestr("public/phase4-results.json", json.dumps(results))
        archive.writestr("public/medqa-summary.json", json.dumps(medqa))
        archive.writestr("public/tmmlu-summary.json", json.dumps(tmmlu))
        archive.writestr("public/stability-summary.json", json.dumps(stability))
        archive.writestr(
            "public/public-predictions.jsonl",
            "".join(json.dumps(row) + "\n" for row in rows),
        )
    private_archive = tmp_path / "private.zip"
    with zipfile.ZipFile(private_archive, "w") as archive:
        archive.writestr(
            "medqa-representative-cases-private.json",
            json.dumps([{"case": index} for index in range(10)]),
        )

    public_hash = hashlib.sha256(public_archive.read_bytes()).hexdigest()
    private_hash = hashlib.sha256(private_archive.read_bytes()).hexdigest()
    manifest = {
        "phase": 4,
        "run_mode": "full",
        "full_evaluation_unlocked": True,
        "user_approval": {"approved_requests": 6},
        "contract_fingerprint": "a" * 64,
        "resumption": {"completed_requests": 6},
        "public_archive": {"sha256": public_hash, "bytes": public_archive.stat().st_size},
        "private_cases_archive": {
            "sha256": private_hash,
            "bytes": private_archive.stat().st_size,
        },
    }
    receipt = {
        "contract_fingerprint": "a" * 64,
        "completed_requests": 6,
        "public_archive_sha256": public_hash,
        "private_cases_archive_sha256": private_hash,
    }
    manifest_path = tmp_path / "manifest.json"
    receipt_path = tmp_path / "receipt.json"
    _write_json(manifest_path, manifest)
    _write_json(receipt_path, receipt)
    return manifest_path, receipt_path, public_archive, private_archive


def _record(row) -> PredictionRecord:
    return PredictionRecord(
        example_id=row["example_id"],
        model=row["model"],
        source=row["source"],
        subject=row["subject"],
        gold=row["gold"],
        prediction=row["prediction"],
        raw_output_sha256=row["raw_output_sha256"],
        latency_seconds=row["latency_seconds"],
        prompt_tokens=row["prompt_tokens"],
        completion_tokens=row["completion_tokens"],
    )


def test_full_evidence_validator_recomputes_public_results(tmp_path: Path) -> None:
    paths = _evidence(tmp_path)

    result = validate_phase4_full_evidence(
        manifest_path=paths[0],
        receipt_path=paths[1],
        public_archive_path=paths[2],
        private_cases_archive_path=paths[3],
        expected_requests=6,
    )

    assert result["status"] == "passed"
    assert result["checks"]["request_count"] == 6
    assert result["checks"]["private_representative_cases"] == 10


def test_full_evidence_validator_rejects_archive_hash_mismatch(tmp_path: Path) -> None:
    paths = _evidence(tmp_path)
    paths[2].write_bytes(paths[2].read_bytes() + b"tamper")

    result = validate_phase4_full_evidence(
        manifest_path=paths[0],
        receipt_path=paths[1],
        public_archive_path=paths[2],
        private_cases_archive_path=paths[3],
        expected_requests=6,
    )

    assert result["status"] == "failed"
    assert "public archive SHA-256 mismatch" in result["errors"]


def test_full_evidence_validator_rejects_private_key_in_public_rows(tmp_path: Path) -> None:
    paths = _evidence(tmp_path, leak_question=True)

    with pytest.raises(ValueError, match="private keys"):
        validate_phase4_full_evidence(
            manifest_path=paths[0],
            receipt_path=paths[1],
            public_archive_path=paths[2],
            private_cases_archive_path=paths[3],
            expected_requests=6,
        )

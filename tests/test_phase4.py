from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from tw_med_qlora.phase4 import (
    build_vllm_serve_command,
    extract_verified_adapter,
    phase4_workload,
    project_evaluation_cost,
)


def _workload():
    return phase4_workload(
        medqa_test_rows=1413,
        tmmlu_test_rows=5573,
        full_model_count=3,
        subject_count=13,
        stability_examples_per_subject=100,
        stability_seeds=[3407, 3408, 3409],
        stability_model_count=2,
    )


def test_phase4_workload_matches_predeclared_generation_count() -> None:
    assert _workload().as_dict() == {
        "medqa_full": 4239,
        "tmmlu_full": 16719,
        "tmmlu_stability": 7800,
        "total": 28758,
    }


def test_cost_projection_uses_measured_rate_startup_and_buffer() -> None:
    result = project_evaluation_cost(
        workload=_workload(),
        measured_requests=60,
        measured_inference_seconds=120,
        measured_server_startup_seconds=30,
        planned_server_starts=3,
        compute_units_per_hour=5.3,
    )

    assert result["projected_seconds"] == pytest.approx(57_606)
    assert result["projected_compute_units_with_buffer"] == pytest.approx(101.7706)
    assert result["estimated_cost"] is None


def _archive(tmp_path: Path, *, base_model: str = "localized/base") -> tuple[Path, str, int]:
    source = tmp_path / "source"
    adapter = source / "delivery" / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": base_model}), encoding="utf-8"
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"synthetic adapter")
    path = tmp_path / "adapter.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for file_path in source.rglob("*"):
            if file_path.is_file():
                archive.write(file_path, file_path.relative_to(source).as_posix())
    return path, hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_size


def test_adapter_archive_is_hash_checked_and_base_verified(tmp_path: Path) -> None:
    archive, digest, size = _archive(tmp_path)
    result = extract_verified_adapter(
        archive,
        tmp_path / "extracted",
        expected_sha256=digest,
        expected_bytes=size,
        expected_base_model_id="localized/base",
    )

    assert result["archive_sha256"] == digest
    assert Path(result["adapter_dir"]).name == "adapter"
    assert result["weight_files"] == ["adapter_model.safetensors"]


def test_adapter_archive_rejects_hash_and_base_mismatch(tmp_path: Path) -> None:
    archive, digest, size = _archive(tmp_path)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        extract_verified_adapter(
            archive,
            tmp_path / "bad-hash",
            expected_sha256="0" * 64,
            expected_bytes=size,
            expected_base_model_id="localized/base",
        )
    with pytest.raises(ValueError, match="adapter/base mismatch"):
        extract_verified_adapter(
            archive,
            tmp_path / "bad-base",
            expected_sha256=digest,
            expected_bytes=size,
            expected_base_model_id="different/base",
        )


def test_vllm_command_pins_revision_quantization_and_optional_adapter(tmp_path: Path) -> None:
    command = build_vllm_serve_command(
        model_id="localized/base",
        model_revision="a" * 40,
        served_model_name="localized-adapter",
        port=8000,
        max_model_length=2048,
        gpu_memory_utilization=0.85,
        seed=3407,
        adapter_name="medical",
        adapter_path=tmp_path / "adapter",
    )
    joined = " ".join(command)

    assert "--quantization bitsandbytes" in joined
    assert "--language-model-only" in command
    assert joined.count("a" * 40) == 2
    assert "--enable-lora" in command
    assert "HF_TOKEN" not in joined


def test_vllm_command_requires_adapter_name_and_path_together(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="provided together"):
        build_vllm_serve_command(
            model_id="base",
            model_revision="a" * 40,
            served_model_name="base",
            port=8000,
            max_model_length=2048,
            gpu_memory_utilization=0.85,
            seed=3407,
            adapter_path=tmp_path,
        )

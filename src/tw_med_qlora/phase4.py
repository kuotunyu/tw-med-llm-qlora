"""Phase 4 workload, adapter, and Colab serving safety helpers."""

from __future__ import annotations

import hashlib
import json
import math
import zipfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any


@dataclass(frozen=True)
class EvaluationWorkload:
    """Predeclared request counts for the two evaluation tracks."""

    medqa_full: int
    tmmlu_full: int
    tmmlu_stability: int

    @property
    def total(self) -> int:
        return self.medqa_full + self.tmmlu_full + self.tmmlu_stability

    def as_dict(self) -> dict[str, int]:
        return {**asdict(self), "total": self.total}


def phase4_workload(
    *,
    medqa_test_rows: int,
    tmmlu_test_rows: int,
    full_model_count: int,
    subject_count: int,
    stability_examples_per_subject: int,
    stability_seeds: Sequence[int],
    stability_model_count: int,
) -> EvaluationWorkload:
    """Calculate the approved Phase 4 generation count without hidden multipliers."""

    values = {
        "medqa_test_rows": medqa_test_rows,
        "tmmlu_test_rows": tmmlu_test_rows,
        "full_model_count": full_model_count,
        "subject_count": subject_count,
        "stability_examples_per_subject": stability_examples_per_subject,
        "stability_model_count": stability_model_count,
    }
    if any(value <= 0 for value in values.values()):
        raise ValueError(f"workload values must be positive: {values}")
    if not stability_seeds:
        raise ValueError("at least one stability seed is required")
    if len(set(stability_seeds)) != len(stability_seeds):
        raise ValueError("stability seeds must be unique")

    return EvaluationWorkload(
        medqa_full=medqa_test_rows * full_model_count,
        tmmlu_full=tmmlu_test_rows * full_model_count,
        tmmlu_stability=(
            subject_count
            * stability_examples_per_subject
            * len(stability_seeds)
            * stability_model_count
        ),
    )


def project_evaluation_cost(
    *,
    workload: EvaluationWorkload,
    measured_requests: int,
    measured_inference_seconds: float,
    measured_server_startup_seconds: float,
    planned_server_starts: int,
    compute_units_per_hour: float,
    buffer_fraction: float = 0.20,
    price_per_compute_unit: float | None = None,
    currency: str | None = None,
) -> dict[str, int | float | str | None]:
    """Project full evaluation from measured generation and server-start timings."""

    numeric = (
        measured_inference_seconds,
        measured_server_startup_seconds,
        compute_units_per_hour,
        buffer_fraction,
    )
    if measured_requests <= 0 or planned_server_starts <= 0:
        raise ValueError("request and server-start counts must be positive")
    if any(not math.isfinite(value) for value in numeric):
        raise ValueError("cost inputs must be finite")
    if measured_inference_seconds <= 0 or measured_server_startup_seconds < 0:
        raise ValueError("measured timings are invalid")
    if compute_units_per_hour <= 0 or buffer_fraction < 0:
        raise ValueError("compute-unit rate must be positive and buffer non-negative")
    if price_per_compute_unit is not None and price_per_compute_unit < 0:
        raise ValueError("price per compute unit cannot be negative")

    seconds_per_request = measured_inference_seconds / measured_requests
    projected_seconds = (
        seconds_per_request * workload.total
        + measured_server_startup_seconds * planned_server_starts
    )
    projected_hours = projected_seconds / 3600
    projected_compute_units = projected_hours * compute_units_per_hour
    buffered_compute_units = projected_compute_units * (1 + buffer_fraction)
    estimated_cost = (
        buffered_compute_units * price_per_compute_unit
        if price_per_compute_unit is not None
        else None
    )
    return {
        "measured_requests": measured_requests,
        "measured_seconds_per_request": seconds_per_request,
        "planned_server_starts": planned_server_starts,
        "projected_seconds": projected_seconds,
        "projected_hours": projected_hours,
        "compute_units_per_hour_user_input": compute_units_per_hour,
        "projected_compute_units": projected_compute_units,
        "buffer_fraction": buffer_fraction,
        "projected_compute_units_with_buffer": buffered_compute_units,
        "price_per_compute_unit_user_input": price_per_compute_unit,
        "estimated_cost": estimated_cost,
        "currency_user_input": currency,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_verified_adapter(
    archive_path: Path,
    destination: Path,
    *,
    expected_sha256: str,
    expected_bytes: int,
    expected_base_model_id: str,
) -> dict[str, Any]:
    """Verify and safely extract the reviewed Phase 3 adapter archive."""

    if not archive_path.is_file():
        raise FileNotFoundError(f"adapter archive not found: {archive_path}")
    actual_bytes = archive_path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"adapter archive size mismatch: expected={expected_bytes}, actual={actual_bytes}"
        )
    actual_sha256 = _file_sha256(archive_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "adapter archive SHA-256 mismatch: "
            f"expected={expected_sha256}, actual={actual_sha256}"
        )

    destination.mkdir(parents=True, exist_ok=True)
    resolved_destination = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            member_path = PurePosixPath(member.filename.replace("\\", "/"))
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"unsafe adapter archive member: {member.filename}")
            target = (destination / Path(*member_path.parts)).resolve()
            if target != resolved_destination and resolved_destination not in target.parents:
                raise ValueError(f"archive member escapes destination: {member.filename}")
        archive.extractall(destination)

    adapter_configs = list(destination.rglob("adapter_config.json"))
    if len(adapter_configs) != 1:
        raise ValueError(
            "expected exactly one adapter_config.json; "
            f"found={len(adapter_configs)}"
        )
    config_path = adapter_configs[0]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    actual_base = config.get("base_model_name_or_path")
    if actual_base != expected_base_model_id:
        raise ValueError(
            "adapter/base mismatch: "
            f"expected={expected_base_model_id}, actual={actual_base}"
        )
    weight_files = sorted(config_path.parent.glob("adapter_model*.safetensors"))
    if not weight_files:
        raise ValueError("adapter archive does not contain safetensors weights")
    return {
        "archive_sha256": actual_sha256,
        "archive_bytes": actual_bytes,
        "adapter_dir": str(config_path.parent),
        "base_model_id": actual_base,
        "adapter_config_sha256": _file_sha256(config_path),
        "weight_files": [path.name for path in weight_files],
    }


def build_vllm_serve_command(
    *,
    model_id: str,
    model_revision: str,
    served_model_name: str,
    port: int,
    max_model_length: int,
    gpu_memory_utilization: float,
    seed: int,
    adapter_name: str | None = None,
    adapter_path: Path | None = None,
    max_lora_rank: int = 16,
) -> list[str]:
    """Build the pinned Colab-only vLLM CLI without embedding secrets."""

    if not model_id or not model_revision or not served_model_name:
        raise ValueError("model identifiers must not be empty")
    if not 1 <= port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    if max_model_length <= 0 or max_lora_rank <= 0:
        raise ValueError("model length and LoRA rank must be positive")
    if not 0 < gpu_memory_utilization < 1:
        raise ValueError("gpu memory utilization must be between zero and one")
    if (adapter_name is None) != (adapter_path is None):
        raise ValueError("adapter name and path must be provided together")

    command = [
        "vllm",
        "serve",
        model_id,
        "--revision",
        model_revision,
        "--tokenizer-revision",
        model_revision,
        "--served-model-name",
        served_model_name,
        "--port",
        str(port),
        "--dtype",
        "bfloat16",
        "--quantization",
        "bitsandbytes",
        "--max-model-len",
        str(max_model_length),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--generation-config",
        "vllm",
        "--language-model-only",
        "--seed",
        str(seed),
    ]
    if adapter_path is not None and adapter_name is not None:
        command.extend(
            [
                "--enable-lora",
                "--max-lora-rank",
                str(max_lora_rank),
                "--lora-modules",
                json.dumps(
                    {"name": adapter_name, "path": str(adapter_path)},
                    separators=(",", ":"),
                ),
            ]
        )
    return command

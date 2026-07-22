"""Validation for content-safe RTX 4090 Phase 5 acceptance manifests."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from .config import ProjectConfig
from .local_inference import ACCEPTANCE_EXPECTED_ANSWER, inference_requirements

_SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"\A[0-9a-f]{40}\Z")
_FORBIDDEN_CONTENT_KEYS = {"prompt", "text", "raw_output"}


class Phase5EvidenceError(ValueError):
    """Raised when a local inference manifest cannot prove the Phase 5 contract."""


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise Phase5EvidenceError(f"{label} must be an object")
    return value


def _finite_nonnegative(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Phase5EvidenceError(f"{label} must be numeric")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0 or (positive and converted <= 0):
        qualifier = "positive" if positive else "non-negative"
        raise Phase5EvidenceError(f"{label} must be finite and {qualifier}")
    return converted


def _assert_no_content_keys(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in _FORBIDDEN_CONTENT_KEYS:
                raise Phase5EvidenceError(f"private content key is forbidden: {path}.{key}")
            _assert_no_content_keys(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_content_keys(child, f"{path}[{index}]")


def validate_phase5_manifest(
    manifest: dict[str, Any],
    *,
    config: ProjectConfig,
) -> dict[str, Any]:
    """Prove the pinned adapter ran successfully on the intended Windows 4090 stack."""

    _assert_no_content_keys(manifest)
    if manifest.get("schema_version") != 1 or manifest.get("phase") != 5:
        raise Phase5EvidenceError("unexpected Phase 5 manifest schema")

    expected = inference_requirements(config)
    base = _mapping(manifest.get("base_model"), "base_model")
    if base.get("model_id") != config.primary.model_id:
        raise Phase5EvidenceError("base model ID does not match the primary profile")
    if base.get("revision") != config.primary.revision:
        raise Phase5EvidenceError("base model revision is not the pinned Phase 3 revision")

    adapter = _mapping(manifest.get("adapter"), "adapter")
    if adapter.get("base_model_name_or_path") != config.primary.model_id:
        raise Phase5EvidenceError("adapter base model does not match the primary profile")
    if adapter.get("base_model_revision") not in {None, config.primary.revision}:
        raise Phase5EvidenceError("adapter declares an incompatible base revision")
    if adapter.get("peft_type") != "LORA":
        raise Phase5EvidenceError("adapter must be a LoRA adapter")
    for key in ("config_sha256",):
        if not _SHA256.fullmatch(str(adapter.get(key, ""))):
            raise Phase5EvidenceError(f"adapter {key} is not SHA-256")
    source_type = adapter.get("source_type")
    if source_type == "local":
        if not _SHA256.fullmatch(str(adapter.get("local_path_sha256", ""))):
            raise Phase5EvidenceError("local adapter path hash is missing")
        if not _SHA256.fullmatch(str(adapter.get("weights_sha256", ""))):
            raise Phase5EvidenceError("local adapter weights hash is missing")
    elif source_type == "huggingface_hub":
        repo_id = adapter.get("repo_id")
        if not isinstance(repo_id, str) or repo_id.count("/") != 1:
            raise Phase5EvidenceError("Hub adapter repo ID is invalid")
        if not _COMMIT.fullmatch(str(adapter.get("resolved_revision", ""))):
            raise Phase5EvidenceError("Hub adapter revision must resolve to a full commit")
    else:
        raise Phase5EvidenceError(f"unknown adapter source_type: {source_type!r}")

    hardware = _mapping(manifest.get("hardware"), "hardware")
    if hardware.get("eligible") is not True or hardware.get("failures") != []:
        raise Phase5EvidenceError("hardware preflight did not pass")
    if hardware.get("os") != "Windows":
        raise Phase5EvidenceError("acceptance did not run on Windows")
    nvidia = _mapping(hardware.get("nvidia_smi"), "hardware.nvidia_smi")
    if expected.gpu_name_contains.casefold() not in str(nvidia.get("name", "")).casefold():
        raise Phase5EvidenceError("acceptance GPU is not the configured RTX 4090")
    detected_vram = _finite_nonnegative(
        nvidia.get("total_vram_gib"), "total_vram_gib"
    )
    if detected_vram < expected.minimum_vram_gib:
        raise Phase5EvidenceError("acceptance GPU has insufficient VRAM")
    torch_runtime = _mapping(hardware.get("torch"), "hardware.torch")
    if torch_runtime.get("cuda_available") is not True:
        raise Phase5EvidenceError("PyTorch CUDA was unavailable")
    if expected.requires_bf16 and torch_runtime.get("bf16_supported") is not True:
        raise Phase5EvidenceError("PyTorch BF16 support was unavailable")

    quantization = _mapping(manifest.get("quantization"), "quantization")
    expected_quantization = config.raw["inference"]["windows_4090"]
    if quantization.get("load_in_4bit") is not True:
        raise Phase5EvidenceError("base model was not loaded in 4-bit")
    if quantization.get("quant_type") != expected_quantization["quantization_type"]:
        raise Phase5EvidenceError("unexpected 4-bit quantization type")
    if quantization.get("compute_dtype") != "bfloat16":
        raise Phase5EvidenceError("inference compute dtype was not bfloat16")
    if quantization.get("attention") != expected_quantization["attention_implementation"]:
        raise Phase5EvidenceError("unexpected attention implementation")

    timing = _mapping(manifest.get("timing"), "timing")
    model_load = _finite_nonnegative(
        timing.get("model_load_seconds"), "model_load_seconds", positive=True
    )
    first_token = _finite_nonnegative(
        timing.get("first_token_seconds"), "first_token_seconds", positive=True
    )
    total_generation = _finite_nonnegative(
        timing.get("total_generation_seconds"),
        "total_generation_seconds",
        positive=True,
    )
    if first_token > total_generation:
        raise Phase5EvidenceError("first-token latency exceeds total generation latency")

    memory = _mapping(manifest.get("memory"), "memory")
    peak_allocated = _finite_nonnegative(
        memory.get("peak_allocated_gib"), "peak_allocated_gib", positive=True
    )
    peak_reserved = _finite_nonnegative(
        memory.get("peak_reserved_gib"), "peak_reserved_gib", positive=True
    )
    if peak_allocated > peak_reserved or peak_reserved > float(nvidia["total_vram_gib"]):
        raise Phase5EvidenceError("reported peak VRAM is inconsistent")

    generation = _mapping(manifest.get("generation"), "generation")
    for key in ("prompt_sha256", "raw_output_sha256"):
        if not _SHA256.fullmatch(str(generation.get(key, ""))):
            raise Phase5EvidenceError(f"generation {key} is not SHA-256")
    if generation.get("parsed_answer") != ACCEPTANCE_EXPECTED_ANSWER:
        raise Phase5EvidenceError("synthetic acceptance answer was not parsed correctly")
    if int(generation.get("prompt_tokens", 0)) <= 0 or int(
        generation.get("completion_tokens", 0)
    ) <= 0:
        raise Phase5EvidenceError("token counts must be positive")

    acceptance = _mapping(manifest.get("acceptance"), "acceptance")
    if acceptance != {
        "probe": "synthetic_unit_mcq_v1",
        "expected_answer": ACCEPTANCE_EXPECTED_ANSWER,
        "passed": True,
    }:
        raise Phase5EvidenceError("synthetic acceptance gate did not pass")

    return {
        "valid": True,
        "phase": 5,
        "gpu": nvidia["name"],
        "base_model": base["model_id"],
        "base_revision": base["revision"],
        "adapter_source_type": source_type,
        "adapter_revision": adapter.get("resolved_revision"),
        "model_load_seconds": model_load,
        "first_token_seconds": first_token,
        "total_generation_seconds": total_generation,
        "peak_allocated_gib": peak_allocated,
        "peak_reserved_gib": peak_reserved,
        "parsed_answer": generation["parsed_answer"],
        "private_content_absent": True,
    }


def validate_phase5_file(path: Path, *, config: ProjectConfig) -> dict[str, Any]:
    """Load one UTF-8 JSON manifest and validate it."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Phase5EvidenceError(f"cannot read Phase 5 manifest: {path}") from exc
    return validate_phase5_manifest(_mapping(payload, "manifest"), config=config)

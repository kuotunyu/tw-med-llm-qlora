import json
from pathlib import Path

import pytest

from tw_med_qlora.config import load_project_config
from tw_med_qlora.local_inference import (
    AdapterContract,
    GenerationResult,
    build_private_safe_manifest,
)
from tw_med_qlora.phase5_evidence import Phase5EvidenceError, validate_phase5_manifest

ROOT = Path(__file__).parents[1]
CONFIG = load_project_config(ROOT / "configs" / "project.toml")


def valid_manifest() -> dict:
    contract = AdapterContract(
        source=str((ROOT / "adapter").resolve()),
        base_model_name_or_path=CONFIG.primary.model_id,
        base_model_revision=None,
        peft_type="LORA",
        task_type="CAUSAL_LM",
        inference_mode=True,
        config_sha256="a" * 64,
        weights_sha256="b" * 64,
        resolved_revision=None,
    )
    result = GenerationResult(
        text="C",
        parsed_answer="C",
        prompt_tokens=35,
        completion_tokens=1,
        first_token_seconds=0.25,
        total_generation_seconds=0.4,
        peak_allocated_gib=8.5,
        peak_reserved_gib=9.0,
    )
    manifest = build_private_safe_manifest(
        result=result,
        prompt="private acceptance prompt",
        base_model=CONFIG.primary.model_id,
        base_revision=CONFIG.primary.revision,
        adapter_contract=contract,
        adapter_revision=None,
        hardware={
            "eligible": True,
            "failures": [],
            "os": "Windows",
            "nvidia_smi": {
                "name": "NVIDIA GeForce RTX 4090",
                "total_vram_gib": 23.99,
            },
            "torch": {
                "cuda_available": True,
                "bf16_supported": True,
                "torch_version": "2.10.0+cu128",
            },
        },
        model_load_seconds=42.0,
    )
    manifest["acceptance"] = {
        "probe": "synthetic_unit_mcq_v1",
        "expected_answer": "C",
        "passed": True,
    }
    return manifest


def test_valid_phase5_manifest_proves_acceptance() -> None:
    report = validate_phase5_manifest(valid_manifest(), config=CONFIG)

    assert report["valid"] is True
    assert report["private_content_absent"] is True
    assert report["parsed_answer"] == "C"


def test_tracked_rtx4090_acceptance_evidence_is_valid() -> None:
    path = ROOT / "reports" / "phase5" / "20260722T131736Z-acceptance.json"
    report = validate_phase5_manifest(
        json.loads(path.read_text(encoding="utf-8")), config=CONFIG
    )

    assert report["valid"] is True
    assert report["gpu"] == "NVIDIA GeForce RTX 4090"
    assert report["parsed_answer"] == "C"
    assert report["private_content_absent"] is True


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("base_model", "revision"), "f" * 40, "base model revision"),
        (("hardware", "eligible"), False, "preflight"),
        (("quantization", "compute_dtype"), "float16", "bfloat16"),
        (("generation", "parsed_answer"), None, "not parsed"),
        (("acceptance", "passed"), False, "acceptance gate"),
    ],
)
def test_phase5_manifest_rejects_contract_failures(
    path: tuple[str, str], value: object, message: str
) -> None:
    manifest = valid_manifest()
    manifest[path[0]][path[1]] = value

    with pytest.raises(Phase5EvidenceError, match=message):
        validate_phase5_manifest(manifest, config=CONFIG)


def test_phase5_manifest_rejects_private_content() -> None:
    manifest = valid_manifest()
    manifest["generation"]["raw_output"] = "C"

    with pytest.raises(Phase5EvidenceError, match="private content key"):
        validate_phase5_manifest(manifest, config=CONFIG)


def test_phase5_manifest_is_json_serializable_without_prompt() -> None:
    serialized = json.dumps(valid_manifest(), ensure_ascii=False)

    assert "private acceptance prompt" not in serialized

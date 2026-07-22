import json
import tomllib
from pathlib import Path

import pytest

from tw_med_qlora.local_inference import (
    AdapterContract,
    GenerationResult,
    InferenceRequirements,
    NvidiaGpu,
    TorchCudaRuntime,
    build_messages,
    build_private_safe_manifest,
    hardware_preflight,
    load_adapter_contract,
    parse_nvidia_smi_row,
    validate_adapter_contract,
)

ROOT = Path(__file__).parents[1]


def test_windows_inference_uses_locked_cuda_128_torch_wheel() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    inference = project["dependency-groups"]["inference"]
    assert "pillow>=12,<13" in inference
    assert "torchvision>=0.25,<0.26" in inference

    expected_sources = [
        {"index": "pytorch-cu128", "marker": "sys_platform == 'win32'"}
    ]
    sources = project["tool"]["uv"]["sources"]
    assert sources["torch"] == expected_sources
    assert sources["torchvision"] == expected_sources

    indexes = {entry["name"]: entry for entry in project["tool"]["uv"]["index"]}
    assert indexes["pytorch-cu128"] == {
        "name": "pytorch-cu128",
        "url": "https://download.pytorch.org/whl/cu128",
        "explicit": True,
    }


BASE = "taide/Gemma-3-TAIDE-12b-Chat-2602"
REVISION = "4de0b93b99f8b61b59c40d019fd593bdd1c42249"
REQUIREMENTS = InferenceRequirements(
    gpu_name_contains="RTX 4090",
    minimum_vram_gib=22.0,
    minimum_compute_capability=(8, 9),
    requires_bf16=True,
    max_new_tokens=64,
)


def eligible_gpu() -> NvidiaGpu:
    return NvidiaGpu(
        name="NVIDIA GeForce RTX 4090",
        total_vram_mib=24564,
        compute_capability=(8, 9),
        driver_version="555.99",
    )


def eligible_torch() -> TorchCudaRuntime:
    return TorchCudaRuntime(
        torch_version="2.10.0+cu128",
        torch_cuda_version="12.8",
        cuda_available=True,
        bf16_supported=True,
        device_name="NVIDIA GeForce RTX 4090",
        compute_capability=(8, 9),
    )


def adapter_contract(**changes: object) -> AdapterContract:
    values = {
        "source": "adapter",
        "base_model_name_or_path": BASE,
        "base_model_revision": None,
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "inference_mode": True,
        "config_sha256": "a" * 64,
        "weights_sha256": "b" * 64,
        "resolved_revision": None,
    }
    values.update(changes)
    return AdapterContract(**values)


def test_parse_nvidia_smi_single_gpu() -> None:
    gpu = parse_nvidia_smi_row("NVIDIA GeForce RTX 4090, 24564, 8.9, 555.99\n")

    assert gpu.name.endswith("RTX 4090")
    assert gpu.total_vram_gib == pytest.approx(23.988, abs=0.001)
    assert gpu.compute_capability == (8, 9)


@pytest.mark.parametrize(
    "output",
    ["", "GPU, 4096, 8.6", "GPU, 4096, 8.x, 555\n", "GPU, 1, 8.9, x\nGPU2, 1, 8.9, x"],
)
def test_parse_nvidia_smi_rejects_ambiguous_rows(output: str) -> None:
    with pytest.raises(ValueError):
        parse_nvidia_smi_row(output)


def test_4090_preflight_requires_matching_cuda_runtime() -> None:
    report = hardware_preflight(
        eligible_gpu(),
        os_name="Windows",
        requirements=REQUIREMENTS,
        torch_runtime=eligible_torch(),
    )

    assert report["eligible"] is True
    assert report["failures"] == []
    assert report["torch"]["bf16_supported"] is True


def test_transfer_laptop_is_rejected_before_model_load() -> None:
    laptop = NvidiaGpu(
        name="NVIDIA GeForce RTX 2050",
        total_vram_mib=4096,
        compute_capability=(8, 6),
        driver_version="536.99",
    )

    report = hardware_preflight(laptop, os_name="Windows", requirements=REQUIREMENTS)

    assert report["eligible"] is False
    assert any("RTX 4090 required" in failure for failure in report["failures"])
    assert any("VRAM required" in failure for failure in report["failures"])


def test_adapter_contract_must_match_base_and_revision() -> None:
    validate_adapter_contract(
        adapter_contract(), expected_base_model=BASE, expected_base_revision=REVISION
    )

    with pytest.raises(RuntimeError, match="adapter/base mismatch"):
        validate_adapter_contract(
            adapter_contract(base_model_name_or_path="other/model"),
            expected_base_model=BASE,
            expected_base_revision=REVISION,
        )
    with pytest.raises(RuntimeError, match="revision mismatch"):
        validate_adapter_contract(
            adapter_contract(base_model_revision="f" * 40),
            expected_base_model=BASE,
            expected_base_revision=REVISION,
        )


def test_load_local_adapter_contract_hashes_config_and_weights(tmp_path: Path) -> None:
    config = {
        "base_model_name_or_path": BASE,
        "revision": None,
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "inference_mode": True,
    }
    (tmp_path / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    (tmp_path / "adapter_model.safetensors").write_bytes(b"adapter")

    contract = load_adapter_contract(str(tmp_path), token=None)

    assert contract.base_model_name_or_path == BASE
    assert len(contract.config_sha256) == 64
    assert len(contract.weights_sha256 or "") == 64


def test_missing_explicit_local_adapter_is_not_treated_as_hub_repo(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="adapter directory does not exist"):
        load_adapter_contract(str(tmp_path / "missing"), token=None)


def test_chat_messages_match_training_text_shape() -> None:
    assert build_messages("  題目  ") == [{"role": "user", "content": "題目"}]
    assert build_messages("題目", "  系統  ") == [
        {"role": "system", "content": "系統"},
        {"role": "user", "content": "題目"},
    ]


def test_manifest_excludes_prompt_and_raw_output() -> None:
    result = GenerationResult(
        text="B",
        parsed_answer="B",
        prompt_tokens=42,
        completion_tokens=1,
        first_token_seconds=0.2,
        total_generation_seconds=0.3,
        peak_allocated_gib=8.0,
        peak_reserved_gib=9.0,
    )
    manifest = build_private_safe_manifest(
        result=result,
        prompt="私人醫療題目",
        base_model=BASE,
        base_revision=REVISION,
        adapter_contract=adapter_contract(),
        adapter_revision=None,
        hardware={"eligible": True},
        model_load_seconds=10.0,
    )
    serialized = json.dumps(manifest, ensure_ascii=False)

    assert "私人醫療題目" not in serialized
    assert '"text"' not in serialized
    assert str(Path("adapter").resolve()) not in serialized
    assert manifest["generation"]["parsed_answer"] == "B"
    assert len(manifest["generation"]["prompt_sha256"]) == 64

"""Windows RTX 4090 inference for the Phase 3 medical QLoRA adapter.

The hardware and adapter checks in this module intentionally use only the Python
standard library. Heavy CUDA dependencies are imported only after every preflight
gate passes, so the transfer laptop can validate the CLI without loading a 12B model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import ProjectConfig, load_project_config
from .evaluation import parse_mcq_answer

GIB = 1024**3
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "project.toml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "local_inference"
ACCEPTANCE_PROMPT = """請只輸出一個大寫選項字母。
研究用測試題：下列何者是人體正常體溫常用的攝氏單位符號？
A. kg
B. mL
C. °C
D. mmHg"""
ACCEPTANCE_EXPECTED_ANSWER = "C"


@dataclass(frozen=True)
class NvidiaGpu:
    """GPU details available without importing PyTorch."""

    name: str
    total_vram_mib: int
    compute_capability: tuple[int, int]
    driver_version: str

    @property
    def total_vram_gib(self) -> float:
        return self.total_vram_mib / 1024


@dataclass(frozen=True)
class InferenceRequirements:
    """Hardware and generation contract loaded from ``configs/project.toml``."""

    gpu_name_contains: str
    minimum_vram_gib: float
    minimum_compute_capability: tuple[int, int]
    requires_bf16: bool
    max_new_tokens: int


def inference_requirements(config: ProjectConfig) -> InferenceRequirements:
    """Validate and expose the single authoritative Windows inference profile."""

    values = config.raw["inference"]["windows_4090"]
    capability = values["minimum_compute_capability"]
    if not isinstance(capability, list) or len(capability) != 2:
        raise ValueError("inference minimum_compute_capability must contain [major, minor]")
    requirements = InferenceRequirements(
        gpu_name_contains=str(values["required_gpu_name"]),
        minimum_vram_gib=float(values["minimum_vram_gib"]),
        minimum_compute_capability=(int(capability[0]), int(capability[1])),
        requires_bf16=bool(values["requires_bf16"]),
        max_new_tokens=int(values["max_new_tokens"]),
    )
    if not requirements.gpu_name_contains:
        raise ValueError("inference required_gpu_name must not be empty")
    if requirements.minimum_vram_gib <= 0 or requirements.max_new_tokens <= 0:
        raise ValueError("inference VRAM and max_new_tokens must be positive")
    return requirements


@dataclass(frozen=True)
class TorchCudaRuntime:
    """CUDA facts that must be verified by the exact PyTorch environment in use."""

    torch_version: str
    torch_cuda_version: str | None
    cuda_available: bool
    bf16_supported: bool
    device_name: str | None
    compute_capability: tuple[int, int] | None


@dataclass(frozen=True)
class AdapterContract:
    """Publication-critical fields read from ``adapter_config.json``."""

    source: str
    base_model_name_or_path: str
    base_model_revision: str | None
    peft_type: str | None
    task_type: str | None
    inference_mode: bool | None
    config_sha256: str
    weights_sha256: str | None
    resolved_revision: str | None


@dataclass(frozen=True)
class GenerationResult:
    """One generated response and its local performance measurements."""

    text: str
    parsed_answer: str | None
    prompt_tokens: int
    completion_tokens: int
    first_token_seconds: float | None
    total_generation_seconds: float
    peak_allocated_gib: float
    peak_reserved_gib: float


def _parse_capability(value: str) -> tuple[int, int]:
    parts = value.strip().split(".", maxsplit=1)
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise ValueError(f"invalid compute capability: {value!r}")
    return int(parts[0]), int(parts[1])


def parse_nvidia_smi_row(output: str) -> NvidiaGpu:
    """Parse one CSV row emitted by the project's fixed ``nvidia-smi`` query."""

    rows = list(csv.reader(line for line in output.splitlines() if line.strip()))
    if len(rows) != 1 or len(rows[0]) != 4:
        raise ValueError("expected exactly one NVIDIA GPU and four nvidia-smi fields")
    name, memory_mib, compute_capability, driver = (item.strip() for item in rows[0])
    if not name or not memory_mib.isdigit() or not driver:
        raise ValueError("nvidia-smi returned incomplete GPU information")
    return NvidiaGpu(
        name=name,
        total_vram_mib=int(memory_mib),
        compute_capability=_parse_capability(compute_capability),
        driver_version=driver,
    )


def probe_nvidia_gpu() -> NvidiaGpu:
    """Inspect a single NVIDIA GPU without initializing CUDA."""

    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,compute_cap,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("nvidia-smi failed; install a supported NVIDIA driver") from exc
    return parse_nvidia_smi_row(completed.stdout)


def inspect_torch_cuda() -> TorchCudaRuntime:
    """Import PyTorch lazily and report the runtime that will load the model."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional inference group
        raise RuntimeError(
            "PyTorch is not installed. Run the documented `uv sync --group inference` "
            "command on the RTX 4090 computer."
        ) from exc

    available = bool(torch.cuda.is_available())
    return TorchCudaRuntime(
        torch_version=str(torch.__version__),
        torch_cuda_version=str(torch.version.cuda) if torch.version.cuda else None,
        cuda_available=available,
        bf16_supported=bool(torch.cuda.is_bf16_supported()) if available else False,
        device_name=str(torch.cuda.get_device_name(0)) if available else None,
        compute_capability=tuple(torch.cuda.get_device_capability(0)) if available else None,
    )


def hardware_preflight(
    gpu: NvidiaGpu,
    *,
    os_name: str,
    requirements: InferenceRequirements,
    torch_runtime: TorchCudaRuntime | None = None,
) -> dict[str, Any]:
    """Return a machine-readable 4090 acceptance decision."""

    failures: list[str] = []
    if os_name != "Windows":
        failures.append(f"Windows required; detected {os_name}")
    if requirements.gpu_name_contains.casefold() not in gpu.name.casefold():
        failures.append(f"{requirements.gpu_name_contains} required; detected {gpu.name}")
    if gpu.total_vram_gib < requirements.minimum_vram_gib:
        failures.append(
            f"at least {requirements.minimum_vram_gib:.0f} GiB VRAM required; "
            f"detected {gpu.total_vram_gib:.1f} GiB"
        )
    if gpu.compute_capability < requirements.minimum_compute_capability:
        failures.append(
            f"compute capability {requirements.minimum_compute_capability[0]}."
            f"{requirements.minimum_compute_capability[1]} required; detected "
            f"{gpu.compute_capability[0]}.{gpu.compute_capability[1]}"
        )

    if torch_runtime is not None:
        if not torch_runtime.cuda_available:
            failures.append("the installed PyTorch build cannot access CUDA")
        if requirements.requires_bf16 and not torch_runtime.bf16_supported:
            failures.append("the installed PyTorch runtime does not support BF16")
        if torch_runtime.device_name and torch_runtime.device_name != gpu.name:
            failures.append(
                "PyTorch and nvidia-smi selected different GPUs: "
                f"{torch_runtime.device_name!r} != {gpu.name!r}"
            )
        if (
            torch_runtime.compute_capability is not None
            and torch_runtime.compute_capability != gpu.compute_capability
        ):
            failures.append("PyTorch and nvidia-smi report different compute capabilities")

    return {
        "eligible": not failures,
        "failures": failures,
        "os": os_name,
        "required": {
            "gpu_name_contains": requirements.gpu_name_contains,
            "minimum_vram_gib": requirements.minimum_vram_gib,
            "minimum_compute_capability": list(requirements.minimum_compute_capability),
            "bf16": requirements.requires_bf16,
        },
        "nvidia_smi": {
            **asdict(gpu),
            "compute_capability": list(gpu.compute_capability),
            "total_vram_gib": gpu.total_vram_gib,
        },
        "torch": (
            {
                **asdict(torch_runtime),
                "compute_capability": (
                    list(torch_runtime.compute_capability)
                    if torch_runtime.compute_capability is not None
                    else None
                ),
            }
            if torch_runtime is not None
            else None
        ),
    }


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _looks_like_local_path(value: str) -> bool:
    candidate = Path(value).expanduser()
    return candidate.is_absolute() or value.startswith((".", "~")) or "\\" in value


def load_adapter_contract(
    adapter: str,
    *,
    token: str | None,
    revision: str | None = None,
) -> AdapterContract:
    """Read an adapter contract from a local directory or a Hub repository."""

    candidate = Path(adapter).expanduser()
    resolved_revision: str | None = None
    if candidate.exists():
        if not candidate.is_dir():
            raise ValueError("--adapter local path must be a directory")
        config_path = candidate / "adapter_config.json"
        source = str(candidate.resolve())
        weights_path = candidate / "adapter_model.safetensors"
        if not weights_path.is_file():
            raise FileNotFoundError(f"missing adapter_model.safetensors in {source}")
        weights_sha256 = _sha256_file(weights_path)
    else:
        if _looks_like_local_path(adapter):
            raise FileNotFoundError(f"adapter directory does not exist: {candidate}")
        try:
            from huggingface_hub import HfApi, hf_hub_download
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("huggingface-hub is required for a remote adapter") from exc
        info = HfApi(token=token).model_info(repo_id=adapter, revision=revision)
        resolved_revision = str(info.sha)
        siblings = {sibling.rfilename for sibling in info.siblings}
        if "adapter_model.safetensors" not in siblings:
            raise FileNotFoundError(
                f"remote adapter {adapter}@{resolved_revision} has no adapter_model.safetensors"
            )
        config_path = Path(
            hf_hub_download(
                adapter,
                "adapter_config.json",
                revision=resolved_revision,
                token=token,
            )
        )
        source = adapter
        weights_sha256 = None

    if not config_path.is_file():
        raise FileNotFoundError(f"missing adapter_config.json in {source}")
    raw_bytes = config_path.read_bytes()
    try:
        values = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("adapter_config.json must be valid UTF-8 JSON") from exc
    base_model = values.get("base_model_name_or_path")
    if not isinstance(base_model, str) or not base_model.strip():
        raise ValueError("adapter_config.json has no base_model_name_or_path")
    base_revision = values.get("revision")
    if base_revision is not None and not isinstance(base_revision, str):
        raise ValueError("adapter revision must be a string or null")
    inference_mode = values.get("inference_mode")
    if inference_mode is not None and not isinstance(inference_mode, bool):
        raise ValueError("adapter inference_mode must be boolean or null")
    return AdapterContract(
        source=source,
        base_model_name_or_path=base_model,
        base_model_revision=base_revision,
        peft_type=values.get("peft_type"),
        task_type=values.get("task_type"),
        inference_mode=inference_mode,
        config_sha256=_sha256_bytes(raw_bytes),
        weights_sha256=weights_sha256,
        resolved_revision=resolved_revision,
    )


def validate_adapter_contract(
    contract: AdapterContract,
    *,
    expected_base_model: str,
    expected_base_revision: str,
) -> None:
    """Reject an adapter that does not belong to the pinned Phase 3 base."""

    if contract.base_model_name_or_path != expected_base_model:
        raise RuntimeError(
            "adapter/base mismatch: "
            f"adapter expects {contract.base_model_name_or_path!r}, "
            f"configured base is {expected_base_model!r}"
        )
    if contract.base_model_revision and contract.base_model_revision != expected_base_revision:
        raise RuntimeError(
            "adapter/base revision mismatch: "
            f"adapter expects {contract.base_model_revision!r}, "
            f"configured revision is {expected_base_revision!r}"
        )
    if contract.peft_type not in {None, "LORA"}:
        raise RuntimeError(f"unsupported PEFT adapter type: {contract.peft_type}")


def build_messages(prompt: str, system_prompt: str | None = None) -> list[dict[str, str]]:
    """Build the same plain-text chat shape used during training."""

    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    messages: list[dict[str, str]] = []
    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append({"role": "user", "content": prompt.strip()})
    return messages


def _load_model_stack(
    *,
    base_model: str,
    base_revision: str,
    adapter: str,
    adapter_revision: str | None,
    token: str | None,
) -> tuple[Any, Any, float]:
    """Load the pinned Gemma 3 base in NF4 and attach a frozen PEFT adapter."""

    try:
        import torch
        from peft import PeftModel
        from transformers import (
            AutoProcessor,
            BitsAndBytesConfig,
            Gemma3ForConditionalGeneration,
        )
    except ImportError as exc:  # pragma: no cover - requires final 4090 environment
        raise RuntimeError(
            "Phase 5 inference dependencies are missing; run `uv sync --group inference`"
        ) from exc

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(
        base_model,
        revision=base_revision,
        token=token,
        padding_side="left",
    )
    model = Gemma3ForConditionalGeneration.from_pretrained(
        base_model,
        revision=base_revision,
        token=token,
        device_map={"": 0},
        dtype=torch.bfloat16,
        quantization_config=quantization,
        attn_implementation="sdpa",
    )
    adapter_kwargs: dict[str, Any] = {
        "is_trainable": False,
        "token": token,
        "low_cpu_mem_usage": True,
    }
    if adapter_revision:
        adapter_kwargs["revision"] = adapter_revision
    model = PeftModel.from_pretrained(model, adapter, **adapter_kwargs)
    model.eval()
    if "default" not in model.peft_config:
        raise RuntimeError("PEFT model did not activate the default adapter")
    trainable, adapter_total = model.get_nb_trainable_parameters()
    if trainable != 0 or adapter_total <= 0:
        raise RuntimeError(
            f"adapter parameter audit failed: trainable={trainable}, total={adapter_total}"
        )
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    if not getattr(tokenizer, "chat_template", None):
        raise RuntimeError("the pinned tokenizer has no chat template")
    torch.cuda.synchronize()
    return model, tokenizer, time.perf_counter() - started


def generate_one(
    model: Any,
    tokenizer: Any,
    *,
    prompt: str,
    system_prompt: str | None,
    max_new_tokens: int,
    stream_to_console: bool,
) -> GenerationResult:
    """Greedily generate one answer while measuring first-token and total latency."""

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    import torch
    from transformers import TextIteratorStreamer

    messages = build_messages(prompt, system_prompt)
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    ).to("cuda")
    prompt_tokens = int(inputs["input_ids"].shape[-1])
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        timeout=120.0,
    )
    generated: dict[str, Any] = {}
    failures: list[BaseException] = []

    def _worker() -> None:
        try:
            with torch.inference_mode():
                generated["tokens"] = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    streamer=streamer,
                )
        except BaseException as exc:  # pragma: no cover - requires CUDA failure
            failures.append(exc)
            streamer.on_finalized_text("", stream_end=True)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    thread = threading.Thread(target=_worker, name="gemma-generation", daemon=True)
    thread.start()
    chunks: list[str] = []
    first_token_seconds: float | None = None
    for chunk in streamer:
        if chunk and first_token_seconds is None:
            first_token_seconds = time.perf_counter() - started
        chunks.append(chunk)
        if stream_to_console:
            print(chunk, end="", flush=True)
    thread.join()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if stream_to_console:
        print()
    if failures:
        raise RuntimeError("generation failed") from failures[0]
    output_tokens = generated["tokens"]
    completion_tokens = int(output_tokens.shape[-1]) - prompt_tokens
    text = "".join(chunks).strip()
    return GenerationResult(
        text=text,
        parsed_answer=parse_mcq_answer(text),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        first_token_seconds=first_token_seconds,
        total_generation_seconds=elapsed,
        peak_allocated_gib=torch.cuda.max_memory_allocated() / GIB,
        peak_reserved_gib=torch.cuda.max_memory_reserved() / GIB,
    )


def build_private_safe_manifest(
    *,
    result: GenerationResult,
    prompt: str,
    base_model: str,
    base_revision: str,
    adapter_contract: AdapterContract,
    adapter_revision: str | None,
    hardware: dict[str, Any],
    model_load_seconds: float,
) -> dict[str, Any]:
    """Build a manifest that contains hashes, never the prompt or raw response."""

    adapter_is_local = Path(adapter_contract.source).is_absolute()
    adapter_record = {
        "source_type": "local" if adapter_is_local else "huggingface_hub",
        "repo_id": None if adapter_is_local else adapter_contract.source,
        "local_path_sha256": (
            _sha256_bytes(adapter_contract.source.encode("utf-8"))
            if adapter_is_local
            else None
        ),
        "base_model_name_or_path": adapter_contract.base_model_name_or_path,
        "base_model_revision": adapter_contract.base_model_revision,
        "peft_type": adapter_contract.peft_type,
        "task_type": adapter_contract.task_type,
        "inference_mode": adapter_contract.inference_mode,
        "config_sha256": adapter_contract.config_sha256,
        "weights_sha256": adapter_contract.weights_sha256,
        "resolved_revision": adapter_contract.resolved_revision,
        "requested_revision": adapter_revision,
    }
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "phase": 5,
        "base_model": {"model_id": base_model, "revision": base_revision},
        "adapter": adapter_record,
        "hardware": hardware,
        "quantization": {
            "load_in_4bit": True,
            "quant_type": "nf4",
            "compute_dtype": "bfloat16",
            "double_quant": True,
            "attention": "sdpa",
        },
        "timing": {
            "model_load_seconds": model_load_seconds,
            "first_token_seconds": result.first_token_seconds,
            "total_generation_seconds": result.total_generation_seconds,
        },
        "memory": {
            "peak_allocated_gib": result.peak_allocated_gib,
            "peak_reserved_gib": result.peak_reserved_gib,
        },
        "generation": {
            "prompt_sha256": _sha256_bytes(prompt.encode("utf-8")),
            "raw_output_sha256": _sha256_bytes(result.text.encode("utf-8")),
            "parsed_answer": result.parsed_answer,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
        },
        "medical_disclaimer": "research use only; not medical advice",
    }


def write_manifest(manifest: dict[str, Any], output_dir: Path) -> Path:
    """Atomically write one ignored local acceptance artifact."""

    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    destination = output_dir / f"{run_id}-phase5-local-inference.json"
    temporary = destination.with_suffix(".json.partial")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line contract without importing optional dependencies."""

    parser = argparse.ArgumentParser(
        description="Windows RTX 4090 4-bit TAIDE base + medical QLoRA inference"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prompt", help="generate one response")
    mode.add_argument("--interactive", action="store_true", help="read independent prompts")
    mode.add_argument(
        "--acceptance",
        action="store_true",
        help="run the fixed synthetic A-D acceptance probe",
    )
    parser.add_argument("--adapter", help="local adapter directory or Hugging Face repo ID")
    parser.add_argument("--adapter-revision", help="optional pinned adapter commit")
    parser.add_argument("--base-model", help="override the configured base model ID")
    parser.add_argument("--base-revision", help="override the pinned base commit")
    parser.add_argument("--system-prompt", help="optional system message")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="check Windows/CUDA/4090 eligibility without loading weights",
    )
    return parser


def _load_environment() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(PROJECT_ROOT / ".env")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point exposed as ``tw-med-local-infer``."""

    _load_environment()
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.preflight_only and not (args.prompt or args.interactive or args.acceptance):
        parser.error("choose --prompt, --interactive, or --acceptance")

    config = load_project_config(DEFAULT_CONFIG_PATH)
    requirements = inference_requirements(config)
    max_new_tokens = args.max_new_tokens or requirements.max_new_tokens
    gpu = probe_nvidia_gpu()
    static_preflight = hardware_preflight(
        gpu,
        os_name=platform.system(),
        requirements=requirements,
    )
    if not static_preflight["eligible"]:
        print(json.dumps(static_preflight, ensure_ascii=False, indent=2))
        return 2
    try:
        torch_runtime = inspect_torch_cuda()
    except RuntimeError as exc:
        failed_report = {
            **static_preflight,
            "eligible": False,
            "failures": [str(exc)],
        }
        print(json.dumps(failed_report, ensure_ascii=False, indent=2))
        return 2
    preflight = hardware_preflight(
        gpu,
        os_name=platform.system(),
        requirements=requirements,
        torch_runtime=torch_runtime,
    )
    print(json.dumps(preflight, ensure_ascii=False, indent=2))
    if not preflight["eligible"]:
        return 2
    if args.preflight_only:
        return 0

    base_model = args.base_model or os.getenv("HF_BASE_MODEL_ID") or config.primary.model_id
    base_revision = (
        args.base_revision or os.getenv("HF_BASE_MODEL_REVISION") or config.primary.revision
    )
    adapter = args.adapter or os.getenv("HF_ADAPTER_REPO_ID") or os.getenv("HF_ADAPTER_PATH")
    adapter_revision = args.adapter_revision or os.getenv("HF_ADAPTER_REVISION") or None
    token = os.getenv("HF_TOKEN") or None
    if not adapter:
        parser.error("set --adapter, HF_ADAPTER_REPO_ID, or HF_ADAPTER_PATH")
    if not token:
        parser.error("HF_TOKEN is required because the pinned base model is gated")

    contract = load_adapter_contract(adapter, token=token, revision=adapter_revision)
    validate_adapter_contract(
        contract,
        expected_base_model=base_model,
        expected_base_revision=base_revision,
    )
    effective_adapter_revision = contract.resolved_revision or adapter_revision
    adapter_load_revision = (
        None if Path(adapter).expanduser().exists() else effective_adapter_revision
    )
    model, tokenizer, load_seconds = _load_model_stack(
        base_model=base_model,
        base_revision=base_revision,
        adapter=adapter,
        adapter_revision=adapter_load_revision,
        token=token,
    )
    print(f"Model and frozen adapter loaded in {load_seconds:.1f}s")

    prompts: list[str] = []
    if args.prompt:
        prompts.append(args.prompt)
    elif args.acceptance:
        prompts.append(ACCEPTANCE_PROMPT)
    else:
        print("互動模式：每題獨立推論；輸入 /quit 離開。研究用途，不構成醫療建議。")
        while True:
            try:
                value = input("\n問題> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if value.casefold() in {"/quit", "/exit"}:
                break
            if value:
                prompts.append(value)
                result = generate_one(
                    model,
                    tokenizer,
                    prompt=value,
                    system_prompt=args.system_prompt,
                    max_new_tokens=max_new_tokens,
                    stream_to_console=True,
                )
                manifest = build_private_safe_manifest(
                    result=result,
                    prompt=value,
                    base_model=base_model,
                    base_revision=base_revision,
                    adapter_contract=contract,
                    adapter_revision=effective_adapter_revision,
                    hardware=preflight,
                    model_load_seconds=load_seconds,
                )
                path = write_manifest(manifest, args.output_dir)
                print(
                    f"TTFT={result.first_token_seconds!s}s, "
                    f"total={result.total_generation_seconds:.3f}s, manifest={path}"
                )
        return 0

    result = generate_one(
        model,
        tokenizer,
        prompt=prompts[0],
        system_prompt=args.system_prompt,
        max_new_tokens=max_new_tokens,
        stream_to_console=True,
    )
    manifest = build_private_safe_manifest(
        result=result,
        prompt=prompts[0],
        base_model=base_model,
        base_revision=base_revision,
        adapter_contract=contract,
        adapter_revision=effective_adapter_revision,
        hardware=preflight,
        model_load_seconds=load_seconds,
    )
    if args.acceptance:
        passed = result.parsed_answer == ACCEPTANCE_EXPECTED_ANSWER
        manifest["acceptance"] = {
            "probe": "synthetic_unit_mcq_v1",
            "expected_answer": ACCEPTANCE_EXPECTED_ANSWER,
            "passed": passed,
        }
    path = write_manifest(manifest, args.output_dir)
    print(
        json.dumps(
            {"parsed_answer": result.parsed_answer, "manifest": str(path)},
            ensure_ascii=False,
        )
    )
    return 0 if not args.acceptance or passed else 3


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

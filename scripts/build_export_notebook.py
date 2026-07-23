"""Build the gated, optional Colab GGUF export notebook."""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path
from textwrap import dedent

import nbformat

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "project.toml"
REQUIREMENTS_PATH = ROOT / "requirements" / "colab-export.txt"
OUTPUT_PATH = ROOT / "notebooks" / "export_gguf.ipynb"


def _markdown(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(dedent(source).strip() + "\n")


def _code(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(dedent(source).strip() + "\n")


def _install_cell() -> str:
    requirements = [
        line.strip()
        for line in REQUIREMENTS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    separator = " \\" + "\n    "
    return "%pip install --quiet" + separator + separator.join(
        json.dumps(requirement) for requirement in requirements
    )


def build_notebook() -> nbformat.NotebookNode:
    with CONFIG_PATH.open("rb") as source:
        config = tomllib.load(source)
    config_literal = repr(json.dumps(config, ensure_ascii=False, sort_keys=True))
    export_config = config["export"]["gguf"]
    configured_export = bool(export_config["enabled"])
    required_approval_code = str(export_config["required_approval_code"])
    approved_compute_units = float(
        export_config["approved_compute_units_with_20pct_buffer"]
    )

    gate = f'''
    # OPTIONAL EXPORT HARD GATE — this is the only cell that may be edited.
    CONFIG_EXPORT_ENABLED = {configured_export!r}
    ENABLE_GGUF_EXPORT = False
    GGUF_EXPORT_APPROVAL = ""
    REQUIRED_GGUF_EXPORT_APPROVAL = {required_approval_code!r}
    APPROVED_COMPUTE_UNIT_LIMIT = {approved_compute_units!r}

    if not CONFIG_EXPORT_ENABLED:
        raise RuntimeError(
            "Repo config keeps optional GGUF export disabled. Obtain explicit approval, "
            "update configs/project.toml, and rebuild this notebook first."
        )
    if not ENABLE_GGUF_EXPORT or GGUF_EXPORT_APPROVAL != REQUIRED_GGUF_EXPORT_APPROVAL:
        raise RuntimeError(
            "GGUF export is locked. Set ENABLE_GGUF_EXPORT=True and copy the exact "
            "approval code in this cell only after the repo gate is enabled."
        )
    print(
        "GGUF Q4_K_M export gate passed. "
        f"Approved compute-unit limit: {{APPROVED_COMPUTE_UNIT_LIMIT:.2f}} CU"
    )
    '''

    setup = f'''
    # ruff: noqa: E501
    import hashlib
    import importlib.metadata
    import json
    import shutil
    import time
    from datetime import UTC, datetime
    from pathlib import Path, PurePosixPath
    from zipfile import ZipFile

    import torch
    from google.colab import drive, userdata
    from huggingface_hub import HfApi, snapshot_download

    PROJECT_CONFIG = json.loads({config_literal})
    export_config = PROJECT_CONFIG["export"]["gguf"]
    if not export_config["enabled"]:
        raise RuntimeError("Embedded repository export gate is disabled")
    if export_config["required_approval_code"] != REQUIRED_GGUF_EXPORT_APPROVAL:
        raise RuntimeError("Notebook approval code does not match embedded repository config")
    if export_config["quantization_method"] != "q4_k_m":
        raise RuntimeError("Only the approved q4_k_m quantization method is allowed")
    if export_config["run_environment"] != "colab_linux":
        raise RuntimeError("GGUF export is restricted to the approved Colab Linux environment")
    if export_config["external_upload_allowed"]:
        raise RuntimeError("External upload must remain disabled for this export")
    if (
        float(export_config["approved_compute_units_with_20pct_buffer"])
        != APPROVED_COMPUTE_UNIT_LIMIT
    ):
        raise RuntimeError("Approved compute-unit limit does not match repository config")

    workflow_started_at_utc = datetime.now(UTC)
    workflow_started_monotonic = time.perf_counter()
    run_id = workflow_started_at_utc.strftime("%Y%m%dT%H%M%SZ")
    HF_TOKEN = userdata.get("HF_TOKEN")
    if not HF_TOKEN:
        raise RuntimeError("Colab Secret HF_TOKEN is required")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    properties = torch.cuda.get_device_properties(0)
    total_vram_gib = properties.total_memory / 1024**3
    gpu = {{
        "name": properties.name,
        "total_vram_gib": total_vram_gib,
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
    }}
    expected_gpu_name = str(export_config["required_gpu_name_contains"])
    if expected_gpu_name.casefold() not in properties.name.casefold():
        raise RuntimeError(
            f"Only the approved {{expected_gpu_name}} profile may run this export: {{gpu}}"
        )
    minimum_vram_gib = float(export_config["minimum_vram_gib"])
    if total_vram_gib < minimum_vram_gib or not gpu["bf16_supported"]:
        raise RuntimeError(
            f"A BF16 GPU with at least {{minimum_vram_gib:.1f}} GiB is required: {{gpu}}"
        )
    free_disk_gib = shutil.disk_usage("/content").free / 1024**3
    minimum_local_disk_gib = float(export_config["minimum_local_disk_gib"])
    if free_disk_gib < minimum_local_disk_gib:
        raise RuntimeError(
            f"At least {{minimum_local_disk_gib:.1f}} GiB free local disk is required: "
            f"{{free_disk_gib:.1f}}"
        )
    drive.mount("/content/drive")
    phase3 = PROJECT_CONFIG["evaluation"]["phase3_adapter"]
    archive_path = Path(phase3["drive_archive"])
    drive_root = Path(export_config["drive_root"])
    drive_root.mkdir(parents=True, exist_ok=True)
    free_drive_gib = shutil.disk_usage(drive_root).free / 1024**3
    minimum_drive_disk_gib = float(export_config["minimum_drive_disk_gib"])
    if free_drive_gib < minimum_drive_disk_gib:
        raise RuntimeError(
            f"At least {{minimum_drive_disk_gib:.1f}} GiB free Drive space is required: "
            f"{{free_drive_gib:.1f}}"
        )
    drive_output = drive_root / run_id
    if drive_output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing export run: {{drive_output}}")
    local_root = Path("/content/tw-med-gguf-export") / run_id
    adapter_dir = local_root / "adapter"
    runtime_adapter_dir = local_root / "runtime_adapter"
    export_dir = local_root / "q4_k_m"
    for directory in (local_root, adapter_dir, export_dir):
        directory.mkdir(parents=True, exist_ok=False)
    print(
        json.dumps(
            {{
                "run_id": run_id,
                "gpu": gpu,
                "free_local_disk_gib": free_disk_gib,
                "free_drive_disk_gib": free_drive_gib,
                "approved_compute_unit_limit": APPROVED_COMPUTE_UNIT_LIMIT,
            }},
            indent=2,
        )
    )
    '''

    extraction = r'''
    def sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    if not archive_path.is_file():
        raise FileNotFoundError(f"Phase 3 full archive not found: {archive_path}")
    if archive_path.stat().st_size != int(phase3["archive_bytes"]):
        raise RuntimeError(
            f"Phase 3 archive size mismatch: {archive_path.stat().st_size} != "
            f"{phase3['archive_bytes']}"
        )
    archive_sha256 = sha256_file(archive_path)
    if archive_sha256 != phase3["archive_sha256"]:
        raise RuntimeError(
            f"Phase 3 archive SHA-256 mismatch: {archive_sha256} != "
            f"{phase3['archive_sha256']}"
        )
    with ZipFile(archive_path) as archive:
        selected = []
        for member in archive.infolist():
            normalized = PurePosixPath(member.filename.replace("\\", "/"))
            if normalized.is_absolute() or ".." in normalized.parts:
                raise RuntimeError(f"Unsafe ZIP member: {member.filename}")
            if normalized.parts and normalized.parts[0] == "adapter" and not member.is_dir():
                selected.append((member, PurePosixPath(*normalized.parts[1:])))
        names = {relative.as_posix() for _, relative in selected}
        required = {"adapter_config.json", "adapter_model.safetensors"}
        if not required.issubset(names):
            raise RuntimeError(f"Phase 3 archive is missing adapter files: {required - names}")
        for member, relative in selected:
            target = adapter_dir.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
    adapter_config_path = adapter_dir / "adapter_config.json"
    adapter_config_sha256 = sha256_file(adapter_config_path)
    adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    model_config = PROJECT_CONFIG["models"]["primary"]
    if adapter_config.get("base_model_name_or_path") != model_config["model_id"]:
        raise RuntimeError("Adapter/base mismatch; export stopped")
    print({"adapter_files": len(selected), "archive_sha256_verified": True})
    '''

    model_load = r'''
    # ruff: noqa: I001  # Unsloth must patch Transformers before PEFT imports it.
    from unsloth import FastModel
    from peft import PeftModel

    model_id = str(model_config["model_id"])
    model_revision = str(model_config["revision"])
    model_info = HfApi(token=HF_TOKEN).model_info(
        repo_id=model_id,
        revision=model_revision,
        files_metadata=True,
    )
    if model_info.sha != model_revision:
        raise RuntimeError(
            "Resolved model revision mismatch: "
            f"expected={model_revision}, actual={model_info.sha}"
        )

    model_siblings = model_info.siblings or []
    remote_weight_files = sorted(
        sibling.rfilename
        for sibling in model_siblings
        if sibling.rfilename.endswith(".safetensors")
    )
    remote_weight_bytes = sum(
        int(sibling.size or 0)
        for sibling in model_siblings
        if sibling.rfilename.endswith(".safetensors")
    )
    if not remote_weight_files or remote_weight_bytes <= 0:
        raise RuntimeError(
            "Hugging Face model metadata has no sized safetensors weights."
        )

    free_disk_bytes_before_download = shutil.disk_usage("/content").free
    required_disk_bytes = remote_weight_bytes + 8 * 1024**3
    if free_disk_bytes_before_download < required_disk_bytes:
        raise RuntimeError(
            "Colab local disk is too small for the pinned model snapshot: "
            f"free={free_disk_bytes_before_download / 1024**3:.2f} GiB, "
            f"required={required_disk_bytes / 1024**3:.2f} GiB. "
            "Factory-reset the runtime or choose a runtime with more local disk."
        )

    snapshot_path = Path(
        snapshot_download(
            repo_id=model_id,
            revision=model_revision,
            token=HF_TOKEN,
            allow_patterns=[
                "*.json",
                "*.jinja",
                "*.model",
                "*.safetensors",
                "*.txt",
            ],
            max_workers=8,
        )
    )
    if snapshot_path.name != model_revision:
        raise RuntimeError(
            "Downloaded snapshot revision mismatch: "
            f"expected={model_revision}, actual={snapshot_path.name}"
        )

    missing_remote_weights = [
        filename
        for filename in remote_weight_files
        if not (snapshot_path / filename).is_file()
        or (snapshot_path / filename).stat().st_size <= 0
    ]
    if missing_remote_weights:
        raise RuntimeError(
            "Incomplete model snapshot; missing or empty weights: "
            f"{missing_remote_weights}"
        )

    index_path = snapshot_path / "model.safetensors.index.json"
    single_weight_path = snapshot_path / "model.safetensors"
    if index_path.is_file():
        weight_index = json.loads(index_path.read_text(encoding="utf-8"))
        indexed_shards = sorted(set(weight_index.get("weight_map", {}).values()))
        if not indexed_shards:
            raise RuntimeError("model.safetensors.index.json has an empty weight_map")
        unsafe_shards = [
            filename
            for filename in indexed_shards
            if Path(filename).name != filename
            or not filename.endswith(".safetensors")
        ]
        if unsafe_shards:
            raise RuntimeError(f"Unsafe shard names in model index: {unsafe_shards}")
        missing_indexed_shards = [
            filename
            for filename in indexed_shards
            if not (snapshot_path / filename).is_file()
            or (snapshot_path / filename).stat().st_size <= 0
        ]
        if missing_indexed_shards:
            raise RuntimeError(
                "Model index references missing or empty shards: "
                f"{missing_indexed_shards}"
            )
    elif single_weight_path.is_file() and single_weight_path.stat().st_size > 0:
        indexed_shards = [single_weight_path.name]
    else:
        raise RuntimeError(
            "Pinned snapshot has neither model.safetensors nor "
            "model.safetensors.index.json"
        )

    snapshot_config_path = snapshot_path / "config.json"
    tokenizer_config_path = snapshot_path / "tokenizer_config.json"
    if not snapshot_config_path.is_file() or not tokenizer_config_path.is_file():
        raise RuntimeError(
            "Pinned snapshot is missing config.json or tokenizer_config.json"
        )
    if not any(
        (snapshot_path / filename).is_file()
        and (snapshot_path / filename).stat().st_size > 0
        for filename in ("tokenizer.json", "tokenizer.model")
    ):
        raise RuntimeError(
            "Pinned snapshot has neither a usable tokenizer.json nor tokenizer.model"
        )
    snapshot_config = json.loads(snapshot_config_path.read_text(encoding="utf-8"))
    snapshot_architectures = snapshot_config.get("architectures") or []
    snapshot_is_vlm = bool(snapshot_config.get("vision_config")) or any(
        str(architecture).endswith("ForConditionalGeneration")
        for architecture in snapshot_architectures
    )
    if snapshot_is_vlm:
        missing_processor_files = [
            filename
            for filename in ("processor_config.json", "preprocessor_config.json")
            if not (snapshot_path / filename).is_file()
            or (snapshot_path / filename).stat().st_size <= 0
        ]
        if missing_processor_files:
            raise RuntimeError(
                "VLM snapshot is missing processor files: "
                f"{missing_processor_files}"
            )

    MODEL_SNAPSHOT_AUDIT = {
        "repo_id": model_id,
        "revision": model_revision,
        "snapshot_path": str(snapshot_path),
        "remote_weight_files": remote_weight_files,
        "indexed_shards": indexed_shards,
        "weight_bytes": remote_weight_bytes,
        "weight_gib": remote_weight_bytes / 1024**3,
        "free_disk_gib_before_download": free_disk_bytes_before_download / 1024**3,
        "tokenizer_source": str(snapshot_path),
        "vlm_processor_required": snapshot_is_vlm,
        "complete": True,
    }
    print(json.dumps(MODEL_SNAPSHOT_AUDIT, ensure_ascii=False, indent=2))

    shutil.copytree(adapter_dir, runtime_adapter_dir)
    runtime_adapter_config_path = runtime_adapter_dir / "adapter_config.json"
    runtime_adapter_config = dict(adapter_config)
    runtime_adapter_config["base_model_name_or_path"] = str(snapshot_path)
    runtime_adapter_config["revision"] = None
    runtime_adapter_config_path.write_text(
        json.dumps(runtime_adapter_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    if sha256_file(adapter_config_path) != adapter_config_sha256:
        raise RuntimeError("Original adapter_config.json changed during runtime rebinding")

    model, processor = FastModel.from_pretrained(
        model_name=str(runtime_adapter_dir),
        max_seq_length=2048,
        load_in_4bit=True,
        load_in_8bit=False,
        full_finetuning=False,
        token=HF_TOKEN,
        tokenizer_name=str(snapshot_path),
        local_files_only=True,
        use_safetensors=True,
    )
    if not isinstance(model, PeftModel):
        raise RuntimeError(
            "Unsloth did not return a PEFT model for the verified adapter; "
            "refusing to export a base-only GGUF"
        )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    trainable, total = model.get_nb_trainable_parameters()
    if trainable != 0 or total <= 0:
        raise RuntimeError(f"Frozen adapter audit failed: trainable={trainable}, total={total}")
    active_adapters_value = getattr(model, "active_adapters", [])
    if callable(active_adapters_value):
        active_adapters_value = active_adapters_value()
    active_adapters = list(active_adapters_value)
    adapter_parameter_names = [
        name for name, _ in model.named_parameters() if "lora_" in name.casefold()
    ]
    if not active_adapters or not adapter_parameter_names:
        raise RuntimeError(
            "Loaded PEFT model has no active adapter or LoRA parameters; export stopped"
        )
    gguf_save_method = model.__dict__.get("save_pretrained_gguf")
    if (
        gguf_save_method is None
        or getattr(gguf_save_method, "__self__", None) is not model
    ):
        raise RuntimeError(
            "Unsloth GGUF save method is not bound to the PEFT wrapper; "
            "refusing to export a base-only GGUF"
        )
    text_tokenizer = getattr(processor, "tokenizer", processor)
    chat_template = (
        getattr(text_tokenizer, "chat_template", None)
        or getattr(processor, "chat_template", None)
    )
    if not isinstance(chat_template, str) or not chat_template.strip():
        raise RuntimeError("The loaded processor/tokenizer has no chat template")
    chat_template_sha256 = hashlib.sha256(chat_template.encode("utf-8")).hexdigest()
    print(
        {
            "adapter_reloaded": True,
            "peft_detected": True,
            "peft_model_class": model.__class__.__name__,
            "active_adapters": active_adapters,
            "lora_parameter_tensors": len(adapter_parameter_names),
            "trainable_parameters": trainable,
            "processor_class": processor.__class__.__name__,
            "tokenizer_class": text_tokenizer.__class__.__name__,
            "chat_template_sha256": chat_template_sha256,
        }
    )
    '''

    export = r'''
    conversion_result = gguf_save_method(
        str(export_dir),
        tokenizer=processor,
        quantization_method=export_config["quantization_method"],
        maximum_memory_usage=float(export_config["maximum_memory_usage"]),
    )
    if not isinstance(conversion_result, dict):
        raise RuntimeError(
            f"Unsloth returned an unexpected GGUF result: {type(conversion_result)!r}"
        )
    reported_gguf_files = sorted(
        {Path(path).resolve() for path in conversion_result.get("gguf_files", [])},
        key=lambda path: path.name.casefold(),
    )
    missing_or_empty_ggufs = [
        str(path)
        for path in reported_gguf_files
        if not path.is_file() or path.stat().st_size <= 0
    ]
    if not reported_gguf_files or missing_or_empty_ggufs:
        raise RuntimeError(
            "Unsloth did not report complete GGUF files: "
            f"files={reported_gguf_files}, invalid={missing_or_empty_ggufs}"
        )
    quantization_label = str(export_config["quantization_method"]).casefold()
    primary_ggufs = [
        path
        for path in reported_gguf_files
        if "mmproj" not in path.name.casefold()
        and quantization_label in path.name.casefold()
    ]
    projector_ggufs = [
        path for path in reported_gguf_files if "mmproj" in path.name.casefold()
    ]
    if len(primary_ggufs) != 1:
        raise RuntimeError(
            "Expected exactly one reported Q4_K_M primary GGUF: "
            f"{reported_gguf_files}"
        )
    if snapshot_is_vlm and len(projector_ggufs) != 1:
        raise RuntimeError(
            "Expected exactly one reported VLM projector GGUF: "
            f"{reported_gguf_files}"
        )
    recognized_ggufs = {path.resolve() for path in primary_ggufs + projector_ggufs}
    if recognized_ggufs != {path.resolve() for path in reported_gguf_files}:
        raise RuntimeError(
            f"Unrecognized GGUF outputs were reported: {reported_gguf_files}"
        )
    if not bool(conversion_result.get("is_vlm")) == snapshot_is_vlm:
        raise RuntimeError(
            "Unsloth GGUF result VLM status does not match the verified snapshot"
        )
    gguf_path = primary_ggufs[0]
    gguf_gib = gguf_path.stat().st_size / 1024**3
    expected_gguf_gib = [
        float(export_config["expected_gguf_gib_lower"]),
        float(export_config["expected_gguf_gib_upper"]),
    ]
    if gguf_gib < 4 or gguf_gib > 20:
        raise RuntimeError(f"Implausible 12B Q4_K_M GGUF size: {gguf_gib:.2f} GiB")
    expected_size_range_match = expected_gguf_gib[0] <= gguf_gib <= expected_gguf_gib[1]
    modelfile = export_dir / "Modelfile"
    modelfile.write_text(
        "\n".join(
            [
                f"FROM ./{gguf_path.name}",
                "PARAMETER temperature 0",
                "PARAMETER seed 3407",
                "PARAMETER num_ctx 2048",
                "PARAMETER num_predict 64",
                "",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )
    drive_output.mkdir(parents=False, exist_ok=False)
    copied = {}
    export_artifacts = [gguf_path, *projector_ggufs, modelfile]
    for source in export_artifacts:
        partial = drive_output / (source.name + ".partial")
        destination = drive_output / source.name
        if partial.exists() or destination.exists():
            raise FileExistsError(f"Refusing to overwrite Drive output: {destination}")
        shutil.copy2(source, partial)
        source_sha256 = sha256_file(source)
        if partial.stat().st_size != source.stat().st_size:
            raise RuntimeError(f"Drive copy size mismatch: {source.name}")
        if sha256_file(partial) != source_sha256:
            raise RuntimeError(f"Drive copy hash mismatch: {source.name}")
        partial.rename(destination)
        copied[source.name] = {
            "sha256": sha256_file(destination),
            "bytes": destination.stat().st_size,
        }
    workflow_elapsed_seconds = time.perf_counter() - workflow_started_monotonic
    receipt = {
        "schema_version": 3,
        "phase": 5,
        "optional_export": "gguf_q4_k_m",
        "run_id": run_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "workflow_elapsed_seconds": workflow_elapsed_seconds,
        "approval": {
            "approved_at": export_config["approved_at"],
            "required_approval_code": export_config["required_approval_code"],
            "approved_compute_units_with_20pct_buffer": (
                export_config["approved_compute_units_with_20pct_buffer"]
            ),
        },
        "base_model_id": model_config["model_id"],
        "base_model_revision": model_config["revision"],
        "model_snapshot": {
            "resolved_revision": model_info.sha,
            "remote_weight_files": len(remote_weight_files),
            "indexed_shards": len(indexed_shards),
            "weight_bytes": remote_weight_bytes,
            "complete": True,
        },
        "adapter_checkpoint": int(phase3["selected_checkpoint"]),
        "phase3_archive_sha256": archive_sha256,
        "adapter_merge": {
            "peft_detected": True,
            "peft_model_class": model.__class__.__name__,
            "active_adapters": active_adapters,
            "lora_parameter_tensors": len(adapter_parameter_names),
            "original_adapter_config_sha256": adapter_config_sha256,
            "runtime_base_model_rebound_to_verified_snapshot": True,
        },
        "quantization_method": export_config["quantization_method"],
        "maximum_memory_usage": export_config["maximum_memory_usage"],
        "gpu": gpu,
        "resources": {
            "free_local_disk_gib_before_export": free_disk_gib,
            "free_drive_disk_gib_before_export": free_drive_gib,
            "gguf_gib": gguf_gib,
            "expected_gguf_gib": expected_gguf_gib,
            "expected_size_range_match": expected_size_range_match,
        },
        "tokenizer": {
            "processor_class": processor.__class__.__name__,
            "tokenizer_class": text_tokenizer.__class__.__name__,
            "chat_template_sha256": chat_template_sha256,
            "bos_token": text_tokenizer.bos_token,
            "eos_token": text_tokenizer.eos_token,
        },
        "gguf": {
            "primary_file": gguf_path.name,
            "projector_files": [path.name for path in projector_ggufs],
            "reported_files": [path.name for path in reported_gguf_files],
            "ollama_import_mode": "text_only_primary_gguf",
            "vlm_projector_archived": bool(projector_ggufs),
        },
        "packages": {
            package: importlib.metadata.version(package)
            for package in (
                "unsloth",
                "unsloth-zoo",
                "transformers",
                "peft",
                "bitsandbytes",
                "accelerate",
            )
        },
        "files": copied,
        "chat_template_warning": "Validate the exported GGUF in Ollama before use.",
        "external_upload_performed": False,
        "published": False,
    }
    receipt_path = drive_output / "gguf-export-receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    print(f"Drive export directory: {drive_output}")
    print("Optional export complete. No model was uploaded or published.")
    '''

    notebook = nbformat.v4.new_notebook(
        cells=[
            _markdown(
                """
                # tw-med-llm-qlora — optional GGUF Q4_K_M export

                This Linux Colab notebook is **not required** for adapter delivery. It is
                approved in repository config but remains disabled in the first code cell for
                every new run. After copying the exact approval code, use an A100 40GB runtime
                and execute from the top. It never pushes a model; each run writes to a unique
                Drive directory for later Windows Ollama validation.
                """
            ),
            _code(gate),
            _markdown("## 1. Install the pinned Phase 3-compatible export stack"),
            _code(_install_cell()),
            _markdown("## 2. Verify A100, Drive, token, and disk"),
            _code(setup),
            _markdown("## 3. Verify and extract only the selected step-700 adapter"),
            _code(extraction),
            _markdown(
                """
                ## 4. Download, verify, and load the pinned base and frozen adapter

                This cell first completes the exact Hugging Face snapshot and verifies every
                safetensors shard, tokenizer, and required processor file. It then gives
                Unsloth the verified local snapshot only, preventing an interrupted cache
                download from being mistaken for a loadable model.
                """
            ),
            _code(model_load),
            _markdown("## 5. Export Q4_K_M and an Ollama Modelfile"),
            _code(export),
        ],
        metadata={
            "accelerator": "GPU",
            "colab": {"gpuType": "A100", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
    )
    for index, cell in enumerate(notebook.cells):
        cell["id"] = f"phase5-export-{index:02d}"
        if cell.cell_type == "code":
            cell.execution_count = None
            cell.outputs = []
    return notebook


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = nbformat.writes(build_notebook(), version=4)
    if args.check:
        if not OUTPUT_PATH.is_file() or OUTPUT_PATH.read_text(encoding="utf-8") != expected:
            raise SystemExit(f"{OUTPUT_PATH} is stale; rebuild it")
        return 0
    OUTPUT_PATH.write_text(expected, encoding="utf-8")
    print(OUTPUT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
    configured_export = bool(config["export"]["gguf"]["enabled"])

    gate = f'''
    # OPTIONAL EXPORT HARD GATE — this is the only cell that may be edited.
    CONFIG_EXPORT_ENABLED = {configured_export!r}
    ENABLE_GGUF_EXPORT = False
    GGUF_EXPORT_APPROVAL = ""
    REQUIRED_GGUF_EXPORT_APPROVAL = "GGUF_Q4_K_M_APPROVED_20260722"

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
    print("GGUF Q4_K_M export gate passed.")
    '''

    setup = f'''
    # ruff: noqa: E501
    import hashlib
    import importlib.metadata
    import json
    import shutil
    from datetime import UTC, datetime
    from pathlib import Path, PurePosixPath
    from zipfile import ZipFile

    import torch
    from google.colab import drive, userdata

    PROJECT_CONFIG = json.loads({config_literal})
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
    if total_vram_gib < 38 or not gpu["bf16_supported"]:
        raise RuntimeError(f"A100-class BF16 GPU with at least 38 GiB is required: {{gpu}}")
    free_disk_gib = shutil.disk_usage("/content").free / 1024**3
    if free_disk_gib < 100:
        raise RuntimeError(f"At least 100 GiB free local disk is required: {{free_disk_gib:.1f}}")
    drive.mount("/content/drive")
    phase3 = PROJECT_CONFIG["evaluation"]["phase3_adapter"]
    export_config = PROJECT_CONFIG["export"]["gguf"]
    archive_path = Path(phase3["drive_archive"])
    local_root = Path("/content/tw-med-gguf-export")
    adapter_dir = local_root / "adapter"
    export_dir = local_root / "q4_k_m"
    drive_output = Path("/content/drive/MyDrive/tw-med-llm-qlora/phase5/gguf-q4-k-m")
    for directory in (local_root, adapter_dir, export_dir):
        directory.mkdir(parents=True, exist_ok=True)
    print(json.dumps({{"gpu": gpu, "free_disk_gib": free_disk_gib}}, indent=2))
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
    adapter_config = json.loads(
        (adapter_dir / "adapter_config.json").read_text(encoding="utf-8")
    )
    model_config = PROJECT_CONFIG["models"]["primary"]
    if adapter_config.get("base_model_name_or_path") != model_config["model_id"]:
        raise RuntimeError("Adapter/base mismatch; export stopped")
    print({"adapter_files": len(selected), "archive_sha256_verified": True})
    '''

    model_load = r'''
    from peft import PeftModel
    from unsloth import FastModel

    base_model, processor = FastModel.from_pretrained(
        model_name=model_config["model_id"],
        revision=model_config["revision"],
        max_seq_length=2048,
        load_in_4bit=True,
        load_in_8bit=False,
        full_finetuning=False,
        token=HF_TOKEN,
    )
    model = PeftModel.from_pretrained(
        base_model,
        str(adapter_dir),
        is_trainable=False,
        low_cpu_mem_usage=True,
    )
    model.eval()
    trainable, total = model.get_nb_trainable_parameters()
    if trainable != 0 or total <= 0:
        raise RuntimeError(f"Frozen adapter audit failed: trainable={trainable}, total={total}")
    if not hasattr(model, "save_pretrained_gguf"):
        raise RuntimeError(
            "Current Unsloth no longer exposes save_pretrained_gguf on this PEFT model; "
            "stop and recheck the official API instead of using an unverified fallback."
        )
    print({"adapter_reloaded": True, "trainable_parameters": trainable})
    '''

    export = r'''
    model.save_pretrained_gguf(
        str(export_dir),
        tokenizer=processor,
        quantization_method="q4_k_m",
        maximum_memory_usage=0.5,
    )
    gguf_files = sorted(export_dir.glob("*.gguf"))
    if len(gguf_files) != 1 or gguf_files[0].stat().st_size <= 0:
        raise RuntimeError(f"Expected exactly one non-empty GGUF file: {gguf_files}")
    gguf_path = gguf_files[0]
    modelfile = export_dir / "Modelfile"
    modelfile.write_text(
        "\n".join(
            [
                f"FROM ./{gguf_path.name}",
                "PARAMETER temperature 0",
                "PARAMETER seed 3407",
                "PARAMETER num_ctx 2048",
                "",
            ]
        ),
        encoding="utf-8",
    )
    drive_output.mkdir(parents=True, exist_ok=True)
    copied = {}
    for source in (gguf_path, modelfile):
        partial = drive_output / (source.name + ".partial")
        destination = drive_output / source.name
        shutil.copy2(source, partial)
        if sha256_file(partial) != sha256_file(source):
            raise RuntimeError(f"Drive copy hash mismatch: {source.name}")
        partial.replace(destination)
        copied[source.name] = {
            "path": str(destination),
            "sha256": sha256_file(destination),
            "bytes": destination.stat().st_size,
        }
    receipt = {
        "schema_version": 1,
        "phase": 5,
        "optional_export": "gguf_q4_k_m",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "base_model_id": model_config["model_id"],
        "base_model_revision": model_config["revision"],
        "adapter_checkpoint": int(phase3["selected_checkpoint"]),
        "phase3_archive_sha256": archive_sha256,
        "gpu": gpu,
        "packages": {
            package: importlib.metadata.version(package)
            for package in ("unsloth", "unsloth-zoo", "transformers", "peft")
        },
        "files": copied,
        "chat_template_warning": "Validate the exported GGUF in Ollama before use.",
        "published": False,
    }
    receipt_path = drive_output / "gguf-export-receipt.json"
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    print("Optional export complete. No model was uploaded or published.")
    '''

    notebook = nbformat.v4.new_notebook(
        cells=[
            _markdown(
                """
                # tw-med-llm-qlora — optional GGUF Q4_K_M export

                This Linux Colab notebook is **not required** for adapter delivery. It is
                disabled in both repository config and the first code cell. After explicit
                approval, use an A100-class runtime and execute from the top. It never pushes
                a model; output stays in Drive for later Windows Ollama validation.
                """
            ),
            _code(gate),
            _markdown("## 1. Install the pinned Phase 3-compatible export stack"),
            _code(_install_cell()),
            _markdown("## 2. Verify A100, Drive, token, and disk"),
            _code(setup),
            _markdown("## 3. Verify and extract only the selected step-700 adapter"),
            _code(extraction),
            _markdown("## 4. Load the pinned base and frozen adapter"),
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

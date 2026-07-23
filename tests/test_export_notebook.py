import ast
import subprocess
import sys
import tomllib
from pathlib import Path

import nbformat

ROOT = Path(__file__).parents[1]
BUILDER = ROOT / "scripts" / "build_export_notebook.py"
NOTEBOOK = ROOT / "notebooks" / "export_gguf.ipynb"
CONFIG = ROOT / "configs" / "project.toml"


def test_export_notebook_is_current_and_clean() -> None:
    subprocess.run([sys.executable, str(BUILDER), "--check"], cwd=ROOT, check=True)
    notebook = nbformat.read(NOTEBOOK, as_version=4)

    assert notebook.metadata["colab"]["gpuType"] == "A100"
    assert all(not cell.get("outputs") for cell in notebook.cells if cell.cell_type == "code")
    assert all(
        cell.get("execution_count") is None
        for cell in notebook.cells
        if cell.cell_type == "code"
    )


def test_export_repo_gate_records_approved_scope() -> None:
    with CONFIG.open("rb") as source:
        export_config = tomllib.load(source)["export"]["gguf"]

    assert export_config["enabled"] is True
    assert export_config["approved_at"] == "2026-07-23"
    assert (
        export_config["required_approval_code"]
        == "GGUF_Q4_K_M_A100_APPROVED_20260723"
    )
    assert export_config["quantization_method"] == "q4_k_m"
    assert export_config["run_environment"] == "colab_linux"
    assert export_config["required_gpu_name_contains"] == "A100"
    assert export_config["minimum_vram_gib"] == 38.0
    assert export_config["minimum_local_disk_gib"] == 100.0
    assert export_config["minimum_drive_disk_gib"] == 20.0
    assert export_config["approved_compute_units_with_20pct_buffer"] == 6.36
    assert export_config["external_upload_allowed"] is False


def test_export_per_run_gate_is_locked_and_api_is_pinned() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells)

    assert "CONFIG_EXPORT_ENABLED = True" in source
    assert "ENABLE_GGUF_EXPORT = False" in source
    assert 'GGUF_EXPORT_APPROVAL = ""' in source
    assert "REQUIRED_GGUF_EXPORT_APPROVAL = " in source
    assert "GGUF_Q4_K_M_A100_APPROVED_20260723" in source
    assert 'quantization_method=export_config["quantization_method"]' in source
    assert "save_pretrained_gguf" in source
    assert "conversion_result = gguf_save_method(" in source
    assert 'conversion_result.get("gguf_files", [])' in source
    assert 'maximum_memory_usage=float(export_config["maximum_memory_usage"])' in source
    assert "push_to_hub" not in source
    assert "upload_folder" not in source
    assert '"external_upload_performed": False' in source
    assert "PARAMETER temperature 0" in source
    assert "PARAMETER num_predict 64" in source


def test_export_notebook_has_resource_and_evidence_gates() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells)

    assert 'expected_gpu_name = str(export_config["required_gpu_name_contains"])' in source
    assert 'minimum_local_disk_gib = float(export_config["minimum_local_disk_gib"])' in source
    assert 'minimum_drive_disk_gib = float(export_config["minimum_drive_disk_gib"])' in source
    assert 'drive_output = drive_root / run_id' in source
    assert "archive_path.stat().st_size" in source
    assert "phase3_archive_sha256" in source
    assert "chat_template_sha256" in source
    assert '"peft_detected": True' in source
    assert '"runtime_base_model_rebound_to_verified_snapshot": True' in source
    assert '"projector_files": [path.name for path in projector_ggufs]' in source
    assert '"ollama_import_mode": "text_only_primary_gguf"' in source
    assert "expected_size_range_match" in source
    assert '"schema_version": 3' in source
    assert "partial.rename(destination)" in source
    assert 'newline="\\n"' in source


def test_export_notebook_verifies_complete_snapshot_before_local_load() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells)

    assert "HfApi(token=HF_TOKEN).model_info(" in source
    assert "files_metadata=True" in source
    assert "snapshot_download(" in source
    assert '"model.safetensors.index.json"' in source
    assert "missing_remote_weights" in source
    assert "missing_indexed_shards" in source
    assert "missing_processor_files" in source
    assert '"complete": True' in source
    assert "shutil.copytree(adapter_dir, runtime_adapter_dir)" in source
    assert 'runtime_adapter_config["base_model_name_or_path"] = str(snapshot_path)' in source
    assert "model_name=str(runtime_adapter_dir)" in source
    assert "tokenizer_name=str(snapshot_path)" in source
    assert "local_files_only=True" in source
    assert "use_safetensors=True" in source
    assert "PeftModel.from_pretrained(" not in source
    assert "isinstance(model, PeftModel)" in source
    assert 'model.__dict__.get("save_pretrained_gguf")' in source
    assert 'getattr(gguf_save_method, "__self__", None) is not model' in source
    assert '"lora_" in name.casefold()' in source
    assert '"model_snapshot": {' in source
    assert source.index("from unsloth import FastModel") < source.index(
        "from peft import PeftModel"
    )


def test_export_python_cells_compile_top_to_bottom() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)

    for cell in notebook.cells:
        if cell.cell_type != "code" or cell.source.lstrip().startswith("%"):
            continue
        ast.parse(cell.source)

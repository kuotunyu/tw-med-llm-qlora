import ast
import subprocess
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).parents[1]
BUILDER = ROOT / "scripts" / "build_export_notebook.py"
NOTEBOOK = ROOT / "notebooks" / "export_gguf.ipynb"


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


def test_export_gate_is_disabled_and_api_is_pinned() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells)

    assert "CONFIG_EXPORT_ENABLED = False" in source
    assert "ENABLE_GGUF_EXPORT = False" in source
    assert 'GGUF_EXPORT_APPROVAL = ""' in source
    assert 'quantization_method="q4_k_m"' in source
    assert "save_pretrained_gguf" in source
    assert "maximum_memory_usage=0.5" in source
    assert "push_to_hub" not in source
    assert "upload_folder" not in source
    assert "PARAMETER temperature 0" in source


def test_export_python_cells_compile_top_to_bottom() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)

    for cell in notebook.cells:
        if cell.cell_type != "code" or cell.source.lstrip().startswith("%"):
            continue
        ast.parse(cell.source)

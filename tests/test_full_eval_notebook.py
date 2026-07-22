from __future__ import annotations

import ast
import json
import subprocess
import sys
import tomllib
from pathlib import Path

import nbformat

ROOT = Path(__file__).parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "evaluate_phase4_full.ipynb"
BUILDER_PATH = ROOT / "scripts" / "build_full_eval_notebook.py"
CONFIG_PATH = ROOT / "configs" / "project.toml"


def _notebook() -> nbformat.NotebookNode:
    return nbformat.read(NOTEBOOK_PATH, as_version=4)


def test_generated_full_eval_notebook_is_current_and_clean() -> None:
    subprocess.run([sys.executable, str(BUILDER_PATH), "--check"], cwd=ROOT, check=True)
    notebook = _notebook()
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]

    assert 'NOTEBOOK_BUILD = "phase4-full-approved-resumable-v1"' in code_cells[0].source
    assert all(cell.execution_count is None for cell in code_cells)
    assert all(cell.outputs == [] for cell in code_cells)
    assert len({cell.id for cell in notebook.cells}) == len(notebook.cells)


def test_full_eval_notebook_python_cells_compile() -> None:
    for cell in _notebook().cells:
        if cell.cell_type == "code":
            compile(cell.source, f"<phase4-full-cell-{cell.id}>", "exec")


def test_full_eval_notebook_has_fixed_approval_workload_and_resume_contract() -> None:
    source = "\n".join(cell.source for cell in _notebook().cells)
    required = [
        'RUN_MODE = "full"',
        "ALLOW_FULL_EVALUATION = True",
        'FULL_EVALUATION_APPROVAL = "PHASE4_FULL_28758_APPROVED_20260722"',
        'approval["approval_phrase"] != "確認解鎖 Phase 4 正式評估"',
        'userdata.get("HF_TOKEN")',
        '"A100" not in gpu_name.upper()',
        "WORKLOAD.total != 28758",
        "planned_requests != WORKLOAD.total",
        'split="test"',
        "expected_test_rows",
        'allow_patterns=allowed_test_patterns',
        '"train_rows_loaded": 0',
        '"validation_rows_loaded": 0',
        "stability_sample(",
        "shuffle_options(",
        "evaluation_request_id(",
        "plan_result_shards(",
        "write_result_shard(",
        "read_verified_result_shard(",
        "atomic_copy_verified(",
        "SHARD_SIZE = int(approval[\"shard_size\"])",
        "ThreadPoolExecutor(",
        "BoxExtractor()",
        "ExactMatchScorer()",
        "parse_mcq_answer(raw_output)",
        'choice.finish_reason == "length"',
        "build_vllm_serve_command(",
        "paired_bootstrap_accuracy_difference(",
        "mcnemar_exact_test(",
        "forgetting_noninferiority(",
        "representative_case_ids(",
        "subject_accuracy(",
        "phase4-results.json",
        "public-predictions.jsonl",
        "medqa-representative-cases-private.json",
        "run_manifest.json",
        "receipt.json",
        '"completed_requests": len(all_public_rows)',
    ]
    for fragment in required:
        assert fragment in source

    assert "push_to_hub" not in source
    assert "OPENAI_API_KEY" not in source
    assert "GOOGLE_API_KEY" not in source


def test_full_eval_notebook_embeds_helpers_verbatim() -> None:
    helper_cell = next(
        cell.source
        for cell in _notebook().cells
        if "EMBEDDED_HELPER_FILES = json.loads(" in cell.source
    )
    tree = ast.parse(helper_cell)
    assignment = tree.body[0]
    assert isinstance(assignment, ast.Assign)
    assert isinstance(assignment.value, ast.Call)
    embedded_files = json.loads(ast.literal_eval(assignment.value.args[0]))
    for name in (
        "types.py",
        "medqa.py",
        "evaluation.py",
        "tmmlu.py",
        "phase4.py",
        "phase4_full.py",
    ):
        helper = (ROOT / "src" / "tw_med_qlora" / name).read_text(encoding="utf-8").strip()
        assert embedded_files[f"tw_med_qlora/{name}"].strip() == helper


def test_full_eval_approval_is_pinned_to_reviewed_calibration() -> None:
    with CONFIG_PATH.open("rb") as source:
        config = tomllib.load(source)
    approval = config["evaluation"]["full_approval"]

    assert approval == {
        "approved": True,
        "approved_at": "2026-07-22",
        "approval_phrase": "確認解鎖 Phase 4 正式評估",
        "required_approval_code": "PHASE4_FULL_28758_APPROVED_20260722",
        "approved_requests": 28_758,
        "calibration_run_id": "20260722T061028Z",
        "calibration_manifest_sha256": (
            "d9c0f23e72808f0a8fc4edc1a7889719637f9390346725d5f72f10c4d3e2cdf2"
        ),
        "calibration_validation_sha256": (
            "6c39aff8cd4e82b80d24a4ce7f7959e7b7fd5ba9546f2432ef3b7c96212b2d85"
        ),
        "shard_size": 250,
        "parallel_workers": 4,
        "compute_units_per_hour": 5.3,
        "projected_hours": 2.979555188311461,
        "projected_compute_units": 15.791642498050743,
        "projected_compute_units_with_20pct_buffer": 18.94997099766089,
    }

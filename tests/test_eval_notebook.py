from __future__ import annotations

import ast
import json
import subprocess
import sys
import tomllib
from pathlib import Path

import nbformat

ROOT = Path(__file__).parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "evaluate_phase4.ipynb"
BUILDER_PATH = ROOT / "scripts" / "build_eval_notebook.py"
REQUIREMENTS_PATH = ROOT / "requirements" / "colab-eval.txt"
CONFIG_PATH = ROOT / "configs" / "project.toml"


def _notebook() -> nbformat.NotebookNode:
    return nbformat.read(NOTEBOOK_PATH, as_version=4)


def test_generated_phase4_notebook_is_current_and_clean() -> None:
    subprocess.run([sys.executable, str(BUILDER_PATH), "--check"], cwd=ROOT, check=True)
    notebook = _notebook()
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]

    assert 'NOTEBOOK_BUILD = "phase4-calibration-policy-v4"' in code_cells[0].source
    assert '"--torch-backend"' in code_cells[0].source
    assert '"cu129"' in code_cells[0].source
    assert all(cell.execution_count is None for cell in code_cells)
    assert all(cell.outputs == [] for cell in code_cells)
    assert len({cell.id for cell in notebook.cells}) == len(notebook.cells)


def test_phase4_notebook_python_cells_compile() -> None:
    code_cells = [
        cell
        for cell in _notebook().cells
        if cell.cell_type == "code" and not cell.source.lstrip().startswith("%")
    ]
    for cell in code_cells:
        compile(cell.source, f"<phase4-cell-{cell.id}>", "exec")


def test_phase4_notebook_enforces_calibration_and_test_isolation() -> None:
    source = "\n".join(cell.source for cell in _notebook().cells)
    required = [
        'RUN_MODE = "calibration"',
        "ALLOW_FULL_EVALUATION = False",
        'REQUIRED_FULL_EVALUATION_APPROVAL = "PHASE4_28758_REQUESTS"',
        'userdata.get("HF_TOKEN")',
        '"A100" not in gpu_name.upper()',
        'allow_patterns=["data/*_val.csv"]',
        'snapshot_path.rglob("*_test.csv")',
        'split="validation"',
        "stratified_calibration_sample(",
        "shuffle_options(",
        "calibration_manifest[\"total\"] != 20",
        "WORKLOAD.total",
        "expected_total_requests",
        "extract_verified_adapter(",
        "BoxExtractor()",
        "ExactMatchScorer()",
        "parse_mcq_answer(raw_output)",
        "GENERATION_MAX_TOKENS",
        "MINIMUM_PARSE_RATE",
        "TOKEN_LIMIT_HITS_FAIL_CALIBRATION",
        "token_limit_hits_count_as_incorrect",
        "max_token_limit_hits",
        "Do not unlock full evaluation",
        "build_vllm_serve_command(",
        'vllm_config["max_model_length"]',
        "ThreadPoolExecutor(max_workers=4)",
        "start_new_session=True",
        "os.killpg(",
        "except ProcessLookupError:",
        "process.poll() is None",
        '"vllm_native_import": True',
        'native_audit["torch_cuda"] != "12.9"',
        "private_archive",
        '"test_files_loaded": 0',
        '"full_evaluation_unlocked": False',
        "project_evaluation_cost(",
        "run_manifest.json",
        "receipt.json",
        "calibration_summary.json",
    ]
    for fragment in required:
        assert fragment in source

    assert 'allow_patterns=["data/*_test.csv"]' not in source
    assert 'split="test"' not in source
    assert "push_to_hub" not in source


def test_phase4_notebook_embeds_tested_helpers_verbatim() -> None:
    helper_cell = next(
        cell.source
        for cell in _notebook().cells
        if "EMBEDDED_HELPER_FILES = json.loads(" in cell.source
    )
    tree = ast.parse(helper_cell)
    assignment = tree.body[0]
    assert isinstance(assignment, ast.Assign)
    assert isinstance(assignment.value, ast.Call)
    payload = ast.literal_eval(assignment.value.args[0])
    embedded_files = json.loads(payload)
    for name in ("types.py", "medqa.py", "evaluation.py", "tmmlu.py", "phase4.py"):
        helper = (ROOT / "src" / "tw_med_qlora" / name).read_text(encoding="utf-8").strip()
        assert embedded_files[f"tw_med_qlora/{name}"].strip() == helper


def test_phase4_dependencies_and_external_revisions_are_pinned() -> None:
    requirements = [
        line.strip()
        for line in REQUIREMENTS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert requirements == [
        "uv==0.11.31",
        (
            "vllm @ https://github.com/vllm-project/vllm/releases/download/"
            "v0.25.1/vllm-0.25.1%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl"
            "#sha256=9e206f370c934a2d4b6b1f05d3d09708d344e05d80260189ef19f60755709431"
        ),
        "twinkle-eval==2.8.0",
        "bitsandbytes==0.49.2",
    ]

    with CONFIG_PATH.open("rb") as source:
        config = tomllib.load(source)
    evaluation = config["evaluation"]
    assert evaluation["twinkle_eval"]["version"] == "2.8.0"
    assert len(evaluation["twinkle_eval"]["revision"]) == 40
    assert evaluation["vllm"]["version"] == "0.25.1"
    assert evaluation["vllm"]["wheel_variant"] == "cu129"
    assert evaluation["vllm"]["expected_torch_cuda"] == "12.9"
    assert len(evaluation["vllm"]["wheel_sha256"]) == 64
    assert evaluation["workload"]["expected_total_requests"] == 28_758
    assert evaluation["generation"] == {
        "max_tokens": 256,
        "minimum_calibration_parse_rate": 0.8,
        "token_limit_hits_fail_calibration": False,
        "token_limit_hits_count_as_incorrect": True,
        "temperature": 0.0,
        "top_p": 1.0,
    }
    assert evaluation["phase3_adapter"]["archive_bytes"] == 113_079_186

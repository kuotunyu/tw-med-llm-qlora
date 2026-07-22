from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

import nbformat

ROOT = Path(__file__).parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "train_qlora.ipynb"
BUILDER_PATH = ROOT / "scripts" / "build_train_notebook.py"
RECOVERY_SCRIPT_PATH = ROOT / "scripts" / "colab_recover_phase3_eval_nan.py"
REQUIREMENTS_PATH = ROOT / "requirements" / "colab-train.txt"
CHECKPOINT_HELPERS_PATH = ROOT / "src" / "tw_med_qlora" / "checkpointing.py"
CONFIG_PATH = ROOT / "configs" / "project.toml"


def _notebook() -> nbformat.NotebookNode:
    return nbformat.read(NOTEBOOK_PATH, as_version=4)


def test_generated_notebook_is_current_and_has_no_saved_outputs() -> None:
    subprocess.run(
        [sys.executable, str(BUILDER_PATH), "--check"],
        cwd=ROOT,
        check=True,
    )
    notebook = _notebook()
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]

    assert code_cells[0].source.startswith("%pip install --quiet")
    assert "REQUIRED_COLAB_PACKAGES" in code_cells[1].source
    assert "importlib.util.find_spec(module_name)" in code_cells[1].source
    assert "Dependency gate passed" in code_cells[1].source
    assert all(cell.execution_count is None for cell in code_cells)
    assert all(cell.outputs == [] for cell in code_cells)


def test_python_notebook_cells_are_syntactically_valid() -> None:
    code_cells = [
        cell
        for cell in _notebook().cells
        if cell.cell_type == "code" and not cell.source.lstrip().startswith("%")
    ]

    for cell in code_cells:
        compile(cell.source, f"<notebook-cell-{cell.id}>", "exec")


def test_phase3_recovery_script_is_syntactically_valid() -> None:
    source = RECOVERY_SCRIPT_PATH.read_text(encoding="utf-8")

    compile(source, RECOVERY_SCRIPT_PATH.name, "exec")
    assert "expected_value = source_value.to(dtype=active_value.dtype)" in source
    assert "torch.equal(active_value, expected_value)" in source


def test_notebook_embeds_the_tested_checkpoint_helpers_verbatim() -> None:
    helper_source = CHECKPOINT_HELPERS_PATH.read_text(encoding="utf-8").strip()
    notebook_sources = [
        cell.source.strip() for cell in _notebook().cells if cell.cell_type == "code"
    ]

    assert any(helper_source in source for source in notebook_sources)


def test_colab_requirements_are_exactly_pinned() -> None:
    requirements = [
        line.strip()
        for line in REQUIREMENTS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    assert requirements
    assert all("==" in requirement for requirement in requirements)
    assert "unsloth[colab-new]==2026.7.4" in requirements
    assert "trl==0.22.2" in requirements
    assert all(not requirement.startswith("hf-transfer==") for requirement in requirements)


def test_notebook_enforces_phase_3_training_and_isolation_contract() -> None:
    notebook = _notebook()
    source = "\n".join(cell.source for cell in notebook.cells)
    trainer_cell = next(
        cell.source
        for cell in notebook.cells
        if cell.cell_type == "code" and "trainer = SFTTrainer(" in cell.source
    )

    required_fragments = [
        'RUN_MODE = "full"',
        "ALLOW_FULL_TRAINING = True",
        'FULL_TRAINING_APPROVAL = "PHASE3_11248_1EPOCH"',
        'REQUIRED_FULL_TRAINING_APPROVAL = "PHASE3_11248_1EPOCH"',
        'FULL_TRAINING_APPROVED_AT = "2026-07-22"',
        "APPROVED_BUFFERED_COMPUTE_UNITS = 31.784883615387425",
        "COMPUTE_UNITS_PER_HOUR = 5.3",
        "CURRENT_COMPUTE_UNITS = 436.2",
        "CALIBRATED_SECONDS_PER_STEP = 20.525462628900005",
        "CALIBRATED_CHECKPOINT_SECONDS_PER_SAVE = 12.12548161999996",
        "CALIBRATED_FULL_EVAL_SECONDS = 434.64561955287155",
        'CALIBRATED_HARDWARE_PROFILE = "primary_40g"',
        "dependency_errors",
        "Colab dependency installation is incomplete or mismatched",
        'read_colab_secret("HF_TOKEN", required=True)',
        "drive.mount(",
        "torch.cuda.is_bf16_supported()",
        'PROJECT_CONFIG["hardware_profiles"]',
        'premium_profiles = {"primary_80g", "primary_40g"}',
        'RUN_MODE == "full" and "A100" not in GPU_NAME.upper()',
        "Full-training GPU profile differs from the reviewed calibration",
        "torch.backends.cuda.matmul.allow_tf32",
        "EXPERIMENT_FINGERPRINT = experiment_fingerprint(TRAINING_CONTRACT)",
        "DRIVE_CHECKPOINT_ROOT",
        "snapshot_download(",
        "files_metadata=True",
        '"model.safetensors.index.json"',
        "missing_indexed_shards",
        "missing_processor_files",
        "tokenizer_name=str(snapshot_path)",
        '"complete": True',
        "FastModel.from_pretrained(",
        "local_files_only=True",
        "use_safetensors=True",
        "load_in_4bit=True",
        "finetune_vision_layers=False",
        "processing_class=text_tokenizer",
        "train_on_responses_only(",
        "is_vision_collator",
        "MASKING_AUDIT",
        "last_response_only=True",
        'max_steps=calibration_steps if RUN_MODE == "calibration" else -1',
        "train_dataset=train_dataset",
        "eval_dataset=eval_dataset",
        'eval_strategy="steps" if RUN_MODE == "full" else "no"',
        "save_only_model=False",
        "save_total_limit=int(",
        "DriveCheckpointCallback",
        "def on_step_end(",
        "archive_checkpoint(",
        "restore_latest_checkpoint(",
        "resume_from_checkpoint=resume_argument",
        '"restore_test_passed": True',
        "model.save_pretrained",
        "tokenizer.save_pretrained",
        "trainer.save_state",
        "adapter_config.json",
        "reloaded_base_model, reloaded_tokenizer = FastModel.from_pretrained(",
        "model_name=str(snapshot_path)",
        "PeftModel.from_pretrained(",
        'adapter_name="default"',
        '"base_source": "pinned_local_snapshot"',
        "globals().pop(variable_name, None)",
        "generation_inputs = reload_text_tokenizer.apply_chat_template(",
        '"generation_tokenizer": type(reload_text_tokenizer).__name__',
        "torch.cuda.max_memory_reserved()",
        "all(math.isfinite(loss)",
        "trainer.evaluate()",
        "set_peft_model_state_dict(",
        "expected_value = source_value.to(dtype=active_value.dtype)",
        '"status": "recovered_from_non_finite_post_train_evaluation"',
        '"selected_adapter_checkpoint": SELECTED_ADAPTER_CHECKPOINT',
        '"rejected_post_train_eval_loss": "NaN"',
        '"seconds_per_step_excluding_checkpoint"',
        '"calibrated_full_eval_seconds"',
        "full_eval_events",
        "trainer_log.csv",
        "trainer_log.json",
        "training_curves.png",
        "MODEL_CARD_DRAFT.md",
        '"full_training_approval": {',
        "COST_ESTIMATE",
        "run_manifest.json",
        '"trainer_log.csv"',
        '"training_curves.png"',
        '"drive_evidence": drive_evidence',
        "drive_manifest",
        "pip-freeze.txt",
    ]
    for fragment in required_fragments:
        assert fragment in source

    assert "push_to_hub=True" not in source
    assert "push_to_hub=False" in source
    assert "UnslothVisionDataCollator" not in source
    assert "generation_inputs = reloaded_tokenizer.apply_chat_template(" not in source
    assert "HF_HUB_ENABLE_HF_TRANSFER" not in source
    assert "test_dataset" not in trainer_cell
    assert "eval_dataset=eval_dataset" in trainer_cell


def test_notebook_embeds_locked_models_data_and_source_hashes() -> None:
    with CONFIG_PATH.open("rb") as config_file:
        config = tomllib.load(config_file)
    source = "\n".join(cell.source for cell in _notebook().cells)

    for profile_name in ("primary", "fallback"):
        profile = config["models"][profile_name]
        assert profile["model_id"] in source
        assert profile["revision"] in source
    for split in ("train", "validation", "test"):
        source_hash = config["data"]["medqa"]["source_sha256"][split]
        assert len(source_hash) == 64
        assert source_hash in source

    assert config["training"]["smoke_examples"] == 100
    assert config["training"]["smoke_steps"] == 10
    assert {profile["name"] for profile in config["hardware_profiles"]} == {
        "primary_80g",
        "primary_40g",
        "primary_24g",
        "fallback_16g",
    }

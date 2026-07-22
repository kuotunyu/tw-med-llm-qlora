"""Build the deterministic Phase 3 Colab notebook from repository settings."""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path
from textwrap import dedent

import nbformat

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "project.toml"
REQUIREMENTS_PATH = ROOT / "requirements" / "colab-train.txt"
CHECKPOINT_HELPERS_PATH = ROOT / "src" / "tw_med_qlora" / "checkpointing.py"
OUTPUT_PATH = ROOT / "notebooks" / "train_qlora.ipynb"


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
    quoted = separator.join(json.dumps(requirement) for requirement in requirements)
    return "%pip install --quiet" + separator + quoted


def _dependency_check_cell() -> str:
    return dedent(
        r"""
        import importlib.metadata
        import importlib.util
        import json

        REQUIRED_COLAB_PACKAGES = {
            "unsloth": ("unsloth", "2026.7.4"),
            "unsloth-zoo": ("unsloth_zoo", "2026.7.4"),
            "transformers": ("transformers", "4.56.2"),
            "trl": ("trl", "0.22.2"),
            "datasets": ("datasets", "4.3.0"),
            "peft": ("peft", "0.19.1"),
            "bitsandbytes": ("bitsandbytes", "0.49.2"),
            "accelerate": ("accelerate", "1.14.0"),
            "huggingface-hub": ("huggingface_hub", "0.35.3"),
        }
        dependency_audit = {}
        dependency_errors = []
        for package_name, (module_name, expected_version) in REQUIRED_COLAB_PACKAGES.items():
            module_available = importlib.util.find_spec(module_name) is not None
            try:
                installed_version = importlib.metadata.version(package_name)
            except importlib.metadata.PackageNotFoundError:
                installed_version = None
            dependency_audit[package_name] = {
                "module": module_name,
                "module_available": module_available,
                "expected_version": expected_version,
                "installed_version": installed_version,
            }
            if not module_available or installed_version != expected_version:
                dependency_errors.append(package_name)

        print(json.dumps(dependency_audit, ensure_ascii=False, indent=2))
        if dependency_errors:
            raise RuntimeError(
                "Colab dependency installation is incomplete or mismatched: "
                f"{dependency_errors}. Rerun section 1 and do not continue to model loading."
            )
        print("Dependency gate passed. Continue from section 2 without restarting the runtime.")
        """
    ).strip()


def _checkpoint_helpers_cell() -> str:
    source = CHECKPOINT_HELPERS_PATH.read_text(encoding="utf-8").strip()
    if "'''" in source:
        raise ValueError("checkpoint helper source cannot contain triple single quotes")
    return (
        "embedded_checkpoint_helpers = r'''\n"
        + source
        + "\n'''\n"
        + 'exec(compile(embedded_checkpoint_helpers, "embedded_checkpointing.py", "exec"))'
    )


def build_notebook() -> nbformat.NotebookNode:
    with CONFIG_PATH.open("rb") as config_file:
        project_config = tomllib.load(config_file)
    config_json = json.dumps(
        project_config,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    indented_config_json = config_json.replace("\n", "\n    ")

    settings_cell = r"""
    # ruff: noqa: E501
    import csv
    import gc
    import hashlib
    import importlib.metadata
    import json
    import math
    import os
    import platform
    import random
    import shutil
    import subprocess
    import sys
    import time
    from collections import Counter
    from datetime import UTC, datetime
    from pathlib import Path
    from textwrap import dedent

    import pyarrow.parquet as parquet
    import torch
    from google.colab import drive, userdata
    from huggingface_hub import HfApi, hf_hub_download, login, snapshot_download

    PROJECT_CONFIG = json.loads(
        r'''
    __PROJECT_CONFIG_JSON__
        '''
    )
    RUN_MODE = "full"
    ALLOW_FULL_TRAINING = True
    FULL_TRAINING_APPROVAL = "PHASE3_11248_1EPOCH"
    REQUIRED_FULL_TRAINING_APPROVAL = "PHASE3_11248_1EPOCH"
    FULL_TRAINING_APPROVED_AT = "2026-07-22"
    APPROVED_BUFFERED_COMPUTE_UNITS = 31.784883615387425
    REQUIRE_PREMIUM_GPU = True
    AUTO_RESUME_FROM_DRIVE = True

    if RUN_MODE not in {"calibration", "full"}:
        raise ValueError("RUN_MODE must be 'calibration' or 'full'")
    if RUN_MODE == "full" and (
        not ALLOW_FULL_TRAINING
        or FULL_TRAINING_APPROVAL != REQUIRED_FULL_TRAINING_APPROVAL
    ):
        raise RuntimeError(
            "Phase 3 full-training gate is locked. Run A100 calibration first and "
            "obtain explicit approval before changing the three gate values."
        )
    if RUN_MODE == "calibration" and ALLOW_FULL_TRAINING:
        raise RuntimeError("Calibration mode cannot enable full training")

    # Reviewed from the A100 calibration resource panel at the approval gate.
    COMPUTE_UNITS_PER_HOUR = 5.3
    CURRENT_COMPUTE_UNITS = 436.2
    PRICE_PER_COMPUTE_UNIT = None
    CURRENCY_LABEL = None
    CALIBRATED_SECONDS_PER_STEP = 20.525462628900005
    CALIBRATED_CHECKPOINT_SECONDS_PER_SAVE = 12.12548161999996
    CALIBRATED_FULL_EVAL_SECONDS = 434.64561955287155
    CALIBRATED_HARDWARE_PROFILE = "primary_40g"

    if COMPUTE_UNITS_PER_HOUR is None or float(COMPUTE_UNITS_PER_HOUR) <= 0:
        raise RuntimeError(
            "Set COMPUTE_UNITS_PER_HOUR from the current A100 resource panel before running."
        )
    if CURRENT_COMPUTE_UNITS is not None and float(CURRENT_COMPUTE_UNITS) < 0:
        raise ValueError("CURRENT_COMPUTE_UNITS cannot be negative")
    if RUN_MODE == "full" and (
        CALIBRATED_SECONDS_PER_STEP is None
        or float(CALIBRATED_SECONDS_PER_STEP) <= 0
    ):
        raise RuntimeError("Full mode requires the reviewed A100 CALIBRATED_SECONDS_PER_STEP")
    if RUN_MODE == "full" and not CALIBRATED_HARDWARE_PROFILE:
        raise RuntimeError("Full mode requires the reviewed CALIBRATED_HARDWARE_PROFILE")
    if RUN_MODE == "full" and (
        CALIBRATED_CHECKPOINT_SECONDS_PER_SAVE is None
        or float(CALIBRATED_CHECKPOINT_SECONDS_PER_SAVE) <= 0
        or CALIBRATED_FULL_EVAL_SECONDS is None
        or float(CALIBRATED_FULL_EVAL_SECONDS) <= 0
    ):
        raise RuntimeError(
            "Full mode requires reviewed checkpoint and full-validation timing inputs"
        )
    if RUN_MODE == "full" and CURRENT_COMPUTE_UNITS is None:
        raise RuntimeError("Full mode requires CURRENT_COMPUTE_UNITS for the preflight budget gate")

    RUN_ID = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    print(
        json.dumps(
            {
                "phase": 3,
                "mode": RUN_MODE,
                "python": platform.python_version(),
                "full_training_enabled": ALLOW_FULL_TRAINING,
            },
            ensure_ascii=False,
        )
    )
    """
    settings_cell = settings_cell.replace(
        "__PROJECT_CONFIG_JSON__",
        indented_config_json,
    )

    cells = [
        _markdown(
            """
            # tw-med-llm-qlora — Phase 3 A100 full training

            A100 40GB calibration 已完成並通過成本閘門；這份 notebook 已依 2026-07-22
            的明確核准設定為 11,248 筆、1 epoch 完整訓練。本 Phase 不會推送 adapter。

            執行前請在 Colab：

            1. 新 runtime 請直接從最上方按「全部執行」，不要從第 6 節開始。
            2. 若同一 runtime 曾執行舊版 notebook，先選「Runtime → Restart session」。
            3. 選擇 **A100 GPU** runtime；必須解析為與 calibration 相同的
               `primary_40g`，其他 profile 會在模型下載前停止。
            4. 在 Secrets 新增 **HF_TOKEN**，並允許此 notebook 存取。
            5. 確認 token 已取得 gated TAIDE 模型權限。
            6. 不需要修改任何程式碼，直接按「全部執行」。
            7. 安裝後不要再次重啟；依賴 gate 通過後會自動繼續。遇到未核准 GPU、
               缺少權重、NaN、無法解析答案時會明確停止。

            研究用途，非醫療建議。MedQA test 只用於去重隔離，不會傳入 trainer；
            adapter 重載驗證使用 validation 題；完整訓練可使用 validation loss，
            但 test 永遠不會傳入 trainer。

            premium GPU 不改模型、資料、learning rate 或 effective batch 16；只提高
            per-device batch 並等比例降低 gradient accumulation，以提升吞吐量。
            """
        ),
        _markdown(
            """
            ## 1. 安裝鎖定依賴

            直接依賴由 requirements/colab-train.txt 產生。Colab 管理 CUDA/PyTorch
            基礎映像，Unsloth 解析相容的 xformers/Triton wheel；實際完整版本會另存
            pip-freeze.txt。
            """
        ),
        _code(_install_cell()),
        _markdown(
            """
            ### 1.1 驗證安裝結果

            這一格不匯入 Unsloth，只檢查 module 與鎖定版本；若安裝格失敗，流程會在
            下載模型前停止。通過後請勿重新啟動 runtime。
            """
        ),
        _code(_dependency_check_cell()),
        _markdown(
            """
            ## 2. 已核准的固定設定與成本閘門

            已依 A100 40GB calibration 寫入 20.53 秒／step、checkpoint 12.13 秒、
            完整 validation 434.65 秒、5.3 CU／小時與核准時 436.2 CU 餘額。
            完整訓練預估 26.49 CU，成本閘門要求至少保留 31.78 CU。請勿修改本格。
            """
        ),
        _code(settings_cell),
        _markdown("## 3. Colab Secrets 與 Drive"),
        _code(
            r"""
            def read_colab_secret(name: str, *, required: bool = False) -> str | None:
                try:
                    value = userdata.get(name)
                except Exception:
                    value = None
                if required and not value:
                    raise RuntimeError(
                        f"缺少 Colab Secret {name}。請在左側鑰匙圖示新增並允許 notebook 存取。"
                    )
                return value


            HF_TOKEN = read_colab_secret("HF_TOKEN", required=True)
            login(token=HF_TOKEN, add_to_git_credential=False)

            WANDB_API_KEY = read_colab_secret("WANDB_API_KEY")
            REPORT_TO = "none"
            if WANDB_API_KEY:
                os.environ["WANDB_API_KEY"] = WANDB_API_KEY
                os.environ.setdefault("WANDB_PROJECT", "tw-med-llm-qlora")
                REPORT_TO = "wandb"

            drive.mount("/content/drive")
            LOCAL_ROOT = Path("/content/tw-med-llm-qlora") / RUN_ID
            DRIVE_BASE = Path("/content/drive/MyDrive/tw-med-llm-qlora/phase3")
            TRAINER_OUTPUT = LOCAL_ROOT / "trainer"
            ADAPTER_DIR = LOCAL_ROOT / "adapter"
            EVIDENCE_DIR = LOCAL_ROOT / "evidence"
            for directory in (TRAINER_OUTPUT, ADAPTER_DIR, EVIDENCE_DIR):
                directory.mkdir(parents=True, exist_ok=True)
            print(f"Local run directory: {LOCAL_ROOT}")
            """
        ),
        _markdown(
            """
            ### 3.1 載入 checkpoint 完整性工具

            這一格是 repo 內已通過 CPU 測試的同一份程式：先在 Colab 本機封裝，
            複製到 Drive 後驗證 SHA-256，再用 partial rename 發布；恢復時驗證
            fingerprint、大小、hash、檔案清單與 optimizer/scheduler/RNG 狀態。
            """
        ),
        _code(_checkpoint_helpers_cell()),
        _markdown("## 4. GPU 偵測與核准路由（模型載入前）"),
        _code(
            r"""
            if not torch.cuda.is_available():
                raise RuntimeError("未偵測到 CUDA GPU；請在 Colab 變更 runtime type。")

            device_properties = torch.cuda.get_device_properties(0)
            GPU_NAME = device_properties.name
            TOTAL_VRAM_GIB = device_properties.total_memory / (1024**3)
            COMPUTE_CAPABILITY = torch.cuda.get_device_capability(0)
            BF16_SUPPORTED = bool(torch.cuda.is_bf16_supported())

            RUNTIME_PROFILE = None
            candidates = sorted(
                PROJECT_CONFIG["hardware_profiles"],
                key=lambda item: float(item["min_vram_gib"]),
                reverse=True,
            )
            for candidate in candidates:
                minimum_capability = tuple(
                    int(value) for value in candidate["min_compute_capability"]
                )
                if TOTAL_VRAM_GIB < float(candidate["min_vram_gib"]):
                    continue
                if COMPUTE_CAPABILITY < minimum_capability:
                    continue
                if bool(candidate["requires_bf16"]) and not BF16_SUPPORTED:
                    continue
                RUNTIME_PROFILE = candidate
                break
            if RUNTIME_PROFILE is None:
                raise RuntimeError(
                    "沒有核准的訓練設定："
                    f"{GPU_NAME}, {TOTAL_VRAM_GIB:.2f} GiB, "
                    f"bf16={BF16_SUPPORTED}, capability={COMPUTE_CAPABILITY}"
                )

            PROFILE_NAME = str(RUNTIME_PROFILE["model_profile"])
            PROFILE = {
                "name": PROFILE_NAME,
                **PROJECT_CONFIG["models"][PROFILE_NAME],
                "hardware_profile": RUNTIME_PROFILE["name"],
                "batch_size": int(RUNTIME_PROFILE["batch_size"]),
                "gradient_accumulation_steps": int(
                    RUNTIME_PROFILE["gradient_accumulation_steps"]
                ),
                "max_sequence_length": int(
                    RUNTIME_PROFILE["max_sequence_length"]
                ),
            }
            USE_BF16 = BF16_SUPPORTED
            USE_FP16 = not USE_BF16
            if PROFILE_NAME == "fallback" and USE_FP16:
                PRECISION_NOTE = "Unsloth Gemma 3 FP16-safe path (T4/no-BF16)"
            else:
                PRECISION_NOTE = "BF16"
            ALLOW_TF32 = bool(RUNTIME_PROFILE["allow_tf32"])
            torch.backends.cuda.matmul.allow_tf32 = ALLOW_TF32
            if ALLOW_TF32:
                torch.set_float32_matmul_precision("high")

            assert (
                int(PROFILE["batch_size"])
                * int(PROFILE["gradient_accumulation_steps"])
                == int(PROJECT_CONFIG["project"]["effective_batch_size"])
            )
            premium_profiles = {"primary_80g", "primary_40g"}
            if REQUIRE_PREMIUM_GPU and PROFILE["hardware_profile"] not in premium_profiles:
                raise RuntimeError(
                    "Phase 3 requires an A100-class premium profile; "
                    f"received {GPU_NAME} / {PROFILE['hardware_profile']}. "
                    "Disconnect and choose A100 before downloading the model."
                )
            if RUN_MODE == "full" and "A100" not in GPU_NAME.upper():
                raise RuntimeError(
                    "This approved full run requires an A100 GPU; "
                    f"received {GPU_NAME}. Disconnect and choose A100."
                )
            if (
                RUN_MODE == "full"
                and PROFILE["hardware_profile"] != CALIBRATED_HARDWARE_PROFILE
            ):
                raise RuntimeError(
                    "Full-training GPU profile differs from the reviewed calibration: "
                    f"current={PROFILE['hardware_profile']}, "
                    f"calibrated={CALIBRATED_HARDWARE_PROFILE}"
                )

            TRAINING_CONTRACT = {
                "model_id": PROFILE["model_id"],
                "model_revision": PROFILE["revision"],
                "dataset_id": PROJECT_CONFIG["data"]["medqa"]["dataset_id"],
                "dataset_revision": PROJECT_CONFIG["data"]["medqa"]["revision"],
                "seed": PROJECT_CONFIG["project"]["seed"],
                "epochs": PROJECT_CONFIG["training"]["num_train_epochs"],
                "effective_batch_size": PROJECT_CONFIG["project"]["effective_batch_size"],
                "hardware_profile": PROFILE["hardware_profile"],
                "batch_size": PROFILE["batch_size"],
                "gradient_accumulation_steps": PROFILE[
                    "gradient_accumulation_steps"
                ],
                "max_sequence_length": PROFILE["max_sequence_length"],
                "learning_rate": PROJECT_CONFIG["training"]["learning_rate"],
                "lora_rank": PROJECT_CONFIG["training"]["lora_rank"],
                "lora_alpha": PROJECT_CONFIG["training"]["lora_alpha"],
            }
            EXPERIMENT_FINGERPRINT = experiment_fingerprint(TRAINING_CONTRACT)
            DRIVE_EXPERIMENT_ROOT = DRIVE_BASE / EXPERIMENT_FINGERPRINT
            if RUN_MODE == "full":
                DRIVE_CHECKPOINT_ROOT = DRIVE_EXPERIMENT_ROOT / "checkpoints"
            else:
                DRIVE_CHECKPOINT_ROOT = (
                    DRIVE_EXPERIMENT_ROOT / "calibrations" / RUN_ID / "checkpoints"
                )
            DRIVE_ARTIFACT_ROOT = DRIVE_EXPERIMENT_ROOT / RUN_MODE / "runs"
            DRIVE_CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
            DRIVE_ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

            full_train_examples = int(
                PROJECT_CONFIG["data"]["medqa"]["expected_train_rows"]
            )
            effective_batch = int(PROJECT_CONFIG["project"]["effective_batch_size"])
            epochs = float(PROJECT_CONFIG["training"]["num_train_epochs"])
            full_steps = math.ceil(full_train_examples * epochs / effective_batch)
            full_checkpoint_events = full_steps // int(
                PROJECT_CONFIG["training"]["save_steps"]
            )
            full_eval_events = (
                full_steps // int(PROJECT_CONFIG["training"]["eval_steps"])
            ) + 1
            if RUN_MODE == "full":
                preflight_seconds = (
                    float(CALIBRATED_SECONDS_PER_STEP) * full_steps
                    + float(CALIBRATED_CHECKPOINT_SECONDS_PER_SAVE)
                    * full_checkpoint_events
                    + float(CALIBRATED_FULL_EVAL_SECONDS) * full_eval_events
                )
                preflight_hours = preflight_seconds / 3600
                preflight_compute_units = (
                    preflight_hours * float(COMPUTE_UNITS_PER_HOUR)
                )
                required_with_buffer = preflight_compute_units * 1.20
                if float(CURRENT_COMPUTE_UNITS) < required_with_buffer:
                    raise RuntimeError(
                        "Insufficient compute-unit buffer for full training: "
                        f"available={CURRENT_COMPUTE_UNITS}, "
                        f"required_with_20pct_buffer={required_with_buffer:.2f}"
                    )
            else:
                preflight_seconds = None
                preflight_hours = None
                preflight_compute_units = None
            print(
                json.dumps(
                    {
                        "gpu": GPU_NAME,
                        "vram_gib": round(TOTAL_VRAM_GIB, 3),
                        "compute_capability": list(COMPUTE_CAPABILITY),
                        "bf16_supported": BF16_SUPPORTED,
                        "model_profile": PROFILE_NAME,
                        "hardware_profile": PROFILE["hardware_profile"],
                        "model_id": PROFILE["model_id"],
                        "revision": PROFILE["revision"],
                        "precision": PRECISION_NOTE,
                        "allow_tf32": ALLOW_TF32,
                        "batch_size": PROFILE["batch_size"],
                        "gradient_accumulation_steps": PROFILE[
                            "gradient_accumulation_steps"
                        ],
                        "max_sequence_length": PROFILE["max_sequence_length"],
                        "experiment_fingerprint": EXPERIMENT_FINGERPRINT,
                        "drive_checkpoint_root": str(DRIVE_CHECKPOINT_ROOT),
                        "full_checkpoint_events": full_checkpoint_events,
                        "full_eval_events_including_final": full_eval_events,
                        "preflight_seconds": preflight_seconds,
                        "preflight_hours": preflight_hours,
                        "preflight_compute_units": preflight_compute_units,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            """
        ),
        _markdown("## 5. 下載、驗證與隔離 MedQA"),
        _code(
            r"""
            CHOICE_KEYS = ("A", "B", "C", "D")
            MEDQA_FIELDS = {
                "meta_info",
                "question",
                "answer_idx",
                "answer",
                "options",
            }


            def sha256_file(path: Path) -> str:
                digest = hashlib.sha256()
                with path.open("rb") as source_file:
                    for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
                        digest.update(chunk)
                return digest.hexdigest()


            def require_text(value: object, field: str) -> str:
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"{field} must be a non-empty string")
                if value.encode("utf-8").decode("utf-8") != value:
                    raise ValueError(f"{field} failed UTF-8 round trip")
                return value


            def stable_id(
                *,
                source: str,
                revision: str,
                split: str,
                question: str,
                choices: dict[str, str],
            ) -> str:
                payload = {
                    "source": source,
                    "revision": revision,
                    "split": split,
                    "question": question,
                    "choices": {key: choices[key] for key in CHOICE_KEYS},
                }
                canonical = json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]


            def parse_medqa_row(
                row: dict[str, object],
                *,
                split: str,
                source: str,
                revision: str,
            ) -> dict[str, object]:
                if set(row) != MEDQA_FIELDS:
                    raise ValueError(f"unexpected fields: {sorted(row)}")
                question = require_text(row["question"], "question")
                answer = require_text(row["answer_idx"], "answer_idx").strip().upper()
                answer_text = require_text(row["answer"], "answer")
                raw_options = row["options"]
                if not isinstance(raw_options, list):
                    raise ValueError("options must be a list")
                choices: dict[str, str] = {}
                for index, option in enumerate(raw_options):
                    if not isinstance(option, dict):
                        raise ValueError(f"options[{index}] must be an object")
                    key = require_text(option.get("key"), f"options[{index}].key").strip().upper()
                    value = require_text(option.get("value"), f"options[{index}].value")
                    if key in choices:
                        raise ValueError(f"duplicate option key: {key}")
                    choices[key] = value
                if tuple(sorted(choices)) != CHOICE_KEYS:
                    raise ValueError("options must contain exactly A, B, C, and D")
                if answer not in choices:
                    raise ValueError("answer_idx must be A, B, C, or D")
                if answer_text != choices[answer]:
                    raise ValueError("answer text does not match selected option")
                return {
                    "id": stable_id(
                        source=source,
                        revision=revision,
                        split=split,
                        question=question,
                        choices=choices,
                    ),
                    "source": source,
                    "revision": revision,
                    "split": split,
                    "question": question,
                    "choices": choices,
                    "answer": answer,
                }


            def normalized_question(question: str) -> str:
                return " ".join(question.split()).casefold()


            def has_ambiguous_choices(example: dict[str, object]) -> bool:
                choices = example["choices"]
                assert isinstance(choices, dict)
                values = [
                    normalized_question(str(choices[key]))
                    for key in CHOICE_KEYS
                ]
                return len(set(values)) != len(values)


            medqa = PROJECT_CONFIG["data"]["medqa"]
            source_id = str(medqa["dataset_id"])
            source_revision = str(medqa["revision"])
            config_name = str(medqa["config"])
            expected_source_hashes = medqa["source_sha256"]
            raw_examples: dict[str, list[dict[str, object]]] = {}
            source_audit: dict[str, dict[str, object]] = {}

            for split in ("train", "validation", "test"):
                filename = f"{config_name}/{split}/0000.parquet"
                parquet_path = Path(
                    hf_hub_download(
                        repo_id=source_id,
                        filename=filename,
                        repo_type="dataset",
                        revision=source_revision,
                        local_dir=LOCAL_ROOT / "data" / "raw",
                        token=HF_TOKEN,
                    )
                )
                actual_hash = sha256_file(parquet_path)
                if actual_hash != expected_source_hashes[split]:
                    raise RuntimeError(
                        f"{split} source hash changed: {actual_hash}; stop and audit."
                    )
                parquet_file = parquet.ParquetFile(parquet_path)
                if set(parquet_file.schema_arrow.names) != MEDQA_FIELDS:
                    raise RuntimeError(f"{split} schema changed; stop and audit.")

                parsed: list[dict[str, object]] = []
                invalid_reasons: Counter[str] = Counter()
                for batch in parquet_file.iter_batches(batch_size=1024):
                    for row in batch.to_pylist():
                        try:
                            parsed.append(
                                parse_medqa_row(
                                    row,
                                    split=split,
                                    source=source_id,
                                    revision=source_revision,
                                )
                            )
                        except ValueError as error:
                            invalid_reasons[str(error)] += 1
                if split == "test" and invalid_reasons:
                    raise RuntimeError(f"test has invalid rows: {dict(invalid_reasons)}")
                if len(parsed) + sum(invalid_reasons.values()) != parquet_file.metadata.num_rows:
                    raise RuntimeError(f"{split} row accounting mismatch")
                raw_examples[split] = parsed
                source_audit[split] = {
                    "raw_rows": parquet_file.metadata.num_rows,
                    "parsed_rows": len(parsed),
                    "invalid_rows": sum(invalid_reasons.values()),
                    "invalid_reasons": dict(sorted(invalid_reasons.items())),
                    "source_sha256": actual_hash,
                }

            ambiguous_counts = {
                split: sum(has_ambiguous_choices(item) for item in raw_examples[split])
                for split in ("train", "validation", "test")
            }
            if ambiguous_counts["test"]:
                raise RuntimeError("test has ambiguous choices and must remain unchanged")

            quality_filtered = {
                "test": list(raw_examples["test"]),
                "validation": [
                    item
                    for item in raw_examples["validation"]
                    if not has_ambiguous_choices(item)
                ],
                "train": [
                    item
                    for item in raw_examples["train"]
                    if not has_ambiguous_choices(item)
                ],
            }

            cleaned: dict[str, list[dict[str, object]]] = {
                "test": list(quality_filtered["test"]),
                "validation": [],
                "train": [],
            }
            seen_questions: dict[str, str] = {}
            for item in cleaned["test"]:
                seen_questions.setdefault(normalized_question(str(item["question"])), "test")
            duplicate_removals: Counter[str] = Counter()
            for split in ("validation", "train"):
                for item in quality_filtered[split]:
                    key = normalized_question(str(item["question"]))
                    winner = seen_questions.get(key)
                    if winner is not None:
                        duplicate_removals[f"{split}->{winner}"] += 1
                        continue
                    cleaned[split].append(item)
                    seen_questions[key] = split

            expected_counts = {
                "train": int(medqa["expected_train_rows"]),
                "validation": int(medqa["expected_validation_rows"]),
                "test": int(medqa["expected_test_rows"]),
            }
            actual_counts = {split: len(cleaned[split]) for split in expected_counts}
            if actual_counts != expected_counts:
                raise RuntimeError(
                    f"clean counts changed: expected {expected_counts}, got {actual_counts}"
                )
            if cleaned["test"] != raw_examples["test"]:
                raise RuntimeError("test must remain row-for-row unchanged")

            split_keys = {
                split: {
                    normalized_question(str(item["question"]))
                    for item in cleaned[split]
                }
                for split in ("train", "validation", "test")
            }
            for left, right in (
                ("train", "validation"),
                ("train", "test"),
                ("validation", "test"),
            ):
                if split_keys[left].intersection(split_keys[right]):
                    raise RuntimeError(f"cross-split overlap remains: {left}/{right}")

            seed = int(PROJECT_CONFIG["project"]["seed"])
            smoke_size = int(PROJECT_CONFIG["training"]["smoke_examples"])
            train_examples = list(cleaned["train"])
            validation_examples = list(cleaned["validation"])
            sample_indices = sorted(
                random.Random(seed).sample(range(len(train_examples)), smoke_size)
            )
            smoke_examples = [train_examples[index] for index in sample_indices]
            validation_probe = validation_examples[0]
            smoke_id_digest = hashlib.sha256(
                "\n".join(str(item["id"]) for item in smoke_examples).encode("utf-8")
            ).hexdigest()
            DATA_AUDIT = {
                "source": source_audit,
                "clean_rows": actual_counts,
                "ambiguous_rows": ambiguous_counts,
                "duplicate_removals": dict(sorted(duplicate_removals.items())),
                "smoke_rows": len(smoke_examples),
                "smoke_ids_sha256": smoke_id_digest,
                "trainer_referenced_splits": ["train", "validation"],
                "calibration_validation_rows": smoke_size,
                "validation_probe_id": validation_probe["id"],
                "test_used_for_training": False,
            }
            expected_trainer_splits = ["train", "validation"]
            if DATA_AUDIT["trainer_referenced_splits"] != expected_trainer_splits:
                raise RuntimeError("trainer split isolation failed")

            # Remove test content from live variables before trainer construction.
            del raw_examples, quality_filtered, cleaned, split_keys, seen_questions
            print(json.dumps(DATA_AUDIT, ensure_ascii=False, indent=2))
            """
        ),
        _markdown(
            """
            ## 6. 完整快取模型、載入 Gemma 3、加入 LoRA

            先把固定 revision 的模型 snapshot 完整下載到 Hugging Face cache，核對
            safetensors index 與所有權重檔，再讓 Unsloth 以離線模式讀取。若下載中斷，
            重跑本 cell 會沿用 Hub cache；不讓 Unsloth 在部分 cache 上自行切換離線重試。
            """
        ),
        _code(
            r"""
            from datasets import Dataset
            from unsloth import FastModel

            model_id = str(PROFILE["model_id"])
            model_revision = str(PROFILE["revision"])
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

            remote_weight_files = sorted(
                sibling.rfilename
                for sibling in model_info.siblings
                if sibling.rfilename.endswith(".safetensors")
            )
            remote_weight_bytes = sum(
                int(sibling.size or 0)
                for sibling in model_info.siblings
                if sibling.rfilename.endswith(".safetensors")
            )
            if not remote_weight_files or remote_weight_bytes <= 0:
                raise RuntimeError(
                    "Hugging Face model metadata has no sized safetensors weights."
                )

            free_disk_bytes = shutil.disk_usage("/content").free
            required_disk_bytes = remote_weight_bytes + 8 * 1024**3
            if free_disk_bytes < required_disk_bytes:
                raise RuntimeError(
                    "Colab local disk is too small for the pinned model snapshot: "
                    f"free={free_disk_bytes / 1024**3:.2f} GiB, "
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
            snapshot_config = json.loads(
                snapshot_config_path.read_text(encoding="utf-8")
            )
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
                "free_disk_gib_before_download": free_disk_bytes / 1024**3,
                "tokenizer_source": str(snapshot_path),
                "vlm_processor_required": snapshot_is_vlm,
                "complete": True,
            }
            print(json.dumps(MODEL_SNAPSHOT_AUDIT, ensure_ascii=False, indent=2))

            model, tokenizer = FastModel.from_pretrained(
                model_name=model_id,
                revision=model_revision,
                max_seq_length=int(PROFILE["max_sequence_length"]),
                load_in_4bit=True,
                load_in_8bit=False,
                full_finetuning=False,
                token=HF_TOKEN,
                tokenizer_name=str(snapshot_path),
                local_files_only=True,
                use_safetensors=True,
            )
            model = FastModel.get_peft_model(
                model,
                finetune_vision_layers=False,
                finetune_language_layers=True,
                finetune_attention_modules=True,
                finetune_mlp_modules=True,
                r=int(PROJECT_CONFIG["training"]["lora_rank"]),
                lora_alpha=int(PROJECT_CONFIG["training"]["lora_alpha"]),
                lora_dropout=float(PROJECT_CONFIG["training"]["lora_dropout"]),
                bias="none",
                random_state=int(PROJECT_CONFIG["project"]["seed"]),
            )

            text_tokenizer = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer


            def build_user_prompt(example: dict[str, object]) -> str:
                choices = example["choices"]
                assert isinstance(choices, dict)
                options = "\n".join(
                    f"{key}. {choices[key]}" for key in CHOICE_KEYS
                )
                return (
                    "請閱讀以下台灣醫療多選題，選出唯一正確答案。\n\n"
                    f"{example['question']}\n\n{options}\n\n"
                    "請只回答 A、B、C 或 D 中的一個字母。"
                )


            def render_training_text(example: dict[str, object]) -> str:
                messages = [
                    {"role": "user", "content": build_user_prompt(example)},
                    {"role": "assistant", "content": str(example["answer"])},
                ]
                rendered = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                bos_token = getattr(text_tokenizer, "bos_token", None) or ""
                if bos_token and rendered.startswith(bos_token):
                    rendered = rendered[len(bos_token) :]
                return rendered


            selected_train_examples = (
                train_examples if RUN_MODE == "full" else smoke_examples
            )
            selected_validation_examples = (
                validation_examples
                if RUN_MODE == "full"
                else validation_examples[: int(PROJECT_CONFIG["training"]["smoke_examples"])]
            )
            rendered_train = [
                render_training_text(item) for item in selected_train_examples
            ]
            rendered_validation = [
                render_training_text(item) for item in selected_validation_examples
            ]
            train_token_lengths = [
                len(text_tokenizer(text, add_special_tokens=True)["input_ids"])
                for text in rendered_train
            ]
            validation_token_lengths = [
                len(text_tokenizer(text, add_special_tokens=True)["input_ids"])
                for text in rendered_validation
            ]
            token_lengths = train_token_lengths + validation_token_lengths
            max_sequence_length = int(PROFILE["max_sequence_length"])
            if max(token_lengths) > max_sequence_length:
                raise RuntimeError(
                    "A rendered train/validation example exceeds max sequence length: "
                    f"{max(token_lengths)}"
                )
            bos_token_id = getattr(text_tokenizer, "bos_token_id", None)
            if bos_token_id is not None:
                first_ids = text_tokenizer(
                    rendered_train[0],
                    add_special_tokens=True,
                )["input_ids"]
                if first_ids.count(bos_token_id) != 1:
                    raise RuntimeError("Expected exactly one BOS token after rendering")

            train_dataset = Dataset.from_list(
                [{"text": text} for text in rendered_train]
            )
            eval_dataset = Dataset.from_list(
                [{"text": text} for text in rendered_validation]
            )
            expected_train_rows = (
                int(PROJECT_CONFIG["data"]["medqa"]["expected_train_rows"])
                if RUN_MODE == "full"
                else int(PROJECT_CONFIG["training"]["smoke_examples"])
            )
            if len(train_dataset) != expected_train_rows:
                raise RuntimeError(
                    f"Training dataset row mismatch: {len(train_dataset)} != {expected_train_rows}"
                )
            expected_validation_rows = (
                int(PROJECT_CONFIG["data"]["medqa"]["expected_validation_rows"])
                if RUN_MODE == "full"
                else int(PROJECT_CONFIG["training"]["smoke_examples"])
            )
            if len(eval_dataset) != expected_validation_rows:
                raise RuntimeError("Validation dataset row mismatch")
            print(
                {
                    "run_mode": RUN_MODE,
                    "train_rows": len(train_dataset),
                    "validation_rows": len(eval_dataset),
                    "train_min_tokens": min(train_token_lengths),
                    "train_max_tokens": max(train_token_lengths),
                    "validation_max_tokens": (
                        max(validation_token_lengths) if validation_token_lengths else None
                    ),
                    "model": PROFILE["model_id"],
                }
            )
            del (
                train_examples,
                validation_examples,
                smoke_examples,
                selected_train_examples,
                selected_validation_examples,
                rendered_train,
                rendered_validation,
            )
            """
        ),
        _markdown("## 7. 建立 response-only trainer、checkpoint callback 與 resume"),
        _code(
            r"""
            from transformers import TrainerCallback
            from trl import SFTConfig, SFTTrainer
            from unsloth.chat_templates import train_on_responses_only

            calibration_steps = int(PROJECT_CONFIG["training"]["smoke_steps"])
            save_steps = (
                int(PROJECT_CONFIG["training"]["save_steps"])
                if RUN_MODE == "full"
                else calibration_steps
            )
            eval_steps = (
                int(PROJECT_CONFIG["training"]["eval_steps"])
                if RUN_MODE == "full"
                else None
            )
            if calibration_steps != 10:
                raise RuntimeError("Phase 3 calibration must remain fixed at 10 steps")

            RESUME_CHECKPOINT = None
            if RUN_MODE == "full" and AUTO_RESUME_FROM_DRIVE:
                RESUME_CHECKPOINT = restore_latest_checkpoint(
                    drive_checkpoint_dir=DRIVE_CHECKPOINT_ROOT,
                    local_output_dir=TRAINER_OUTPUT,
                    fingerprint=EXPERIMENT_FINGERPRINT,
                )
            CHECKPOINT_SYNC_RECORDS = []


            class DriveCheckpointCallback(TrainerCallback):
                def __init__(self):
                    self.checkpoint_cycle_started = None

                def on_step_end(self, args, state, control, **kwargs):
                    if state.global_step > 0 and state.global_step % int(args.save_steps) == 0:
                        self.checkpoint_cycle_started = time.perf_counter()
                    return control

                def on_save(self, args, state, control, **kwargs):
                    if not getattr(state, "is_world_process_zero", True):
                        return control
                    checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
                    sync_started = time.perf_counter()
                    record = archive_checkpoint(
                        checkpoint_dir=checkpoint_dir,
                        drive_checkpoint_dir=DRIVE_CHECKPOINT_ROOT,
                        fingerprint=EXPERIMENT_FINGERPRINT,
                        keep=int(PROJECT_CONFIG["training"]["save_total_limit"]),
                    )
                    record["sync_wall_seconds"] = time.perf_counter() - sync_started
                    record["checkpoint_cycle_wall_seconds"] = (
                        time.perf_counter() - self.checkpoint_cycle_started
                        if self.checkpoint_cycle_started is not None
                        else record["sync_wall_seconds"]
                    )
                    self.checkpoint_cycle_started = None
                    CHECKPOINT_SYNC_RECORDS.append(record)
                    print(json.dumps({"checkpoint_synced": record}, ensure_ascii=False))
                    return control

            trainer = SFTTrainer(
                model=model,
                # This dataset is text-only. The inner tokenizer keeps TRL on its
                # language-model collator even when the base config has a vision tower.
                processing_class=text_tokenizer,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                callbacks=[DriveCheckpointCallback()],
                args=SFTConfig(
                    output_dir=str(TRAINER_OUTPUT),
                    dataset_text_field="text",
                    max_length=int(PROFILE["max_sequence_length"]),
                    per_device_train_batch_size=int(PROFILE["batch_size"]),
                    gradient_accumulation_steps=int(
                        PROFILE["gradient_accumulation_steps"]
                    ),
                    num_train_epochs=float(
                        PROJECT_CONFIG["training"]["num_train_epochs"]
                    ),
                    max_steps=calibration_steps if RUN_MODE == "calibration" else -1,
                    learning_rate=float(PROJECT_CONFIG["training"]["learning_rate"]),
                    warmup_ratio=float(PROJECT_CONFIG["training"]["warmup_ratio"]),
                    lr_scheduler_type=str(
                        PROJECT_CONFIG["training"]["lr_scheduler_type"]
                    ),
                    optim=str(PROJECT_CONFIG["training"]["optimizer"]),
                    bf16=USE_BF16,
                    fp16=USE_FP16,
                    logging_steps=1 if RUN_MODE == "calibration" else 10,
                    logging_first_step=True,
                    eval_strategy="steps" if RUN_MODE == "full" else "no",
                    eval_steps=eval_steps,
                    per_device_eval_batch_size=int(PROFILE["batch_size"]),
                    eval_accumulation_steps=1,
                    prediction_loss_only=True,
                    save_strategy="steps",
                    save_steps=save_steps,
                    save_total_limit=int(
                        PROJECT_CONFIG["training"]["save_total_limit"]
                    ),
                    save_only_model=False,
                    restore_callback_states_from_checkpoint=False,
                    report_to=REPORT_TO,
                    push_to_hub=False,
                    packing=False,
                    dataset_num_proc=1,
                    gradient_checkpointing=True,
                    seed=int(PROJECT_CONFIG["project"]["seed"]),
                    data_seed=int(PROJECT_CONFIG["project"]["seed"]),
                ),
            )
            trainer_collator = trainer.data_collator
            collator_objects = [
                getattr(trainer_collator, attribute, None)
                for attribute in ("processor", "tokenizer")
            ]
            is_vision_collator = (
                "Vision" in type(trainer_collator).__name__
                or any(
                    item is not None and hasattr(item, "image_processor")
                    for item in collator_objects
                )
            )
            if is_vision_collator:
                raise RuntimeError(
                    "Text-only smoke test unexpectedly selected a vision collator: "
                    f"{type(trainer_collator).__name__}"
                )
            trainer = train_on_responses_only(
                trainer,
                tokenizer=text_tokenizer,
                last_response_only=True,
                num_proc=1,
            )
            if len(trainer.train_dataset) != expected_train_rows:
                raise RuntimeError(
                    "Response masking dropped rows; inspect truncation/template before training"
                )
            if len(trainer.eval_dataset) != expected_validation_rows:
                raise RuntimeError("Response masking changed validation rows")
            first_labels = trainer.train_dataset[0]["labels"]
            if hasattr(first_labels, "tolist"):
                first_labels = first_labels.tolist()
            masked_tokens = sum(label == -100 for label in first_labels)
            response_tokens = sum(label != -100 for label in first_labels)
            if not masked_tokens or not response_tokens:
                raise RuntimeError("Response-only label mask is invalid")
            first_validation_labels = trainer.eval_dataset[0]["labels"]
            if hasattr(first_validation_labels, "tolist"):
                first_validation_labels = first_validation_labels.tolist()
            validation_masked_tokens = sum(
                label == -100 for label in first_validation_labels
            )
            validation_response_tokens = sum(
                label != -100 for label in first_validation_labels
            )
            if not validation_masked_tokens or not validation_response_tokens:
                raise RuntimeError("Validation response-only label mask is invalid")
            MASKING_AUDIT = {
                "processing_class": type(text_tokenizer).__name__,
                "data_collator": type(trainer_collator).__name__,
                "vision_collator": False,
                "masked_prompt_tokens": masked_tokens,
                "response_loss_tokens": response_tokens,
                "validation_masked_prompt_tokens": validation_masked_tokens,
                "validation_response_loss_tokens": validation_response_tokens,
                "last_response_only": True,
                "test_passed_to_trainer": False,
                "train_rows": len(trainer.train_dataset),
                "validation_rows": len(trainer.eval_dataset),
                "resume_checkpoint": (
                    str(RESUME_CHECKPOINT) if RESUME_CHECKPOINT is not None else None
                ),
            }
            print(json.dumps(MASKING_AUDIT, ensure_ascii=False, indent=2))
            """
        ),
        _markdown("## 8. 執行 calibration／完整訓練並驗證 checkpoint 恢復"),
        _code(
            r"""
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            train_started = time.perf_counter()
            resume_argument = (
                str(RESUME_CHECKPOINT) if RESUME_CHECKPOINT is not None else None
            )
            train_result = trainer.train(resume_from_checkpoint=resume_argument)
            TRAINING_WALL_SECONDS = time.perf_counter() - train_started

            TRAIN_LOG_HISTORY = list(trainer.state.log_history)
            logged_losses = [
                float(item["loss"])
                for item in TRAIN_LOG_HISTORY
                if "loss" in item
            ]
            training_loss = float(train_result.training_loss)
            if not logged_losses or not all(math.isfinite(loss) for loss in logged_losses):
                raise RuntimeError(f"Non-finite or missing step loss: {logged_losses}")
            if not math.isfinite(training_loss):
                raise RuntimeError(f"Non-finite training loss: {training_loss}")
            eval_losses = [
                float(item["eval_loss"])
                for item in TRAIN_LOG_HISTORY
                if "eval_loss" in item
            ]
            if not all(math.isfinite(loss) for loss in eval_losses):
                raise RuntimeError(f"Non-finite validation loss: {eval_losses}")

            final_eval_started = time.perf_counter()
            POST_TRAIN_EVAL_ATTEMPT = {
                key: float(value) if isinstance(value, (int, float)) else value
                for key, value in trainer.evaluate().items()
            }
            FINAL_EVAL_WALL_SECONDS = time.perf_counter() - final_eval_started
            TRAIN_LOG_HISTORY = list(trainer.state.log_history)
            finite_eval_records = [
                dict(item)
                for item in TRAIN_LOG_HISTORY
                if "eval_loss" in item
                and math.isfinite(float(item["eval_loss"]))
            ]
            if not finite_eval_records:
                raise RuntimeError("No finite validation result is available")

            completed_global_step = int(trainer.state.global_step)
            final_eval_loss = float(POST_TRAIN_EVAL_ATTEMPT["eval_loss"])
            SELECTED_ADAPTER_CHECKPOINT = f"checkpoint-{completed_global_step}"
            EVALUATION_SELECTION = {
                "status": "accepted_post_train_evaluation",
                "completed_global_step": completed_global_step,
                "selected_adapter_checkpoint": SELECTED_ADAPTER_CHECKPOINT,
                "selected_adapter_global_step": completed_global_step,
                "selected_validation_step": completed_global_step,
                "selection_reason": "post-training validation loss is finite",
            }
            if math.isfinite(final_eval_loss):
                FINAL_EVAL_METRICS = {
                    **POST_TRAIN_EVAL_ATTEMPT,
                    "source": "post_train_full_validation",
                    "step": completed_global_step,
                    "selected_checkpoint": SELECTED_ADAPTER_CHECKPOINT,
                }
            else:
                if RUN_MODE != "full":
                    raise RuntimeError("Calibration final validation loss is non-finite")
                bad_lora_parameters = [
                    name
                    for name, parameter in model.named_parameters()
                    if "lora_" in name
                    and not torch.isfinite(parameter.detach()).all().item()
                ]
                if bad_lora_parameters:
                    raise RuntimeError(
                        "Post-training validation is non-finite and LoRA weights are invalid: "
                        f"{bad_lora_parameters[:5]}"
                    )
                selected_eval = finite_eval_records[-1]
                selected_step = int(selected_eval["step"])
                selected_checkpoint = TRAINER_OUTPUT / f"checkpoint-{selected_step}"
                if not selected_checkpoint.is_dir():
                    raise RuntimeError(
                        "The last finite validation checkpoint is unavailable: "
                        f"{selected_checkpoint}"
                    )
                selected_records = [
                    record
                    for record in CHECKPOINT_SYNC_RECORDS
                    if int(record["global_step"]) == selected_step
                ]
                if len(selected_records) != 1:
                    raise RuntimeError("Validated checkpoint sync record is unavailable")
                selected_record = selected_records[0]
                selected_archive = DRIVE_CHECKPOINT_ROOT / selected_record["archive"]
                if (
                    not selected_archive.is_file()
                    or selected_archive.stat().st_size
                    != int(selected_record["archive_bytes"])
                    or sha256_file(selected_archive)
                    != selected_record["archive_sha256"]
                ):
                    raise RuntimeError("Validated Drive checkpoint failed integrity checks")

                from peft import get_peft_model_state_dict, set_peft_model_state_dict
                from safetensors.torch import load_file

                source_adapter_state = load_file(
                    str(selected_checkpoint / "adapter_model.safetensors"),
                    device="cpu",
                )
                set_peft_model_state_dict(
                    model,
                    source_adapter_state,
                    adapter_name="default",
                )
                active_adapter_state = get_peft_model_state_dict(
                    model,
                    adapter_name="default",
                )
                if set(active_adapter_state) != set(source_adapter_state):
                    raise RuntimeError("Validated adapter key set does not match")
                mismatched_adapter_keys = []
                adapter_dtype_conversions = set()
                for key, source_value in source_adapter_state.items():
                    active_value = active_adapter_state[key].detach().cpu()
                    expected_value = source_value.to(dtype=active_value.dtype)
                    adapter_dtype_conversions.add(
                        f"{source_value.dtype}->{active_value.dtype}"
                    )
                    if not torch.equal(active_value, expected_value):
                        mismatched_adapter_keys.append(key)
                if mismatched_adapter_keys:
                    raise RuntimeError(
                        "Validated adapter reload mismatch: "
                        f"{mismatched_adapter_keys[:5]}"
                    )

                SELECTED_ADAPTER_CHECKPOINT = selected_checkpoint.name
                FINAL_EVAL_METRICS = {
                    **selected_eval,
                    "source": "scheduled_full_validation",
                    "selected_checkpoint": SELECTED_ADAPTER_CHECKPOINT,
                    "selected_adapter_step": selected_step,
                }
                EVALUATION_SELECTION = {
                    "status": "recovered_from_non_finite_post_train_evaluation",
                    "completed_global_step": completed_global_step,
                    "selected_adapter_checkpoint": SELECTED_ADAPTER_CHECKPOINT,
                    "selected_adapter_global_step": selected_step,
                    "selected_validation_step": selected_step,
                    "selection_reason": (
                        "post-training validation returned non-finite loss; "
                        "selected the last checkpoint with a finite full validation"
                    ),
                    "rejected_post_train_eval_loss": "NaN",
                    "rejected_adapter_parameter_audit": "all_finite",
                    "adapter_dtype_conversions": sorted(adapter_dtype_conversions),
                }
                TRAIN_LOG_HISTORY = [
                    dict(item)
                    for item in TRAIN_LOG_HISTORY
                    if not (
                        "eval_loss" in item
                        and not math.isfinite(float(item["eval_loss"]))
                    )
                ]

            eval_losses = [
                float(item["eval_loss"])
                for item in TRAIN_LOG_HISTORY
                if "eval_loss" in item
            ]
            if not eval_losses or not all(math.isfinite(loss) for loss in eval_losses):
                raise RuntimeError(f"Accepted validation losses are invalid: {eval_losses}")

            RECOVERY_TEST_OUTPUT = LOCAL_ROOT / "checkpoint-recovery-test"
            restored_checkpoint = restore_latest_checkpoint(
                drive_checkpoint_dir=DRIVE_CHECKPOINT_ROOT,
                local_output_dir=RECOVERY_TEST_OUTPUT,
                fingerprint=EXPERIMENT_FINGERPRINT,
            )
            if restored_checkpoint is None:
                raise RuntimeError("Checkpoint callback produced no recoverable Drive archive")
            restored_state = json.loads(
                (restored_checkpoint / "trainer_state.json").read_text(encoding="utf-8")
            )
            restored_step = int(restored_state["global_step"])
            if restored_step != int(restored_checkpoint.name.rsplit("-", 1)[1]):
                raise RuntimeError("Restored trainer_state global_step mismatch")
            CHECKPOINT_AUDIT = {
                "drive_root": str(DRIVE_CHECKPOINT_ROOT),
                "resume_requested": AUTO_RESUME_FROM_DRIVE,
                "resumed_from": resume_argument,
                "archives_written_this_session": CHECKPOINT_SYNC_RECORDS,
                "restore_test_passed": True,
                "restored_checkpoint": restored_checkpoint.name,
                "restored_global_step": restored_step,
                "selected_adapter_checkpoint": SELECTED_ADAPTER_CHECKPOINT,
                "selected_adapter_global_step": int(
                    EVALUATION_SELECTION["selected_adapter_global_step"]
                ),
                "evaluation_selection": EVALUATION_SELECTION,
                "retention_limit": int(
                    PROJECT_CONFIG["training"]["save_total_limit"]
                ),
            }
            shutil.rmtree(RECOVERY_TEST_OUTPUT)

            PEAK_ALLOCATED_GIB = torch.cuda.max_memory_allocated() / (1024**3)
            PEAK_RESERVED_GIB = torch.cuda.max_memory_reserved() / (1024**3)
            completed_steps_this_session = int(trainer.state.global_step) - (
                int(RESUME_CHECKPOINT.name.rsplit("-", 1)[1])
                if RESUME_CHECKPOINT is not None
                else 0
            )
            CHECKPOINT_SYNC_WALL_SECONDS = sum(
                float(record["sync_wall_seconds"])
                for record in CHECKPOINT_SYNC_RECORDS
            )
            CHECKPOINT_CYCLE_WALL_SECONDS = sum(
                float(record["checkpoint_cycle_wall_seconds"])
                for record in CHECKPOINT_SYNC_RECORDS
            )
            training_loop_without_checkpoint_seconds = max(
                TRAINING_WALL_SECONDS - CHECKPOINT_CYCLE_WALL_SECONDS,
                0.0,
            )
            TRAINING_METRICS = {
                "mode": RUN_MODE,
                "global_step": int(trainer.state.global_step),
                "completed_steps_this_session": completed_steps_this_session,
                "selected_adapter_step": int(
                    EVALUATION_SELECTION["selected_adapter_global_step"]
                ),
                "wall_seconds": TRAINING_WALL_SECONDS,
                "seconds_per_step_this_session": (
                    TRAINING_WALL_SECONDS / completed_steps_this_session
                    if completed_steps_this_session > 0
                    else None
                ),
                "training_loop_without_checkpoint_seconds": (
                    training_loop_without_checkpoint_seconds
                ),
                "seconds_per_step_excluding_checkpoint": (
                    training_loop_without_checkpoint_seconds
                    / completed_steps_this_session
                    if completed_steps_this_session > 0
                    else None
                ),
                "checkpoint_sync_wall_seconds": CHECKPOINT_SYNC_WALL_SECONDS,
                "checkpoint_cycle_wall_seconds": CHECKPOINT_CYCLE_WALL_SECONDS,
                "checkpoint_seconds_per_save": (
                    CHECKPOINT_CYCLE_WALL_SECONDS / len(CHECKPOINT_SYNC_RECORDS)
                    if CHECKPOINT_SYNC_RECORDS
                    else None
                ),
                "final_eval_wall_seconds": FINAL_EVAL_WALL_SECONDS,
                "final_eval_rows": len(trainer.eval_dataset),
                "training_loss": training_loss,
                "logged_losses": logged_losses,
                "logged_eval_losses": eval_losses,
                "final_eval_metrics": FINAL_EVAL_METRICS,
                "post_train_eval_attempt": {
                    **POST_TRAIN_EVAL_ATTEMPT,
                    "eval_loss": (
                        float(POST_TRAIN_EVAL_ATTEMPT["eval_loss"])
                        if math.isfinite(float(POST_TRAIN_EVAL_ATTEMPT["eval_loss"]))
                        else "NaN"
                    ),
                },
                "evaluation_selection": EVALUATION_SELECTION,
                "peak_allocated_gib": PEAK_ALLOCATED_GIB,
                "peak_reserved_gib": PEAK_RESERVED_GIB,
                "oom": False,
                "all_losses_finite": True,
            }
            print(json.dumps(TRAINING_METRICS, ensure_ascii=False, indent=2))
            print(json.dumps(CHECKPOINT_AUDIT, ensure_ascii=False, indent=2))
            """
        ),
        _markdown("## 9. 保存 adapter、trainer state、曲線、環境與模型卡草稿"),
        _code(
            r"""
            import matplotlib.pyplot as plt

            ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(ADAPTER_DIR))
            tokenizer.save_pretrained(str(ADAPTER_DIR))
            trainer.save_state()

            adapter_config_path = ADAPTER_DIR / "adapter_config.json"
            if not adapter_config_path.exists():
                raise RuntimeError("adapter_config.json was not saved")
            adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
            if adapter_config.get("base_model_name_or_path") != PROFILE["model_id"]:
                raise RuntimeError(
                    "Saved adapter base model mismatch: "
                    f"{adapter_config.get('base_model_name_or_path')}"
                )

            log_keys = sorted({key for row in TRAIN_LOG_HISTORY for key in row})
            with (EVIDENCE_DIR / "trainer_log.csv").open(
                "w",
                encoding="utf-8",
                newline="",
            ) as log_file:
                writer = csv.DictWriter(log_file, fieldnames=log_keys)
                writer.writeheader()
                writer.writerows(TRAIN_LOG_HISTORY)
            (EVIDENCE_DIR / "trainer_log.json").write_text(
                json.dumps(TRAIN_LOG_HISTORY, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            loss_rows = [
                (int(row["step"]), float(row["loss"]))
                for row in TRAIN_LOG_HISTORY
                if "step" in row and "loss" in row
            ]
            learning_rate_rows = [
                (int(row["step"]), float(row["learning_rate"]))
                for row in TRAIN_LOG_HISTORY
                if "step" in row and "learning_rate" in row
            ]
            eval_loss_rows = [
                (int(row["step"]), float(row["eval_loss"]))
                for row in TRAIN_LOG_HISTORY
                if "step" in row and "eval_loss" in row
            ]
            figure, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
            axes[0].plot(
                [step for step, _ in loss_rows],
                [loss for _, loss in loss_rows],
                marker="o",
                label="train loss",
            )
            if eval_loss_rows:
                axes[0].plot(
                    [step for step, _ in eval_loss_rows],
                    [loss for _, loss in eval_loss_rows],
                    marker="s",
                    label="validation loss",
                )
            axes[0].set_ylabel("Loss")
            axes[0].set_title(f"Phase 3 {RUN_MODE} loss")
            axes[0].legend()
            axes[0].grid(alpha=0.25)
            axes[1].plot(
                [step for step, _ in learning_rate_rows],
                [rate for _, rate in learning_rate_rows],
                color="tab:orange",
            )
            axes[1].set_xlabel("Step")
            axes[1].set_ylabel("Learning rate")
            axes[1].grid(alpha=0.25)
            figure.tight_layout()
            figure.savefig(EVIDENCE_DIR / "training_curves.png", dpi=160)
            plt.close(figure)

            freeze_result = subprocess.run(
                [sys.executable, "-m", "pip", "freeze"],
                check=True,
                capture_output=True,
                text=True,
            )
            (EVIDENCE_DIR / "pip-freeze.txt").write_text(
                freeze_result.stdout,
                encoding="utf-8",
            )
            model_card_draft = dedent(f'''---
            base_model: {PROFILE['model_id']}
            library_name: peft
            tags:
              - qlora
              - traditional-chinese
              - medical-mcq
            ---

            # tw-med-llm-qlora adapter（草稿）

            本 adapter 僅供研究與教育用途，不構成醫療建議、診斷或治療依據。

            - Base model：`{PROFILE['model_id']}`，revision `{PROFILE['revision']}`
            - Dataset：`{PROJECT_CONFIG['data']['medqa']['dataset_id']}`，revision
              `{PROJECT_CONFIG['data']['medqa']['revision']}`
            - Training mode：`{RUN_MODE}`
            - Selected adapter checkpoint：`{SELECTED_ADAPTER_CHECKPOINT}`
            - Seed：`{PROJECT_CONFIG['project']['seed']}`
            - Method：4-bit QLoRA，rank/alpha
              {PROJECT_CONFIG['training']['lora_rank']}/{PROJECT_CONFIG['training']['lora_alpha']}

            Adapter 的使用與再發布必須遵守基底模型條款。本 repo 的 MIT 授權只適用於
            程式碼，不代表模型權重採 MIT。資料卡目前標示的授權資訊與原始資料 repo
            不完全一致；本專案不重新散布題目。正式評估、限制與引用將於 Phase 4/5
            補齊後才發布。
            ''').strip() + "\n"
            (EVIDENCE_DIR / "MODEL_CARD_DRAFT.md").write_text(
                model_card_draft,
                encoding="utf-8",
            )
            print(f"Saved adapter: {ADAPTER_DIR}")
            """
        ),
        _markdown("## 10. 釋放模型、重載 adapter、validation 嚴格解析"),
        _code(
            r"""
            import re

            from peft import PeftModel

            for variable_name in (
                "trainer",
                "model",
                "tokenizer",
                "text_tokenizer",
                "train_dataset",
                "eval_dataset",
                "reloaded_base_model",
                "reloaded_model",
                "reloaded_tokenizer",
            ):
                globals().pop(variable_name, None)
            gc.collect()
            torch.cuda.empty_cache()

            # Reload the exact pinned base first. Unsloth's automatic PEFT path does
            # not propagate the base revision while offline, so it may look for a
            # non-cached `main` even though the pinned snapshot is complete.
            reloaded_base_model, reloaded_tokenizer = FastModel.from_pretrained(
                model_name=str(snapshot_path),
                max_seq_length=int(PROFILE["max_sequence_length"]),
                load_in_4bit=True,
                load_in_8bit=False,
                full_finetuning=False,
                token=HF_TOKEN,
                tokenizer_name=str(snapshot_path),
                local_files_only=True,
                use_safetensors=True,
            )
            reloaded_model = PeftModel.from_pretrained(
                reloaded_base_model,
                str(ADAPTER_DIR),
                adapter_name="default",
                is_trainable=False,
                local_files_only=True,
            )
            if "default" not in reloaded_model.peft_config:
                raise RuntimeError("Reloaded PEFT model has no default adapter")
            reloaded_peft_config = reloaded_model.peft_config["default"]
            if reloaded_peft_config.base_model_name_or_path != PROFILE["model_id"]:
                raise RuntimeError(
                    "Reloaded adapter publication base mismatch: "
                    f"{reloaded_peft_config.base_model_name_or_path}"
                )
            adapter_parameter_count = sum(
                parameter.numel()
                for name, parameter in reloaded_model.named_parameters()
                if "lora_" in name
            )
            trainable_adapter_parameter_count = sum(
                parameter.numel()
                for name, parameter in reloaded_model.named_parameters()
                if "lora_" in name and parameter.requires_grad
            )
            if adapter_parameter_count <= 0 or trainable_adapter_parameter_count != 0:
                raise RuntimeError(
                    "Reloaded adapter parameter audit failed: "
                    f"adapter={adapter_parameter_count}, "
                    f"trainable={trainable_adapter_parameter_count}"
                )
            reloaded_model.eval()
            reload_text_tokenizer = (
                reloaded_tokenizer.tokenizer
                if hasattr(reloaded_tokenizer, "tokenizer")
                else reloaded_tokenizer
            )
            if not getattr(reload_text_tokenizer, "chat_template", None):
                raise RuntimeError("Reloaded text tokenizer has no chat template")
            generation_messages = [
                {
                    "role": "user",
                    "content": build_user_prompt(validation_probe),
                }
            ]
            generation_inputs = reload_text_tokenizer.apply_chat_template(
                generation_messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
            ).to("cuda")
            prompt_length = generation_inputs["input_ids"].shape[-1]
            with torch.inference_mode():
                generated = reloaded_model.generate(
                    **generation_inputs,
                    do_sample=False,
                    max_new_tokens=4,
                    use_cache=True,
                )
            new_tokens = generated[0, prompt_length:]
            generated_text = reload_text_tokenizer.decode(
                new_tokens,
                skip_special_tokens=True,
            )

            strict_match = re.fullmatch(r"\s*([A-D])\s*[。.]?\s*", generated_text)
            parsed_answer = strict_match.group(1) if strict_match else None
            if parsed_answer is None:
                raise RuntimeError(
                    f"Reloaded adapter output is not strict A-D: {generated_text!r}"
                )
            RELOAD_CHECK = {
                "adapter_reloaded": True,
                "base_source": "pinned_local_snapshot",
                "published_base_model_id": reloaded_peft_config.base_model_name_or_path,
                "adapter_parameters": adapter_parameter_count,
                "trainable_adapter_parameters": trainable_adapter_parameter_count,
                "generation_tokenizer": type(reload_text_tokenizer).__name__,
                "probe_split": "validation",
                "probe_id": validation_probe["id"],
                "gold": validation_probe["answer"],
                "prediction": parsed_answer,
                "raw_output_sha256": hashlib.sha256(
                    generated_text.encode("utf-8")
                ).hexdigest(),
                "strict_parse": True,
            }
            print(json.dumps(RELOAD_CHECK, ensure_ascii=False, indent=2))
            """
        ),
        _markdown("## 11. 成本預估、manifest 與 Drive 原子封裝"),
        _code(
            r"""
            def optional_positive(value: float | None, name: str) -> float | None:
                if value is not None and value <= 0:
                    raise ValueError(f"{name} must be positive when provided")
                return value


            compute_units_per_hour = optional_positive(
                COMPUTE_UNITS_PER_HOUR,
                "COMPUTE_UNITS_PER_HOUR",
            )
            price_per_compute_unit = optional_positive(
                PRICE_PER_COMPUTE_UNIT,
                "PRICE_PER_COMPUTE_UNIT",
            )
            if price_per_compute_unit is not None and compute_units_per_hour is None:
                raise ValueError(
                    "PRICE_PER_COMPUTE_UNIT requires COMPUTE_UNITS_PER_HOUR"
                )

            current_compute_units = (
                float(CURRENT_COMPUTE_UNITS)
                if CURRENT_COMPUTE_UNITS is not None
                else None
            )
            calibrated_seconds_per_step = (
                float(TRAINING_METRICS["seconds_per_step_excluding_checkpoint"])
                if RUN_MODE == "calibration"
                else float(CALIBRATED_SECONDS_PER_STEP)
            )
            calibrated_checkpoint_seconds = (
                float(TRAINING_METRICS["checkpoint_seconds_per_save"])
                if RUN_MODE == "calibration"
                else float(CALIBRATED_CHECKPOINT_SECONDS_PER_SAVE)
            )
            calibrated_full_eval_seconds = (
                float(TRAINING_METRICS["final_eval_wall_seconds"])
                / int(TRAINING_METRICS["final_eval_rows"])
                * int(PROJECT_CONFIG["data"]["medqa"]["expected_validation_rows"])
                if RUN_MODE == "calibration"
                else float(CALIBRATED_FULL_EVAL_SECONDS)
            )
            projected_seconds = (
                calibrated_seconds_per_step * full_steps
                + calibrated_checkpoint_seconds * full_checkpoint_events
                + calibrated_full_eval_seconds * full_eval_events
            )
            projected_hours = projected_seconds / 3600
            projected_compute_units = projected_hours * compute_units_per_hour
            projected_cost = (
                projected_compute_units * price_per_compute_unit
                if price_per_compute_unit is not None
                else None
            )
            actual_session_hours = (
                TRAINING_WALL_SECONDS + FINAL_EVAL_WALL_SECONDS
            ) / 3600
            actual_session_compute_units = (
                actual_session_hours * compute_units_per_hour
            )
            COST_ESTIMATE = {
                "basis": "measured Phase 3 A100 calibration",
                "full_train_examples": full_train_examples,
                "effective_batch_size": effective_batch,
                "epochs": epochs,
                "full_steps": full_steps,
                "calibrated_seconds_per_step": calibrated_seconds_per_step,
                "calibrated_checkpoint_seconds_per_save": (
                    calibrated_checkpoint_seconds
                ),
                "calibrated_full_eval_seconds": calibrated_full_eval_seconds,
                "checkpoint_events": full_checkpoint_events,
                "eval_events_including_final": full_eval_events,
                "projected_seconds": projected_seconds,
                "projected_hours": projected_hours,
                "compute_units_per_hour_user_input": compute_units_per_hour,
                "projected_compute_units": projected_compute_units,
                "projected_compute_units_with_20pct_buffer": (
                    projected_compute_units * 1.20
                ),
                "current_compute_units_user_input": current_compute_units,
                "projected_remaining_compute_units": (
                    current_compute_units - projected_compute_units
                    if current_compute_units is not None
                    else None
                ),
                "price_per_compute_unit_user_input": price_per_compute_unit,
                "estimated_cost": projected_cost,
                "currency_user_input": CURRENCY_LABEL,
                "actual_session_hours": actual_session_hours,
                "actual_session_compute_units": actual_session_compute_units,
            }

            package_names = [
                "unsloth",
                "unsloth-zoo",
                "transformers",
                "trl",
                "datasets",
                "peft",
                "bitsandbytes",
                "accelerate",
                "huggingface-hub",
                "xformers",
                "triton",
            ]
            package_versions = {}
            for package_name in package_names:
                try:
                    package_versions[package_name] = importlib.metadata.version(package_name)
                except importlib.metadata.PackageNotFoundError:
                    package_versions[package_name] = None
            package_versions["torch"] = torch.__version__

            RUN_MANIFEST = {
                "schema_version": 1,
                "phase": 3,
                "run_id": RUN_ID,
                "created_at_utc": datetime.now(UTC).isoformat(),
                "run_mode": RUN_MODE,
                "full_training_enabled": RUN_MODE == "full",
                "full_training_approval_verified": (
                    RUN_MODE != "full"
                    or FULL_TRAINING_APPROVAL == REQUIRED_FULL_TRAINING_APPROVAL
                ),
                "full_training_approval": {
                    "approved_at": FULL_TRAINING_APPROVED_AT,
                    "approved_buffered_compute_units": (
                        APPROVED_BUFFERED_COMPUTE_UNITS
                    ),
                    "compute_units_per_hour_at_approval": (
                        COMPUTE_UNITS_PER_HOUR
                    ),
                    "compute_units_balance_at_approval": CURRENT_COMPUTE_UNITS,
                },
                "experiment_fingerprint": EXPERIMENT_FINGERPRINT,
                "training_contract": TRAINING_CONTRACT,
                "gpu": {
                    "name": GPU_NAME,
                    "total_vram_gib": TOTAL_VRAM_GIB,
                    "compute_capability": list(COMPUTE_CAPABILITY),
                    "bf16_supported": BF16_SUPPORTED,
                    "precision": PRECISION_NOTE,
                    "allow_tf32": ALLOW_TF32,
                },
                "model": {
                    "model_profile": PROFILE_NAME,
                    "hardware_profile": PROFILE["hardware_profile"],
                    "model_id": PROFILE["model_id"],
                    "revision": PROFILE["revision"],
                    "load_in_4bit": True,
                    "adapter_base_verified": True,
                    "snapshot_audit": MODEL_SNAPSHOT_AUDIT,
                },
                "training": {
                    "batch_size": PROFILE["batch_size"],
                    "gradient_accumulation_steps": PROFILE[
                        "gradient_accumulation_steps"
                    ],
                    "max_sequence_length": PROFILE["max_sequence_length"],
                    "lora_rank": PROJECT_CONFIG["training"]["lora_rank"],
                    "lora_alpha": PROJECT_CONFIG["training"]["lora_alpha"],
                    "learning_rate": PROJECT_CONFIG["training"]["learning_rate"],
                    "seed": PROJECT_CONFIG["project"]["seed"],
                    "response_only_loss": True,
                    "masking_audit": MASKING_AUDIT,
                    "metrics": TRAINING_METRICS,
                },
                "data": {
                    "dataset_id": PROJECT_CONFIG["data"]["medqa"]["dataset_id"],
                    "revision": PROJECT_CONFIG["data"]["medqa"]["revision"],
                    "audit": DATA_AUDIT,
                },
                "reload_check": RELOAD_CHECK,
                "checkpoint_audit": CHECKPOINT_AUDIT,
                "cost_estimate": COST_ESTIMATE,
                "packages": package_versions,
                "outputs": {
                    "adapter": str(ADAPTER_DIR),
                    "trainer": str(TRAINER_OUTPUT),
                    "evidence": str(EVIDENCE_DIR),
                    "drive_checkpoints": str(DRIVE_CHECKPOINT_ROOT),
                },
            }
            manifest_path = EVIDENCE_DIR / "run_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    RUN_MANIFEST,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

            delivery_root = LOCAL_ROOT.parent / f"{RUN_ID}-delivery"
            if delivery_root.exists():
                shutil.rmtree(delivery_root)
            shutil.copytree(ADAPTER_DIR, delivery_root / "adapter")
            shutil.copytree(EVIDENCE_DIR, delivery_root / "evidence")
            trainer_state_delivery = delivery_root / "trainer-state"
            trainer_state_delivery.mkdir(parents=True)
            for trainer_file in TRAINER_OUTPUT.iterdir():
                if trainer_file.is_file():
                    shutil.copy2(trainer_file, trainer_state_delivery / trainer_file.name)
            if not (trainer_state_delivery / "trainer_state.json").is_file():
                raise RuntimeError("Final delivery is missing trainer_state.json")

            archive_base = LOCAL_ROOT.parent / f"{RUN_ID}-phase3-{RUN_MODE}"
            archive_path = Path(
                shutil.make_archive(
                    str(archive_base),
                    "zip",
                    root_dir=delivery_root,
                )
            )
            shutil.rmtree(delivery_root)
            DRIVE_ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
            drive_archive = DRIVE_ARTIFACT_ROOT / archive_path.name
            partial_archive = drive_archive.with_suffix(".zip.partial")
            shutil.copy2(archive_path, partial_archive)
            partial_archive.replace(drive_archive)
            drive_manifest = DRIVE_ARTIFACT_ROOT / f"{RUN_ID}-run-manifest.json"
            partial_manifest = drive_manifest.with_suffix(".json.partial")
            shutil.copy2(manifest_path, partial_manifest)
            partial_manifest.replace(drive_manifest)
            drive_evidence = {}
            for evidence_name in (
                "trainer_log.csv",
                "trainer_log.json",
                "training_curves.png",
                "MODEL_CARD_DRAFT.md",
                "pip-freeze.txt",
            ):
                source_evidence = EVIDENCE_DIR / evidence_name
                if not source_evidence.is_file() or source_evidence.stat().st_size <= 0:
                    raise RuntimeError(f"Missing final evidence file: {evidence_name}")
                drive_evidence_path = (
                    DRIVE_ARTIFACT_ROOT / f"{RUN_ID}-{evidence_name}"
                )
                partial_evidence = drive_evidence_path.with_suffix(
                    drive_evidence_path.suffix + ".partial"
                )
                shutil.copy2(source_evidence, partial_evidence)
                partial_evidence.replace(drive_evidence_path)
                drive_evidence[evidence_name] = {
                    "path": str(drive_evidence_path),
                    "sha256": sha256_file(drive_evidence_path),
                    "bytes": drive_evidence_path.stat().st_size,
                }
            archive_sha256 = sha256_file(drive_archive)
            receipt = {
                "phase": 3,
                "run_mode": RUN_MODE,
                "experiment_fingerprint": EXPERIMENT_FINGERPRINT,
                "drive_archive": str(drive_archive),
                "drive_manifest": str(drive_manifest),
                "archive_sha256": archive_sha256,
                "archive_bytes": drive_archive.stat().st_size,
                "drive_evidence": drive_evidence,
            }
            (DRIVE_ARTIFACT_ROOT / f"{RUN_ID}-receipt.json").write_text(
                json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            print(json.dumps(COST_ESTIMATE, ensure_ascii=False, indent=2))
            print(json.dumps(receipt, ensure_ascii=False, indent=2))
            if RUN_MODE == "calibration":
                print(
                    "\nA100 calibration 完成。完整訓練仍是鎖定狀態；"
                    "請把 run_manifest.json、receipt.json 與最後輸出回傳確認。"
                )
            else:
                print(
                    "\nPhase 3 完整訓練完成。請回傳 run_manifest.json、receipt.json、"
                    "trainer_log.csv 與 training_curves.png；不要在 Phase 4 前載入 test。"
                )
            """
        ),
        _markdown(
            """
            ## Phase 3 停止點

            完整訓練成功後停在這裡，不會推送 Hub，也不會自動進入 Phase 4。
            請回傳 run manifest、receipt、trainer log 與 training curve；MedQA test 與
            TMMLU+ 仍維持未載入狀態，等 Phase 3 證據驗收後再處理。
            """
        ),
    ]

    for index, cell in enumerate(cells):
        cell["id"] = f"phase3-{index:02d}"
        if cell.cell_type == "code":
            cell["execution_count"] = None
            cell["outputs"] = []

    return nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "accelerator": "GPU",
            "colab": {
                "name": "train_qlora.ipynb",
                "provenance": [],
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
    )


def rendered_notebook() -> str:
    return nbformat.writes(build_notebook(), version=4) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if the checked-in notebook differs from generated content.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rendered = rendered_notebook()
    if args.check:
        if not OUTPUT_PATH.exists() or OUTPUT_PATH.read_text(encoding="utf-8") != rendered:
            raise SystemExit("notebooks/train_qlora.ipynb is stale; run the notebook builder")
        print(f"Notebook is current: {OUTPUT_PATH.relative_to(ROOT)}")
        return

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(rendered, encoding="utf-8", newline="\n")
    print(f"Wrote {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

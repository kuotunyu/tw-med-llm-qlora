"""Build the approved, resumable Phase 4 full-evaluation Colab notebook."""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path
from textwrap import dedent

import nbformat

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "project.toml"
REQUIREMENTS_PATH = ROOT / "requirements" / "colab-eval.txt"
OUTPUT_PATH = ROOT / "notebooks" / "evaluate_phase4_full.ipynb"
EMBEDDED_HELPERS = (
    "types.py",
    "medqa.py",
    "evaluation.py",
    "tmmlu.py",
    "phase4.py",
    "phase4_full.py",
)


def _markdown(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(dedent(source).strip() + "\n")


def _code(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(dedent(source).strip() + "\n")


def _install_cell(project_config: dict) -> str:
    requirements = [
        line.strip()
        for line in REQUIREMENTS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    uv_requirements = [item for item in requirements if item.startswith("uv==")]
    runtime_requirements = [item for item in requirements if not item.startswith("uv==")]
    if len(uv_requirements) != 1:
        raise ValueError("colab-eval.txt must pin exactly one uv version")
    vllm_config = project_config["evaluation"]["vllm"]
    if not any(vllm_config["wheel_url"] in item for item in runtime_requirements):
        raise ValueError("Pinned vLLM wheel URL is missing from colab-eval.txt")
    return dedent(
        f"""
        import json
        import subprocess
        import sys
        import time

        NOTEBOOK_BUILD = "phase4-full-approved-resumable-v1"
        NOTEBOOK_STARTED_PERF = time.perf_counter()
        already_loaded = sorted(name for name in ("torch", "vllm") if name in sys.modules)
        if already_loaded:
            raise RuntimeError(
                "請先『中斷連線並刪除執行階段』，重新連線 A100 後再按全部執行；"
                f"目前已載入會污染 CUDA 安裝的模組：{{already_loaded}}"
            )

        uv_requirement = {json.dumps(uv_requirements[0])}
        runtime_requirements = json.loads(
            {json.dumps(json.dumps(runtime_requirements, ensure_ascii=False))}
        )
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", uv_requirement],
            check=True,
        )
        install_command = [
            "uv",
            "pip",
            "install",
            "--system",
            "--upgrade",
            "--torch-backend",
            {json.dumps(vllm_config["torch_backend"])},
            *runtime_requirements,
        ]
        print(json.dumps({{
            "notebook_build": NOTEBOOK_BUILD,
            "torch_backend": {json.dumps(vllm_config["torch_backend"])},
            "vllm_wheel": {json.dumps(vllm_config["wheel_url"])},
        }}, indent=2))
        subprocess.run(install_command, check=True)
        """
    ).strip()


def _embedded_helpers_cell() -> str:
    files = {"tw_med_qlora/__init__.py": ""}
    source_root = ROOT / "src" / "tw_med_qlora"
    for name in EMBEDDED_HELPERS:
        files[f"tw_med_qlora/{name}"] = (source_root / name).read_text(encoding="utf-8")
    payload = json.dumps(files, ensure_ascii=False)
    return dedent(
        f"""
        # ruff: noqa: E402, E501, I001
        EMBEDDED_HELPER_FILES = json.loads({json.dumps(payload)})
        HELPER_ROOT = Path("/content/tw-med-eval-helpers")
        for relative_name, source in EMBEDDED_HELPER_FILES.items():
            target = HELPER_ROOT / relative_name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(source, encoding="utf-8")
        if str(HELPER_ROOT) not in sys.path:
            sys.path.insert(0, str(HELPER_ROOT))

        from tw_med_qlora.evaluation import (
            PredictionRecord,
            accuracy_summary,
            forgetting_noninferiority,
            mcnemar_exact_test,
            paired_bootstrap_accuracy_difference,
            parse_mcq_answer,
            representative_case_ids,
            subject_accuracy,
        )
        from tw_med_qlora.medqa import (
            content_fingerprint,
            file_sha256,
            iter_parquet_rows,
            medqa_row_to_example,
        )
        from tw_med_qlora.phase4 import (
            build_vllm_serve_command,
            extract_verified_adapter,
            phase4_workload,
        )
        from tw_med_qlora.phase4_full import (
            atomic_copy_verified,
            canonical_json,
            evaluation_request_id,
            plan_result_shards,
            read_verified_result_shard,
            sha256_bytes,
            write_result_shard,
        )
        from tw_med_qlora.tmmlu import (
            SubjectExample,
            read_tmmlu_csv,
            shuffle_options,
            stability_sample,
        )
        print("Repository-tested Phase 4 full-evaluation helpers loaded.")
        """
    ).strip()


def build_notebook() -> nbformat.NotebookNode:
    with CONFIG_PATH.open("rb") as config_file:
        project_config = tomllib.load(config_file)
    approval = project_config["evaluation"]["full_approval"]
    if not approval["approved"] or approval["approved_requests"] != 28_758:
        raise RuntimeError("Phase 4 full evaluation has not been approved")
    config_json = json.dumps(
        project_config,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    cells = [
        _markdown(
            r"""
            # tw-med-llm-qlora — Phase 4 正式雙軌評估

            這是已通過人工成本閘門的 **A100 正式評估 notebook**。它固定執行：

            - MedQA test：1,413 題 × 3 個模型；
            - TMMLU+ 13 科 test：5,573 題 × 3 個模型；
            - TMMLU+ 穩定度：每科至多 100 題 × 3 seeds × base/adapter；
            - 合計恰好 28,758 次生成。

            生成會每 250 次封裝成一個具 SHA-256 與 request graph 驗證的 Drive ZIP。
            Colab 中斷後，只要重新開這一份 notebook、選 A100 並按「全部執行」，就會驗證
            並略過完成分片；不要改任何 code，也不要從中段開始。

            原始題目與完整輸出只留在私人 Drive 分片；公開結果只含 hash ID、gold、prediction、
            parse/correct、token/latency 與統計表。研究用途，非醫療建議。
            """
        ),
        _markdown(
            """
            ## 1. 安裝固定 CUDA 12.9 評估環境

            請從全新的 A100 runtime 按「全部執行」。安裝後不要再次執行這格；若中斷，刪除
            runtime、重連 A100，再由最上方全部執行即可。
            """
        ),
        _code(_install_cell(project_config)),
        _code(
            r"""
            import importlib.metadata
            import importlib.util
            import json
            import subprocess
            import sys

            REQUIRED_EVAL_PACKAGES = {
                "vllm": ("vllm", "0.25.1+cu129"),
                "twinkle-eval": ("twinkle_eval", "2.8.0"),
                "bitsandbytes": ("bitsandbytes", "0.49.2"),
            }
            dependency_audit = {}
            dependency_errors = []
            for package_name, (module_name, expected_version) in REQUIRED_EVAL_PACKAGES.items():
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
            if dependency_errors:
                raise RuntimeError(
                    f"評估依賴缺少或版本不符：{dependency_errors}。請刪除 runtime 後重來。"
                )

            native_probe = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import importlib.metadata,json,torch,vllm;"
                        "print(json.dumps({'vllm_version':importlib.metadata.version('vllm'),"
                        "'torch_version':torch.__version__,'torch_cuda':torch.version.cuda,"
                        "'cuda_available':torch.cuda.is_available(),"
                        "'gpu_name':torch.cuda.get_device_name(0) "
                        "if torch.cuda.is_available() else None}))"
                    ),
                ],
                capture_output=True,
                text=True,
                timeout=180,
            )
            if native_probe.returncode != 0:
                print(native_probe.stdout)
                print(native_probe.stderr)
                raise RuntimeError("vLLM CUDA 原生匯入失敗；請刪除 runtime 後重新全部執行。")
            native_audit = json.loads(
                [line for line in native_probe.stdout.splitlines() if line.strip()][-1]
            )
            if native_audit["torch_cuda"] != "12.9" or not native_audit["cuda_available"]:
                raise RuntimeError(f"CUDA runtime 不符：{native_audit}")
            dependency_audit["native_cuda_preflight"] = native_audit
            print(json.dumps(dependency_audit, ensure_ascii=False, indent=2))
            """
        ),
        _markdown(
            """
            ## 2. 固定核准、Secrets、Drive 與 A100 閘門

            只需在 Colab Secrets 開啟 `HF_TOKEN`。本 notebook 已寫入人工核准碼，無需尋找或
            修改 `RUN_MODE`。它不會推送 HF adapter，也不會呼叫付費 LLM API。
            """
        ),
        _code(
            r"""
            import concurrent.futures
            import hashlib
            import json
            import os
            import platform
            import shutil
            import signal
            import statistics
            import subprocess
            import sys
            import time
            import urllib.error
            import urllib.request
            from datetime import UTC, datetime
            from pathlib import Path

            import torch
            from google.colab import drive, userdata
            from huggingface_hub import hf_hub_download, snapshot_download

            PROJECT_CONFIG = json.loads(r'''__PROJECT_CONFIG_JSON__''')
            RUN_MODE = "full"
            ALLOW_FULL_EVALUATION = True
            FULL_EVALUATION_APPROVAL = "PHASE4_FULL_28758_APPROVED_20260722"
            approval = PROJECT_CONFIG["evaluation"]["full_approval"]
            if RUN_MODE != "full" or not ALLOW_FULL_EVALUATION:
                raise RuntimeError("Phase 4 正式評估 gate 未解鎖")
            if FULL_EVALUATION_APPROVAL != approval["required_approval_code"]:
                raise RuntimeError("Phase 4 正式評估人工核准碼不符")
            if approval["approval_phrase"] != "確認解鎖 Phase 4 正式評估":
                raise RuntimeError("Phase 4 人工核准證據不符")

            if not torch.cuda.is_available():
                raise RuntimeError("Phase 4 正式評估需要 A100 GPU runtime")
            gpu_name = torch.cuda.get_device_name(0)
            gpu_properties = torch.cuda.get_device_properties(0)
            gpu_vram_gib = gpu_properties.total_memory / 1024**3
            if "A100" not in gpu_name.upper() or gpu_vram_gib < 38:
                raise RuntimeError(
                    f"需要 A100 >=38 GiB；目前為 {gpu_name} ({gpu_vram_gib:.2f} GiB)"
                )
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("正式評估設定需要 BF16")

            HF_TOKEN = userdata.get("HF_TOKEN")
            if not HF_TOKEN:
                raise RuntimeError("Colab Secret HF_TOKEN 缺少或未授權此 notebook")
            os.environ["HF_TOKEN"] = HF_TOKEN
            os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
            drive.mount("/content/drive")

            RUN_ID = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            LOCAL_ROOT = Path("/content/tw-med-phase4-full") / RUN_ID
            LOCAL_SHARD_ROOT = LOCAL_ROOT / "shards"
            PUBLIC_ROOT = LOCAL_ROOT / "public"
            PRIVATE_ROOT = LOCAL_ROOT / "private"
            LOG_ROOT = PRIVATE_ROOT / "server-logs"
            for directory in (LOCAL_SHARD_ROOT, PUBLIC_ROOT, PRIVATE_ROOT, LOG_ROOT):
                directory.mkdir(parents=True, exist_ok=True)

            hardware_audit = {
                "gpu_name": gpu_name,
                "gpu_vram_gib": gpu_vram_gib,
                "compute_capability": list(torch.cuda.get_device_capability(0)),
                "bf16_supported": torch.cuda.is_bf16_supported(),
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "platform": platform.platform(),
            }
            print(json.dumps({
                "phase": 4,
                "run_mode": RUN_MODE,
                "approved_requests": approval["approved_requests"],
                "calibration_run_id": approval["calibration_run_id"],
                "hardware": hardware_audit,
            }, ensure_ascii=False, indent=2))
            """.replace("__PROJECT_CONFIG_JSON__", config_json)
        ),
        _markdown("## 3. 載入 repo 測試過的 helper 與固定 request contract"),
        _code(_embedded_helpers_cell()),
        _code(
            r"""
            evaluation_config = PROJECT_CONFIG["evaluation"]
            workload_config = evaluation_config["workload"]
            all_subjects = (
                evaluation_config["medical_subjects"]
                + evaluation_config["control_subjects"]
            )
            WORKLOAD = phase4_workload(
                medqa_test_rows=int(workload_config["medqa_test_rows"]),
                tmmlu_test_rows=int(workload_config["tmmlu_test_rows"]),
                full_model_count=int(workload_config["full_model_count"]),
                subject_count=len(all_subjects),
                stability_examples_per_subject=int(
                    evaluation_config["stability_examples_per_subject"]
                ),
                stability_seeds=evaluation_config["stability_seeds"],
                stability_model_count=int(workload_config["stability_model_count"]),
            )
            if WORKLOAD.total != 28758 or WORKLOAD.total != int(approval["approved_requests"]):
                raise RuntimeError(f"Request contract 不是核准的 28,758：{WORKLOAD.as_dict()}")

            model_config = PROJECT_CONFIG["models"]["primary"]
            CONTRACT = {
                "schema_version": 1,
                "approval_code": FULL_EVALUATION_APPROVAL,
                "workload": WORKLOAD.as_dict(),
                "models": {
                    "original-instruct": {
                        "id": model_config["baseline_id"],
                        "revision": model_config["baseline_revision"],
                    },
                    "localized-base": {
                        "id": model_config["model_id"],
                        "revision": model_config["revision"],
                    },
                    "localized-medical-adapter": evaluation_config["phase3_adapter"],
                },
                "data": {
                    "medqa": PROJECT_CONFIG["data"]["medqa"],
                    "tmmluplus": PROJECT_CONFIG["data"]["tmmluplus"],
                },
                "generation": evaluation_config["generation"],
                "statistics": {
                    "bootstrap_iterations": evaluation_config["bootstrap_iterations"],
                    "forgetting_margin_percentage_points": evaluation_config[
                        "forgetting_margin_percentage_points"
                    ],
                    "medical_subjects": evaluation_config["medical_subjects"],
                    "control_subjects": evaluation_config["control_subjects"],
                },
                "full_shuffle_seed": evaluation_config["full_shuffle_seed"],
                "stability_seeds": evaluation_config["stability_seeds"],
                "stability_examples_per_subject": evaluation_config[
                    "stability_examples_per_subject"
                ],
                "calibration_manifest_sha256": approval["calibration_manifest_sha256"],
                "calibration_validation_sha256": approval["calibration_validation_sha256"],
            }
            CONTRACT_FINGERPRINT = sha256_bytes(canonical_json(CONTRACT).encode("utf-8"))
            DRIVE_ROOT = (
                Path("/content/drive/MyDrive/tw-med-llm-qlora/phase4/full")
                / CONTRACT_FINGERPRINT
            )
            DRIVE_SHARD_ROOT = DRIVE_ROOT / "shards"
            DRIVE_FINAL_ROOT = DRIVE_ROOT / "final" / "runs"
            DRIVE_SHARD_ROOT.mkdir(parents=True, exist_ok=True)
            DRIVE_FINAL_ROOT.mkdir(parents=True, exist_ok=True)
            print(json.dumps({
                "contract_fingerprint": CONTRACT_FINGERPRINT,
                "workload": WORKLOAD.as_dict(),
                "drive_root": str(DRIVE_ROOT),
                "resume_policy": "verify completed ZIP shards, rerun only missing shards",
            }, ensure_ascii=False, indent=2))
            """
        ),
        _markdown("## 4. 驗證 Phase 3 step-700 adapter"),
        _code(
            r"""
            adapter_contract = evaluation_config["phase3_adapter"]
            adapter_audit = extract_verified_adapter(
                Path(adapter_contract["drive_archive"]),
                Path("/content/tw-med-phase4-adapter"),
                expected_sha256=adapter_contract["archive_sha256"],
                expected_bytes=int(adapter_contract["archive_bytes"]),
                expected_base_model_id=adapter_contract["base_model_id"],
            )
            ADAPTER_DIR = Path(adapter_audit["adapter_dir"])
            print(json.dumps(adapter_audit, ensure_ascii=False, indent=2))
            """
        ),
        _markdown(
            """
            ## 5. 下載並固定正式資料

            這是 Phase 4 正式 test，因此只下載 MedQA test 與核准的 13 個 TMMLU+ test CSV。
            不載入 MedQA train/validation，也不再做 prompt 或超參數選擇。
            """
        ),
        _code(
            r"""
            medqa_config = PROJECT_CONFIG["data"]["medqa"]
            medqa_filename = f"{medqa_config['config']}/test/0000.parquet"
            medqa_path = Path(
                hf_hub_download(
                    repo_id=medqa_config["dataset_id"],
                    filename=medqa_filename,
                    repo_type="dataset",
                    revision=medqa_config["revision"],
                    local_dir=LOCAL_ROOT / "data" / "medqa",
                    token=HF_TOKEN,
                )
            )
            if file_sha256(medqa_path) != medqa_config["source_sha256"]["test"]:
                raise RuntimeError("MedQA test Parquet SHA-256 已改變，停止正式評估")
            medqa_examples = [
                medqa_row_to_example(
                    row,
                    split="test",
                    source=medqa_config["dataset_id"],
                    revision=medqa_config["revision"],
                )
                for row in iter_parquet_rows(medqa_path)
            ]
            if len(medqa_examples) != int(medqa_config["expected_test_rows"]):
                raise RuntimeError(f"MedQA test 筆數不符：{len(medqa_examples)}")
            medqa_items = [
                SubjectExample(subject="medqa_total", example=example)
                for example in medqa_examples
            ]

            tmmlu_config = PROJECT_CONFIG["data"]["tmmluplus"]
            tmmlu_root = LOCAL_ROOT / "data" / "tmmluplus"
            allowed_test_patterns = [f"data/{subject}_test.csv" for subject in all_subjects]
            tmmlu_snapshot = Path(
                snapshot_download(
                    repo_id=tmmlu_config["dataset_id"],
                    repo_type="dataset",
                    revision=tmmlu_config["revision"],
                    allow_patterns=allowed_test_patterns,
                    local_dir=tmmlu_root,
                    token=HF_TOKEN,
                )
            )
            unexpected_csv = [
                path for path in tmmlu_snapshot.rglob("*.csv")
                if path.name not in {f"{subject}_test.csv" for subject in all_subjects}
            ]
            if unexpected_csv:
                raise RuntimeError(f"下載到未核准的 TMMLU+ CSV：{unexpected_csv}")

            tmmlu_by_subject = {}
            tmmlu_counts = {}
            for subject in all_subjects:
                rows = read_tmmlu_csv(
                    tmmlu_snapshot / "data" / f"{subject}_test.csv",
                    subject=subject,
                    split="test",
                    source=tmmlu_config["dataset_id"],
                    revision=tmmlu_config["revision"],
                )
                tmmlu_by_subject[subject] = rows
                tmmlu_counts[subject] = len(rows)
            if sum(tmmlu_counts.values()) != int(workload_config["tmmlu_test_rows"]):
                raise RuntimeError(f"TMMLU+ 13 科筆數不符：{tmmlu_counts}")

            full_seed = int(evaluation_config["full_shuffle_seed"])
            stability_source = stability_sample(
                tmmlu_by_subject,
                per_subject=int(evaluation_config["stability_examples_per_subject"]),
                sample_seed=int(PROJECT_CONFIG["project"]["seed"]),
            )
            ITEMS_BY_SUITE = {
                "medqa-full": medqa_items,
                "tmmlu-full": [
                    shuffle_options(item, seed=full_seed)
                    for subject in all_subjects
                    for item in tmmlu_by_subject[subject]
                ],
            }
            for option_seed in evaluation_config["stability_seeds"]:
                ITEMS_BY_SUITE[f"tmmlu-stability-{option_seed}"] = [
                    shuffle_options(item, seed=int(option_seed))
                    for subject in all_subjects
                    for item in stability_source[subject]
                ]

            REQUEST_ITEM = {}
            for suite, items in ITEMS_BY_SUITE.items():
                option_seed = int(suite.rsplit("-", 1)[-1]) if suite.startswith(
                    "tmmlu-stability-"
                ) else (full_seed if suite == "tmmlu-full" else None)
                for item in items:
                    request_id = evaluation_request_id(
                        suite=suite,
                        example_id=item.example.id,
                        option_seed=option_seed,
                    )
                    if request_id in REQUEST_ITEM:
                        raise RuntimeError(f"正式評估 request ID 重複：{request_id}")
                    REQUEST_ITEM[request_id] = {
                        "suite": suite,
                        "option_seed": option_seed,
                        "item": item,
                    }

            data_audit = {
                "medqa_test_rows": len(medqa_items),
                "medqa_content_fingerprint": content_fingerprint(medqa_examples),
                "tmmlu_test_rows": sum(tmmlu_counts.values()),
                "tmmlu_test_counts": tmmlu_counts,
                "stability_rows_per_seed": sum(len(rows) for rows in stability_source.values()),
                "loaded_splits": ["medqa:test", "tmmluplus:test"],
                "train_rows_loaded": 0,
                "validation_rows_loaded": 0,
                "prompt_or_hyperparameter_selection_after_test_load": False,
            }
            print(json.dumps(data_audit, ensure_ascii=False, indent=2))
            """
        ),
        _markdown("## 6. 建立可驗證、可續跑的 Drive 分片計畫"),
        _code(
            r"""
            SHARD_SIZE = int(approval["shard_size"])
            PARALLEL_WORKERS = int(approval["parallel_workers"])
            MODEL_SUITES = {
                "original-instruct": ["medqa-full", "tmmlu-full"],
                "localized-base": [
                    "medqa-full",
                    "tmmlu-full",
                    *[f"tmmlu-stability-{seed}" for seed in evaluation_config["stability_seeds"]],
                ],
                "localized-medical-adapter": [
                    "medqa-full",
                    "tmmlu-full",
                    *[f"tmmlu-stability-{seed}" for seed in evaluation_config["stability_seeds"]],
                ],
            }
            PLANS_BY_MODEL = {}
            for model_label, suites in MODEL_SUITES.items():
                plans = []
                for suite in suites:
                    request_ids = [
                        evaluation_request_id(
                            suite=suite,
                            example_id=item.example.id,
                            option_seed=(
                                int(suite.rsplit("-", 1)[-1])
                                if suite.startswith("tmmlu-stability-")
                                else (full_seed if suite == "tmmlu-full" else None)
                            ),
                        )
                        for item in ITEMS_BY_SUITE[suite]
                    ]
                    plans.extend(
                        plan_result_shards(
                            suite=suite,
                            model=model_label,
                            request_ids=request_ids,
                            shard_size=SHARD_SIZE,
                            contract_fingerprint=CONTRACT_FINGERPRINT,
                        )
                    )
                PLANS_BY_MODEL[model_label] = plans

            planned_requests = sum(
                len(plan.request_ids)
                for plans in PLANS_BY_MODEL.values()
                for plan in plans
            )
            if planned_requests != WORKLOAD.total:
                raise RuntimeError(
                    f"分片計畫產生 {planned_requests} requests，不是核准的 {WORKLOAD.total}"
                )

            def drive_shard_path(plan):
                return DRIVE_SHARD_ROOT / plan.model / plan.suite / plan.filename

            completed_plans = set()
            completed_requests = 0
            for plans in PLANS_BY_MODEL.values():
                for plan in plans:
                    path = drive_shard_path(plan)
                    if not path.exists():
                        continue
                    read_verified_result_shard(path, expected_plan=plan)
                    completed_plans.add(plan.fingerprint)
                    completed_requests += len(plan.request_ids)
            print(json.dumps({
                "approved_requests": WORKLOAD.total,
                "planned_shards": sum(len(plans) for plans in PLANS_BY_MODEL.values()),
                "verified_completed_shards": len(completed_plans),
                "verified_completed_requests": completed_requests,
                "remaining_requests": WORKLOAD.total - completed_requests,
            }, ensure_ascii=False, indent=2))
            """
        ),
        _markdown(
            """
            ## 7. Twinkle Eval 稽核、vLLM server 與分片執行器

            parser、prompt、temperature、max tokens 與校準完全相同。每個完整分片才會原子同步
            至 Drive；若中斷，最多重做當下未完成的 250 次生成。
            """
        ),
        _code(
            r"""
            from openai import OpenAI
            from twinkle_eval.metrics.extractors.box import BoxExtractor
            from twinkle_eval.metrics.scorers.exact import ExactMatchScorer

            box_extractor = BoxExtractor()
            exact_scorer = ExactMatchScorer()
            if box_extractor.extract(r"\boxed{A}") != "A":
                raise RuntimeError("Twinkle Eval BoxExtractor contract failed")
            if not exact_scorer.score("A", "A") or exact_scorer.score("A", "B"):
                raise RuntimeError("Twinkle Eval ExactMatchScorer contract failed")

            ACTIVE_SERVER = None
            ACTIVE_LOG_HANDLE = None

            def wait_for_server(process, *, port, timeout_seconds=1800):
                started = time.perf_counter()
                last_notice = -30
                while time.perf_counter() - started < timeout_seconds:
                    if process.poll() is not None:
                        raise RuntimeError(
                            f"vLLM server exited with code {process.returncode}; "
                            "inspect private log"
                        )
                    try:
                        with urllib.request.urlopen(
                            f"http://127.0.0.1:{port}/health", timeout=5
                        ) as response:
                            if response.status == 200:
                                return time.perf_counter() - started
                    except (urllib.error.URLError, TimeoutError):
                        pass
                    elapsed = int(time.perf_counter() - started)
                    if elapsed - last_notice >= 30:
                        print(f"Waiting for vLLM server: {elapsed}s elapsed")
                        last_notice = elapsed
                    time.sleep(5)
                raise TimeoutError("vLLM server did not become healthy within 1800s")

            def start_server(command, *, label, port):
                global ACTIVE_SERVER, ACTIVE_LOG_HANDLE
                if ACTIVE_SERVER is not None:
                    raise RuntimeError("A vLLM server is already active")
                log_path = LOG_ROOT / f"{label}.log"
                ACTIVE_LOG_HANDLE = log_path.open("w", encoding="utf-8")
                environment = os.environ.copy()
                environment["HF_TOKEN"] = HF_TOKEN
                ACTIVE_SERVER = subprocess.Popen(
                    command,
                    stdout=ACTIVE_LOG_HANDLE,
                    stderr=subprocess.STDOUT,
                    env=environment,
                    text=True,
                    start_new_session=True,
                )
                try:
                    startup_seconds = wait_for_server(ACTIVE_SERVER, port=port)
                except Exception:
                    ACTIVE_LOG_HANDLE.flush()
                    print(log_path.read_text(encoding="utf-8", errors="replace")[-6000:])
                    stop_server()
                    raise
                print(f"{label} server ready after {startup_seconds:.1f}s")
                return startup_seconds

            def stop_server():
                global ACTIVE_SERVER, ACTIVE_LOG_HANDLE
                process = ACTIVE_SERVER
                ACTIVE_SERVER = None
                stopped_running_process = False
                if process is not None and process.poll() is None:
                    stopped_running_process = True
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        process.wait(timeout=10)
                log_handle = ACTIVE_LOG_HANDLE
                ACTIVE_LOG_HANDLE = None
                if log_handle is not None and not log_handle.closed:
                    log_handle.close()
                if stopped_running_process:
                    time.sleep(5)

            def visible_prompt(item):
                example = item.example
                choices = "\n".join(
                    f"{key}. {example.choices[key]}" for key in ("A", "B", "C", "D")
                )
                return f"{example.question}\n\n{choices}"

            generation_config = evaluation_config["generation"]
            SYSTEM_PROMPT = (
                "請選擇唯一最佳答案。不要解釋或重述題目；只輸出單一大寫 "
                r"A–D 字母，或一個 LaTeX 答案框，例如 \boxed{A}。"
            )

            def evaluate_one(client, *, served_name, public_label, request_id):
                request = REQUEST_ITEM[request_id]
                item = request["item"]
                prompt = visible_prompt(item)
                started = time.perf_counter()
                response = client.chat.completions.create(
                    model=served_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=float(generation_config["temperature"]),
                    top_p=float(generation_config["top_p"]),
                    max_tokens=int(generation_config["max_tokens"]),
                    seed=int(PROJECT_CONFIG["project"]["seed"]),
                )
                latency = time.perf_counter() - started
                choice = response.choices[0]
                raw_output = choice.message.content or ""
                prediction = parse_mcq_answer(raw_output)
                max_token_limit_hit = choice.finish_reason == "length"
                if max_token_limit_hit:
                    prediction = None
                twinkle_prediction = box_extractor.extract(raw_output)
                if prediction is not None and r"\boxed" in raw_output:
                    if twinkle_prediction != prediction:
                        raise RuntimeError("Strict parser and Twinkle box extractor disagree")
                if prediction is not None:
                    score = bool(exact_scorer.score(prediction, item.example.answer))
                    if score != (prediction == item.example.answer):
                        raise RuntimeError("Twinkle exact scorer contract drifted")
                usage = response.usage
                raw_digest = hashlib.sha256(raw_output.encode("utf-8")).hexdigest()
                public = {
                    "request_id": request_id,
                    "example_id": item.example.id,
                    "suite": request["suite"],
                    "option_seed": request["option_seed"],
                    "model": public_label,
                    "source": item.example.source,
                    "subject": item.subject,
                    "gold": item.example.answer,
                    "prediction": prediction,
                    "parsed": prediction is not None,
                    "correct": prediction == item.example.answer,
                    "raw_output_sha256": raw_digest,
                    "latency_seconds": latency,
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "finish_reason": choice.finish_reason,
                    "max_token_limit_hit": max_token_limit_hit,
                }
                private = {
                    "request_id": request_id,
                    "example_id": item.example.id,
                    "suite": request["suite"],
                    "option_seed": request["option_seed"],
                    "model": public_label,
                    "subject": item.subject,
                    "gold": item.example.answer,
                    "question": item.example.question,
                    "choices": dict(item.example.choices),
                    "system_prompt": SYSTEM_PROMPT,
                    "prompt": prompt,
                    "raw_output": raw_output,
                }
                return public, private

            def run_pending_plans(*, port, served_name, public_label):
                client = OpenAI(
                    api_key="local-eval",
                    base_url=f"http://127.0.0.1:{port}/v1",
                    timeout=600,
                    max_retries=5,
                )
                plans = PLANS_BY_MODEL[public_label]
                pending = [plan for plan in plans if plan.fingerprint not in completed_plans]
                for plan in pending:
                    started = time.perf_counter()
                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=PARALLEL_WORKERS
                    ) as executor:
                        futures = [
                            executor.submit(
                                evaluate_one,
                                client,
                                served_name=served_name,
                                public_label=public_label,
                                request_id=request_id,
                            )
                            for request_id in plan.request_ids
                        ]
                        pairs = [future.result() for future in futures]
                    public_rows = [pair[0] for pair in pairs]
                    private_rows = [pair[1] for pair in pairs]
                    local_path = LOCAL_SHARD_ROOT / plan.filename
                    write_result_shard(
                        local_path,
                        plan=plan,
                        public_rows=public_rows,
                        private_rows=private_rows,
                    )
                    destination = drive_shard_path(plan)
                    receipt = atomic_copy_verified(local_path, destination)
                    read_verified_result_shard(destination, expected_plan=plan)
                    completed_plans.add(plan.fingerprint)
                    elapsed = time.perf_counter() - started
                    total_done = sum(
                        len(candidate.request_ids)
                        for model_plans in PLANS_BY_MODEL.values()
                        for candidate in model_plans
                        if candidate.fingerprint in completed_plans
                    )
                    print(json.dumps({
                        "saved_shard": plan.filename,
                        "model": public_label,
                        "suite": plan.suite,
                        "rows": len(plan.request_ids),
                        "shard_seconds": elapsed,
                        "drive_sha256": receipt["sha256"],
                        "completed_requests": total_done,
                        "remaining_requests": WORKLOAD.total - total_done,
                    }, ensure_ascii=False))
                return len(pending)

            print("Formal evaluation runner is ready.")
            """
        ),
        _markdown(
            """
            ## 8. 執行三個模型（自動續跑）

            原廠模型使用第一個 server；台灣 base 與 adapter 共用第二個 server。若某個模型的
            所有 Drive 分片都已完成，就不會啟動其 server。請保持瀏覽器與 runtime 連線。
            """
        ),
        _code(
            r"""
            vllm_config = evaluation_config["vllm"]
            PORT = 8000
            server_startup_seconds = {}

            original_name = "original-instruct"
            if any(
                plan.fingerprint not in completed_plans
                for plan in PLANS_BY_MODEL[original_name]
            ):
                original_command = build_vllm_serve_command(
                    model_id=model_config["baseline_id"],
                    model_revision=model_config["baseline_revision"],
                    served_model_name=original_name,
                    port=PORT,
                    max_model_length=int(vllm_config["max_model_length"]),
                    gpu_memory_utilization=float(vllm_config["gpu_memory_utilization"]),
                    seed=int(PROJECT_CONFIG["project"]["seed"]),
                )
                try:
                    server_startup_seconds[original_name] = start_server(
                        original_command, label=original_name, port=PORT
                    )
                    run_pending_plans(
                        port=PORT,
                        served_name=original_name,
                        public_label=original_name,
                    )
                finally:
                    stop_server()
            else:
                print("original-instruct 已完成，略過 server 啟動。")

            localized_name = "localized-base"
            adapter_name = "localized-medical-adapter"
            localized_pending = any(
                plan.fingerprint not in completed_plans
                for label in (localized_name, adapter_name)
                for plan in PLANS_BY_MODEL[label]
            )
            if localized_pending:
                localized_command = build_vllm_serve_command(
                    model_id=model_config["model_id"],
                    model_revision=model_config["revision"],
                    served_model_name=localized_name,
                    port=PORT,
                    max_model_length=int(vllm_config["max_model_length"]),
                    gpu_memory_utilization=float(vllm_config["gpu_memory_utilization"]),
                    seed=int(PROJECT_CONFIG["project"]["seed"]),
                    adapter_name=adapter_name,
                    adapter_path=ADAPTER_DIR,
                    max_lora_rank=int(vllm_config["max_lora_rank"]),
                )
                try:
                    server_startup_seconds["localized-with-adapter"] = start_server(
                        localized_command, label="localized-with-adapter", port=PORT
                    )
                    for served_name in (localized_name, adapter_name):
                        run_pending_plans(
                            port=PORT,
                            served_name=served_name,
                            public_label=served_name,
                        )
                finally:
                    stop_server()
            else:
                print("localized-base 與 adapter 已完成，略過 server 啟動。")

            session_wall_seconds = time.perf_counter() - NOTEBOOK_STARTED_PERF
            expected_plan_fingerprints = {
                plan.fingerprint
                for plans in PLANS_BY_MODEL.values()
                for plan in plans
            }
            if completed_plans != expected_plan_fingerprints:
                missing = expected_plan_fingerprints.difference(completed_plans)
                raise RuntimeError(
                    f"本次尚有 {len(missing)} 個分片未完成。重新開 A100 並按全部執行即可續跑。"
                )
            print(json.dumps({
                "all_shards_complete": True,
                "completed_requests": WORKLOAD.total,
                "session_wall_seconds": session_wall_seconds,
                "server_startup_seconds": server_startup_seconds,
            }, ensure_ascii=False, indent=2))
            """
        ),
        _markdown("## 9. 重建公開結果並計算雙軌統計"),
        _code(
            r"""
            all_public_rows = []
            all_private_by_request_model = {}
            for plans in PLANS_BY_MODEL.values():
                for plan in plans:
                    public_rows, private_rows, _ = read_verified_result_shard(
                        drive_shard_path(plan), expected_plan=plan
                    )
                    all_public_rows.extend(public_rows)
                    for row in private_rows:
                        key = (row["model"], row["request_id"])
                        if key in all_private_by_request_model:
                            raise RuntimeError(f"重複 private result：{key}")
                        all_private_by_request_model[key] = row
            if len(all_public_rows) != WORKLOAD.total:
                raise RuntimeError(f"結果筆數不是 28,758：{len(all_public_rows)}")

            def prediction_records(*, model, suite):
                rows = [
                    row for row in all_public_rows
                    if row["model"] == model and row["suite"] == suite
                ]
                return [
                    PredictionRecord(
                        example_id=row["example_id"],
                        model=row["model"],
                        source=row["source"],
                        subject=row["subject"],
                        gold=row["gold"],
                        prediction=row["prediction"],
                        raw_output_sha256=row["raw_output_sha256"],
                        latency_seconds=float(row["latency_seconds"]),
                        prompt_tokens=row["prompt_tokens"],
                        completion_tokens=row["completion_tokens"],
                    )
                    for row in rows
                ]

            model_labels = ["original-instruct", "localized-base", "localized-medical-adapter"]
            medqa_records = {
                model: prediction_records(model=model, suite="medqa-full")
                for model in model_labels
            }
            MEDQA_SUMMARY = {
                "split": "test",
                "subject_breakdown": "not_reported_source_has_no_native_subject_labels",
                "models": {
                    model: {
                        **accuracy_summary(records),
                        "max_token_limit_hits": sum(
                            row["max_token_limit_hit"]
                            for row in all_public_rows
                            if row["model"] == model and row["suite"] == "medqa-full"
                        ),
                    }
                    for model, records in medqa_records.items()
                },
                "adapter_minus_localized_base": paired_bootstrap_accuracy_difference(
                    medqa_records["localized-base"],
                    medqa_records["localized-medical-adapter"],
                    iterations=int(evaluation_config["bootstrap_iterations"]),
                    seed=int(PROJECT_CONFIG["project"]["seed"]),
                ),
                "mcnemar_adapter_vs_localized_base": mcnemar_exact_test(
                    medqa_records["localized-base"],
                    medqa_records["localized-medical-adapter"],
                ),
                "representative_cases": representative_case_ids(
                    medqa_records["localized-base"],
                    medqa_records["localized-medical-adapter"],
                    limit=10,
                    seed=int(PROJECT_CONFIG["project"]["seed"]),
                ),
            }

            tmmlu_records = {
                model: prediction_records(model=model, suite="tmmlu-full")
                for model in model_labels
            }
            TMMLU_SUMMARY = {
                "split": "test",
                "option_seed": full_seed,
                "models": {
                    model: {
                        "overall": accuracy_summary(records),
                        "by_subject": subject_accuracy(records),
                    }
                    for model, records in tmmlu_records.items()
                },
            }
            for model in model_labels:
                by_subject = TMMLU_SUMMARY["models"][model]["by_subject"]
                TMMLU_SUMMARY["models"][model]["medical_subject_macro_accuracy"] = statistics.mean(
                    by_subject[subject]["accuracy"]
                    for subject in evaluation_config["medical_subjects"]
                )
                TMMLU_SUMMARY["models"][model]["control_subject_macro_accuracy"] = statistics.mean(
                    by_subject[subject]["accuracy"]
                    for subject in evaluation_config["control_subjects"]
                )

            base_controls = [
                row for row in tmmlu_records["localized-base"]
                if row.subject in evaluation_config["control_subjects"]
            ]
            adapter_controls = [
                row for row in tmmlu_records["localized-medical-adapter"]
                if row.subject in evaluation_config["control_subjects"]
            ]
            TMMLU_SUMMARY["catastrophic_forgetting"] = forgetting_noninferiority(
                base_controls,
                adapter_controls,
                margin_percentage_points=float(
                    evaluation_config["forgetting_margin_percentage_points"]
                ),
                iterations=int(evaluation_config["bootstrap_iterations"]),
                seed=int(PROJECT_CONFIG["project"]["seed"]),
            )

            STABILITY_SUMMARY = {
                "split": "test",
                "sample_seed": int(PROJECT_CONFIG["project"]["seed"]),
                "option_seeds": evaluation_config["stability_seeds"],
                "models": {},
            }
            for model in ("localized-base", "localized-medical-adapter"):
                subject_seed_accuracy = {subject: {} for subject in all_subjects}
                for option_seed in evaluation_config["stability_seeds"]:
                    records = prediction_records(
                        model=model, suite=f"tmmlu-stability-{option_seed}"
                    )
                    summaries = subject_accuracy(records)
                    for subject in all_subjects:
                        subject_seed_accuracy[subject][str(option_seed)] = summaries[subject]
                STABILITY_SUMMARY["models"][model] = {}
                for subject, seed_summaries in subject_seed_accuracy.items():
                    accuracies = [summary["accuracy"] for summary in seed_summaries.values()]
                    STABILITY_SUMMARY["models"][model][subject] = {
                        "seeds": seed_summaries,
                        "accuracy_mean": statistics.mean(accuracies),
                        "accuracy_population_std": statistics.pstdev(accuracies),
                    }

            parse_warnings = []
            minimum_parse_rate = float(generation_config["minimum_calibration_parse_rate"])
            for suite_summary_name, summary in (
                ("medqa", MEDQA_SUMMARY["models"]),
                (
                    "tmmlu",
                    {model: TMMLU_SUMMARY["models"][model]["overall"] for model in model_labels},
                ),
            ):
                for model, values in summary.items():
                    if values["parse_rate"] < minimum_parse_rate:
                        parse_warnings.append({
                            "suite": suite_summary_name,
                            "model": model,
                            "parse_rate": values["parse_rate"],
                            "threshold": minimum_parse_rate,
                        })

            PHASE4_RESULTS = {
                "schema_version": 1,
                "phase": 4,
                "run_mode": "full",
                "contract_fingerprint": CONTRACT_FINGERPRINT,
                "generation_requests": len(all_public_rows),
                "generation_contract": {
                    "parser": "standalone_A-D_or_exactly_one_simple_boxed_A-D",
                    "scorer": "Twinkle Eval exact_match audit",
                    **generation_config,
                },
                "medqa": MEDQA_SUMMARY,
                "tmmluplus": TMMLU_SUMMARY,
                "stability": STABILITY_SUMMARY,
                "parse_rate_warnings": parse_warnings,
                "cost_accounting": {
                    "basis": "parser-v3 A100 calibration",
                    "projected_hours": approval["projected_hours"],
                    "compute_units_per_hour": approval["compute_units_per_hour"],
                    "projected_compute_units": approval["projected_compute_units"],
                    "projected_compute_units_with_20pct_buffer": approval[
                        "projected_compute_units_with_20pct_buffer"
                    ],
                    "this_session_wall_hours": session_wall_seconds / 3600,
                    "this_session_compute_units_estimate": (
                        session_wall_seconds
                        / 3600
                        * float(approval["compute_units_per_hour"])
                    ),
                    "resume_note": (
                        "若曾中斷續跑，總 CU 需加總各 Colab session；"
                        "本欄只估算完成這次 notebook session。"
                    ),
                },
                "limitations": [
                    "parse failures and token-limit outputs count as incorrect",
                    "MedQA has no native subject labels, so only total accuracy is reported",
                    "results apply to the pinned revisions and quantized serving stack",
                ],
            }
            print(json.dumps({
                "medqa_models": MEDQA_SUMMARY["models"],
                "tmmlu_models": {
                    model: TMMLU_SUMMARY["models"][model]["overall"]
                    for model in model_labels
                },
                "catastrophic_forgetting": TMMLU_SUMMARY["catastrophic_forgetting"],
                "parse_rate_warnings": parse_warnings,
            }, ensure_ascii=False, indent=2))
            """
        ),
        _markdown("## 10. 封存結果與下載收據"),
        _code(
            r"""
            def write_json(path, value):
                path.write_text(
                    json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8",
                )

            write_json(PUBLIC_ROOT / "phase4-results.json", PHASE4_RESULTS)
            write_json(PUBLIC_ROOT / "medqa-summary.json", MEDQA_SUMMARY)
            write_json(PUBLIC_ROOT / "tmmlu-summary.json", TMMLU_SUMMARY)
            write_json(PUBLIC_ROOT / "stability-summary.json", STABILITY_SUMMARY)
            with (PUBLIC_ROOT / "public-predictions.jsonl").open(
                "w", encoding="utf-8", newline="\n"
            ) as target:
                for row in sorted(
                    all_public_rows,
                    key=lambda item: (item["model"], item["suite"], item["request_id"]),
                ):
                    target.write(canonical_json(row) + "\n")

            private_cases = []
            base_medqa_rows = {
                row["example_id"]: row
                for row in all_public_rows
                if row["model"] == "localized-base" and row["suite"] == "medqa-full"
            }
            adapter_medqa_rows = {
                row["example_id"]: row
                for row in all_public_rows
                if row["model"] == "localized-medical-adapter" and row["suite"] == "medqa-full"
            }
            for case in MEDQA_SUMMARY["representative_cases"]:
                example_id = case["example_id"]
                base_row = base_medqa_rows[example_id]
                adapter_row = adapter_medqa_rows[example_id]
                private_cases.append({
                    "public_case": case,
                    "localized_base": all_private_by_request_model[
                        ("localized-base", base_row["request_id"])
                    ],
                    "localized_medical_adapter": all_private_by_request_model[
                        ("localized-medical-adapter", adapter_row["request_id"])
                    ],
                })
            write_json(PRIVATE_ROOT / "medqa-representative-cases-private.json", private_cases)

            public_archive_base = LOCAL_ROOT / f"{RUN_ID}-phase4-full-public"
            public_archive = Path(
                shutil.make_archive(
                    str(public_archive_base),
                    "zip",
                    root_dir=LOCAL_ROOT,
                    base_dir="public",
                )
            )
            private_cases_archive_base = LOCAL_ROOT / f"{RUN_ID}-phase4-cases-private"
            private_cases_archive = Path(
                shutil.make_archive(
                    str(private_cases_archive_base),
                    "zip",
                    root_dir=PRIVATE_ROOT,
                    base_dir="medqa-representative-cases-private.json",
                )
            )
            final_manifest = {
                "schema_version": 1,
                "phase": 4,
                "run_mode": "full",
                "created_at_utc": datetime.now(UTC).isoformat(),
                "full_evaluation_unlocked": True,
                "user_approval": {
                    "approved_at": approval["approved_at"],
                    "approval_phrase": approval["approval_phrase"],
                    "approved_requests": approval["approved_requests"],
                },
                "contract": CONTRACT,
                "contract_fingerprint": CONTRACT_FINGERPRINT,
                "hardware": hardware_audit,
                "dependencies": dependency_audit,
                "adapter": adapter_audit,
                "data": data_audit,
                "twinkle_eval_contract": {
                    **evaluation_config["twinkle_eval"],
                    "strict_parser": "standalone A-D or exactly one simple boxed A-D",
                    "runner_shuffle_options": False,
                },
                "resumption": {
                    "shard_size": SHARD_SIZE,
                    "completed_shards": len(completed_plans),
                    "completed_requests": len(all_public_rows),
                    "drive_shard_root": str(DRIVE_SHARD_ROOT),
                },
                "session_wall_seconds": session_wall_seconds,
                "server_startup_seconds": server_startup_seconds,
                "cost_accounting": PHASE4_RESULTS["cost_accounting"],
                "public_archive": {
                    "sha256": file_sha256(public_archive),
                    "bytes": public_archive.stat().st_size,
                },
                "private_cases_archive": {
                    "sha256": file_sha256(private_cases_archive),
                    "bytes": private_cases_archive.stat().st_size,
                },
            }
            manifest_path = LOCAL_ROOT / "run_manifest.json"
            write_json(manifest_path, final_manifest)

            drive_public_archive = DRIVE_FINAL_ROOT / public_archive.name
            drive_private_cases = DRIVE_FINAL_ROOT / private_cases_archive.name
            drive_manifest = DRIVE_FINAL_ROOT / f"{RUN_ID}-run-manifest.json"
            atomic_copy_verified(public_archive, drive_public_archive)
            atomic_copy_verified(private_cases_archive, drive_private_cases)
            shutil.copy2(manifest_path, drive_manifest)
            if file_sha256(drive_manifest) != file_sha256(manifest_path):
                raise RuntimeError("Drive run manifest SHA-256 verification failed")

            receipt = {
                "phase": 4,
                "run_mode": "full",
                "contract_fingerprint": CONTRACT_FINGERPRINT,
                "drive_public_archive": str(drive_public_archive),
                "drive_private_cases_archive": str(drive_private_cases),
                "drive_manifest": str(drive_manifest),
                "public_archive_sha256": final_manifest["public_archive"]["sha256"],
                "private_cases_archive_sha256": final_manifest[
                    "private_cases_archive"
                ]["sha256"],
                "completed_requests": len(all_public_rows),
                "raw_shards_remain_private": str(DRIVE_SHARD_ROOT),
            }
            receipt_path = LOCAL_ROOT / "receipt.json"
            write_json(receipt_path, receipt)
            drive_receipt = DRIVE_FINAL_ROOT / f"{RUN_ID}-receipt.json"
            shutil.copy2(receipt_path, drive_receipt)
            if file_sha256(drive_receipt) != file_sha256(receipt_path):
                raise RuntimeError("Drive receipt SHA-256 verification failed")
            receipt["drive_receipt"] = str(drive_receipt)
            print(json.dumps(receipt, ensure_ascii=False, indent=2))
            print(
                "\nPhase 4 正式評估完成。請到上方三個 drive_* 路徑下載："
                "run_manifest.json、receipt.json、phase4-full-public.zip，"
                "再下載 phase4-cases-private.zip 供 10 題私下錯誤分析。"
            )
            """
        ),
    ]
    notebook = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "accelerator": "GPU",
            "colab": {"gpuType": "A100", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
    )
    for index, cell in enumerate(notebook.cells):
        cell.id = f"phase4-full-{index:02d}"
    return notebook


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    generated = nbformat.writes(build_notebook())
    if args.check:
        if not OUTPUT_PATH.is_file() or OUTPUT_PATH.read_text(encoding="utf-8") != generated:
            raise SystemExit(f"{OUTPUT_PATH} is stale; rebuild it")
        return
    OUTPUT_PATH.write_text(generated, encoding="utf-8")


if __name__ == "__main__":
    main()

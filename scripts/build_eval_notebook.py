"""Build the deterministic Phase 4 A100 calibration notebook."""

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
OUTPUT_PATH = ROOT / "notebooks" / "evaluate_phase4.ipynb"
EMBEDDED_HELPERS = (
    "types.py",
    "medqa.py",
    "evaluation.py",
    "tmmlu.py",
    "phase4.py",
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

        NOTEBOOK_BUILD = "phase4-calibration-policy-v4"
        already_loaded = sorted(
            name for name in ("torch", "vllm") if name in sys.modules
        )
        if already_loaded:
            raise RuntimeError(
                "This notebook must start in a fresh Colab runtime before changing the "
                f"CUDA stack; already loaded: {{already_loaded}}. Use Runtime > "
                "Disconnect and delete runtime, reconnect to A100, then Run all."
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
            parse_mcq_answer,
        )
        from tw_med_qlora.phase4 import (
            build_vllm_serve_command,
            extract_verified_adapter,
            phase4_workload,
            project_evaluation_cost,
        )
        from tw_med_qlora.tmmlu import (
            read_tmmlu_csv,
            shuffle_options,
            stratified_calibration_sample,
            write_twinkle_dataset,
        )
        print("Repository-tested Phase 4 helpers loaded.")
        """
    ).strip()


def build_notebook() -> nbformat.NotebookNode:
    with CONFIG_PATH.open("rb") as config_file:
        project_config = tomllib.load(config_file)
    config_json = json.dumps(
        project_config,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    cells = [
        _markdown(
            r"""
            # tw-med-llm-qlora — Phase 4 A100 評估校準（CUDA 12.9 修正版）

            這份 notebook **只執行 20 題 TMMLU+ validation 校準**，不讀取 test 題目，
            也不會解鎖完整 28,758 次生成。它會用完全相同的 20 題依序測試：

            1. 同尺寸原廠 instruct；
            2. 台灣在地化 base；
            3. 同一台灣 base + Phase 3 醫療 adapter。

            A100 上以 vLLM 4-bit OpenAI-compatible server 執行，答案用 Twinkle Eval
            `box` extractor 與 exact-match 檢查。完整題目和原始輸出只封存到私人 Drive ZIP；
            可公開 manifest 只有 hash ID、答案標籤、摘要、耗時與成本估算。

            此修正版固定安裝 vLLM 官方 CUDA 12.9 wheel，並在下載模型前驗證原生 CUDA
            library 可匯入。舊版曾誤裝需要 `libcudart.so.13` 的 PyPI wheel。

            > 研究用途，非醫療建議。請先刪除舊 runtime，再從新 A100 runtime 按
            >「全部執行」，不要從中段開始，也不要沿用已載入舊 CUDA 套件的 session。
            """
        ),
        _markdown(
            """
            ## 1. 安裝鎖定依賴

            本節使用 vLLM 官方 release 的 `cu129` wheel，並交由 uv 選擇相符的 PyTorch
            CUDA 12.9 套件。請務必從乾淨 runtime 執行；不需要手改任何程式碼。
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
            print(json.dumps(dependency_audit, ensure_ascii=False, indent=2))
            if dependency_errors:
                raise RuntimeError(
                    "Phase 4 dependencies are missing or mismatched: "
                    f"{dependency_errors}. Delete the runtime and run this notebook "
                    "from the top; do not repair the environment in place."
                )

            native_probe_code = r'''
            import importlib.metadata
            import json
            import torch
            import vllm

            payload = {
                "vllm_version": importlib.metadata.version("vllm"),
                "torch_version": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            }
            print(json.dumps(payload, sort_keys=True))
            '''
            native_probe = subprocess.run(
                [sys.executable, "-c", native_probe_code],
                capture_output=True,
                text=True,
                timeout=180,
            )
            if native_probe.returncode != 0:
                print(native_probe.stdout)
                print(native_probe.stderr)
                raise RuntimeError(
                    "vLLM native CUDA preflight failed before any model download. "
                    "Delete the runtime, reconnect to A100, and run this corrected "
                    "notebook from the top."
                )
            native_lines = [line for line in native_probe.stdout.splitlines() if line.strip()]
            native_audit = json.loads(native_lines[-1])
            if native_audit["torch_cuda"] != "12.9" or not native_audit["cuda_available"]:
                raise RuntimeError(f"Unexpected CUDA runtime after installation: {native_audit}")
            dependency_audit["native_cuda_preflight"] = native_audit
            print(json.dumps({
                "notebook_build": "phase4-calibration-cu129-v2",
                "vllm_native_import": True,
                **native_audit,
            }, ensure_ascii=False, indent=2))
            """
        ),
        _markdown(
            """
            ## 2. 固定設定、Secrets、Drive 與 A100 硬體閘門

            不需要修改 code。只需在 Colab Secrets 開啟 `HF_TOKEN` 的 notebook 存取權，並使用
            A100 GPU。完整評估硬閘門在這份 notebook 中固定關閉。
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
            import subprocess
            import sys
            import time
            import urllib.error
            import urllib.request
            from datetime import UTC, datetime
            from pathlib import Path

            import torch
            from google.colab import drive, userdata
            from huggingface_hub import snapshot_download

            PROJECT_CONFIG = json.loads(r'''__PROJECT_CONFIG_JSON__''')
            RUN_MODE = "calibration"
            ALLOW_FULL_EVALUATION = False
            FULL_EVALUATION_APPROVAL = None
            REQUIRED_FULL_EVALUATION_APPROVAL = "PHASE4_28758_REQUESTS"
            COMPUTE_UNITS_PER_HOUR = 5.3
            PRICE_PER_COMPUTE_UNIT = None
            CURRENCY_LABEL = None

            if RUN_MODE != "calibration":
                raise RuntimeError("This reviewed notebook permits calibration mode only")
            if ALLOW_FULL_EVALUATION or FULL_EVALUATION_APPROVAL is not None:
                raise RuntimeError(
                    "Full Phase 4 evaluation remains locked until calibration review"
                )

            if not torch.cuda.is_available():
                raise RuntimeError("Phase 4 calibration requires an A100 GPU runtime")
            gpu_name = torch.cuda.get_device_name(0)
            gpu_properties = torch.cuda.get_device_properties(0)
            gpu_vram_gib = gpu_properties.total_memory / 1024**3
            if "A100" not in gpu_name.upper() or gpu_vram_gib < 38:
                raise RuntimeError(
                    f"Expected A100 >=38 GiB; detected {gpu_name} ({gpu_vram_gib:.2f} GiB)"
                )
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("The reviewed Phase 4 profile requires BF16 support")

            HF_TOKEN = userdata.get("HF_TOKEN")
            if not HF_TOKEN:
                raise RuntimeError(
                    "Colab Secret HF_TOKEN is missing or not enabled for this notebook"
                )
            os.environ["HF_TOKEN"] = HF_TOKEN
            os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
            drive.mount("/content/drive")

            RUN_ID = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            RUN_ROOT = Path("/content/tw-med-phase4-calibration") / RUN_ID
            PRIVATE_ROOT = RUN_ROOT / "private"
            PUBLIC_ROOT = RUN_ROOT / "public"
            LOG_ROOT = PRIVATE_ROOT / "server-logs"
            for directory in (PRIVATE_ROOT, PUBLIC_ROOT, LOG_ROOT):
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
            print(json.dumps(hardware_audit, ensure_ascii=False, indent=2))
            """.replace("__PROJECT_CONFIG_JSON__", config_json)
        ),
        _markdown("## 3. 載入 repo 測試過的評估 helper"),
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
            if WORKLOAD.total != int(workload_config["expected_total_requests"]):
                raise RuntimeError("Phase 4 request-count contract changed unexpectedly")
            print(json.dumps(WORKLOAD.as_dict(), ensure_ascii=False, indent=2))
            """
        ),
        _markdown(
            """
            ## 4. 驗證並解開 Phase 3 adapter

            這裡只接受已審核的 Drive ZIP、固定大小與 SHA-256，並檢查
            `adapter_config.json` 的 base model ID。
            """
        ),
        _code(
            r"""
            adapter_contract = evaluation_config["phase3_adapter"]
            adapter_archive = Path(adapter_contract["drive_archive"])
            adapter_audit = extract_verified_adapter(
                adapter_archive,
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
            ## 5. 只下載 TMMLU+ validation，固定抽 20 題

            所有 13 科至少各一題。此 cell 的 allow-pattern 只允許 `_val.csv`；若目錄出現
            `_test.csv` 會立刻停止。
            """
        ),
        _code(
            r"""
            tmmlu_config = PROJECT_CONFIG["data"]["tmmluplus"]
            tmmlu_root = Path("/content/tmmluplus-validation")
            snapshot_path = Path(
                snapshot_download(
                    repo_id=tmmlu_config["dataset_id"],
                    repo_type="dataset",
                    revision=tmmlu_config["revision"],
                    allow_patterns=["data/*_val.csv"],
                    local_dir=tmmlu_root,
                    token=HF_TOKEN,
                )
            )
            forbidden_test_files = list(snapshot_path.rglob("*_test.csv"))
            if forbidden_test_files:
                raise RuntimeError(
                    f"Calibration downloaded forbidden test files: {forbidden_test_files}"
                )

            validation_by_subject = {}
            validation_counts = {}
            for subject in all_subjects:
                path = snapshot_path / "data" / f"{subject}_val.csv"
                rows = read_tmmlu_csv(
                    path,
                    subject=subject,
                    split="validation",
                    source=tmmlu_config["dataset_id"],
                    revision=tmmlu_config["revision"],
                )
                validation_by_subject[subject] = rows
                validation_counts[subject] = len(rows)

            calibration_by_subject = stratified_calibration_sample(
                validation_by_subject,
                total=int(evaluation_config["calibration_examples"]),
                seed=int(PROJECT_CONFIG["project"]["seed"]),
            )
            calibration_manifest = write_twinkle_dataset(
                PRIVATE_ROOT / "tmmlu-calibration",
                calibration_by_subject,
                option_seed=int(evaluation_config["full_shuffle_seed"]),
            )
            if calibration_manifest["total"] != 20:
                raise RuntimeError(
                    "Calibration must contain exactly 20 unique validation questions"
                )
            calibration_ids = [
                item.example.id
                for subject in all_subjects
                for item in calibration_by_subject[subject]
            ]
            if len(set(calibration_ids)) != 20:
                raise RuntimeError("Calibration sample contains duplicate IDs")
            print(json.dumps({
                "split": "validation",
                "subjects": len(all_subjects),
                "rows": calibration_manifest["total"],
                "ordered_ids_sha256": hashlib.sha256(
                    "\n".join(calibration_ids).encode("utf-8")
                ).hexdigest(),
                "validation_counts": validation_counts,
                "test_files_loaded": 0,
            }, ensure_ascii=False, indent=2))
            """
        ),
        _markdown("## 6. Twinkle Eval extractor/scorer 與 vLLM server 工具"),
        _code(
            r"""
            from openai import OpenAI
            from twinkle_eval.metrics.extractors.box import BoxExtractor
            from twinkle_eval.metrics.scorers.exact import ExactMatchScorer

            box_extractor = BoxExtractor()
            exact_scorer = ExactMatchScorer()
            if box_extractor.extract(r"\boxed{A}") != "A":
                raise RuntimeError("Twinkle Eval box extractor contract failed")
            if not exact_scorer.score("A", "A") or exact_scorer.score("A", "B"):
                raise RuntimeError("Twinkle Eval exact scorer contract failed")

            ACTIVE_SERVER = None
            ACTIVE_LOG_HANDLE = None

            def wait_for_server(process, *, port, timeout_seconds=1800):
                started = time.perf_counter()
                last_notice = -30
                while time.perf_counter() - started < timeout_seconds:
                    if process.poll() is not None:
                        raise RuntimeError(
                            f"vLLM server exited with code {process.returncode}; "
                            "inspect its private log"
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
                raise TimeoutError(f"vLLM server did not become healthy within {timeout_seconds}s")

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
                    tail = log_path.read_text(encoding="utf-8", errors="replace")[-6000:]
                    print(tail)
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
            GENERATION_MAX_TOKENS = int(generation_config["max_tokens"])
            MINIMUM_PARSE_RATE = float(
                generation_config["minimum_calibration_parse_rate"]
            )
            TOKEN_LIMIT_HITS_FAIL_CALIBRATION = bool(
                generation_config["token_limit_hits_fail_calibration"]
            )
            if generation_config["token_limit_hits_count_as_incorrect"] is not True:
                raise RuntimeError("Token-limit outputs must remain strict parse failures")
            SYSTEM_PROMPT = (
                "請選擇唯一最佳答案。不要解釋或重述題目；只輸出單一大寫 "
                r"A–D 字母，或一個 LaTeX 答案框，例如 \boxed{A}。"
            )

            def evaluate_one(client, *, model_name, public_label, item):
                started = time.perf_counter()
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": visible_prompt(item)},
                    ],
                    temperature=float(generation_config["temperature"]),
                    top_p=float(generation_config["top_p"]),
                    max_tokens=GENERATION_MAX_TOKENS,
                    seed=int(PROJECT_CONFIG["project"]["seed"]),
                )
                latency = time.perf_counter() - started
                raw_output = response.choices[0].message.content or ""
                prediction = parse_mcq_answer(raw_output)
                twinkle_box_prediction = box_extractor.extract(raw_output)
                if (
                    prediction is not None
                    and r"\boxed" in raw_output
                    and twinkle_box_prediction != prediction
                ):
                    raise RuntimeError(
                        "Strict parser and Twinkle BoxExtractor disagree on a boxed response"
                    )
                if prediction is not None:
                    exact_match = bool(
                        exact_scorer.score(prediction, item.example.answer)
                    )
                    if exact_match != (prediction == item.example.answer):
                        raise RuntimeError("Twinkle exact-match scorer contract drifted")
                usage = response.usage
                public = PredictionRecord(
                    example_id=item.example.id,
                    model=public_label,
                    source=item.example.source,
                    subject=item.subject,
                    gold=item.example.answer,
                    prediction=prediction,
                    raw_output_sha256=hashlib.sha256(raw_output.encode("utf-8")).hexdigest(),
                    latency_seconds=latency,
                    prompt_tokens=getattr(usage, "prompt_tokens", None),
                    completion_tokens=getattr(usage, "completion_tokens", None),
                )
                private = {
                    "example_id": item.example.id,
                    "model": public_label,
                    "subject": item.subject,
                    "gold": item.example.answer,
                    "raw_output": raw_output,
                }
                return public, private

            calibration_items = [
                shuffle_options(
                    item,
                    seed=int(evaluation_config["full_shuffle_seed"]),
                )
                for subject in all_subjects
                for item in calibration_by_subject[subject]
            ]

            def evaluate_model(*, port, served_name, public_label):
                client = OpenAI(api_key="local-eval", base_url=f"http://127.0.0.1:{port}/v1")
                started = time.perf_counter()
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                    futures = [
                        executor.submit(
                            evaluate_one,
                            client,
                            model_name=served_name,
                            public_label=public_label,
                            item=item,
                        )
                        for item in calibration_items
                    ]
                    pairs = [future.result() for future in futures]
                elapsed = time.perf_counter() - started
                public_records = sorted((pair[0] for pair in pairs), key=lambda row: row.example_id)
                private_records = sorted(
                    (pair[1] for pair in pairs), key=lambda row: row["example_id"]
                )
                private_path = PRIVATE_ROOT / f"{public_label}-raw.jsonl"
                with private_path.open("w", encoding="utf-8", newline="\n") as target:
                    for record in private_records:
                        target.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
                        target.write("\n")
                return public_records, elapsed

            print("Twinkle Eval and vLLM calibration helpers are ready.")
            """
        ),
        _markdown(
            """
            ## 7. 校準三個模型

            會啟動兩次 server：原廠模型一次；台灣 base 與 adapter 共用另一次。
            模型首次下載可能需要數分鐘，cell 每 30 秒會印一次等待狀態。
            """
        ),
        _code(
            r"""
            model_config = PROJECT_CONFIG["models"]["primary"]
            vllm_config = evaluation_config["vllm"]
            PORT = 8000
            startup_timings = []
            inference_timings = {}
            public_records_by_model = {}

            original_name = "original-instruct"
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
                startup_timings.append(
                    start_server(original_command, label=original_name, port=PORT)
                )
                records, elapsed = evaluate_model(
                    port=PORT,
                    served_name=original_name,
                    public_label=original_name,
                )
                public_records_by_model[original_name] = records
                inference_timings[original_name] = elapsed
            finally:
                stop_server()

            localized_name = "localized-base"
            adapter_name = "localized-medical-adapter"
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
                startup_timings.append(
                    start_server(localized_command, label="localized-with-adapter", port=PORT)
                )
                for served_name, public_label in (
                    (localized_name, localized_name),
                    (adapter_name, adapter_name),
                ):
                    records, elapsed = evaluate_model(
                        port=PORT,
                        served_name=served_name,
                        public_label=public_label,
                    )
                    public_records_by_model[public_label] = records
                    inference_timings[public_label] = elapsed
            finally:
                stop_server()

            if set(public_records_by_model) != {original_name, localized_name, adapter_name}:
                raise RuntimeError("All three calibration models must complete")
            print(json.dumps({
                "startup_seconds": startup_timings,
                "inference_seconds": inference_timings,
                "requests": sum(len(rows) for rows in public_records_by_model.values()),
            }, ensure_ascii=False, indent=2))
            """
        ),
        _markdown("## 8. 安全摘要、成本預估與 Drive 封存"),
        _code(
            r"""
            calibration_summary = {
                "label": "validation calibration only; not a Phase 4 test result",
                "split": "validation",
                "unique_questions": 20,
                "models": {
                    model_name: {
                        **accuracy_summary(records),
                        "completion_tokens_total": sum(
                            record.completion_tokens or 0 for record in records
                        ),
                        "max_token_limit_hits": sum(
                            record.completion_tokens is not None
                            and record.completion_tokens >= GENERATION_MAX_TOKENS
                            for record in records
                        ),
                    }
                    for model_name, records in public_records_by_model.items()
                },
                "generation_contract": {
                    "parser": "standalone_A-D_or_exactly_one_simple_boxed_A-D",
                    "scorer": exact_scorer.get_name(),
                    "max_tokens": GENERATION_MAX_TOKENS,
                    "minimum_parse_rate": MINIMUM_PARSE_RATE,
                    "token_limit_hits_fail_calibration": (
                        TOKEN_LIMIT_HITS_FAIL_CALIBRATION
                    ),
                    "token_limit_hits_count_as_incorrect": True,
                },
                "inference_seconds": inference_timings,
                "server_startup_seconds": startup_timings,
            }
            parse_gate_failures = {
                model_name: summary["parse_rate"]
                for model_name, summary in calibration_summary["models"].items()
                if summary["parse_rate"] < MINIMUM_PARSE_RATE
            }
            observed_token_limit_hits = {
                model_name: summary["max_token_limit_hits"]
                for model_name, summary in calibration_summary["models"].items()
                if summary["max_token_limit_hits"] > 0
            }
            generation_gate_passed = not (
                parse_gate_failures
                or (
                    TOKEN_LIMIT_HITS_FAIL_CALIBRATION
                    and observed_token_limit_hits
                )
            )
            calibration_summary["generation_gate"] = {
                "passed": generation_gate_passed,
                "parse_rate_failures": parse_gate_failures,
                "observed_max_token_limit_hits": observed_token_limit_hits,
                "max_token_limit_failures": (
                    observed_token_limit_hits
                    if TOKEN_LIMIT_HITS_FAIL_CALIBRATION
                    else {}
                ),
                "failure_action": (
                    None
                    if generation_gate_passed
                    else "Do not unlock full evaluation; review archived evidence."
                ),
            }
            measured_requests = sum(len(rows) for rows in public_records_by_model.values())
            cost_estimate = project_evaluation_cost(
                workload=WORKLOAD,
                measured_requests=measured_requests,
                measured_inference_seconds=sum(inference_timings.values()),
                measured_server_startup_seconds=sum(startup_timings) / len(startup_timings),
                planned_server_starts=2,
                compute_units_per_hour=float(COMPUTE_UNITS_PER_HOUR),
                price_per_compute_unit=PRICE_PER_COMPUTE_UNIT,
                currency=CURRENCY_LABEL,
            )
            public_predictions = {
                model_name: [record.as_public_dict() for record in records]
                for model_name, records in public_records_by_model.items()
            }
            (PUBLIC_ROOT / "calibration_summary.json").write_text(
                json.dumps(
                    calibration_summary, ensure_ascii=False, indent=2, allow_nan=False
                )
                + "\n",
                encoding="utf-8",
            )
            (PUBLIC_ROOT / "public_predictions.json").write_text(
                json.dumps(
                    public_predictions, ensure_ascii=False, indent=2, allow_nan=False
                )
                + "\n",
                encoding="utf-8",
            )
            (PUBLIC_ROOT / "cost_estimate.json").write_text(
                json.dumps(cost_estimate, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            with (PRIVATE_ROOT / "pip-freeze.txt").open("w", encoding="utf-8") as target:
                subprocess.run(
                    [sys.executable, "-m", "pip", "freeze"],
                    check=True,
                    stdout=target,
                )

            private_archive_base = RUN_ROOT / f"{RUN_ID}-phase4-calibration-private"
            private_archive = Path(
                shutil.make_archive(
                    str(private_archive_base),
                    "zip",
                    root_dir=RUN_ROOT,
                    base_dir="private",
                )
            )

            def file_sha256(path):
                digest = hashlib.sha256()
                with path.open("rb") as source:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        digest.update(block)
                return digest.hexdigest()

            manifest = {
                "schema_version": 1,
                "phase": 4,
                "run_mode": RUN_MODE,
                "created_at_utc": datetime.now(UTC).isoformat(),
                "full_evaluation_unlocked": False,
                "test_files_loaded": 0,
                "project_seed": PROJECT_CONFIG["project"]["seed"],
                "hardware": hardware_audit,
                "models": {
                    "original": {
                        "id": model_config["baseline_id"],
                        "revision": model_config["baseline_revision"],
                    },
                    "localized_base": {
                        "id": model_config["model_id"],
                        "revision": model_config["revision"],
                    },
                    "adapter": adapter_audit,
                },
                "data": {
                    "dataset_id": tmmlu_config["dataset_id"],
                    "revision": tmmlu_config["revision"],
                    "split": "validation",
                    "unique_questions": 20,
                    "subject_count": len(all_subjects),
                    "ordered_ids_sha256": hashlib.sha256(
                        "\n".join(calibration_ids).encode("utf-8")
                    ).hexdigest(),
                },
                "dependencies": dependency_audit,
                "twinkle_eval_contract": {
                    "repository": evaluation_config["twinkle_eval"]["repository"],
                    "revision": evaluation_config["twinkle_eval"]["revision"],
                    "extractor": (
                        "strict standalone A-D or exactly one simple boxed A-D; "
                        f"Twinkle audit={box_extractor.get_name()}"
                    ),
                    "scorer": exact_scorer.get_name(),
                    "shuffle_options_inside_runner": False,
                },
                "workload": WORKLOAD.as_dict(),
                "calibration_summary": calibration_summary,
                "cost_estimate": cost_estimate,
                "private_archive": {
                    "sha256": file_sha256(private_archive),
                    "bytes": private_archive.stat().st_size,
                },
            }
            manifest_path = PUBLIC_ROOT / "run_manifest.json"
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )

            drive_root = Path(
                "/content/drive/MyDrive/tw-med-llm-qlora/phase4/calibration/runs"
            )
            drive_root.mkdir(parents=True, exist_ok=True)
            drive_archive = drive_root / private_archive.name
            drive_manifest = drive_root / f"{RUN_ID}-run-manifest.json"
            drive_summary = drive_root / f"{RUN_ID}-calibration-summary.json"
            shutil.copy2(private_archive, drive_archive)
            shutil.copy2(manifest_path, drive_manifest)
            shutil.copy2(PUBLIC_ROOT / "calibration_summary.json", drive_summary)
            if file_sha256(drive_archive) != manifest["private_archive"]["sha256"]:
                raise RuntimeError("Drive private archive SHA-256 verification failed")

            receipt = {
                "phase": 4,
                "run_mode": "calibration",
                "drive_private_archive": str(drive_archive),
                "drive_manifest": str(drive_manifest),
                "drive_calibration_summary": str(drive_summary),
                "archive_sha256": manifest["private_archive"]["sha256"],
                "archive_bytes": manifest["private_archive"]["bytes"],
                "full_evaluation_unlocked": False,
            }
            receipt_path = PUBLIC_ROOT / "receipt.json"
            receipt_path.write_text(
                json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            drive_receipt = drive_root / f"{RUN_ID}-receipt.json"
            shutil.copy2(receipt_path, drive_receipt)

            print(json.dumps(calibration_summary, ensure_ascii=False, indent=2))
            print(json.dumps(cost_estimate, ensure_ascii=False, indent=2))
            print(
                json.dumps(
                    {**receipt, "drive_receipt": str(drive_receipt)},
                    ensure_ascii=False,
                    indent=2,
                )
            )
            if generation_gate_passed:
                print("\nPhase 4 的 20 題 validation 校準完成；生成協定閘門通過。")
            else:
                print(
                    "\nPhase 4 校準檔案已安全封存，但生成協定閘門未通過；"
                    "完整評估仍鎖定。"
                )
            print(
                "請下載 run_manifest.json、receipt.json、calibration_summary.json，"
                "以及 phase4-calibration-private.zip 回傳確認；私人 ZIP 會留在 "
                "gitignored 目錄，不會提交 Git。"
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
        cell.id = f"phase4-{index:02d}"
        if cell.cell_type == "code":
            cell.execution_count = None
            cell.outputs = []
    return notebook


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    generated = nbformat.writes(build_notebook())
    if args.check:
        if not OUTPUT_PATH.exists() or OUTPUT_PATH.read_text(encoding="utf-8") != generated:
            raise SystemExit(f"Generated notebook is stale: {OUTPUT_PATH}")
        return
    OUTPUT_PATH.write_text(generated, encoding="utf-8")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()

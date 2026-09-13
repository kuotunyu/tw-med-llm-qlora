"""Windows RTX 4090 probe CLI for the TMMLU+ v1.1 limited evaluation.

``uv run python -m tw_med_qlora.cli.tmmlu_v1_1_local_eval --run-id <ID> --mode smoke|probe``

Reads ``reports/tmmlu-v1.1/<ID>/mapping.json`` and the pinned v1.1 CSV snapshot, plans the
probe deterministically, generates with Transformers + bitsandbytes NF4, and writes:

* ``reports/tmmlu-v1.1/<ID>/local-inference/<mode>-public-predictions.jsonl`` (content-safe)
* ``reports/private/tmmlu-v1.1/<ID>/<mode>-private-predictions.jsonl`` (gitignored)
* ``reports/tmmlu-v1.1/<ID>/local-inference/<mode>-runtime.json`` and, for ``probe``,
  ``probe-analysis.json``

This is a separate backend experiment; results are labelled and never merged into the
historical Phase 4 rows.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tw_med_qlora.cli.tmmlu_v1_1_report import (
    DEFAULT_V1_1_CONFIG_PATH,
    PROJECT_ROOT,
    _read_json,
    _settings,
)
from tw_med_qlora.local_inference import (
    hardware_preflight,
    inference_requirements,
    inspect_torch_cuda,
    probe_nvidia_gpu,
)
from tw_med_qlora.medqa import file_sha256
from tw_med_qlora.tmmlu_local_eval import (
    EVIDENCE_KIND,
    SYSTEM_PROMPT,
    ProbeItem,
    analyze_probe,
    build_probe_plan,
    model_specs,
    run_model_probe,
    write_jsonl,
)
from tw_med_qlora.tmmlu_rescore import MODEL_LABELS, load_historical_tmmlu_rows
from tw_med_qlora.tmmlu_revision import load_revision_rows
from tw_med_qlora.tmmlu_v1_1_report import resolve_run_dir


def _load_environment() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover
        return
    load_dotenv(PROJECT_ROOT / ".env")


def _smoke_plan(plan: list[ProbeItem]) -> list[ProbeItem]:
    uncovered = [item for item in plan if item.historical_example_id is None]
    covered = [item for item in plan if item.historical_example_id is not None]
    return uncovered[:1] + covered[:1]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _package_versions() -> dict[str, str | None]:
    from importlib import metadata

    versions: dict[str, str | None] = {}
    for name in ("torch", "transformers", "peft", "bitsandbytes", "accelerate", "huggingface-hub"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def main(argv: list[str] | None = None) -> int:
    _load_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", choices=["smoke", "probe"], required=True)
    parser.add_argument("--models", nargs="*", choices=MODEL_LABELS, default=list(MODEL_LABELS))
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "project.toml")
    parser.add_argument("--v1-1-config", type=Path, default=DEFAULT_V1_1_CONFIG_PATH)
    parser.add_argument("--adapter-revision", default=os.getenv("HF_ADAPTER_REVISION") or None)
    args = parser.parse_args(argv)

    settings = _settings(args.config, args.v1_1_config)
    eval_settings = settings["eval"]
    probe_settings = eval_settings["local_probe"]
    run_dir = resolve_run_dir(
        PROJECT_ROOT, report_root=eval_settings["report_root"], run_id=args.run_id
    )
    mapping_path = run_dir / "mapping.json"
    if not mapping_path.is_file():
        raise SystemExit(f"mapping missing; run the report CLI diff first: {mapping_path}")
    mapping = _read_json(mapping_path)
    subjects = settings["medical_subjects"] + settings["control_subjects"]
    data = settings["data"]
    new_rows, new_files = load_revision_rows(
        PROJECT_ROOT / eval_settings["data_root"] / data["revision"],
        version="new",
        revision=data["revision"],
        subjects=subjects,
    )
    for name, info in mapping["files"]["new"].items():
        if new_files[name]["sha256"] != info["sha256"]:
            raise SystemExit(f"v1.1 CSV drifted since the mapping was built: {name}")

    plan = build_probe_plan(
        new_rows,
        mapping,
        subjects=subjects,
        per_subject=int(probe_settings["probe_per_subject"]),
        seed=int(probe_settings["probe_seed"]),
        option_seed=int(eval_settings["option_seed"]),
    )
    if args.mode == "smoke":
        plan = _smoke_plan(plan)
    print(f"planned {len(plan)} questions per model ({args.mode})", flush=True)

    config = settings["config"]
    requirements = inference_requirements(config)
    gpu = probe_nvidia_gpu()
    torch_runtime = inspect_torch_cuda()
    preflight = hardware_preflight(
        gpu, os_name=platform.system(), requirements=requirements, torch_runtime=torch_runtime
    )
    if not preflight["eligible"]:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return 2

    token = os.getenv("HF_TOKEN") or None
    if not token:
        raise SystemExit("HF_TOKEN is required for the gated base models")
    max_new_tokens = int(probe_settings["max_new_tokens"])
    backend_label = str(probe_settings["backend_label"])
    specs = model_specs(config.raw)

    public_dir = run_dir / "local-inference"
    private_dir = PROJECT_ROOT / eval_settings["private_root"] / args.run_id
    public_rows: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    per_model_runtime: dict[str, Any] = {}
    started = datetime.now(UTC)
    for label in args.models:
        rows, private, info = run_model_probe(
            model_label=label,
            spec=specs[label],
            adapter_revision=args.adapter_revision,
            items=plan,
            max_new_tokens=max_new_tokens,
            token=token,
            backend_label=backend_label,
        )
        public_rows.extend(rows)
        private_rows.extend(private)
        per_model_runtime[label] = {**specs[label], **info}
        write_jsonl(public_dir / f"{args.mode}-public-predictions.jsonl", public_rows)
        write_jsonl(private_dir / f"{args.mode}-private-predictions.jsonl", private_rows)
        print(f"[{label}] saved {len(rows)} rows", flush=True)

    runtime = {
        "schema_version": 1,
        "evidence_kind": EVIDENCE_KIND,
        "mode": args.mode,
        "run_id": args.run_id,
        "backend_label": backend_label,
        "started_at_utc": started.isoformat(),
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "planned_questions_per_model": len(plan),
        "plan_reasons": sorted({item.reason for item in plan}),
        "generation": {
            "decoding": "greedy",
            "max_new_tokens": max_new_tokens,
            "token_limit_rule": "completion_tokens >= max_new_tokens -> prediction=None",
            "system_prompt_sha256": __import__("hashlib")
            .sha256(SYSTEM_PROMPT.encode("utf-8"))
            .hexdigest(),
            "option_seed": int(eval_settings["option_seed"]),
        },
        "quantization": {
            "load_in_4bit": True,
            "quant_type": "nf4",
            "compute_dtype": "bfloat16",
            "double_quant": True,
            "attention": "sdpa",
        },
        "hardware": preflight,
        "packages": _package_versions(),
        "python": platform.python_version(),
        "models": per_model_runtime,
        "dataset": mapping["dataset"],
        "outputs": {
            "public_predictions": str(
                (public_dir / f"{args.mode}-public-predictions.jsonl").relative_to(PROJECT_ROOT)
            ),
            "public_predictions_sha256": file_sha256(
                public_dir / f"{args.mode}-public-predictions.jsonl"
            ),
            "private_predictions": str(
                (private_dir / f"{args.mode}-private-predictions.jsonl").relative_to(PROJECT_ROOT)
            ),
            "private_predictions_sha256": file_sha256(
                private_dir / f"{args.mode}-private-predictions.jsonl"
            ),
        },
    }
    _write_json(public_dir / f"{args.mode}-runtime.json", runtime)

    if args.mode == "probe":
        historical_rows = _all_historical_rows(eval_settings)
        analysis = analyze_probe(
            public_rows, historical_rows, backend_label=backend_label, runtime=runtime
        )
        _write_json(public_dir / "probe-analysis.json", analysis)
        print(
            json.dumps(
                {
                    m: v["prediction_agreement_with_historical"]
                    for m, v in analysis["models"].items()
                },
                ensure_ascii=False,
            )
        )
    return 0


def _all_historical_rows(eval_settings: dict[str, Any]) -> list[dict[str, Any]]:
    import zipfile

    archive = PROJECT_ROOT / eval_settings["historical_public_archive"]
    # Reuse the verified loader for the SHA check, then read every suite for the ceiling.
    load_historical_tmmlu_rows(
        archive,
        expected_sha256=eval_settings["historical_public_archive_sha256"],
        expected_rows_per_model=int(eval_settings["expected_v1_0_test_rows"]),
    )
    with zipfile.ZipFile(archive) as handle:
        lines = handle.read("public/public-predictions.jsonl").decode("utf-8").splitlines()
    return [json.loads(line) for line in lines if line.startswith("{")]


if __name__ == "__main__":
    sys.exit(main())

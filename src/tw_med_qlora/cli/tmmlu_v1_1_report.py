"""CLI for the TMMLU+ v1.1 limited evaluation: diff, rescore, render, manifest.

Run with ``uv run python -m tw_med_qlora.cli.tmmlu_v1_1_report <command> --run-id <ID>``.
All outputs live under ``reports/tmmlu-v1.1/<ID>/``; the frozen Phase 4 evidence is read
only and never written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tw_med_qlora.config import load_project_config
from tw_med_qlora.medqa import file_sha256
from tw_med_qlora.phase4_full import canonical_json
from tw_med_qlora.tmmlu_rescore import analyze, load_historical_tmmlu_rows
from tw_med_qlora.tmmlu_revision import (
    build_revision_mapping,
    fetch_subject_tests,
    load_revision_rows,
    mapping_sha256,
    verify_tag_revision,
)
from tw_med_qlora.tmmlu_v1_1_report import (
    extract_generated_section,
    inject_generated_section,
    render_report,
    resolve_run_dir,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "project.toml"
DEFAULT_V1_1_CONFIG_PATH = PROJECT_ROOT / "configs" / "tmmlu_v1_1.toml"
DEFAULT_DOC_PATH = PROJECT_ROOT / "docs" / "tmmlu-v1.1-limited-evaluation.md"
MAPPING_NAME = "mapping.json"
ANALYSIS_NAME = "analysis.json"
REPORT_NAME = "report.md"
MANIFEST_NAME = "run-manifest.json"
PROBE_ANALYSIS_RELATIVE = Path("local-inference") / "probe-analysis.json"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(rendered, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _settings(config_path: Path, v1_1_config_path: Path | None = None) -> dict[str, Any]:
    """Merge the frozen project config with the separate v1.1 evaluation config."""

    config = load_project_config(config_path)
    raw = config.raw
    evaluation = raw["evaluation"]
    v1_1_path = v1_1_config_path or DEFAULT_V1_1_CONFIG_PATH
    v1_1 = tomllib.loads(v1_1_path.read_text(encoding="utf-8"))
    if v1_1["data"]["baseline_revision"] != raw["data"]["tmmluplus"]["revision"]:
        raise ValueError("v1.1 baseline_revision must equal the frozen [data.tmmluplus].revision")
    return {
        "config": config,
        "data": v1_1["data"],
        "eval": v1_1["evaluation"],
        "medical_subjects": list(evaluation["medical_subjects"]),
        "control_subjects": list(evaluation["control_subjects"]),
        "bootstrap_iterations": int(evaluation["bootstrap_iterations"]),
        "margin": float(evaluation["forgetting_margin_percentage_points"]),
        "v1_1_config_path": v1_1_path,
    }


def _load_environment() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - optional dependency
        return
    load_dotenv(PROJECT_ROOT / ".env")


def command_diff(args: argparse.Namespace, settings: dict[str, Any], run_dir: Path) -> Path:
    data = settings["data"]
    subjects = settings["medical_subjects"] + settings["control_subjects"]
    token = os.getenv("HF_TOKEN") or None
    refs = verify_tag_revision(
        dataset_id=data["dataset_id"],
        tag=data["tag"],
        expected_revision=data["revision"],
        token=token,
    )
    data_root = PROJECT_ROOT / settings["eval"]["data_root"]
    snapshots = {}
    for label, revision in (("old", data["baseline_revision"]), ("new", data["revision"])):
        snapshots[label] = fetch_subject_tests(
            dataset_id=data["dataset_id"],
            revision=revision,
            subjects=subjects,
            root=data_root,
            token=token,
        )
    old_rows, old_files = load_revision_rows(
        snapshots["old"], version="old", revision=data["baseline_revision"], subjects=subjects
    )
    new_rows, new_files = load_revision_rows(
        snapshots["new"], version="new", revision=data["revision"], subjects=subjects
    )
    mapping = build_revision_mapping(old_rows, new_rows)
    expected_old = int(settings["eval"]["expected_v1_0_test_rows"])
    if mapping["counts"]["old"]["total"] != expected_old:
        raise RuntimeError(
            f"v1.0 row count {mapping['counts']['old']['total']} != expected {expected_old}"
        )
    payload = {
        "dataset": {
            "dataset_id": data["dataset_id"],
            "split": "test",
            "old_revision": data["baseline_revision"],
            "new_revision": data["revision"],
            "new_tag": data["tag"],
            "hub_refs_at_run": refs,
            "subjects": subjects,
        },
        "files": {"old": old_files, "new": new_files},
        **mapping,
    }
    path = run_dir / MAPPING_NAME
    _write_json(path, payload)
    print(json.dumps(payload["counts"]["old"] | {"by_subject": "..."}, ensure_ascii=False))
    print(json.dumps(payload["counts"]["new"] | {"by_subject": "..."}, ensure_ascii=False))
    print(f"mapping written: {path}")
    return path


def command_rescore(args: argparse.Namespace, settings: dict[str, Any], run_dir: Path) -> Path:
    eval_settings = settings["eval"]
    mapping = _read_json(run_dir / MAPPING_NAME)
    archive = PROJECT_ROOT / eval_settings["historical_public_archive"]
    rows, seed = load_historical_tmmlu_rows(
        archive,
        expected_sha256=eval_settings["historical_public_archive_sha256"],
        expected_rows_per_model=int(eval_settings["expected_v1_0_test_rows"]),
    )
    if seed != int(eval_settings["option_seed"]):
        raise RuntimeError(
            f"historical option seed {seed} != configured {eval_settings['option_seed']}"
        )
    frozen = _read_json(
        PROJECT_ROOT / "reports" / "phase4" / "full" / "public" / "tmmlu-summary.json"
    )
    analysis = analyze(
        rows,
        mapping,
        seed=seed,
        medical_subjects=settings["medical_subjects"],
        control_subjects=settings["control_subjects"],
        iterations=settings["bootstrap_iterations"],
        margin_percentage_points=settings["margin"],
        frozen_tmmlu_summary=frozen,
    )
    if analysis["frozen_reproduction_check"]["status"] != "passed":
        raise RuntimeError(
            "v1_0_full did not reproduce the frozen summary: "
            f"{analysis['frozen_reproduction_check']['mismatches'][:5]}"
        )
    analysis["inputs"] = {
        "historical_public_archive": eval_settings["historical_public_archive"],
        "historical_public_archive_sha256": eval_settings["historical_public_archive_sha256"],
        "historical_run_id": eval_settings["historical_run_id"],
        "historical_contract_fingerprint": eval_settings["historical_contract_fingerprint"],
        "mapping_sha256": mapping_sha256(mapping),
        "old_revision": mapping["dataset"]["old_revision"],
        "new_revision": mapping["dataset"]["new_revision"],
    }
    path = run_dir / ANALYSIS_NAME
    _write_json(path, analysis)
    print(f"analysis written: {path}")
    return path


def command_render(args: argparse.Namespace, settings: dict[str, Any], run_dir: Path) -> Path:
    analysis = _read_json(run_dir / ANALYSIS_NAME)
    probe_path = run_dir / PROBE_ANALYSIS_RELATIVE
    probe = _read_json(probe_path) if probe_path.is_file() else None
    generated = render_report(
        analysis,
        medical_subjects=settings["medical_subjects"],
        control_subjects=settings["control_subjects"],
        probe=probe,
        run_id=args.run_id,
    )
    report_path = run_dir / REPORT_NAME
    report_path.write_text(generated, encoding="utf-8", newline="\n")
    print(f"report written: {report_path}")
    if args.doc is not None:
        document = args.doc.read_text(encoding="utf-8")
        updated = inject_generated_section(document, generated)
        args.doc.write_text(updated, encoding="utf-8", newline="\n")
        if extract_generated_section(updated) != generated:
            raise RuntimeError("generated section round trip failed")
        print(f"document updated: {args.doc}")
    return report_path


def _git(*arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *arguments], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _package_versions() -> dict[str, str | None]:
    from importlib import metadata

    versions: dict[str, str | None] = {}
    for name in ("huggingface-hub", "pytest", "ruff"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def command_manifest(args: argparse.Namespace, settings: dict[str, Any], run_dir: Path) -> Path:
    from tw_med_qlora.tmmlu_local_eval import SYSTEM_PROMPT, visible_prompt_template

    mapping = _read_json(run_dir / MAPPING_NAME)
    analysis = _read_json(run_dir / ANALYSIS_NAME)
    artifacts = {}
    for name in (MAPPING_NAME, ANALYSIS_NAME, REPORT_NAME):
        path = run_dir / name
        artifacts[name] = {"sha256": file_sha256(path), "bytes": path.stat().st_size}
    probe_path = run_dir / PROBE_ANALYSIS_RELATIVE
    if probe_path.is_file():
        artifacts[PROBE_ANALYSIS_RELATIVE.as_posix()] = {
            "sha256": file_sha256(probe_path),
            "bytes": probe_path.stat().st_size,
        }
    eval_settings = settings["eval"]
    manifest = {
        "schema_version": 1,
        "run_id": args.run_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "purpose": "TMMLU+ v1.1 limited evaluation: historical outputs rescored, "
        "plus a separately labelled local probe when present",
        "evidence_kinds": sorted(
            {analysis["evidence_kind"]}
            | ({_read_json(probe_path)["evidence_kind"]} if probe_path.is_file() else set())
        ),
        "code": {
            "git_head": _git("rev-parse", "HEAD"),
            "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "git_dirty_paths": [
                line for line in (_git("status", "--porcelain") or "").splitlines() if line
            ],
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": _package_versions(),
        },
        "dataset": mapping["dataset"],
        "dataset_files": mapping["files"],
        "historical_evidence": analysis["inputs"],
        "prompt_contract": {
            "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
            "user_prompt_template": visible_prompt_template(),
            "option_seed": analysis["option_seed"],
            "few_shot": False,
            "parser": "standalone_A-D_or_exactly_one_simple_boxed_A-D",
            "token_limit_hits_count_as_incorrect": True,
        },
        "statistics": analysis["statistics"],
        "coverage": analysis["coverage"],
        "frozen_reproduction_check": analysis["frozen_reproduction_check"],
        "artifacts": artifacts,
        "config_snapshot": {
            "configs/tmmlu_v1_1.toml": {
                "sha256": file_sha256(settings["v1_1_config_path"]),
                "data": settings["data"],
                "evaluation": eval_settings,
            },
            "configs/project.toml": {
                "sha256": file_sha256(args.config),
                "evaluation.medical_subjects": settings["medical_subjects"],
                "evaluation.control_subjects": settings["control_subjects"],
                "evaluation.bootstrap_iterations": settings["bootstrap_iterations"],
                "evaluation.forgetting_margin_percentage_points": settings["margin"],
                "evaluation.full_shuffle_seed": settings["config"].raw["evaluation"][
                    "full_shuffle_seed"
                ],
            },
        },
    }
    path = run_dir / MANIFEST_NAME
    _write_json(path, manifest)
    print(f"manifest written: {path}")
    print(canonical_json({"artifacts": artifacts}))
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["diff", "rescore", "render", "manifest", "all"])
    parser.add_argument("--run-id", required=True, help="UTC run id, e.g. 20260914T120000Z")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--v1-1-config", type=Path, default=DEFAULT_V1_1_CONFIG_PATH)
    parser.add_argument(
        "--doc",
        type=Path,
        default=DEFAULT_DOC_PATH,
        help="Markdown document whose generated section is refreshed by render",
    )
    parser.add_argument("--no-doc", action="store_true", help="render report.md only")
    return parser


def main(argv: list[str] | None = None) -> int:
    _load_environment()
    args = build_parser().parse_args(argv)
    if args.no_doc:
        args.doc = None
    settings = _settings(args.config, args.v1_1_config)
    run_dir = resolve_run_dir(
        PROJECT_ROOT, report_root=settings["eval"]["report_root"], run_id=args.run_id
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    commands = {
        "diff": command_diff,
        "rescore": command_rescore,
        "render": command_render,
        "manifest": command_manifest,
    }
    order = ["diff", "rescore", "render", "manifest"] if args.command == "all" else [args.command]
    for name in order:
        commands[name](args, settings, run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

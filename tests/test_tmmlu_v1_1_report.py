from __future__ import annotations

import json
from pathlib import Path

import pytest

from tw_med_qlora.tmmlu_v1_1_report import (
    BEGIN_MARKER,
    END_MARKER,
    extract_generated_section,
    inject_generated_section,
    render_report,
    resolve_run_dir,
)

ROOT = Path(__file__).parents[1]
DOC_PATH = ROOT / "docs" / "tmmlu-v1.1-limited-evaluation.md"
REPORT_ROOT = ROOT / "reports" / "tmmlu-v1.1"


def _config_subjects() -> tuple[list[str], list[str]]:
    import tomllib

    config = tomllib.loads((ROOT / "configs" / "project.toml").read_text(encoding="utf-8"))
    evaluation = config["evaluation"]
    return list(evaluation["medical_subjects"]), list(evaluation["control_subjects"])


def _committed_runs() -> list[Path]:
    if not REPORT_ROOT.is_dir():
        return []
    return sorted(path for path in REPORT_ROOT.iterdir() if (path / "analysis.json").is_file())


def test_inject_and_extract_round_trip() -> None:
    document = f"# 標題\n\n{BEGIN_MARKER}\n舊內容\n{END_MARKER}\n\n尾段\n"
    generated = "| a |\n|---|\n| 1 |\n"

    updated = inject_generated_section(document, generated)

    assert updated.startswith("# 標題")
    assert updated.endswith("尾段\n")
    assert "舊內容" not in updated
    assert extract_generated_section(updated) == generated
    with pytest.raises(ValueError, match="generated section"):
        inject_generated_section("no markers", generated)


def test_resolve_run_dir_refuses_frozen_and_foreign_paths() -> None:
    run_dir = resolve_run_dir(ROOT, report_root="reports/tmmlu-v1.1", run_id="20260914T000000Z")

    assert run_dir == (REPORT_ROOT / "20260914T000000Z").resolve()
    with pytest.raises(ValueError, match="report_root"):
        resolve_run_dir(ROOT, report_root="reports/phase4", run_id="20260914T000000Z")
    with pytest.raises(ValueError, match="run_id"):
        resolve_run_dir(ROOT, report_root="reports/tmmlu-v1.1", run_id="../phase4")
    with pytest.raises(ValueError, match="run_id"):
        resolve_run_dir(ROOT, report_root="reports/tmmlu-v1.1", run_id="latest")


@pytest.mark.skipif(not _committed_runs(), reason="no committed v1.1 run yet")
def test_committed_document_tables_match_saved_evidence() -> None:
    """The document's generated block must equal a fresh render of the saved analysis."""

    medical, control = _config_subjects()
    run_dir = _committed_runs()[-1]
    analysis = json.loads((run_dir / "analysis.json").read_text(encoding="utf-8"))
    probe_path = run_dir / "local-inference" / "probe-analysis.json"
    probe = json.loads(probe_path.read_text(encoding="utf-8")) if probe_path.is_file() else None

    generated = render_report(
        analysis,
        medical_subjects=medical,
        control_subjects=control,
        probe=probe,
        run_id=run_dir.name,
    )

    assert (run_dir / "report.md").read_text(encoding="utf-8") == generated
    document = DOC_PATH.read_text(encoding="utf-8")
    assert extract_generated_section(document) == generated
    assert analysis["frozen_reproduction_check"]["status"] == "passed"
    assert analysis["evidence_kind"] == "historical_outputs_rescored_under_v1_1"
    manifest = json.loads((run_dir / "run-manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset"]["new_revision"] == "94d86f1d013d59f389b710a902af30b7a0aa8c8f"
    assert manifest["dataset"]["old_revision"] == "81d53e38340c9ade988f7fed8996da6554b504f3"
    assert manifest["artifacts"]["analysis.json"]["sha256"]


@pytest.mark.skipif(not _committed_runs(), reason="no committed v1.1 run yet")
def test_committed_run_keeps_frozen_evidence_untouched() -> None:
    frozen_summary = json.loads(
        (ROOT / "reports" / "phase4" / "full" / "public" / "tmmlu-summary.json").read_text(
            encoding="utf-8"
        )
    )
    run_dir = _committed_runs()[-1]
    analysis = json.loads((run_dir / "analysis.json").read_text(encoding="utf-8"))
    full = analysis["variants"]["v1_0_full"]["models"]
    for model, summary in frozen_summary["models"].items():
        assert full[model]["overall"] == summary["overall"]
    archive = ROOT / "reports" / "phase4" / "full" / "20260722T070936Z-phase4-full-public.zip"
    assert archive.is_file()

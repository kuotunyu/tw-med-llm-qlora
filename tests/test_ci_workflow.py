from pathlib import Path

ROOT = Path(__file__).parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
GIT_ATTRIBUTES = ROOT / ".gitattributes"


def workflow_text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_ci_is_read_only_and_uses_no_secrets() -> None:
    text = workflow_text()

    assert "permissions:\n  contents: read" in text
    assert "pull_request_target" not in text
    assert "secrets." not in text
    assert "id-token: write" not in text
    assert "upload" not in text.casefold()


def test_ci_pins_uv_and_covers_windows_and_linux() -> None:
    text = workflow_text()

    assert "actions/checkout@v7" in text
    assert "astral-sh/setup-uv@08807647e7069bb48b6ef5acd8ec9567f424441b" in text
    assert 'version: "0.11.30"' in text
    assert 'python-version: "3.11"' in text
    assert "ubuntu-latest" in text
    assert "windows-latest" in text


def test_ci_runs_locked_quality_and_notebook_checks() -> None:
    text = workflow_text()

    assert "uv sync --locked --group dev" in text
    assert "uv run ruff check ." in text
    assert "uv run pytest -q" in text
    assert (
        "uv build --no-sources --clear --out-dir outputs/ci-dist --no-create-gitignore"
        in text
    )
    assert "uv run python scripts/audit_release_artifacts.py outputs/ci-dist" in text
    assert (
        "uv run python scripts/smoke_release_install.py outputs/ci-dist "
        "--venv outputs/ci-release-venv"
    ) in text
    for builder in (
        "build_train_notebook.py",
        "build_eval_notebook.py",
        "build_full_eval_notebook.py",
        "build_export_notebook.py",
    ):
        assert f"uv run python scripts/{builder} --check" in text


def test_hash_verified_reports_keep_lf_on_windows_checkout() -> None:
    attributes = GIT_ATTRIBUTES.read_text(encoding="utf-8")

    assert "reports/**/*.json text eol=lf" in attributes
    assert "reports/**/*.csv text eol=lf" in attributes

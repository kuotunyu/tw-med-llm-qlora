import json
import os
import re
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).parents[1]
SKIPPED_DIRECTORIES = {
    ".codex",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "artifacts",
    "outputs",
    "private",
}
LOCAL_ONLY_PATHS = {
    Path("AGENTS.md"),
    Path("PROJECT_PLAN.md"),
    Path("tests/test_project_skill.py"),
}
TEXT_SUFFIXES = {
    ".cff",
    ".ipynb",
    ".jinja",
    ".json",
    ".md",
    ".ps1",
    ".py",
    ".toml",
    ".yaml",
    ".yml",
}
TEXT_FILE_NAMES = {
    ".editorconfig",
    ".env.example",
    ".gitattributes",
    ".gitignore",
    ".python-version",
    "LICENSE",
}
SECRET_PATTERNS = {
    "Hugging Face token": re.compile(r"hf_[A-Za-z0-9]{20,}"),
    "OpenAI key": re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    "Google key": re.compile(r"AIza[0-9A-Za-z_-]{30,}"),
    "Discord webhook": re.compile(
        r"discord(?:app)?\.com/api/webhooks/[0-9]+/[A-Za-z0-9_-]+"
    ),
}
FORBIDDEN_REPORT_KEYS = {
    "GOOGLE_API_KEY",
    "HF_TOKEN",
    "OPENAI_API_KEY",
    "answer_text",
    "choices",
    "prompt",
    "question",
    "raw_output",
}


def public_files(
    *,
    suffixes: set[str] | None = None,
    names: set[str] | None = None,
) -> list[Path]:
    files: list[Path] = []
    for directory, child_directories, filenames in os.walk(ROOT):
        child_directories[:] = [
            name for name in child_directories if name not in SKIPPED_DIRECTORIES
        ]
        for filename in filenames:
            path = Path(directory) / filename
            relative = path.relative_to(ROOT)
            if relative in LOCAL_ONLY_PATHS:
                continue
            if path.name == ".env":
                continue
            if (
                (suffixes is None and names is None)
                or (suffixes is not None and path.suffix.lower() in suffixes)
                or (names is not None and path.name in names)
            ):
                files.append(path)
    return files


def nested_keys(value: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            keys.add(str(key))
            keys.update(nested_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(nested_keys(child))
    return keys


def test_public_markdown_local_links_resolve() -> None:
    link_pattern = re.compile(r"!?\[[^\]]*]\(([^)]+)\)")
    broken: list[str] = []
    markdown_files = public_files(suffixes={".md"})

    for markdown in markdown_files:
        text = markdown.read_text(encoding="utf-8")
        for match in link_pattern.finditer(text):
            target = match.group(1).strip().strip("<>")
            if target.startswith(("http://", "https://", "mailto:", "#", "{{")):
                continue
            path_text = unquote(target.split("#", maxsplit=1)[0])
            if path_text and not (markdown.parent / path_text).exists():
                relative = markdown.relative_to(ROOT).as_posix()
                broken.append(f"{relative}: {target}")

    assert markdown_files
    assert not broken, "\n".join(broken)


def test_readme_contains_required_public_sections() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "## 個人動機" not in readme
    assert "Windows RTX 4090 實機驗收" not in readme
    assert "```mermaid" in readme
    assert "## 模型選型" in readme
    assert "## 快速開始" in readme
    assert "## 評估結果" in readme
    assert "## 訓練曲線與成本" in readme
    assert "## 引用" in readme
    assert "## 授權" in readme


def test_public_release_metadata_is_present_and_consistent() -> None:
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")

    assert "cff-version: 1.2.0" in citation
    assert 'version: "0.2.0"' in citation
    assert 'date-released: "2026-07-24"' in citation
    assert "https://github.com/kuotunyu/tw-med-llm-qlora" in citation
    assert "## [0.2.0] - 2026-07-24" in changelog
    assert "## [0.1.0] - 2026-07-23" in changelog


def test_public_text_files_do_not_contain_live_secret_shapes() -> None:
    findings: list[str] = []
    for path in public_files(suffixes=TEXT_SUFFIXES, names=TEXT_FILE_NAMES):
        text = path.read_text(encoding="utf-8")
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                relative = path.relative_to(ROOT).as_posix()
                findings.append(f"{label}: {relative}")

    assert not findings, "\n".join(findings)


def test_public_snapshot_has_no_oversized_files() -> None:
    limit = 50 * 1024 * 1024
    forbidden_model_suffixes = {".bin", ".gguf", ".pt", ".pth", ".safetensors"}
    public = public_files()
    oversized = [
        f"{path.relative_to(ROOT).as_posix()}: {path.stat().st_size}"
        for path in public
        if path.stat().st_size > limit
    ]
    model_artifacts = [
        path.relative_to(ROOT).as_posix()
        for path in public
        if path.suffix.lower() in forbidden_model_suffixes
    ]

    assert not oversized, "\n".join(oversized)
    assert not model_artifacts, "\n".join(model_artifacts)


def test_public_reports_are_parseable_and_exclude_private_content_keys() -> None:
    report_files = sorted((ROOT / "reports").rglob("*.json"))
    findings: list[str] = []

    for path in report_files:
        report = json.loads(path.read_text(encoding="utf-8"))
        forbidden = nested_keys(report) & FORBIDDEN_REPORT_KEYS
        if forbidden:
            relative = path.relative_to(ROOT).as_posix()
            findings.append(f"{relative}: {sorted(forbidden)}")

    assert report_files
    assert not findings, "\n".join(findings)


def test_ignore_rules_cover_secrets_and_large_model_artifacts() -> None:
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()

    assert ".env" in ignore
    assert ".env.*" in ignore
    assert "!.env.example" in ignore
    assert "artifacts/" in ignore
    assert "reports/private/" in ignore
    assert "AGENTS.md" in ignore
    assert "PROJECT_PLAN.md" in ignore
    assert ".codex/" in ignore
    assert "tests/test_project_skill.py" in ignore
    assert "*.safetensors" in ignore
    assert "*.gguf" in ignore

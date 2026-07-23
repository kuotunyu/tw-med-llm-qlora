import importlib.util
import io
import tarfile
import tomllib
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "audit_release_artifacts.py"
SPEC = importlib.util.spec_from_file_location("audit_release_artifacts", SCRIPT)
assert SPEC and SPEC.loader
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)

DIST_INFO = f"{AUDIT.RELEASE_ROOT}.dist-info"
SDIST_ROOT = AUDIT.RELEASE_ROOT
METADATA = """\
Metadata-Version: 2.4
Name: tw-med-llm-qlora
Version: 0.2.0
Requires-Python: >=3.11,<3.12
License-Expression: MIT
License-File: LICENSE
Project-URL: Changelog, https://github.com/kuotunyu/tw-med-llm-qlora/blob/main/CHANGELOG.md
Project-URL: Homepage, https://github.com/kuotunyu/tw-med-llm-qlora
Project-URL: Issues, https://github.com/kuotunyu/tw-med-llm-qlora/issues
Project-URL: Model, https://huggingface.co/steven0226/tw-med-llm-qlora-adapter

"""
ENTRY_POINTS = """\
[console_scripts]
tw-med-local-infer = tw_med_qlora.local_inference:main
tw-med-phase5-status = tw_med_qlora.cli.phase5_status:main
tw-med-publish-adapter = tw_med_qlora.cli.publish_adapter:main
tw-med-validate-phase5 = tw_med_qlora.cli.validate_phase5_evidence:main
tw-med-verify-public-adapter = tw_med_qlora.cli.verify_public_adapter:main
"""


def add_tar_bytes(archive: tarfile.TarFile, name: str, payload: bytes = b"x") -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    archive.addfile(info, io.BytesIO(payload))


def write_valid_archives(directory: Path, *, extra_sdist: str | None = None) -> None:
    wheel = directory / f"{AUDIT.RELEASE_ROOT}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, mode="w") as archive:
        archive.writestr("tw_med_qlora/__init__.py", "")
        archive.writestr(f"{DIST_INFO}/METADATA", METADATA)
        archive.writestr(f"{DIST_INFO}/entry_points.txt", ENTRY_POINTS)
        archive.writestr(f"{DIST_INFO}/licenses/LICENSE", "MIT")

    sdist = directory / f"{AUDIT.RELEASE_ROOT}.tar.gz"
    with tarfile.open(sdist, mode="w:gz") as archive:
        add_tar_bytes(archive, f"{SDIST_ROOT}/LICENSE", b"MIT")
        add_tar_bytes(archive, f"{SDIST_ROOT}/CHANGELOG.md")
        add_tar_bytes(archive, f"{SDIST_ROOT}/CITATION.cff")
        add_tar_bytes(archive, f"{SDIST_ROOT}/MANIFEST.in")
        add_tar_bytes(archive, f"{SDIST_ROOT}/PKG-INFO", METADATA.encode())
        add_tar_bytes(archive, f"{SDIST_ROOT}/README.md")
        add_tar_bytes(archive, f"{SDIST_ROOT}/pyproject.toml")
        add_tar_bytes(archive, f"{SDIST_ROOT}/src/tw_med_qlora/__init__.py")
        if extra_sdist:
            add_tar_bytes(archive, f"{SDIST_ROOT}/{extra_sdist}")


def test_release_metadata_and_manifest_use_current_contract() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8").splitlines()

    assert project["build-system"]["requires"] == ["setuptools>=77"]
    assert project["project"]["version"] == "0.2.0"
    assert project["project"]["license"] == "MIT"
    assert project["project"]["license-files"] == ["LICENSE"]
    assert project["project"]["scripts"] == AUDIT.EXPECTED_SCRIPTS
    assert project["project"]["urls"] == AUDIT.EXPECTED_PROJECT_URLS
    for name in ("CHANGELOG.md", "CITATION.cff"):
        assert f"include {name}" in manifest
    for directory in ("tests", "reports", "notebooks", "scripts", ".codex"):
        assert f"prune {directory}" in manifest
    for name in ("AGENTS.md", "PROJECT_PLAN.md", ".env"):
        assert f"exclude {name}" in manifest


def test_release_artifact_auditor_accepts_minimal_contract(tmp_path: Path) -> None:
    write_valid_archives(tmp_path)

    result = AUDIT.audit_directory(tmp_path)

    assert result["release"] == "v0.2.0"
    assert result["forbidden_content_absent"] is True
    assert result["wheel"]["console_scripts"] == AUDIT.EXPECTED_SCRIPTS


def test_release_metadata_accepts_normalized_python_specifier_order() -> None:
    normalized = METADATA.replace(">=3.11,<3.12", "<3.12,>=3.11")

    result = AUDIT.parse_metadata(normalized)

    assert result["requires_python"] == "<3.12,>=3.11"


@pytest.mark.parametrize(
    "forbidden",
    [
        "tests/test_project_skill.py",
        "reports/private/run.json",
        "artifacts/adapter_model.safetensors",
        "PROJECT_PLAN.md",
    ],
)
def test_release_artifact_auditor_rejects_private_or_large_content(
    tmp_path: Path,
    forbidden: str,
) -> None:
    write_valid_archives(tmp_path, extra_sdist=forbidden)

    with pytest.raises(AUDIT.ReleaseAuditError, match="forbidden"):
        AUDIT.audit_directory(tmp_path)

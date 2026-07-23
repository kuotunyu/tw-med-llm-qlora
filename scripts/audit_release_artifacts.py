"""Audit one wheel and one source distribution without extracting either archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import zipfile
from email.parser import Parser
from pathlib import Path, PurePosixPath
from typing import Any

EXPECTED_NAME = "tw-med-llm-qlora"
EXPECTED_VERSION = "0.2.0"
NORMALIZED_NAME = EXPECTED_NAME.replace("-", "_")
RELEASE_ROOT = f"{NORMALIZED_NAME}-{EXPECTED_VERSION}"
EXPECTED_SCRIPTS = {
    "tw-med-local-infer": "tw_med_qlora.local_inference:main",
    "tw-med-phase5-status": "tw_med_qlora.cli.phase5_status:main",
    "tw-med-publish-adapter": "tw_med_qlora.cli.publish_adapter:main",
    "tw-med-validate-phase5": "tw_med_qlora.cli.validate_phase5_evidence:main",
    "tw-med-verify-public-adapter": "tw_med_qlora.cli.verify_public_adapter:main",
}
EXPECTED_PROJECT_URLS = {
    "Changelog": "https://github.com/kuotunyu/tw-med-llm-qlora/blob/main/CHANGELOG.md",
    "Homepage": "https://github.com/kuotunyu/tw-med-llm-qlora",
    "Issues": "https://github.com/kuotunyu/tw-med-llm-qlora/issues",
    "Model": "https://huggingface.co/steven0226/tw-med-llm-qlora-adapter",
}
FORBIDDEN_PARTS = {
    ".codex",
    "artifacts",
    "checkpoints",
    "configs",
    "model_card",
    "notebooks",
    "outputs",
    "private",
    "reports",
    "requirements",
    "runs",
    "scripts",
    "tests",
}
FORBIDDEN_NAMES = {
    ".env",
    "AGENTS.md",
    "PROJECT_PLAN.md",
    "test_project_skill.py",
}
FORBIDDEN_SUFFIXES = {
    ".bin",
    ".gguf",
    ".pt",
    ".pth",
    ".safetensors",
}
SDIST_ALLOWED_ROOTS = {
    "CHANGELOG.md",
    "CITATION.cff",
    "LICENSE",
    "MANIFEST.in",
    "PKG-INFO",
    "README.md",
    "pyproject.toml",
    "setup.cfg",
    "src",
}


class ReleaseAuditError(RuntimeError):
    """Raised when an artifact violates the current release contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validated_members(names: list[str], *, strip_sdist_root: bool) -> list[PurePosixPath]:
    members: list[PurePosixPath] = []
    roots: set[str] = set()
    for name in names:
        normalized = PurePosixPath(name.replace("\\", "/"))
        if normalized.is_absolute() or ".." in normalized.parts:
            raise ReleaseAuditError(f"unsafe archive member: {name}")
        if not normalized.parts:
            continue
        roots.add(normalized.parts[0])
        members.append(normalized)

    if strip_sdist_root:
        if len(roots) != 1:
            raise ReleaseAuditError(f"sdist must have one root directory: {sorted(roots)}")
        root = next(iter(roots))
        members = [
            PurePosixPath(*member.parts[1:])
            for member in members
            if len(member.parts) > 1
        ]
        if root != RELEASE_ROOT:
            raise ReleaseAuditError(f"unexpected sdist root: {root}")

    for member in members:
        if not member.parts:
            continue
        if FORBIDDEN_PARTS.intersection(member.parts):
            raise ReleaseAuditError(f"forbidden directory in artifact: {member}")
        if member.name in FORBIDDEN_NAMES or member.suffix.casefold() in FORBIDDEN_SUFFIXES:
            raise ReleaseAuditError(f"forbidden file in artifact: {member}")
    return members


def parse_metadata(text: str) -> dict[str, Any]:
    metadata = Parser().parsestr(text)
    result = {
        "name": metadata["Name"],
        "version": metadata["Version"],
        "requires_python": metadata["Requires-Python"],
        "license_expression": metadata["License-Expression"],
        "license_files": metadata.get_all("License-File", []),
        "project_urls": dict(
            value.split(", ", maxsplit=1)
            for value in metadata.get_all("Project-URL", [])
        ),
    }
    expected = {
        "name": EXPECTED_NAME,
        "version": EXPECTED_VERSION,
        "license_expression": "MIT",
    }
    for key, value in expected.items():
        if result[key] != value:
            raise ReleaseAuditError(f"metadata {key}: expected {value!r}, got {result[key]!r}")
    requires_python = {
        specifier.strip() for specifier in (result["requires_python"] or "").split(",")
    }
    if requires_python != {">=3.11", "<3.12"}:
        raise ReleaseAuditError(
            "metadata requires_python: expected Python 3.11 only, "
            f"got {result['requires_python']!r}"
        )
    if "LICENSE" not in result["license_files"]:
        raise ReleaseAuditError("metadata does not declare LICENSE")
    if result["project_urls"] != EXPECTED_PROJECT_URLS:
        raise ReleaseAuditError(
            "metadata project_urls: "
            f"expected {EXPECTED_PROJECT_URLS!r}, got {result['project_urls']!r}"
        )
    return result


def parse_entry_points(text: str) -> dict[str, str]:
    scripts: dict[str, str] = {}
    section: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
        elif line and not line.startswith("#") and section == "console_scripts":
            name, target = line.split("=", maxsplit=1)
            scripts[name.strip()] = target.strip()
    if scripts != EXPECTED_SCRIPTS:
        raise ReleaseAuditError(
            f"console scripts differ: expected {EXPECTED_SCRIPTS}, got {scripts}"
        )
    return scripts


def audit_wheel(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        members = validated_members(archive.namelist(), strip_sdist_root=False)
        roots = {member.parts[0] for member in members if member.parts}
        dist_info = f"{RELEASE_ROOT}.dist-info"
        if roots != {"tw_med_qlora", dist_info}:
            raise ReleaseAuditError(f"unexpected wheel roots: {sorted(roots)}")
        required = {
            PurePosixPath("tw_med_qlora/__init__.py"),
            PurePosixPath(f"{dist_info}/METADATA"),
            PurePosixPath(f"{dist_info}/entry_points.txt"),
            PurePosixPath(f"{dist_info}/licenses/LICENSE"),
        }
        missing = required.difference(members)
        if missing:
            raise ReleaseAuditError(f"wheel is missing: {sorted(map(str, missing))}")
        metadata = parse_metadata(archive.read(f"{dist_info}/METADATA").decode("utf-8"))
        scripts = parse_entry_points(
            archive.read(f"{dist_info}/entry_points.txt").decode("utf-8")
        )
    return {
        "filename": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "files": len(members),
        "metadata": metadata,
        "console_scripts": scripts,
    }


def audit_sdist(path: Path) -> dict[str, Any]:
    with tarfile.open(path, mode="r:gz") as archive:
        members = validated_members(archive.getnames(), strip_sdist_root=True)
        roots = {member.parts[0] for member in members if member.parts}
        unexpected = roots.difference(SDIST_ALLOWED_ROOTS)
        if unexpected:
            raise ReleaseAuditError(f"unexpected sdist roots: {sorted(unexpected)}")
        required = {
            PurePosixPath("CHANGELOG.md"),
            PurePosixPath("CITATION.cff"),
            PurePosixPath("LICENSE"),
            PurePosixPath("MANIFEST.in"),
            PurePosixPath("PKG-INFO"),
            PurePosixPath("README.md"),
            PurePosixPath("pyproject.toml"),
            PurePosixPath("src/tw_med_qlora/__init__.py"),
        }
        missing = required.difference(members)
        if missing:
            raise ReleaseAuditError(f"sdist is missing: {sorted(map(str, missing))}")
        member = archive.getmember(f"{RELEASE_ROOT}/PKG-INFO")
        extracted = archive.extractfile(member)
        if extracted is None:
            raise ReleaseAuditError("cannot read sdist PKG-INFO")
        metadata = parse_metadata(extracted.read().decode("utf-8"))
    return {
        "filename": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "files": len(members),
        "metadata": metadata,
    }


def audit_directory(directory: Path) -> dict[str, Any]:
    wheels = sorted(directory.glob("*.whl"))
    sdists = sorted(directory.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ReleaseAuditError(
            f"expected one wheel and one sdist, found {len(wheels)} wheel(s) "
            f"and {len(sdists)} sdist(s)"
        )
    return {
        "schema_version": 1,
        "release": f"v{EXPECTED_VERSION}",
        "wheel": audit_wheel(wheels[0]),
        "sdist": audit_sdist(sdists[0]),
        "forbidden_content_absent": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = audit_directory(args.directory)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

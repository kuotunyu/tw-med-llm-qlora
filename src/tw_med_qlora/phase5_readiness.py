"""Machine-readable completion audit for the remaining Phase 5 gates."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

from .config import ProjectConfig
from .local_inference import load_adapter_contract, validate_adapter_contract
from .phase5_evidence import validate_phase5_file
from .publication import render_model_card, validate_repo_target

_SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"\A[0-9a-f]{40}\Z")


class Phase5ReadinessError(ValueError):
    """Raised when supplied completion evidence is malformed or contradictory."""


def _requirement(
    requirement_id: str,
    status: str,
    detail: str,
    *,
    required_for: str,
) -> dict[str, str]:
    if status not in {"pass", "pending", "fail"}:
        raise ValueError(f"invalid readiness status: {status}")
    return {
        "id": requirement_id,
        "status": status,
        "required_for": required_for,
        "detail": detail,
    }


def _git(repo_root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", "-C", str(repo_root), *arguments],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except FileNotFoundError as exc:
        raise Phase5ReadinessError("git is required for the Phase 5 audit") from exc


def validate_publication_receipt(
    receipt_path: Path,
    *,
    config: ProjectConfig,
) -> dict[str, Any]:
    """Validate the local, content-safe proof returned after a Hub upload."""

    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Phase5ReadinessError(f"cannot read publication receipt: {receipt_path}") from exc
    if not isinstance(receipt, dict) or receipt.get("schema_version") != 1:
        raise Phase5ReadinessError("unexpected publication receipt schema")
    if receipt.get("phase") != 5 or receipt.get("token_recorded") is not False:
        raise Phase5ReadinessError("publication receipt phase or token policy is invalid")
    publication = config.raw["publication"]
    repo_id = str(publication.get("adapter_repo_id", ""))
    visibility = str(
        publication.get(
            "phase5_receipt_visibility",
            publication.get("visibility", ""),
        )
    )
    if receipt.get("repo_id") != repo_id or receipt.get("visibility") != visibility:
        raise Phase5ReadinessError("publication receipt target does not match project config")
    if not _COMMIT.fullmatch(str(receipt.get("resolved_revision", ""))):
        raise Phase5ReadinessError("published adapter revision must be a full commit")
    expected_prefix = f"https://huggingface.co/{repo_id}/commit/"
    if not str(receipt.get("commit_url", "")).startswith(expected_prefix):
        raise Phase5ReadinessError("publication commit URL does not match the adapter repo")
    files = receipt.get("files")
    if not isinstance(files, dict):
        raise Phase5ReadinessError("publication receipt files must be an object")
    required = {"README.md", "adapter_config.json", "adapter_model.safetensors"}
    if not required.issubset(files):
        missing = sorted(required.difference(files))
        raise Phase5ReadinessError(f"publication receipt is missing files: {missing}")
    for name, metadata in files.items():
        if not isinstance(name, str) or not isinstance(metadata, dict):
            raise Phase5ReadinessError("publication receipt file entry is malformed")
        if not _SHA256.fullmatch(str(metadata.get("sha256", ""))):
            raise Phase5ReadinessError(f"publication file hash is invalid: {name}")
        size = metadata.get("bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise Phase5ReadinessError(f"publication file size is invalid: {name}")
    acceptance = receipt.get("acceptance")
    if not isinstance(acceptance, dict) or acceptance.get("valid") is not True:
        raise Phase5ReadinessError("publication receipt has no validated 4090 acceptance")
    return {
        "valid": True,
        "repo_id": repo_id,
        "visibility": visibility,
        "resolved_revision": receipt["resolved_revision"],
        "files": len(files),
        "token_absent": True,
    }


def assess_phase5_readiness(
    *,
    repo_root: Path,
    config: ProjectConfig,
    adapter_dir: Path | None = None,
    acceptance_manifest: Path | None = None,
    publication_receipt: Path | None = None,
) -> dict[str, Any]:
    """Evaluate every required local, 4090, repository, and publication proof."""

    repo_root = repo_root.resolve()
    requirements: list[dict[str, str]] = []

    status = _git(repo_root, "status", "--porcelain=v1", "--untracked-files=all")
    if status.returncode != 0:
        requirements.append(
            _requirement(
                "source.clean_worktree",
                "fail",
                "Git repository is unreadable",
                required_for="handoff",
            )
        )
        git_commit = None
    else:
        clean = not status.stdout.strip()
        requirements.append(
            _requirement(
                "source.clean_worktree",
                "pass" if clean else "fail",
                "worktree clean" if clean else "commit or remove local changes",
                required_for="handoff",
            )
        )
        commit = _git(repo_root, "rev-parse", "HEAD")
        git_commit = commit.stdout.strip() if commit.returncode == 0 else None

    required_paths = {
        "artifact.readme": repo_root / "README.md",
        "artifact.model_card": repo_root / "model_card" / "README.md",
        "artifact.local_inference": repo_root / "src" / "tw_med_qlora" / "local_inference.py",
        "artifact.acceptance_script": repo_root / "scripts" / "run_phase5_acceptance.ps1",
        "artifact.ci": repo_root / ".github" / "workflows" / "ci.yml",
    }
    for requirement_id, path in required_paths.items():
        requirements.append(
            _requirement(
                requirement_id,
                "pass" if path.is_file() and path.stat().st_size > 0 else "fail",
                path.relative_to(repo_root).as_posix(),
                required_for="handoff",
            )
        )

    try:
        template = (repo_root / "model_card" / "README.md").read_text(encoding="utf-8")
        render_model_card(
            template,
            repo_id="owner/adapter",
            github_url="https://github.com/owner/repository",
        )
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        card_status, card_detail = "fail", f"model card validation failed: {exc}"
    else:
        card_status = "pass"
        card_detail = "required license, safety, and result disclosures present"
    requirements.append(
        _requirement("publication.model_card", card_status, card_detail, required_for="publication")
    )

    adapter_validation: dict[str, Any] | None = None
    if adapter_dir is None:
        requirements.append(
            _requirement(
                "adapter.step700",
                "pending",
                "provide the extracted Phase 3 step-700 adapter directory",
                required_for="handoff",
            )
        )
    else:
        try:
            contract = load_adapter_contract(str(adapter_dir), token=None)
            validate_adapter_contract(
                contract,
                expected_base_model=config.primary.model_id,
                expected_base_revision=config.primary.revision,
            )
        except (OSError, ValueError, RuntimeError) as exc:
            requirements.append(
                _requirement("adapter.step700", "fail", str(exc), required_for="handoff")
            )
        else:
            adapter_validation = {
                "base_model_id": contract.base_model_name_or_path,
                "base_model_revision": contract.base_model_revision,
                "config_sha256": contract.config_sha256,
                "weights_sha256": contract.weights_sha256,
            }
            requirements.append(
                _requirement(
                    "adapter.step700",
                    "pass",
                    "adapter/base contract valid",
                    required_for="handoff",
                )
            )

    acceptance_validation: dict[str, Any] | None = None
    if acceptance_manifest is None:
        requirements.append(
            _requirement(
                "acceptance.rtx4090",
                "pending",
                "run the fixed acceptance probe on Windows RTX 4090",
                required_for="publication",
            )
        )
    else:
        try:
            acceptance_validation = validate_phase5_file(acceptance_manifest, config=config)
        except (OSError, ValueError) as exc:
            requirements.append(
                _requirement("acceptance.rtx4090", "fail", str(exc), required_for="publication")
            )
        else:
            requirements.append(
                _requirement(
                    "acceptance.rtx4090",
                    "pass",
                    "validated Windows RTX 4090 manifest",
                    required_for="publication",
                )
            )

    publication = config.raw["publication"]
    repo_id = str(publication.get("adapter_repo_id", ""))
    visibility = str(publication.get("visibility", ""))
    github_url = str(publication.get("github_repository_url", ""))
    try:
        validate_repo_target(repo_id, visibility, github_url)
    except ValueError:
        target_status, target_detail = "pending", "set exact HF repo ID, visibility, and GitHub URL"
    else:
        target_status, target_detail = "pass", f"{repo_id} ({visibility})"
    requirements.append(
        _requirement(
            "publication.targets",
            target_status,
            target_detail,
            required_for="publication",
        )
    )
    enabled = publication.get("enabled") is True
    requirements.append(
        _requirement(
            "publication.explicit_gate",
            "pass" if enabled else "pending",
            "publication.enabled=true" if enabled else "publication.enabled remains false",
            required_for="publication",
        )
    )

    remote = _git(repo_root, "remote", "get-url", "origin")
    origin_url = remote.stdout.strip() if remote.returncode == 0 else None
    if not github_url:
        remote_status, remote_detail = "pending", "GitHub repository URL is not configured"
    elif origin_url in {github_url, f"{github_url}.git"}:
        remote_status, remote_detail = "pass", origin_url
    else:
        remote_status = "pending"
        remote_detail = "origin remote does not match configured GitHub URL"
    requirements.append(
        _requirement(
            "publication.git_origin",
            remote_status,
            remote_detail,
            required_for="publication",
        )
    )

    publication_validation: dict[str, Any] | None = None
    if publication_receipt is None:
        requirements.append(
            _requirement(
                "publication.hf_receipt",
                "pending",
                "adapter has not been published",
                required_for="completion",
            )
        )
    else:
        try:
            publication_validation = validate_publication_receipt(
                publication_receipt,
                config=config,
            )
        except (OSError, ValueError) as exc:
            requirements.append(
                _requirement("publication.hf_receipt", "fail", str(exc), required_for="completion")
            )
        else:
            requirements.append(
                _requirement(
                    "publication.hf_receipt",
                    "pass",
                    "validated Hugging Face commit receipt",
                    required_for="completion",
                )
            )

    by_id = {item["id"]: item for item in requirements}
    handoff_ids = {
        "source.clean_worktree",
        *required_paths,
        "adapter.step700",
    }
    publication_ids = {
        "publication.model_card",
        "acceptance.rtx4090",
        "publication.targets",
        "publication.explicit_gate",
        "publication.git_origin",
    }
    ready_for_handoff = all(by_id[item]["status"] == "pass" for item in handoff_ids)
    ready_for_publication = ready_for_handoff and all(
        by_id[item]["status"] == "pass" for item in publication_ids
    )
    phase5_complete = ready_for_publication and by_id["publication.hf_receipt"]["status"] == "pass"
    return {
        "schema_version": 1,
        "phase": 5,
        "git_commit": git_commit,
        "ready_for_handoff": ready_for_handoff,
        "ready_for_publication": ready_for_publication,
        "phase5_complete": phase5_complete,
        "requirements": requirements,
        "adapter_validation": adapter_validation,
        "acceptance_validation": acceptance_validation,
        "publication_validation": publication_validation,
        "optional_gguf_required": False,
    }

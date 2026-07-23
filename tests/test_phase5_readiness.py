import json
import shutil
import subprocess
from pathlib import Path

import pytest

import tw_med_qlora.phase5_readiness as readiness_module
from tw_med_qlora.config import load_project_config
from tw_med_qlora.phase5_readiness import (
    Phase5ReadinessError,
    assess_phase5_readiness,
    validate_publication_receipt,
)

ROOT = Path(__file__).parents[1]


def run_git(repo: Path, *arguments: str) -> None:
    subprocess.run(["git", "-C", str(repo), *arguments], check=True, capture_output=True)


def readiness_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    required = (
        "model_card/README.md",
        "src/tw_med_qlora/local_inference.py",
        "scripts/run_phase5_acceptance.ps1",
        ".github/workflows/ci.yml",
    )
    repo.mkdir()
    (repo / "README.md").write_text("research\n", encoding="utf-8")
    for relative in required:
        destination = repo / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if relative == "model_card/README.md":
            shutil.copy2(ROOT / relative, destination)
        else:
            destination.write_text("placeholder\n", encoding="utf-8")
    run_git(repo, "init", "--quiet")
    run_git(repo, "config", "user.name", "Test")
    run_git(repo, "config", "user.email", "test@example.invalid")
    run_git(repo, "add", ".")
    run_git(repo, "commit", "--quiet", "-m", "readiness fixture")
    return repo


def adapter_dir(tmp_path: Path) -> Path:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": "taide/Gemma-3-TAIDE-12b-Chat-2602",
                "revision": "4de0b93b99f8b61b59c40d019fd593bdd1c42249",
                "peft_type": "LORA",
                "inference_mode": True,
            }
        ),
        encoding="utf-8",
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    return adapter


def config():
    return load_project_config(ROOT / "configs" / "project.toml")


def prepublication_config():
    project = config()
    project.raw["publication"].update(
        {
            "enabled": False,
            "adapter_repo_id": "",
            "visibility": "private",
            "github_repository_url": "",
        }
    )
    return project


def requirements_by_id(report: dict) -> dict:
    return {item["id"]: item for item in report["requirements"]}


def test_readiness_reports_external_gates_as_pending(tmp_path: Path) -> None:
    report = assess_phase5_readiness(
        repo_root=readiness_repo(tmp_path), config=prepublication_config()
    )
    checks = requirements_by_id(report)

    assert checks["source.clean_worktree"]["status"] == "pass"
    assert checks["publication.model_card"]["status"] == "pass"
    assert checks["adapter.step700"]["status"] == "pending"
    assert checks["acceptance.rtx4090"]["status"] == "pending"
    assert checks["publication.targets"]["status"] == "pending"
    assert report["ready_for_handoff"] is False
    assert report["ready_for_publication"] is False
    assert report["phase5_complete"] is False


def test_readiness_can_prove_handoff_without_claiming_publication(tmp_path: Path) -> None:
    report = assess_phase5_readiness(
        repo_root=readiness_repo(tmp_path),
        config=config(),
        adapter_dir=adapter_dir(tmp_path),
    )

    assert report["ready_for_handoff"] is True
    assert report["ready_for_publication"] is False
    assert report["adapter_validation"]["weights_sha256"]


def test_readiness_rejects_dirty_worktree(tmp_path: Path) -> None:
    repo = readiness_repo(tmp_path)
    (repo / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    report = assess_phase5_readiness(repo_root=repo, config=config())

    assert requirements_by_id(report)["source.clean_worktree"]["status"] == "fail"
    assert report["ready_for_handoff"] is False


def publication_config():
    project = config()
    project.raw["publication"].update(
        {
            "enabled": True,
            "adapter_repo_id": "owner/adapter",
            "visibility": "public",
            "phase5_receipt_visibility": "public",
            "github_repository_url": "https://github.com/owner/repository",
        }
    )
    return project


def publication_receipt(tmp_path: Path, *, token_recorded: bool = False) -> Path:
    metadata = {"sha256": "a" * 64, "bytes": 10}
    receipt = {
        "schema_version": 1,
        "phase": 5,
        "repo_id": "owner/adapter",
        "visibility": "public",
        "commit_url": "https://huggingface.co/owner/adapter/commit/" + "b" * 40,
        "resolved_revision": "b" * 40,
        "files": {
            "README.md": metadata,
            "adapter_config.json": metadata,
            "adapter_model.safetensors": metadata,
        },
        "acceptance": {"valid": True},
        "token_recorded": token_recorded,
    }
    path = tmp_path / "publication.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    return path


def test_publication_receipt_contract_is_content_safe(tmp_path: Path) -> None:
    validated = validate_publication_receipt(
        publication_receipt(tmp_path),
        config=publication_config(),
    )

    assert validated["valid"] is True
    assert validated["resolved_revision"] == "b" * 40
    assert validated["token_absent"] is True


def test_archived_private_receipt_remains_valid_after_phase7_target_change() -> None:
    validated = validate_publication_receipt(
        ROOT / "reports" / "phase5" / "20260722T165957Z-publication-receipt.json",
        config=config(),
    )

    assert validated["valid"] is True
    assert validated["visibility"] == "private"


def test_publication_receipt_rejects_token_policy_violation(tmp_path: Path) -> None:
    with pytest.raises(Phase5ReadinessError, match="token policy"):
        validate_publication_receipt(
            publication_receipt(tmp_path, token_recorded=True),
            config=publication_config(),
        )


def test_readiness_requires_every_gate_before_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = readiness_repo(tmp_path)
    run_git(repo, "remote", "add", "origin", "https://github.com/owner/repository")
    acceptance = tmp_path / "acceptance.json"
    publication = tmp_path / "publication.json"
    acceptance.write_text("{}", encoding="utf-8")
    publication.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        readiness_module,
        "validate_phase5_file",
        lambda *_args, **_kwargs: {"valid": True, "gpu": "NVIDIA GeForce RTX 4090"},
    )
    monkeypatch.setattr(
        readiness_module,
        "validate_publication_receipt",
        lambda *_args, **_kwargs: {"valid": True, "repo_id": "owner/adapter"},
    )

    report = assess_phase5_readiness(
        repo_root=repo,
        config=publication_config(),
        adapter_dir=adapter_dir(tmp_path),
        acceptance_manifest=acceptance,
        publication_receipt=publication,
    )

    assert report["ready_for_handoff"] is True
    assert report["ready_for_publication"] is True
    assert report["phase5_complete"] is True

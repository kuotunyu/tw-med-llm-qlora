import hashlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

import tw_med_qlora.phase7_publication as phase7_module
from tw_med_qlora.config import load_project_config
from tw_med_qlora.phase7_publication import (
    AnonymousDownloadProbe,
    Phase7PublicationError,
    validate_phase7_receipt,
    verify_public_adapter,
)

ROOT = Path(__file__).parents[1]
REPO_ID = "steven0226/tw-med-llm-qlora-adapter"
REVISION = "b" * 40
ARCHIVED_RECEIPT = (
    ROOT / "reports" / "phase7" / "20260723T163936Z-publication-receipt.json"
)
ARCHIVED_VALIDATION = (
    ROOT / "reports" / "phase7" / "20260723T164059Z-public-validation.json"
)


def receipt_path(tmp_path: Path, remote: Path) -> Path:
    files: dict[str, dict[str, int | str]] = {}
    for name in (
        "README.md",
        "TAIDE-GEMMA-LICENSE.pdf",
        "adapter_config.json",
        "adapter_model.safetensors",
    ):
        payload = f"payload:{name}".encode()
        (remote / name).write_bytes(payload)
        files[name] = {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }
    (remote / ".gitattributes").write_text("*.safetensors filter=lfs\n", encoding="utf-8")
    receipt = {
        "schema_version": 2,
        "phase": 7,
        "repo_id": REPO_ID,
        "visibility": "public",
        "gated": "auto",
        "resolved_revision": REVISION,
        "files": files,
        "removed_files": ["tokenizer_config.json"],
        "remote_files_verified": True,
        "token_recorded": False,
    }
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    return path


def test_phase7_receipt_requires_exact_minimal_file_allowlist(tmp_path: Path) -> None:
    remote = tmp_path / "remote"
    remote.mkdir()
    path = receipt_path(tmp_path, remote)
    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipt["files"]["tokenizer_config.json"] = {
        "sha256": "a" * 64,
        "bytes": 10,
    }
    path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(Phase7PublicationError, match="file allowlist"):
        validate_phase7_receipt(
            path,
            config=load_project_config(ROOT / "configs" / "project.toml"),
        )


def test_public_verification_checks_both_authenticated_and_anonymous_views(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote = tmp_path / "remote"
    remote.mkdir()
    receipt = receipt_path(tmp_path, remote)
    calls: list[str] = []

    class FakeApi:
        def __init__(self, *, token: str | bool) -> None:
            self.token = token

        def whoami(self) -> dict[str, str]:
            assert self.token == "write-token"
            calls.append("authenticated")
            return {"name": "steven0226"}

        def repo_info(self, **_kwargs: object) -> SimpleNamespace:
            calls.append("anonymous-metadata" if self.token is False else "private-metadata")
            return SimpleNamespace(private=False, gated="auto", sha=REVISION)

        def list_repo_files(self, **_kwargs: object) -> list[str]:
            assert self.token == "write-token"
            return sorted(path.name for path in remote.iterdir())

        def hf_hub_download(self, **kwargs: object) -> str:
            assert self.token == "write-token"
            return str(remote / str(kwargs["filename"]))

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.HfApi = FakeApi
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    monkeypatch.setattr(
        phase7_module,
        "probe_anonymous_download",
        lambda **_kwargs: AnonymousDownloadProbe(denied=True, status=401),
    )
    output = tmp_path / "validation.json"
    report = verify_public_adapter(
        receipt_path=receipt,
        config=load_project_config(ROOT / "configs" / "project.toml"),
        token="write-token",
        output_path=output,
    )

    assert calls == ["authenticated", "private-metadata", "anonymous-metadata"]
    assert report["anonymous_metadata_visible"] is True
    assert report["anonymous_download_denied"] is True
    assert report["authenticated_hashes_verified"] is True
    assert report["token_recorded"] is False
    assert output.is_file()


def test_archived_phase7_evidence_is_complete_and_content_safe() -> None:
    receipt = validate_phase7_receipt(
        ARCHIVED_RECEIPT,
        config=load_project_config(ROOT / "configs" / "project.toml"),
    )
    validation = json.loads(ARCHIVED_VALIDATION.read_text(encoding="utf-8"))

    assert receipt["resolved_revision"] == (
        "b1d8f74291da75d0719b5a3ea0d088ee8236e096"
    )
    assert receipt["visibility_before"] == "private"
    assert receipt["visibility_transition_performed"] is True
    assert len(receipt["files"]) == 4
    assert len(receipt["removed_files"]) == 9
    assert validation == {
        "schema_version": 1,
        "phase": 7,
        "created_at_utc": "2026-07-23T16:40:59.871271+00:00",
        "repo_id": REPO_ID,
        "visibility": "public",
        "gated": "auto",
        "resolved_revision": receipt["resolved_revision"],
        "files": 4,
        "authenticated_hashes_verified": True,
        "anonymous_metadata_visible": True,
        "anonymous_download_denied": True,
        "anonymous_denial_status": 401,
        "token_recorded": False,
    }

"""Independent, content-safe verification for the Phase 7 public gated adapter."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .config import ProjectConfig

_SHA256_LENGTH = 64
_COMMIT_LENGTH = 40
_EXPECTED_FILES = {
    "README.md",
    "TAIDE-GEMMA-LICENSE.pdf",
    "adapter_config.json",
    "adapter_model.safetensors",
}
_ALLOWED_REMOTE_EXTRAS = {".gitattributes"}


class Phase7PublicationError(ValueError):
    """Raised when public gated publication evidence is missing or inconsistent."""


@dataclass(frozen=True)
class AnonymousDownloadProbe:
    denied: bool
    status: int


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(
        self,
        req: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_gate(value: Any) -> str:
    return "false" if value in {False, None, "false"} else str(value)


def validate_phase7_receipt(
    receipt_path: Path,
    *,
    config: ProjectConfig,
) -> dict[str, Any]:
    """Validate the local receipt before using it as remote verification input."""

    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Phase7PublicationError(f"cannot read Phase 7 receipt: {receipt_path}") from exc
    if not isinstance(receipt, dict):
        raise Phase7PublicationError("Phase 7 receipt must be an object")
    if receipt.get("schema_version") != 2 or receipt.get("phase") != 7:
        raise Phase7PublicationError("unexpected Phase 7 receipt schema")
    if receipt.get("token_recorded") is not False:
        raise Phase7PublicationError("Phase 7 receipt token policy is invalid")
    publication = config.raw["publication"]
    expected_target = {
        "repo_id": str(publication.get("adapter_repo_id", "")),
        "visibility": str(publication.get("visibility", "")),
        "gated": str(publication.get("gated", "false")),
    }
    for key, expected in expected_target.items():
        if receipt.get(key) != expected:
            raise Phase7PublicationError(f"Phase 7 receipt {key} does not match config")
    revision = str(receipt.get("resolved_revision", ""))
    if len(revision) != _COMMIT_LENGTH or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise Phase7PublicationError("Phase 7 receipt revision is not a full commit")
    files = receipt.get("files")
    if not isinstance(files, dict) or set(files) != _EXPECTED_FILES:
        raise Phase7PublicationError("Phase 7 receipt file allowlist is invalid")
    for name, metadata in files.items():
        if not isinstance(metadata, dict):
            raise Phase7PublicationError(f"Phase 7 receipt metadata is invalid: {name}")
        digest = str(metadata.get("sha256", ""))
        size = metadata.get("bytes")
        if len(digest) != _SHA256_LENGTH or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise Phase7PublicationError(f"Phase 7 receipt hash is invalid: {name}")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise Phase7PublicationError(f"Phase 7 receipt size is invalid: {name}")
    if receipt.get("remote_files_verified") is not True:
        raise Phase7PublicationError("Phase 7 receipt lacks remote file verification")
    if not isinstance(receipt.get("removed_files"), list):
        raise Phase7PublicationError("Phase 7 receipt removed_files must be a list")
    return receipt


def probe_anonymous_download(
    *,
    repo_id: str,
    revision: str,
    filename: str = "adapter_config.json",
    timeout_seconds: float = 20,
) -> AnonymousDownloadProbe:
    """Prove that a tokenless HTTP client cannot resolve a gated model file."""

    url = f"https://huggingface.co/{repo_id}/resolve/{revision}/{filename}"
    request = Request(url, method="GET", headers={"User-Agent": "tw-med-phase7-audit/1"})
    opener = build_opener(_NoRedirect)
    try:
        response = opener.open(request, timeout=timeout_seconds)
    except HTTPError as exc:
        location = str(exc.headers.get("Location", ""))
        denied = exc.code in {401, 403} or (
            300 <= exc.code < 400 and "/login" in location
        )
        return AnonymousDownloadProbe(denied=denied, status=exc.code)
    with response:
        return AnonymousDownloadProbe(denied=False, status=int(response.status))


def verify_public_adapter(
    *,
    receipt_path: Path,
    config: ProjectConfig,
    token: str,
    output_path: Path,
) -> dict[str, Any]:
    """Verify public metadata, automatic gating, files, hashes, and anonymous denial."""

    if not token:
        raise Phase7PublicationError("HF_TOKEN is required for authenticated hash verification")
    receipt = validate_phase7_receipt(receipt_path, config=config)
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - optional remote verification dependency
        raise Phase7PublicationError("huggingface-hub is required for verification") from exc

    repo_id = str(receipt["repo_id"])
    revision = str(receipt["resolved_revision"])
    authenticated = HfApi(token=token)
    anonymous = HfApi(token=False)
    authenticated.whoami()
    private_info = authenticated.repo_info(repo_id=repo_id, repo_type="model")
    public_info = anonymous.repo_info(repo_id=repo_id, repo_type="model")
    for label, info in (("authenticated", private_info), ("anonymous", public_info)):
        if bool(info.private):
            raise Phase7PublicationError(f"{label} repository metadata still reports private")
        if _normalize_gate(getattr(info, "gated", False)) != "auto":
            raise Phase7PublicationError(f"{label} repository metadata is not automatic gated")
        if str(info.sha) != revision:
            raise Phase7PublicationError(f"{label} repository revision does not match receipt")

    remote_files = set(
        authenticated.list_repo_files(
            repo_id=repo_id,
            repo_type="model",
            revision=revision,
        )
    )
    expected_remote = _EXPECTED_FILES | _ALLOWED_REMOTE_EXTRAS
    if remote_files != expected_remote:
        raise Phase7PublicationError(
            "remote file allowlist mismatch: "
            f"missing={sorted(expected_remote - remote_files)}, "
            f"unexpected={sorted(remote_files - expected_remote)}"
        )
    for name, expected in receipt["files"].items():
        downloaded = Path(
            authenticated.hf_hub_download(
                repo_id=repo_id,
                filename=name,
                repo_type="model",
                revision=revision,
            )
        )
        if downloaded.stat().st_size != int(expected["bytes"]):
            raise Phase7PublicationError(f"authenticated file size mismatch: {name}")
        if _sha256_file(downloaded) != expected["sha256"]:
            raise Phase7PublicationError(f"authenticated file SHA-256 mismatch: {name}")

    anonymous_probe = probe_anonymous_download(repo_id=repo_id, revision=revision)
    if not anonymous_probe.denied:
        raise Phase7PublicationError(
            f"anonymous file download was not denied (HTTP {anonymous_probe.status})"
        )
    report = {
        "schema_version": 1,
        "phase": 7,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "repo_id": repo_id,
        "visibility": "public",
        "gated": "auto",
        "resolved_revision": revision,
        "files": len(_EXPECTED_FILES),
        "authenticated_hashes_verified": True,
        "anonymous_metadata_visible": True,
        "anonymous_download_denied": True,
        "anonymous_denial_status": anonymous_probe.status,
        "token_recorded": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report

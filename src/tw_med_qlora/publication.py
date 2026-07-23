"""Safe-by-default planning and execution for publishing the PEFT adapter."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import ProjectConfig
from .local_inference import load_adapter_contract, validate_adapter_contract
from .phase5_evidence import validate_phase5_file

_REPO_ID = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_SECRET = re.compile(r"(?:hf|gho|sk)-?_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}")
_PRIVATE_ADAPTER_FILES = {
    "adapter_config.json",
    "adapter_model.safetensors",
    "added_tokens.json",
    "chat_template.jinja",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
}
_PUBLIC_ADAPTER_FILES = {
    "adapter_config.json",
    "adapter_model.safetensors",
}
_PUBLIC_LICENSE_FILE = "TAIDE-GEMMA-LICENSE.pdf"
_GATED_CHOICES = {"false", "auto", "manual"}
_REQUIRED_CARD_TEXT = {
    "不構成醫療建議",
    "gemma-version-taide-models-license-agreement",
    "Gemma is provided under and subject to the Gemma Terms of Use found at "
    "ai.google.dev/gemma/terms",
    "TAIDE-GEMMA-LICENSE.pdf",
    "taide.tw",
    "未對本 adapter 背書",
    "MIT License 只涵蓋程式碼",
    "72.05%",
    "61.53%",
    "step 700",
}


class PublicationError(ValueError):
    """Raised when a publication request is incomplete or unsafe."""


@dataclass(frozen=True)
class PublicationPlan:
    repo_id: str
    visibility: str
    gated: str
    github_repository_url: str
    adapter_dir: str
    license_path: str
    base_model_id: str
    base_model_revision: str
    confirmation_code: str
    files: dict[str, dict[str, int | str]]
    delete_files: tuple[str, ...]
    rendered_model_card: str

    def public_summary(self) -> dict[str, Any]:
        return {
            "repo_id": self.repo_id,
            "visibility": self.visibility,
            "gated": self.gated,
            "github_repository_url": self.github_repository_url,
            "base_model_id": self.base_model_id,
            "base_model_revision": self.base_model_revision,
            "confirmation_code": self.confirmation_code,
            "files": self.files,
            "delete_files": list(self.delete_files),
            "planned_visibility_transition": (
                f"private -> public gated({self.gated})"
                if self.visibility == "public"
                else "remain private"
            ),
            "ready_for_dry_run": True,
            "external_mutation_performed": False,
        }


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def publication_confirmation_code(
    repo_id: str,
    visibility: str,
    gated: str = "false",
) -> str:
    """Bind explicit approval to exactly one target, visibility, and gate mode."""

    digest = hashlib.sha256(f"phase7:{repo_id}:{visibility}:{gated}".encode()).hexdigest()[:12]
    return f"PUBLISH_ADAPTER_{digest.upper()}"


def validate_repo_target(
    repo_id: str,
    visibility: str,
    github_url: str,
    gated: str | None = None,
) -> None:
    if not _REPO_ID.fullmatch(repo_id):
        raise PublicationError("repo_id must have the form owner/name")
    if visibility not in {"private", "public"}:
        raise PublicationError("visibility must be private or public")
    if gated is not None:
        if gated not in _GATED_CHOICES:
            raise PublicationError("gated must be false, auto, or manual")
        if visibility == "private" and gated != "false":
            raise PublicationError("a private repository cannot use a public access gate")
        if visibility == "public" and gated == "false":
            raise PublicationError("public adapter publication must remain gated")
    if not github_url.startswith("https://github.com/") or github_url.count("/") < 4:
        raise PublicationError("github_repository_url must be a full GitHub repository URL")


def render_model_card(template: str, *, repo_id: str, github_url: str) -> str:
    """Render and validate the adapter card without inventing publication targets."""

    rendered = template.replace("{{HF_ADAPTER_REPO_ID}}", repo_id).replace(
        "{{GITHUB_REPOSITORY_URL}}", github_url.rstrip("/")
    )
    if "{{" in rendered or "}}" in rendered:
        raise PublicationError("model card contains an unresolved placeholder")
    missing = sorted(text for text in _REQUIRED_CARD_TEXT if text not in rendered)
    if missing:
        raise PublicationError(f"model card is missing required disclosures: {missing}")
    if not rendered.startswith("---\n") or "license: other" not in rendered[:500]:
        raise PublicationError("model card metadata must declare the non-MIT weight license")
    if _SECRET.search(rendered):
        raise PublicationError("model card appears to contain a credential")
    return rendered


def build_publication_plan(
    *,
    adapter_dir: Path,
    model_card_template: Path,
    repo_id: str,
    visibility: str,
    gated: str = "false",
    github_url: str,
    config: ProjectConfig,
) -> PublicationPlan:
    """Audit every file that would be sent to the Hub; perform no external mutation."""

    validate_repo_target(repo_id, visibility, github_url, gated)
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"adapter directory not found: {adapter_dir}")
    contract = load_adapter_contract(str(adapter_dir), token=None)
    validate_adapter_contract(
        contract,
        expected_base_model=config.primary.model_id,
        expected_base_revision=config.primary.revision,
    )
    template = model_card_template.read_text(encoding="utf-8")
    rendered = render_model_card(template, repo_id=repo_id, github_url=github_url)

    files: dict[str, dict[str, int | str]] = {}
    adapter_allowlist = (
        _PUBLIC_ADAPTER_FILES if visibility == "public" else _PRIVATE_ADAPTER_FILES
    )
    for name in sorted(adapter_allowlist):
        path = adapter_dir / name
        if not path.exists():
            continue
        if path.is_symlink() or not path.is_file():
            raise PublicationError(f"adapter publication file must be regular: {name}")
        files[name] = {"sha256": _sha256_file(path), "bytes": path.stat().st_size}
    for required in ("adapter_config.json", "adapter_model.safetensors"):
        if required not in files:
            raise PublicationError(f"missing required publication file: {required}")
    license_path = model_card_template.parent / _PUBLIC_LICENSE_FILE
    if not license_path.is_file() or license_path.is_symlink():
        raise PublicationError(f"missing required publication license: {license_path}")
    files[_PUBLIC_LICENSE_FILE] = {
        "sha256": _sha256_file(license_path),
        "bytes": license_path.stat().st_size,
    }
    card_bytes = rendered.encode("utf-8")
    files["README.md"] = {"sha256": _sha256_bytes(card_bytes), "bytes": len(card_bytes)}
    if any(int(metadata["bytes"]) <= 0 for metadata in files.values()):
        raise PublicationError("publication contains an empty file")
    delete_files = (
        tuple(sorted(_PRIVATE_ADAPTER_FILES - _PUBLIC_ADAPTER_FILES))
        if visibility == "public"
        else ()
    )
    return PublicationPlan(
        repo_id=repo_id,
        visibility=visibility,
        gated=gated,
        github_repository_url=github_url.rstrip("/"),
        adapter_dir=str(adapter_dir.resolve()),
        license_path=str(license_path.resolve()),
        base_model_id=config.primary.model_id,
        base_model_revision=config.primary.revision,
        confirmation_code=publication_confirmation_code(repo_id, visibility, gated),
        files=files,
        delete_files=delete_files,
        rendered_model_card=rendered,
    )


def assert_publication_execution_gate(
    plan: PublicationPlan,
    *,
    config: ProjectConfig,
    confirmation_code: str | None,
    acceptance_manifest: Path | None,
) -> dict[str, Any]:
    """Require every user-controlled gate before creating or updating a repository."""

    settings = config.raw["publication"]
    if settings.get("enabled") is not True:
        raise PublicationError("publication.enabled is false")
    if settings.get("adapter_repo_id") != plan.repo_id:
        raise PublicationError("configured adapter_repo_id does not match the plan")
    if settings.get("visibility") != plan.visibility:
        raise PublicationError("configured visibility does not match the plan")
    if str(settings.get("gated", "false")) != plan.gated:
        raise PublicationError("configured gated mode does not match the plan")
    if settings.get("github_repository_url", "").rstrip("/") != plan.github_repository_url:
        raise PublicationError("configured GitHub URL does not match the plan")
    if confirmation_code != plan.confirmation_code:
        raise PublicationError("publication confirmation code is missing or incorrect")
    if acceptance_manifest is None:
        raise PublicationError("a validated RTX 4090 acceptance manifest is required")
    return validate_phase5_file(acceptance_manifest, config=config)


def _stage_folder(plan: PublicationPlan, destination: Path) -> None:
    adapter_dir = Path(plan.adapter_dir)
    destination.mkdir(parents=True)
    for name in plan.files:
        if name == "README.md":
            continue
        source = Path(plan.license_path) if name == _PUBLIC_LICENSE_FILE else adapter_dir / name
        shutil.copy2(source, destination / name)
    (destination / "README.md").write_text(
        plan.rendered_model_card,
        encoding="utf-8",
        newline="\n",
    )


def _normalize_remote_gate(value: Any) -> str:
    return "false" if value in {False, None, "false"} else str(value)


def _verify_remote_files(
    api: Any,
    *,
    plan: PublicationPlan,
    revision: str,
) -> None:
    remote_files = set(
        api.list_repo_files(
            repo_id=plan.repo_id,
            repo_type="model",
            revision=revision,
        )
    )
    missing = sorted(set(plan.files) - remote_files)
    stale = sorted(set(plan.delete_files) & remote_files)
    unexpected = sorted(remote_files - set(plan.files) - {".gitattributes"})
    if missing:
        raise PublicationError(f"remote publication is missing files: {missing}")
    if stale:
        raise PublicationError(f"remote publication retained files scheduled for deletion: {stale}")
    if unexpected:
        raise PublicationError(f"remote publication contains unexpected files: {unexpected}")
    for name, expected in plan.files.items():
        downloaded = Path(
            api.hf_hub_download(
                repo_id=plan.repo_id,
                filename=name,
                repo_type="model",
                revision=revision,
            )
        )
        if downloaded.stat().st_size != int(expected["bytes"]):
            raise PublicationError(f"remote publication size mismatch: {name}")
        if _sha256_file(downloaded) != expected["sha256"]:
            raise PublicationError(f"remote publication SHA-256 mismatch: {name}")


def execute_publication(
    plan: PublicationPlan,
    *,
    token: str,
    config: ProjectConfig,
    confirmation_code: str,
    acceptance_manifest: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Create/update exactly one approved repo and return a content-safe receipt."""

    acceptance = assert_publication_execution_gate(
        plan,
        config=config,
        confirmation_code=confirmation_code,
        acceptance_manifest=acceptance_manifest,
    )
    if not token:
        raise PublicationError("HF_TOKEN with write permission is required")
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.errors import RepositoryNotFoundError
    except ImportError as exc:  # pragma: no cover - optional execution dependency
        raise PublicationError("huggingface-hub is required for publication") from exc

    api = HfApi(token=token)
    api.whoami()
    try:
        existing = api.repo_info(repo_id=plan.repo_id, repo_type="model")
    except RepositoryNotFoundError:
        existing = None
    requested_private = plan.visibility == "private"
    if existing is not None:
        current_gate = _normalize_remote_gate(getattr(existing, "gated", False))
        if requested_private and not bool(existing.private):
            raise PublicationError("existing repository is public but the approved plan is private")
        if not requested_private and not bool(existing.private) and current_gate != plan.gated:
            raise PublicationError(
                "existing public repository gate does not match the approved plan"
            )
    visibility_before = (
        "absent" if existing is None else ("private" if bool(existing.private) else "public")
    )
    gated_before = (
        "false"
        if existing is None
        else _normalize_remote_gate(getattr(existing, "gated", False))
    )
    api.create_repo(
        repo_id=plan.repo_id,
        repo_type="model",
        private=True if plan.visibility == "public" else requested_private,
        exist_ok=True,
    )
    with tempfile.TemporaryDirectory(prefix="tw-med-adapter-publish-") as temporary:
        stage = Path(temporary) / "adapter"
        _stage_folder(plan, stage)
        commit = api.upload_folder(
            folder_path=stage,
            repo_id=plan.repo_id,
            repo_type="model",
            commit_message=(
                "Prepare validated public gated step-700 adapter"
                if plan.visibility == "public"
                else "Upload validated Phase 3 step-700 adapter"
            ),
            delete_patterns=list(plan.delete_files) or None,
        )
    _verify_remote_files(api, plan=plan, revision=str(commit.oid))
    if plan.visibility == "public":
        api.update_repo_settings(
            repo_id=plan.repo_id,
            repo_type="model",
            private=False,
            gated=plan.gated,
        )
    info = api.repo_info(repo_id=plan.repo_id, repo_type="model")
    final_visibility = "private" if bool(info.private) else "public"
    final_gate = _normalize_remote_gate(getattr(info, "gated", False))
    if final_visibility != plan.visibility or final_gate != plan.gated:
        raise PublicationError("final repository visibility or gate does not match the plan")
    is_public_transition = plan.visibility == "public"
    receipt = {
        "schema_version": 2 if is_public_transition else 1,
        "phase": 7 if is_public_transition else 5,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "repo_id": plan.repo_id,
        "visibility": plan.visibility,
        "gated": plan.gated,
        "visibility_before": visibility_before,
        "gated_before": gated_before,
        "commit_url": str(commit.commit_url),
        "resolved_revision": str(info.sha),
        "files": plan.files,
        "removed_files": list(plan.delete_files),
        "remote_files_verified": True,
        "visibility_transition_performed": (
            is_public_transition and visibility_before != "public"
        ),
        "acceptance": acceptance,
        "token_recorded": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / (
        "phase7-publication-receipt.json"
        if is_public_transition
        else "phase5-publication-receipt.json"
    )
    destination.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return receipt

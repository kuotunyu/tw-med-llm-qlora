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
_ALLOWED_ADAPTER_FILES = {
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
_REQUIRED_CARD_TEXT = {
    "不構成醫療建議",
    "gemma-version-taide-models-license-agreement",
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
    github_repository_url: str
    adapter_dir: str
    base_model_id: str
    base_model_revision: str
    confirmation_code: str
    files: dict[str, dict[str, int | str]]
    rendered_model_card: str

    def public_summary(self) -> dict[str, Any]:
        return {
            "repo_id": self.repo_id,
            "visibility": self.visibility,
            "github_repository_url": self.github_repository_url,
            "base_model_id": self.base_model_id,
            "base_model_revision": self.base_model_revision,
            "confirmation_code": self.confirmation_code,
            "files": self.files,
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


def publication_confirmation_code(repo_id: str, visibility: str) -> str:
    """Bind explicit approval to exactly one target and visibility."""

    digest = hashlib.sha256(f"phase5:{repo_id}:{visibility}".encode()).hexdigest()[:12]
    return f"PUBLISH_ADAPTER_{digest.upper()}"


def validate_repo_target(repo_id: str, visibility: str, github_url: str) -> None:
    if not _REPO_ID.fullmatch(repo_id):
        raise PublicationError("repo_id must have the form owner/name")
    if visibility not in {"private", "public"}:
        raise PublicationError("visibility must be private or public")
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
    github_url: str,
    config: ProjectConfig,
) -> PublicationPlan:
    """Audit every file that would be sent to the Hub; perform no external mutation."""

    validate_repo_target(repo_id, visibility, github_url)
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
    for name in sorted(_ALLOWED_ADAPTER_FILES):
        path = adapter_dir / name
        if not path.exists():
            continue
        if path.is_symlink() or not path.is_file():
            raise PublicationError(f"adapter publication file must be regular: {name}")
        files[name] = {"sha256": _sha256_file(path), "bytes": path.stat().st_size}
    for required in ("adapter_config.json", "adapter_model.safetensors"):
        if required not in files:
            raise PublicationError(f"missing required publication file: {required}")
    card_bytes = rendered.encode("utf-8")
    files["README.md"] = {"sha256": _sha256_bytes(card_bytes), "bytes": len(card_bytes)}
    if any(int(metadata["bytes"]) <= 0 for metadata in files.values()):
        raise PublicationError("publication contains an empty file")
    return PublicationPlan(
        repo_id=repo_id,
        visibility=visibility,
        github_repository_url=github_url.rstrip("/"),
        adapter_dir=str(adapter_dir.resolve()),
        base_model_id=config.primary.model_id,
        base_model_revision=config.primary.revision,
        confirmation_code=publication_confirmation_code(repo_id, visibility),
        files=files,
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
        shutil.copy2(adapter_dir / name, destination / name)
    (destination / "README.md").write_text(
        plan.rendered_model_card,
        encoding="utf-8",
        newline="\n",
    )


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
    if existing is not None and bool(existing.private) != requested_private:
        raise PublicationError("existing repository visibility does not match the approved plan")
    api.create_repo(
        repo_id=plan.repo_id,
        repo_type="model",
        private=requested_private,
        exist_ok=True,
    )
    with tempfile.TemporaryDirectory(prefix="tw-med-adapter-publish-") as temporary:
        stage = Path(temporary) / "adapter"
        _stage_folder(plan, stage)
        commit = api.upload_folder(
            folder_path=stage,
            repo_id=plan.repo_id,
            repo_type="model",
            commit_message="Upload validated Phase 3 step-700 adapter",
        )
    info = api.repo_info(repo_id=plan.repo_id, repo_type="model")
    receipt = {
        "schema_version": 1,
        "phase": 5,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "repo_id": plan.repo_id,
        "visibility": plan.visibility,
        "commit_url": str(commit.commit_url),
        "resolved_revision": str(info.sha),
        "files": plan.files,
        "acceptance": acceptance,
        "token_recorded": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "phase5-publication-receipt.json"
    destination.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return receipt

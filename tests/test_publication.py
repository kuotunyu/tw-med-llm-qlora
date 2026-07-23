import hashlib
import json
import shutil
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

import tw_med_qlora.publication as publication_module
from tw_med_qlora.config import load_project_config
from tw_med_qlora.publication import (
    PublicationError,
    _stage_folder,
    assert_publication_execution_gate,
    build_publication_plan,
    execute_publication,
    publication_confirmation_code,
    render_model_card,
)

ROOT = Path(__file__).parents[1]
TAIDE_LICENSE_SHA256 = "39e4c7c020250cd1b9e6d0651745d7381359584c562dbe2c73dce5333f446ce0"


def adapter_dir(tmp_path: Path) -> Path:
    path = tmp_path / "adapter"
    path.mkdir()
    (path / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": "taide/Gemma-3-TAIDE-12b-Chat-2602",
                "revision": None,
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "inference_mode": True,
            }
        ),
        encoding="utf-8",
    )
    (path / "adapter_model.safetensors").write_bytes(b"weights")
    (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (path / "ignored-secret.txt").write_text("not uploaded", encoding="utf-8")
    return path


def build_plan(tmp_path: Path):
    return build_publication_plan(
        adapter_dir=adapter_dir(tmp_path),
        model_card_template=ROOT / "model_card" / "README.md",
        repo_id="researcher/tw-med-adapter",
        visibility="private",
        gated="false",
        github_url="https://github.com/researcher/tw-med-llm-qlora",
        config=load_project_config(ROOT / "configs" / "project.toml"),
    )


def test_publication_plan_is_dry_and_allowlisted(tmp_path: Path) -> None:
    plan = build_plan(tmp_path)
    summary = plan.public_summary()

    assert summary["external_mutation_performed"] is False
    assert set(plan.files) == {
        "README.md",
        "TAIDE-GEMMA-LICENSE.pdf",
        "adapter_config.json",
        "adapter_model.safetensors",
        "tokenizer_config.json",
    }
    assert plan.gated == "false"
    assert plan.delete_files == ()
    assert "ignored-secret.txt" not in plan.files
    assert "{{" not in plan.rendered_model_card
    assert "researcher/tw-med-adapter" in plan.rendered_model_card


def test_staged_publication_bytes_match_dry_run_hashes(tmp_path: Path) -> None:
    plan = build_plan(tmp_path)
    destination = tmp_path / "stage"

    _stage_folder(plan, destination)

    for name, expected in plan.files.items():
        payload = (destination / name).read_bytes()
        assert len(payload) == expected["bytes"]
        assert hashlib.sha256(payload).hexdigest() == expected["sha256"]


def test_model_card_requires_target_and_disclosures() -> None:
    template = (ROOT / "model_card" / "README.md").read_text(encoding="utf-8")
    rendered = render_model_card(
        template,
        repo_id="owner/model",
        github_url="https://github.com/owner/repo",
    )
    assert "不構成醫療建議" in rendered
    assert "TAIDE-GEMMA-LICENSE.pdf" in rendered
    assert "extra_gated_fields:" in rendered
    assert (
        "Gemma is provided under and subject to the Gemma Terms of Use found at "
        "ai.google.dev/gemma/terms"
    ) in rendered

    with pytest.raises(PublicationError, match="required disclosures"):
        render_model_card(
            template.replace("72.05%", "missing"),
            repo_id="owner/model",
            github_url="https://github.com/owner/repo",
        )


def test_taide_license_pdf_matches_official_snapshot() -> None:
    payload = (ROOT / "model_card" / "TAIDE-GEMMA-LICENSE.pdf").read_bytes()

    assert len(payload) == 242366
    assert hashlib.sha256(payload).hexdigest() == TAIDE_LICENSE_SHA256


def test_confirmation_code_is_bound_to_visibility() -> None:
    private = publication_confirmation_code("owner/model", "private")
    public = publication_confirmation_code("owner/model", "public", "auto")
    manual = publication_confirmation_code("owner/model", "public", "manual")

    assert private.startswith("PUBLISH_ADAPTER_")
    assert private != public
    assert public != manual


def test_public_plan_is_gated_and_excludes_base_derived_tokenizer(
    tmp_path: Path,
) -> None:
    plan = build_publication_plan(
        adapter_dir=adapter_dir(tmp_path),
        model_card_template=ROOT / "model_card" / "README.md",
        repo_id="researcher/tw-med-adapter",
        visibility="public",
        gated="auto",
        github_url="https://github.com/researcher/tw-med-llm-qlora",
        config=load_project_config(ROOT / "configs" / "project.toml"),
    )

    assert set(plan.files) == {
        "README.md",
        "TAIDE-GEMMA-LICENSE.pdf",
        "adapter_config.json",
        "adapter_model.safetensors",
    }
    assert "tokenizer_config.json" in plan.delete_files
    assert "chat_template.jinja" in plan.delete_files
    assert "adapter_model.safetensors" not in plan.delete_files


def test_public_plan_without_access_gate_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(PublicationError, match="must remain gated"):
        build_publication_plan(
            adapter_dir=adapter_dir(tmp_path),
            model_card_template=ROOT / "model_card" / "README.md",
            repo_id="researcher/tw-med-adapter",
            visibility="public",
            gated="false",
            github_url="https://github.com/researcher/tw-med-llm-qlora",
            config=load_project_config(ROOT / "configs" / "project.toml"),
        )


def test_public_execution_verifies_files_before_visibility_transition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote = tmp_path / "remote"
    remote.mkdir()
    (remote / ".gitattributes").write_text("*.safetensors filter=lfs\n", encoding="utf-8")
    (remote / "tokenizer_config.json").write_text("legacy\n", encoding="utf-8")
    events: list[str] = []

    class FakeApi:
        private = True
        gated: bool | str = False
        sha = "a" * 40

        def __init__(self, *, token: str) -> None:
            assert token == "write-token"

        def whoami(self) -> dict[str, str]:
            return {"name": "steven0226"}

        def repo_info(self, **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(private=self.private, gated=self.gated, sha=self.sha)

        def create_repo(self, **kwargs: object) -> None:
            assert kwargs["private"] is True
            events.append("create-private")

        def upload_folder(self, **kwargs: object) -> SimpleNamespace:
            assert self.private is True
            events.append("upload-while-private")
            for name in kwargs["delete_patterns"] or []:
                path = remote / str(name)
                if path.exists():
                    path.unlink()
            for source in Path(str(kwargs["folder_path"])).iterdir():
                shutil.copy2(source, remote / source.name)
            self.sha = "b" * 40
            return SimpleNamespace(
                oid=self.sha,
                commit_url=f"https://huggingface.co/owner/model/commit/{self.sha}",
            )

        def list_repo_files(self, **_kwargs: object) -> list[str]:
            events.append("verify-while-private")
            assert self.private is True
            return sorted(path.name for path in remote.iterdir())

        def hf_hub_download(self, **kwargs: object) -> str:
            assert self.private is True
            return str(remote / str(kwargs["filename"]))

        def update_repo_settings(self, **kwargs: object) -> None:
            assert events[-1] == "verify-while-private"
            assert kwargs["private"] is False
            assert kwargs["gated"] == "auto"
            events.append("switch-public-gated")
            self.private = False
            self.gated = "auto"

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.HfApi = FakeApi
    fake_errors = types.ModuleType("huggingface_hub.errors")
    fake_errors.RepositoryNotFoundError = type("RepositoryNotFoundError", (Exception,), {})
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", fake_errors)
    monkeypatch.setattr(
        publication_module,
        "validate_phase5_file",
        lambda *_args, **_kwargs: {"valid": True, "phase": 5},
    )

    config = load_project_config(ROOT / "configs" / "project.toml")
    plan = build_publication_plan(
        adapter_dir=adapter_dir(tmp_path),
        model_card_template=ROOT / "model_card" / "README.md",
        repo_id="steven0226/tw-med-llm-qlora-adapter",
        visibility="public",
        gated="auto",
        github_url="https://github.com/kuotunyu/tw-med-llm-qlora",
        config=config,
    )
    receipt = execute_publication(
        plan,
        token="write-token",
        config=config,
        confirmation_code=plan.confirmation_code,
        acceptance_manifest=tmp_path / "acceptance.json",
        output_dir=tmp_path / "outputs",
    )

    assert events == [
        "create-private",
        "upload-while-private",
        "verify-while-private",
        "switch-public-gated",
    ]
    assert receipt["phase"] == 7
    assert receipt["visibility"] == "public"
    assert receipt["gated"] == "auto"
    assert receipt["remote_files_verified"] is True
    assert "tokenizer_config.json" not in {path.name for path in remote.iterdir()}
    assert (tmp_path / "outputs" / "phase7-publication-receipt.json").is_file()


def test_execution_gate_can_be_closed_explicitly(tmp_path: Path) -> None:
    config = load_project_config(ROOT / "configs" / "project.toml")
    config.raw["publication"]["enabled"] = False
    plan = build_plan(tmp_path)

    with pytest.raises(PublicationError, match="publication.enabled is false"):
        assert_publication_execution_gate(
            plan,
            config=config,
            confirmation_code=plan.confirmation_code,
            acceptance_manifest=tmp_path / "acceptance.json",
        )


def test_repository_publication_gate_matches_confirmed_public_gated_target() -> None:
    publication = load_project_config(ROOT / "configs" / "project.toml").raw["publication"]

    assert publication == {
        "enabled": True,
        "adapter_repo_id": "steven0226/tw-med-llm-qlora-adapter",
        "visibility": "public",
        "gated": "auto",
        "phase5_receipt_visibility": "private",
        "github_repository_url": "https://github.com/kuotunyu/tw-med-llm-qlora",
        "requires_explicit_repo_id": True,
        "requires_explicit_visibility_confirmation": True,
    }


def test_invalid_repo_id_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(PublicationError, match="owner/name"):
        build_publication_plan(
            adapter_dir=adapter_dir(tmp_path),
            model_card_template=ROOT / "model_card" / "README.md",
            repo_id="missing-owner",
            visibility="private",
            github_url="https://github.com/owner/repo",
            config=load_project_config(ROOT / "configs" / "project.toml"),
        )

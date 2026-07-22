import hashlib
import json
from pathlib import Path

import pytest

from tw_med_qlora.config import load_project_config
from tw_med_qlora.publication import (
    PublicationError,
    _stage_folder,
    assert_publication_execution_gate,
    build_publication_plan,
    publication_confirmation_code,
    render_model_card,
)

ROOT = Path(__file__).parents[1]


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
        github_url="https://github.com/researcher/tw-med-llm-qlora",
        config=load_project_config(ROOT / "configs" / "project.toml"),
    )


def test_publication_plan_is_dry_and_allowlisted(tmp_path: Path) -> None:
    plan = build_plan(tmp_path)
    summary = plan.public_summary()

    assert summary["external_mutation_performed"] is False
    assert set(plan.files) == {
        "README.md",
        "adapter_config.json",
        "adapter_model.safetensors",
        "tokenizer_config.json",
    }
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

    with pytest.raises(PublicationError, match="required disclosures"):
        render_model_card(
            template.replace("72.05%", "missing"),
            repo_id="owner/model",
            github_url="https://github.com/owner/repo",
        )


def test_confirmation_code_is_bound_to_visibility() -> None:
    private = publication_confirmation_code("owner/model", "private")
    public = publication_confirmation_code("owner/model", "public")

    assert private.startswith("PUBLISH_ADAPTER_")
    assert private != public


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


def test_repository_publication_gate_matches_confirmed_private_target() -> None:
    publication = load_project_config(ROOT / "configs" / "project.toml").raw["publication"]

    assert publication == {
        "enabled": True,
        "adapter_repo_id": "steven0226/tw-med-llm-qlora-adapter",
        "visibility": "private",
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

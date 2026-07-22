import re
from pathlib import Path

ROOT = Path(__file__).parents[1]
SKILL_ROOT = ROOT / ".codex" / "skills" / "tw-med-qlora-workflow"


def test_project_skill_has_valid_required_structure() -> None:
    expected = {
        "SKILL.md",
        "agents/openai.yaml",
        "references/phase-checklist.md",
    }
    actual = {
        path.relative_to(SKILL_ROOT).as_posix()
        for path in SKILL_ROOT.rglob("*")
        if path.is_file()
    }
    assert actual == expected


def test_project_skill_frontmatter_and_references() -> None:
    skill = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")
    assert skill.startswith("---\n")
    frontmatter = skill.split("---", maxsplit=2)[1]
    assert re.search(r"(?m)^name: tw-med-qlora-workflow$", frontmatter)
    assert re.search(r"(?m)^description: .{20,}$", frontmatter)

    local_links = re.findall(r"\[[^]]+\]\(([^)]+)\)", skill)
    assert local_links
    for relative in local_links:
        assert not relative.startswith(("http://", "https://", "/"))
        assert (SKILL_ROOT / relative).is_file()


def test_project_skill_agent_metadata_invokes_the_skill() -> None:
    metadata = (SKILL_ROOT / "agents" / "openai.yaml").read_text(encoding="utf-8")
    assert 'display_name: "TW Med QLoRA Workflow"' in metadata
    assert 'default_prompt: "Use $tw-med-qlora-workflow ' in metadata

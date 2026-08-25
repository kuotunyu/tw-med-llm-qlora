from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_portfolio_closure_records_artifact_license_and_safety_boundaries() -> None:
    closure_path = ROOT / "docs" / "portfolio-closure.md"
    assert closure_path.is_file(), "缺少 portfolio closure 文件"
    readme_path = ROOT / "README.md"
    assert readme_path.is_file(), "缺少 README 文件"

    text = closure_path.read_text(encoding="utf-8")
    readme_text = readme_path.read_text(encoding="utf-8")
    required_sections = (
        "## Released artifacts",
        "## Evaluation evidence",
        "## License chain",
        "## Safety boundary",
        "## Reproduction boundary",
    )
    required_markers = (
        "steven0226/tw-med-llm-qlora-adapter",
        "b1d8f74291da75d0719b5a3ea0d088ee8236e096",
        "step 700",
        "MIT covers repository code only",
        "adapter, TAIDE/Gemma base, and datasets retain their own upstream terms",
        "does not constitute medical advice",
    )

    for marker in (*required_sections, *required_markers):
        assert marker in text, f"portfolio closure 缺少必要標記：{marker}"

    canonical_url = (
        "https://github.com/kuotunyu/tw-med-llm-qlora/blob/main/docs/portfolio-closure.md"
    )
    assert canonical_url in readme_text, "README 缺少 canonical portfolio closure 連結"
    assert "(docs/portfolio-closure.md)" not in readme_text, "README 不應使用相對結案文件連結"

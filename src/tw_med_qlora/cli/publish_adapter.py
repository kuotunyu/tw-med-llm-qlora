"""Plan an adapter publication; execute only after every Phase 5 gate passes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from tw_med_qlora.config import load_project_config
from tw_med_qlora.publication import (
    assert_publication_execution_gate,
    build_publication_plan,
    execute_publication,
)

ROOT = Path(__file__).parents[3]


def main() -> int:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adapter_dir", type=Path)
    parser.add_argument("--repo-id", default=os.getenv("HF_ADAPTER_REPO_ID"))
    parser.add_argument("--visibility", choices=("private", "public"), default="private")
    parser.add_argument("--github-url", default=os.getenv("GITHUB_REPOSITORY_URL"))
    parser.add_argument("--model-card", type=Path, default=ROOT / "model_card" / "README.md")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "project.toml")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirmation-code")
    parser.add_argument("--acceptance-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "publication")
    args = parser.parse_args()
    if not args.repo_id or not args.github_url:
        parser.error("--repo-id and --github-url are required")

    config = load_project_config(args.config)
    plan = build_publication_plan(
        adapter_dir=args.adapter_dir,
        model_card_template=args.model_card,
        repo_id=args.repo_id,
        visibility=args.visibility,
        github_url=args.github_url,
        config=config,
    )
    if not args.execute:
        print(json.dumps(plan.public_summary(), ensure_ascii=False, indent=2))
        print("Dry-run only. No repository was created and no files were uploaded.")
        return 0

    assert_publication_execution_gate(
        plan,
        config=config,
        confirmation_code=args.confirmation_code,
        acceptance_manifest=args.acceptance_manifest,
    )
    receipt = execute_publication(
        plan,
        token=os.getenv("HF_TOKEN", ""),
        config=config,
        confirmation_code=args.confirmation_code or "",
        acceptance_manifest=args.acceptance_manifest,
        output_dir=args.output_dir,
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

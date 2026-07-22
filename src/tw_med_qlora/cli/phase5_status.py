"""Report exactly which Phase 5 completion gates are passed, pending, or failed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tw_med_qlora.config import load_project_config
from tw_med_qlora.phase5_readiness import assess_phase5_readiness

ROOT = Path(__file__).parents[3]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--acceptance-manifest", type=Path)
    parser.add_argument("--publication-receipt", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "project.toml")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    report = assess_phase5_readiness(
        repo_root=ROOT,
        config=load_project_config(args.config),
        adapter_dir=args.adapter,
        acceptance_manifest=args.acceptance_manifest,
        publication_receipt=args.publication_receipt,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    if args.require_complete and not report["phase5_complete"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Validate Phase 4 A100 calibration artifacts and diagnose parser compatibility."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tw_med_qlora.phase4_evidence import validate_phase4_calibration_evidence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--calibration-summary", type=Path, required=True)
    parser.add_argument("--private-archive", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/project.toml"))
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = validate_phase4_calibration_evidence(
        manifest_path=args.manifest,
        receipt_path=args.receipt,
        calibration_summary_path=args.calibration_summary,
        private_archive_path=args.private_archive,
        config_path=args.config,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()

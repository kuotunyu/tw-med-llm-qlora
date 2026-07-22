"""CLI for validating Phase 2 Colab smoke-test artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tw_med_qlora.phase2_evidence import validate_phase2_evidence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/project.toml"))
    parser.add_argument("--compute-units-per-hour", type=float, required=True)
    parser.add_argument("--current-compute-units", type=float, required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = validate_phase2_evidence(
        manifest_path=args.manifest,
        receipt_path=args.receipt,
        config_path=args.config,
        compute_units_per_hour=args.compute_units_per_hour,
        current_compute_units=args.current_compute_units,
    )
    rendered = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()

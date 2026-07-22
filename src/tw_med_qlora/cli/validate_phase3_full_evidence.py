"""CLI for validating completed Phase 3 full-training artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tw_med_qlora.phase3_full_evidence import validate_phase3_full_evidence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--trainer-log", type=Path, required=True)
    parser.add_argument("--training-curves", type=Path, required=True)
    parser.add_argument("--calibration-validation", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/project.toml"))
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = validate_phase3_full_evidence(
        manifest_path=args.manifest,
        receipt_path=args.receipt,
        trainer_log_path=args.trainer_log,
        training_curves_path=args.training_curves,
        calibration_validation_path=args.calibration_validation,
        config_path=args.config,
    )
    rendered = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()

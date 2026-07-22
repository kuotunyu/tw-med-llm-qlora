"""CLI for validating a downloaded Phase 4 full-evaluation delivery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tw_med_qlora.phase4_full_evidence import validate_phase4_full_evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--public-archive", type=Path, required=True)
    parser.add_argument("--private-cases-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = validate_phase4_full_evidence(
        manifest_path=args.manifest,
        receipt_path=args.receipt,
        public_archive_path=args.public_archive,
        private_cases_archive_path=args.private_cases_archive,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    print(rendered, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

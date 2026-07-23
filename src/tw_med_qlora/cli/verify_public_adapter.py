"""Verify a Phase 7 Hugging Face public gated adapter without exposing credentials."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tw_med_qlora.config import load_project_config
from tw_med_qlora.phase7_publication import verify_public_adapter

ROOT = Path(__file__).parents[3]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "project.toml")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "publication" / "phase7-public-validation.json",
    )
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
    except ImportError:
        load_dotenv = None
    if load_dotenv is not None:
        load_dotenv(ROOT / ".env")

    report = verify_public_adapter(
        receipt_path=args.receipt,
        config=load_project_config(args.config),
        token=os.getenv("HF_TOKEN", ""),
        output_path=args.output,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

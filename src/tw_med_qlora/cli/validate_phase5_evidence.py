"""Validate a content-safe manifest returned from the RTX 4090 computer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tw_med_qlora.config import load_project_config
from tw_med_qlora.phase5_evidence import validate_phase5_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parents[3] / "configs" / "project.toml",
    )
    args = parser.parse_args()
    result = validate_phase5_file(args.manifest, config=load_project_config(args.config))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

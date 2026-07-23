"""Install the release wheel without dependencies and verify every console script."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

CONSOLE_SCRIPTS = (
    "tw-med-local-infer",
    "tw-med-validate-phase5",
    "tw-med-phase5-status",
    "tw-med-publish-adapter",
)


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, capture_output=True, text=True)


def smoke_install(directory: Path, venv: Path) -> dict[str, Any]:
    wheels = sorted(directory.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected one wheel, found {len(wheels)}")

    run(["uv", "venv", "--clear", "--python", "3.11", str(venv)])
    executable_dir = venv / ("Scripts" if os.name == "nt" else "bin")
    python = executable_dir / ("python.exe" if os.name == "nt" else "python")
    run(["uv", "pip", "install", "--python", str(python), "--no-deps", str(wheels[0])])

    probe = run(
        [
            str(python),
            "-c",
            (
                "import importlib.metadata as m, json, tw_med_qlora; "
                "print(json.dumps({'version': m.version('tw-med-llm-qlora'), "
                "'module': tw_med_qlora.__name__, "
                "'distributions': sorted(d.metadata['Name'] for d in m.distributions())}))"
            ),
        ]
    )
    package = json.loads(probe.stdout)
    if package != {
        "version": "0.1.0",
        "module": "tw_med_qlora",
        "distributions": ["tw-med-llm-qlora"],
    }:
        raise RuntimeError(f"unexpected clean-environment package state: {package}")

    commands: dict[str, dict[str, Any]] = {}
    suffix = ".exe" if os.name == "nt" else ""
    for name in CONSOLE_SCRIPTS:
        completed = run([str(executable_dir / f"{name}{suffix}"), "--help"])
        first_line = completed.stdout.splitlines()[0] if completed.stdout else ""
        if "usage:" not in first_line:
            raise RuntimeError(f"{name} did not render argparse help")
        commands[name] = {"exit_code": completed.returncode, "first_line": first_line}

    return {
        "schema_version": 1,
        "python": "3.11",
        "package": package,
        "console_scripts": commands,
        "dependency_isolation": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--venv", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = smoke_install(args.directory, args.venv)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

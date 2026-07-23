import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "run_ollama_acceptance.ps1"


def test_ollama_acceptance_validates_export_before_local_import() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert 'gguf-export-receipt.json' in source
    assert 'optional_export -ne "gguf_q4_k_m"' in source
    assert 'quantization_method -ne "q4_k_m"' in source
    assert "ExpectedBaseModelRevision" in source
    assert "ExpectedPhase3ArchiveSha256" in source
    assert "ExpectedAdapterCheckpoint" in source
    assert "approved_compute_units_with_20pct_buffer" in source
    assert "GGUF SHA-256 does not match the export receipt." in source
    assert "Modelfile SHA-256 does not match the export receipt." in source
    assert "& ollama create" in source
    assert "& ollama show $ModelName --modelfile" in source
    assert "& ollama run" in source
    assert "& ollama ps" in source
    assert '100%\\s+GPU' in source
    assert "function Get-FileSha256" in source
    assert "[System.IO.File]::OpenRead" in source
    assert "Get-FileHash" not in source
    assert "raw_output_recorded = $false" in source
    assert "external_upload_performed = $false" in source
    assert "Set-Content" not in source
    assert "[System.IO.File]::WriteAllText" in source


def test_ollama_acceptance_script_parses_when_powershell_is_available() -> None:
    powershell = shutil.which("powershell") or shutil.which("pwsh")
    if powershell is None:
        pytest.skip("PowerShell is not installed in this test environment")

    script_path = str(SCRIPT).replace("'", "''")
    command = (
        "$tokens = $null; $errors = $null; "
        f"[System.Management.Automation.Language.Parser]::ParseFile('{script_path}', "
        "[ref]$tokens, [ref]$errors) > $null; "
        "if ($errors.Count -gt 0) { "
        "$errors | ForEach-Object { Write-Error $_.Message }; exit 1 }"
    )
    subprocess.run(
        [powershell, "-NoProfile", "-NonInteractive", "-Command", command],
        cwd=ROOT,
        check=True,
    )


@pytest.mark.skipif(os.name != "nt", reason="The acceptance entry point is Windows-only")
def test_ollama_acceptance_contract_with_fake_cli(tmp_path: Path) -> None:
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("Windows PowerShell is not installed")

    gguf = tmp_path / "tw-med-q4-k-m.gguf"
    gguf.write_bytes(b"GGUF-test-payload")
    modelfile = tmp_path / "Modelfile"
    modelfile.write_text(
        "\n".join(
            [
                f"FROM ./{gguf.name}",
                "PARAMETER temperature 0",
                "PARAMETER seed 3407",
                "PARAMETER num_ctx 2048",
                "PARAMETER num_predict 64",
                "",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )

    def file_record(path: Path) -> dict[str, int | str]:
        payload = path.read_bytes()
        return {
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    export_receipt = {
        "schema_version": 2,
        "optional_export": "gguf_q4_k_m",
        "quantization_method": "q4_k_m",
        "base_model_id": "taide/Gemma-3-TAIDE-12b-Chat-2602",
        "base_model_revision": "4de0b93b99f8b61b59c40d019fd593bdd1c42249",
        "phase3_archive_sha256": (
            "2c537dfd3049319286c678a3ca3aa72e3f20baa7e0f44bde93ff7ee4dc47e43e"
        ),
        "adapter_checkpoint": 700,
        "approval": {"approved_compute_units_with_20pct_buffer": 6.36},
        "files": {
            gguf.name: file_record(gguf),
            "Modelfile": file_record(modelfile),
        },
        "published": False,
        "external_upload_performed": False,
    }
    (tmp_path / "gguf-export-receipt.json").write_text(
        json.dumps(export_receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    fake_ollama = tmp_path / "ollama.cmd"
    fake_ollama.write_text(
        "\n".join(
            [
                "@echo off",
                'if /I "%~1"=="--version" (echo ollama version is 0.32.0& exit /b 0)',
                'if /I "%~1"=="list" (echo NAME ID SIZE MODIFIED& exit /b 0)',
                'if /I "%~1"=="create" (exit /b 0)',
                'if /I "%~1"=="show" (echo FROM .\\tw-med-q4-k-m.gguf& exit /b 0)',
                'if /I "%~1"=="run" (echo C& exit /b 0)',
                (
                    'if /I "%~1"=="ps" '
                    "(echo NAME ID SIZE PROCESSOR UNTIL& "
                    "echo tw-med-taide-12b-q4-k-m:latest abc 8GB 100%% GPU 4m& "
                    "exit /b 0)"
                ),
                "exit /b 1",
                "",
            ]
        ),
        encoding="ascii",
        newline="\r\n",
    )

    environment = os.environ.copy()
    environment["PATH"] = str(tmp_path) + os.pathsep + environment["PATH"]
    result = subprocess.run(
        [
            powershell,
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-ExportDirectory",
            str(tmp_path),
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert result.returncode == 0, result.stdout + result.stderr

    acceptance_bytes = (tmp_path / "ollama-acceptance.json").read_bytes()
    assert not acceptance_bytes.startswith(b"\xef\xbb\xbf")
    assert b"\r\n" not in acceptance_bytes
    acceptance = json.loads(acceptance_bytes)
    assert acceptance["schema_version"] == 2
    assert acceptance["passed"] is True
    assert acceptance["gpu_fully_loaded"] is True
    assert acceptance["gguf_sha256"] == file_record(gguf)["sha256"]
    assert acceptance["output_sha256"] == hashlib.sha256(b"C").hexdigest()
    assert acceptance["raw_output_recorded"] is False
    assert acceptance["external_upload_performed"] is False

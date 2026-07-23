import hashlib
import json
import os
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "run_ollama_vlm_acceptance.ps1"


def test_vlm_acceptance_enforces_multimodal_and_content_safe_contract() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert 'optional_export -ne "gguf_q4_k_m"' in source
    assert "adapter_merge.peft_detected" in source
    assert "lora_parameter_tensors" in source
    assert "gguf.primary_file" in source
    assert "gguf.projector_files" in source
    assert "vlm_projector_archived" in source
    assert "Text acceptance and export receipt refer to different" in source
    assert '"FROM ./$primaryName"' in source
    assert '"FROM ./$($projectorNames[0])"' in source
    assert "/api/show" in source
    assert "/api/chat" in source
    assert "OllamaApiBaseUrl must use HTTP on a loopback host." in source
    assert '$capabilities -notcontains "completion"' in source
    assert '$capabilities -notcontains "vision"' in source
    assert 'projector_info."general.architecture"' in source
    assert "System.Drawing.Bitmap" in source
    assert "synthetic_red_square_v1" in source
    assert '$answer -ceq "RED"' in source
    assert "RED[.]?" not in source
    assert "100%\\s+GPU" in source
    assert "fixture_recorded = $false" in source
    assert "raw_output_recorded = $false" in source
    assert "imported_modelfile_recorded = $false" in source
    assert "ollama_ps_recorded = $false" in source
    assert "external_upload_performed = $false" in source
    assert "Get-FileHash" not in source
    assert "Set-Content" not in source


def test_vlm_acceptance_script_parses_when_powershell_is_available() -> None:
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


@pytest.mark.skipif(os.name != "nt", reason="The VLM acceptance entry point is Windows-only")
def test_vlm_acceptance_rejects_non_loopback_api(tmp_path: Path) -> None:
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("Windows PowerShell is not installed")

    fake_ollama = tmp_path / "ollama.cmd"
    fake_ollama.write_text(
        "@echo off\r\nexit /b 0\r\n",
        encoding="ascii",
        newline="",
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
            "-OllamaApiBaseUrl",
            "https://example.invalid",
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    assert result.returncode != 0
    assert "OllamaApiBaseUrl must use HTTP on a loopback host." in (
        result.stdout + result.stderr
    )
    assert not (tmp_path / "ollama-vlm-acceptance.json").exists()


@pytest.mark.skipif(os.name != "nt", reason="The VLM acceptance entry point is Windows-only")
def test_vlm_acceptance_contract_with_fake_cli_and_loopback_api(
    tmp_path: Path,
) -> None:
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("Windows PowerShell is not installed")

    primary = tmp_path / "tw-med-q4-k-m.gguf"
    primary.write_bytes(b"GGUF-primary-test-payload")
    projector = tmp_path / "tw-med-BF16-mmproj.gguf"
    projector.write_bytes(b"GGUF-projector-test-payload")

    def file_record(path: Path) -> dict[str, int | str]:
        payload = path.read_bytes()
        return {
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    export_receipt = {
        "schema_version": 3,
        "optional_export": "gguf_q4_k_m",
        "base_model_id": "taide/Gemma-3-TAIDE-12b-Chat-2602",
        "base_model_revision": "4de0b93b99f8b61b59c40d019fd593bdd1c42249",
        "adapter_checkpoint": 700,
        "adapter_merge": {
            "peft_detected": True,
            "lora_parameter_tensors": 42,
        },
        "gguf": {
            "primary_file": primary.name,
            "projector_files": [projector.name],
            "vlm_projector_archived": True,
        },
        "files": {
            primary.name: file_record(primary),
            projector.name: file_record(projector),
        },
        "published": False,
        "external_upload_performed": False,
    }
    export_path = tmp_path / "gguf-export-receipt.json"
    export_path.write_text(
        json.dumps(export_receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    text_acceptance = {
        "passed": True,
        "gpu_fully_loaded": True,
        "gguf_sha256": file_record(primary)["sha256"],
        "external_upload_performed": False,
    }
    text_acceptance_path = tmp_path / "ollama-acceptance.json"
    text_acceptance_path.write_text(
        json.dumps(text_acceptance, ensure_ascii=False, indent=2) + "\n",
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
                (
                    'if /I "%~1"=="ps" '
                    "(echo NAME ID SIZE PROCESSOR UNTIL& "
                    "echo tw-med-taide-12b-q4-k-m-vlm:latest abc 9GB "
                    "100%% GPU 4m& exit /b 0)"
                ),
                "exit /b 1",
                "",
            ]
        ),
        encoding="ascii",
        newline="\r\n",
    )

    class OllamaHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            content_length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(content_length))
            if self.path == "/api/show":
                assert request["model"] == "tw-med-taide-12b-q4-k-m-vlm"
                payload = {
                    "capabilities": ["completion", "vision"],
                    "projector_info": {"general.architecture": "clip"},
                    "modelfile": (
                        f"FROM ./{primary.name}\n"
                        f"FROM ./{projector.name}\n"
                    ),
                }
            elif self.path == "/api/chat":
                assert request["model"] == "tw-med-taide-12b-q4-k-m-vlm"
                assert request["stream"] is False
                assert len(request["messages"][0]["images"]) == 1
                payload = {
                    "message": {"content": "RED"},
                    "total_duration": 714_377_200,
                    "load_duration": 165_865_300,
                    "prompt_eval_count": 292,
                    "eval_count": 2,
                }
            else:
                self.send_error(404)
                return
            response = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), OllamaHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    environment = os.environ.copy()
    environment["PATH"] = str(tmp_path) + os.pathsep + environment["PATH"]
    try:
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
                "-OllamaApiBaseUrl",
                f"http://127.0.0.1:{server.server_port}",
            ],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)

    assert result.returncode == 0, result.stdout + result.stderr
    receipt_bytes = (tmp_path / "ollama-vlm-acceptance.json").read_bytes()
    assert not receipt_bytes.startswith(b"\xef\xbb\xbf")
    assert b"\r\n" not in receipt_bytes
    receipt = json.loads(receipt_bytes)
    assert receipt["passed"] is True
    assert receipt["capabilities"] == ["completion", "vision"]
    assert receipt["projector_architecture"] == "clip"
    assert receipt["imported_from_records"] == 2
    assert receipt["gpu_fully_loaded"] is True
    assert receipt["gguf_sha256"] == file_record(primary)["sha256"]
    assert receipt["projector_sha256"] == file_record(projector)["sha256"]
    assert receipt["output_sha256"] == hashlib.sha256(b"RED").hexdigest()
    assert receipt["fixture_recorded"] is False
    assert receipt["raw_output_recorded"] is False
    assert receipt["external_upload_performed"] is False
    assert not list(tmp_path.glob(".vlm-acceptance-*"))

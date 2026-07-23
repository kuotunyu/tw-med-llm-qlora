import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
EXPORT_EVIDENCE = (
    ROOT / "reports" / "phase5" / "20260723T080232Z-gguf-export-receipt.json"
)
OLLAMA_EVIDENCE = (
    ROOT / "reports" / "phase5" / "20260723T081846Z-ollama-acceptance.json"
)
OLLAMA_VLM_EVIDENCE = (
    ROOT / "reports" / "phase5" / "20260723T104400Z-ollama-vlm-acceptance.json"
)


def test_archived_gguf_and_ollama_evidence_is_content_safe() -> None:
    export_bytes = EXPORT_EVIDENCE.read_bytes()
    text_acceptance_bytes = OLLAMA_EVIDENCE.read_bytes()
    export = json.loads(export_bytes)
    acceptance = json.loads(text_acceptance_bytes)
    vlm_acceptance = json.loads(OLLAMA_VLM_EVIDENCE.read_bytes())

    assert export["schema_version"] == 3
    assert export["run_id"] == "20260723T074001Z"
    assert export["adapter_merge"]["peft_detected"] is True
    assert export["adapter_merge"]["lora_parameter_tensors"] == 672
    assert export["resources"]["expected_size_range_match"] is True
    assert export["gguf"]["vlm_projector_archived"] is True
    assert export["external_upload_performed"] is False
    assert export["published"] is False

    assert acceptance["export_receipt_sha256"] == hashlib.sha256(
        export_bytes
    ).hexdigest()
    assert acceptance["gguf_sha256"] == export["files"][
        export["gguf"]["primary_file"]
    ]["sha256"]
    assert acceptance["projector_count"] == 1
    assert acceptance["vlm_processor_required"] is True
    assert acceptance["gpu_fully_loaded"] is True
    assert acceptance["passed"] is True
    assert acceptance["raw_output_recorded"] is False
    assert acceptance["external_upload_performed"] is False

    assert vlm_acceptance["export_receipt_sha256"] == hashlib.sha256(
        export_bytes
    ).hexdigest()
    assert vlm_acceptance["text_acceptance_receipt_sha256"] == hashlib.sha256(
        text_acceptance_bytes
    ).hexdigest()
    assert vlm_acceptance["gguf_sha256"] == export["files"][
        export["gguf"]["primary_file"]
    ]["sha256"]
    projector = export["gguf"]["projector_files"][0]
    assert vlm_acceptance["projector_sha256"] == export["files"][projector]["sha256"]
    assert vlm_acceptance["capabilities"] == ["completion", "vision"]
    assert vlm_acceptance["projector_architecture"] == "clip"
    assert vlm_acceptance["imported_from_records"] == 2
    assert vlm_acceptance["gpu_fully_loaded"] is True
    assert vlm_acceptance["probe"] == "synthetic_red_square_v1"
    assert vlm_acceptance["expected_answer"] == "RED"
    assert vlm_acceptance["output_sha256"] == hashlib.sha256(b"RED").hexdigest()
    assert vlm_acceptance["total_seconds"] > 0
    assert vlm_acceptance["api_total_seconds"] > 0
    assert vlm_acceptance["passed"] is True
    assert vlm_acceptance["fixture_recorded"] is False
    assert vlm_acceptance["raw_output_recorded"] is False
    assert vlm_acceptance["imported_modelfile_recorded"] is False
    assert vlm_acceptance["ollama_ps_recorded"] is False
    assert vlm_acceptance["external_upload_performed"] is False

    serialized = json.dumps(
        {
            "export": export,
            "acceptance": acceptance,
            "vlm_acceptance": vlm_acceptance,
        },
        ensure_ascii=False,
    )
    assert "C:\\" not in serialized
    assert "/content/" not in serialized
    assert '"raw_output":' not in serialized

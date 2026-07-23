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


def test_archived_gguf_and_ollama_evidence_is_content_safe() -> None:
    export_bytes = EXPORT_EVIDENCE.read_bytes()
    export = json.loads(export_bytes)
    acceptance = json.loads(OLLAMA_EVIDENCE.read_bytes())

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

    serialized = json.dumps(
        {"export": export, "acceptance": acceptance},
        ensure_ascii=False,
    )
    assert "C:\\" not in serialized
    assert "/content/" not in serialized
    assert '"raw_output":' not in serialized

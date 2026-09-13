from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
MEDIA = ROOT / "media" / "manim"


def _load_module():
    name = "media_manim_results_data"
    spec = importlib.util.spec_from_file_location(name, MEDIA / "results_data.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module  # dataclasses resolve the module through sys.modules
    spec.loader.exec_module(module)
    return module


def test_animation_numbers_come_from_saved_evidence() -> None:
    module = _load_module()
    data = module.load_results()
    phase4 = json.loads(module.PHASE4_RESULTS.read_text(encoding="utf-8"))
    analysis = json.loads(module.V1_1_ANALYSIS.read_text(encoding="utf-8"))

    adapter = phase4["medqa"]["models"]["localized-medical-adapter"]
    assert data.medqa.questions == adapter["total"]
    assert data.medqa.accuracy_pct[2] == adapter["accuracy"] * 100
    base = phase4["tmmluplus"]["models"]["localized-base"]["overall"]
    assert data.tmmlu.accuracy_pct[1] == base["accuracy"] * 100
    assert data.tmmlu.parse_rate_pct[1] == base["parse_rate"] * 100
    assert data.control_conclusion == phase4["tmmluplus"]["catastrophic_forgetting"]["conclusion"]
    rescored = analysis["variants"]["retained_new_gold"]
    assert data.tmmlu_v1_1_questions == rescored["questions_per_model"]
    assert data.tmmlu_v1_1_adapter_accuracy_pct == (
        rescored["models"]["localized-medical-adapter"]["overall"]["accuracy"] * 100
    )


def test_scene_script_contains_no_hand_typed_result_numbers() -> None:
    source = (MEDIA / "results_scene.py").read_text(encoding="utf-8")
    assert "load_results" in source
    # Percentages or pp deltas typed as literals would bypass the evidence files.
    assert not re.search(r"\b\d{2}\.\d{2}\b", source)
    assert (ROOT / "media" / "results.gif").is_file()

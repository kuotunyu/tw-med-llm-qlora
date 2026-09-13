"""Load the numbers shown in the README results animation from saved evidence.

Standard library only, so tests can import it without Manim. Every value comes from the
frozen Phase 4 public results or the TMMLU+ v1.1 rescoring analysis; nothing is typed by
hand in the animation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PHASE4_RESULTS = ROOT / "reports" / "phase4" / "full" / "public" / "phase4-results.json"
V1_1_ANALYSIS = ROOT / "reports" / "tmmlu-v1.1" / "20260913T170025Z" / "analysis.json"

MODEL_ORDER = ("original-instruct", "localized-base", "localized-medical-adapter")
MODEL_NAMES = {
    "original-instruct": "原始 Instruct",
    "localized-base": "TAIDE Base",
    "localized-medical-adapter": "Step 700 Adapter",
}


@dataclass(frozen=True)
class Panel:
    title: str
    questions: int
    accuracy_pct: tuple[float, float, float]
    parse_rate_pct: tuple[float, float, float]
    adapter_minus_base_pp: float


@dataclass(frozen=True)
class ResultsData:
    medqa: Panel
    tmmlu: Panel
    control_conclusion: str
    control_ci_lower_pp: float
    tmmlu_v1_1_adapter_accuracy_pct: float
    tmmlu_v1_1_questions: int
    tmmlu_v1_1_control_conclusion: str


def _panel(models: dict, *, title: str, questions: int, accuracy_key) -> Panel:
    accuracy = tuple(accuracy_key(models[m]) * 100 for m in MODEL_ORDER)
    parse = tuple(_parse_rate(models[m]) * 100 for m in MODEL_ORDER)
    return Panel(
        title=title,
        questions=questions,
        accuracy_pct=accuracy,
        parse_rate_pct=parse,
        adapter_minus_base_pp=accuracy[2] - accuracy[1],
    )


def _parse_rate(model: dict) -> float:
    return model["overall"]["parse_rate"] if "overall" in model else model["parse_rate"]


def load_results(
    phase4_path: Path = PHASE4_RESULTS, v1_1_path: Path = V1_1_ANALYSIS
) -> ResultsData:
    phase4 = json.loads(phase4_path.read_text(encoding="utf-8"))
    v1_1 = json.loads(v1_1_path.read_text(encoding="utf-8"))
    medqa_models = phase4["medqa"]["models"]
    tmmlu_models = phase4["tmmluplus"]["models"]
    forgetting = phase4["tmmluplus"]["catastrophic_forgetting"]
    rescored = v1_1["variants"]["retained_new_gold"]
    control_v1_1 = v1_1["paired"]["retained_new_gold"]["control_noninferiority_adapter_vs_base"]
    return ResultsData(
        medqa=_panel(
            medqa_models,
            title="MedQA 臺灣醫師國考 Test",
            questions=medqa_models["localized-medical-adapter"]["total"],
            accuracy_key=lambda m: m["accuracy"],
        ),
        tmmlu=_panel(
            tmmlu_models,
            title="TMMLU+ 13 科 Test",
            questions=tmmlu_models["localized-medical-adapter"]["overall"]["total"],
            accuracy_key=lambda m: m["overall"]["accuracy"],
        ),
        control_conclusion=forgetting["conclusion"],
        control_ci_lower_pp=forgetting["ci_lower_percentage_points"],
        tmmlu_v1_1_adapter_accuracy_pct=(
            rescored["models"]["localized-medical-adapter"]["overall"]["accuracy"] * 100
        ),
        tmmlu_v1_1_questions=rescored["questions_per_model"],
        tmmlu_v1_1_control_conclusion=control_v1_1["conclusion"],
    )


if __name__ == "__main__":
    print(load_results())

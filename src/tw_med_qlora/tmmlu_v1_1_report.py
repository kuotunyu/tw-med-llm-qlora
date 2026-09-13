"""Render the TMMLU+ v1.1 limited-evaluation tables from saved per-question evidence.

Every number in the generated Markdown comes from ``analysis.json`` (historical outputs
rescored under v1.1) or from the separately labelled local-probe analysis. Nothing is
typed by hand, and the document section between the markers is replaced wholesale.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tw_med_qlora.tmmlu_rescore import (
    ADAPTER_MODEL,
    BASE_MODEL,
    INSTRUCT_MODEL,
    MODEL_LABELS,
    VARIANT_RETAINED_NEW_GOLD,
    VARIANT_RETAINED_NEW_GOLD_WITH_DUPLICATES,
    VARIANT_RETAINED_OLD_GOLD,
    VARIANT_V1_0_FULL,
)

BEGIN_MARKER = "<!-- generated:tmmlu-v1.1:begin -->"
END_MARKER = "<!-- generated:tmmlu-v1.1:end -->"
REPORT_ROOT_NAME = "tmmlu-v1.1"
RUN_ID_PATTERN = re.compile(r"\A\d{8}T\d{6}Z\Z")

MODEL_NAMES = {
    INSTRUCT_MODEL: "原始 Instruct",
    BASE_MODEL: "TAIDE Base",
    ADAPTER_MODEL: "Step 700 Adapter",
}
VARIANT_NAMES = {
    VARIANT_V1_0_FULL: "v1.0 全量（凍結結果重算）",
    VARIANT_RETAINED_OLD_GOLD: "保留題組 × v1.0 答案",
    VARIANT_RETAINED_NEW_GOLD: "保留題組 × v1.1 答案（主結果）",
    VARIANT_RETAINED_NEW_GOLD_WITH_DUPLICATES: "敏感度：加入依列序配對的重複列",
}
GROUP_NAMES = {"all_13": "13 科全部", "medical_8": "醫學 8 科", "control_5": "控制 5 科"}


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _pp(value: float) -> str:
    return f"{value:+.2f} pp"


def _ci(stats: Mapping[str, Any]) -> str:
    return (
        f"[{stats['ci_lower_percentage_points']:+.2f}, {stats['ci_upper_percentage_points']:+.2f}]"
    )


def _p_value(value: float) -> str:
    return f"{value:.2e}" if value < 1e-3 else f"{value:.4f}"


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def render_coverage(analysis: Mapping[str, Any]) -> str:
    rows = []
    for group, values in analysis["coverage"].items():
        rows.append(
            [
                GROUP_NAMES.get(group, group),
                str(values["v1_0_total"]),
                str(values["v1_1_total"]),
                str(values["v1_0_retained_with_identical_content"]),
                str(values["v1_0_answer_revised"]),
                str(values["v1_0_removed_or_modified"]),
                str(values["v1_1_added_or_modified_uncovered"]),
                str(values["v1_1_duplicate_ambiguous"]),
                str(values["v1_1_covered_by_rescoring"]),
            ]
        )
    return _table(
        [
            "範圍",
            "v1.0 題數",
            "v1.1 題數",
            "內容完全相同",
            "其中答案修訂",
            "v1.0 刪除或修改",
            "v1.1 新增或修改（未覆蓋）",
            "重複內容歧義",
            "可重新計分的 v1.1 題數",
        ],
        rows,
    )


def render_variant_table(analysis: Mapping[str, Any]) -> str:
    rows = []
    for variant, payload in analysis["variants"].items():
        for model in MODEL_LABELS:
            summary = payload["models"][model]
            overall = summary["overall"]
            length = sum(s["parse_fail_length"] for s in summary["by_subject"].values())
            other = sum(s["parse_fail_other"] for s in summary["by_subject"].values())
            rows.append(
                [
                    VARIANT_NAMES.get(variant, variant),
                    MODEL_NAMES[model],
                    str(overall["total"]),
                    _pct(overall["accuracy"]),
                    _pct(summary["medical_subject_macro_accuracy"]),
                    _pct(summary["control_subject_macro_accuracy"]),
                    _pct(overall["parse_rate"]),
                    str(length),
                    str(other),
                ]
            )
    return _table(
        [
            "計分方式",
            "模型",
            "題數",
            "13 科整體",
            "醫學 8 科 macro",
            "控制 5 科 macro",
            "Parse rate",
            "256-token 截斷",
            "其他解析失敗",
        ],
        rows,
    )


def render_subject_table(
    analysis: Mapping[str, Any],
    *,
    medical_subjects: Sequence[str],
    control_subjects: Sequence[str],
) -> str:
    frozen = analysis["variants"][VARIANT_V1_0_FULL]["models"]
    current = analysis["variants"][VARIANT_RETAINED_NEW_GOLD]["models"]
    rows = []
    for group, subjects in (("醫學", medical_subjects), ("控制", control_subjects)):
        for subject in subjects:
            adapter_new = current[ADAPTER_MODEL]["by_subject"][subject]
            adapter_old = frozen[ADAPTER_MODEL]["by_subject"][subject]
            base_new = current[BASE_MODEL]["by_subject"][subject]
            rows.append(
                [
                    group,
                    subject,
                    f"{adapter_old['total']} → {adapter_new['total']}",
                    _pct(current[INSTRUCT_MODEL]["by_subject"][subject]["accuracy"]),
                    _pct(base_new["accuracy"]),
                    _pct(adapter_new["accuracy"]),
                    _pp((adapter_new["accuracy"] - adapter_old["accuracy"]) * 100),
                    _pct(base_new["parse_rate"]),
                    f"{base_new['parse_fail_length']} / {base_new['parse_fail_other']}",
                ]
            )
    return _table(
        [
            "群組",
            "科目",
            "題數 v1.0 → v1.1 覆蓋",
            "Instruct",
            "Base",
            "Adapter",
            "Adapter Δ vs v1.0",
            "Base parse rate",
            "Base 截斷 / 其他失敗",
        ],
        rows,
    )


def render_paired_table(analysis: Mapping[str, Any]) -> str:
    rows = []
    for variant, payload in analysis["paired"].items():
        control = payload["control_noninferiority_adapter_vs_base"]
        rows.append(
            [
                VARIANT_NAMES.get(variant, variant),
                "控制 5 科 Adapter − Base（subject-macro）",
                str(control["questions"]),
                _pp(control["observed_difference_percentage_points"]),
                _ci(control),
                f"−2 pp 判準：{control['conclusion']}",
            ]
        )
        for key, label in (
            ("medical_macro_adapter_vs_base", "醫學 8 科 Adapter − Base（subject-macro）"),
            ("medical_macro_adapter_vs_instruct", "醫學 8 科 Adapter − Instruct（subject-macro）"),
            ("overall_micro_adapter_vs_base", "13 科整體 Adapter − Base（micro）"),
            ("overall_micro_adapter_vs_instruct", "13 科整體 Adapter − Instruct（micro）"),
        ):
            stats = payload[key]
            mcnemar = stats["mcnemar_pooled"]
            rows.append(
                [
                    VARIANT_NAMES.get(variant, variant),
                    label,
                    str(stats["questions"]),
                    _pp(stats["observed_difference_percentage_points"]),
                    _ci(stats),
                    "McNemar p = "
                    f"{_p_value(mcnemar['two_sided_exact_p_value'])}"
                    f"（{mcnemar['base_wrong_adapter_correct']} 進步 / "
                    f"{mcnemar['base_correct_adapter_wrong']} 退步）",
                ]
            )
    return _table(["計分方式", "比較", "配對題數", "觀察差值", "Bootstrap 95% CI", "判讀"], rows)


def render_decomposition_table(analysis: Mapping[str, Any]) -> str:
    rows = []
    metric_names = {
        "overall_micro": "13 科整體",
        "medical_macro": "醫學 8 科 macro",
        "control_macro": "控制 5 科 macro",
    }
    for model in MODEL_LABELS:
        for metric, values in analysis["decomposition"][model].items():
            rows.append(
                [
                    MODEL_NAMES[model],
                    metric_names[metric],
                    _pct(values["v1_0_full"]),
                    _pct(values["retained_old_gold"]),
                    _pct(values["retained_new_gold"]),
                    _pp(values["removed_question_effect_pp"]),
                    _pp(values["answer_revision_effect_pp"]),
                    _pp(values["total_change_pp"]),
                ]
            )
    return _table(
        [
            "模型",
            "指標",
            "v1.0 全量",
            "保留題組 × v1.0 答案",
            "保留題組 × v1.1 答案",
            "刪題效應",
            "答案修訂效應",
            "合計變化",
        ],
        rows,
    )


def render_version_delta_table(analysis: Mapping[str, Any]) -> str:
    rows = []
    for model, values in analysis["version_delta_same_outputs"].items():
        flips = values["answer_revision_flips"]
        for metric, label in (
            ("overall_micro", "13 科整體"),
            ("medical_macro", "醫學 8 科 macro"),
            ("control_macro", "控制 5 科 macro"),
        ):
            stats = values[metric]
            rows.append(
                [
                    MODEL_NAMES[model],
                    label,
                    str(values["questions"]),
                    f"{flips['gold_changed']}（轉對 {flips['to_correct']} / "
                    f"轉錯 {flips['to_wrong']}）",
                    _pp(stats["observed_difference_percentage_points"]),
                    _ci(stats),
                ]
            )
    return _table(
        ["模型", "指標", "配對題數", "答案修訂題數（結果翻轉）", "v1.1 − v1.0", "Bootstrap 95% CI"],
        rows,
    )


def render_all_parsed_table(analysis: Mapping[str, Any]) -> str:
    rows = []
    for variant, payload in analysis["all_parsed_slice"].items():
        for model in MODEL_LABELS:
            summary = payload["models"][model]
            rows.append(
                [
                    VARIANT_NAMES.get(variant, variant),
                    MODEL_NAMES[model],
                    str(payload["questions"]),
                    _pct(summary["overall"]["accuracy"]),
                    _pct(summary["medical_subject_macro_accuracy"]),
                    _pct(summary["control_subject_macro_accuracy"]),
                ]
            )
    return _table(
        ["計分方式", "模型", "三模型皆解析題數", "13 科整體", "醫學 8 科 macro", "控制 5 科 macro"],
        rows,
    )


def render_probe_table(probe: Mapping[str, Any]) -> str:
    rows = []
    for model in MODEL_LABELS:
        values = probe["models"].get(model)
        if values is None:
            continue
        agreement = values["prediction_agreement_with_historical"]
        ceiling = values.get("in_backend_ceiling_full_vs_stability_3407")
        rows.append(
            [
                MODEL_NAMES[model],
                str(values["probe_questions_with_historical_prompt"]),
                f"{agreement['agree']} / {agreement['total']} = {_pct(agreement['rate'])}",
                f"[{agreement['wilson_ci_lower'] * 100:.1f}%, "
                f"{agreement['wilson_ci_upper'] * 100:.1f}%]",
                (
                    f"{ceiling['agree']} / {ceiling['total']} = {_pct(ceiling['rate'])}"
                    if ceiling
                    else "n/a"
                ),
                str(values["prompt_tokens_identical_to_historical"]),
                _pct(values["accuracy_under_v1_1_gold"]["accuracy"]),
                str(values["uncovered_v1_1_questions"]["total"]),
                str(values["uncovered_v1_1_questions"]["correct"]),
            ]
        )
    return _table(
        [
            "模型",
            "探針題數（歷史 prompt）",
            "與歷史 prediction 一致",
            "Wilson 95% CI",
            "同 backend 上限（full vs stability-3407）",
            "prompt_tokens 相同",
            "探針 v1.1 正確率",
            "v1.1 未覆蓋題",
            "其中答對",
        ],
        rows,
    )


def render_report(
    analysis: Mapping[str, Any],
    *,
    medical_subjects: Sequence[str],
    control_subjects: Sequence[str],
    probe: Mapping[str, Any] | None = None,
    run_id: str,
) -> str:
    """Render every generated table for the limited evaluation."""

    check = analysis.get("frozen_reproduction_check") or {}
    parts = [
        f"_下列表格由 `reports/tmmlu-v1.1/{run_id}/analysis.json` 逐題證據自動產生；"
        f"證據種類：`{analysis['evidence_kind']}`；"
        f"凍結數字重現檢查：`{check.get('status', 'not_run')}`。_",
        "",
        "#### 表 1：新舊題目覆蓋率",
        "",
        render_coverage(analysis),
        "",
        "#### 表 2：三種計分方式下的整體指標",
        "",
        render_variant_table(analysis),
        "",
        "#### 表 3：各科結果（保留題組 × v1.1 答案）",
        "",
        render_subject_table(
            analysis, medical_subjects=medical_subjects, control_subjects=control_subjects
        ),
        "",
        "#### 表 4：配對統計與預先定義判準",
        "",
        render_paired_table(analysis),
        "",
        "#### 表 5：差異分解（刪題 vs 答案修訂）",
        "",
        render_decomposition_table(analysis),
        "",
        "#### 表 6：同一模型、同一輸出、僅答案版本不同",
        "",
        render_version_delta_table(analysis),
        "",
        "#### 表 7：三模型皆可解析子集（描述性，未消除格式混淆）",
        "",
        render_all_parsed_table(analysis),
    ]
    if probe is not None:
        parts += [
            "",
            f"#### 表 8：本機新推論探針（獨立實驗，backend = `{probe['backend_label']}`）",
            "",
            f"_證據種類：`{probe['evidence_kind']}`；此表不得與表 2–7 的歷史重計分合併解讀。_",
            "",
            render_probe_table(probe),
        ]
    return "\n".join(parts) + "\n"


def inject_generated_section(document: str, generated: str) -> str:
    """Replace the marker-delimited generated block inside a Markdown document."""

    begin = document.find(BEGIN_MARKER)
    end = document.find(END_MARKER)
    if begin < 0 or end < 0 or end < begin:
        raise ValueError("document lacks a well-formed generated section")
    head = document[: begin + len(BEGIN_MARKER)]
    tail = document[end:]
    return f"{head}\n{generated.rstrip()}\n{tail}"


def extract_generated_section(document: str) -> str:
    """Return the current generated block for round-trip verification."""

    begin = document.find(BEGIN_MARKER)
    end = document.find(END_MARKER)
    if begin < 0 or end < 0 or end < begin:
        raise ValueError("document lacks a well-formed generated section")
    return document[begin + len(BEGIN_MARKER) : end].strip("\n") + "\n"


def resolve_run_dir(project_root: Path, *, report_root: str, run_id: str) -> Path:
    """Return the versioned run directory and refuse anything outside it."""

    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("run_id must look like 20260914T120000Z")
    root = (project_root / report_root).resolve()
    expected_root = (project_root / "reports" / REPORT_ROOT_NAME).resolve()
    if root != expected_root:
        raise ValueError(f"report_root must be reports/{REPORT_ROOT_NAME}, got {report_root}")
    run_dir = (root / run_id).resolve()
    if run_dir.parent != root:
        raise ValueError("run directory escaped the report root")
    frozen = (project_root / "reports" / "phase4").resolve()
    if frozen == run_dir or frozen in run_dir.parents:
        raise ValueError("refusing to write inside the frozen reports/phase4 evidence")
    return run_dir

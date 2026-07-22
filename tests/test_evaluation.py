from __future__ import annotations

from collections.abc import Iterable

import pytest

from tw_med_qlora.evaluation import (
    PredictionRecord,
    accuracy_summary,
    forgetting_noninferiority,
    mcnemar_exact_test,
    paired_bootstrap_accuracy_difference,
    parse_mcq_answer,
    prediction_record,
    representative_case_ids,
    subject_accuracy,
)


@pytest.mark.parametrize(
    "text",
    [
        "A",
        " B ",
        "C.",
        "D。\n",
        r"\boxed{A}",
        r"推理內容。最後答案是 \boxed{B}",
        r"Reasoning first; therefore \boxed{ C }.",
    ],
)
def test_strict_answer_parser_accepts_one_unambiguous_choice(text: str) -> None:
    assert parse_mcq_answer(text) in {"A", "B", "C", "D"}


@pytest.mark.parametrize(
    "text",
    [
        "",
        "a",
        "答案：A",
        "A 或 B",
        "A，因為……",
        "AB",
        "E",
        r"\boxed{E}",
        r"\boxed{A} 或 \boxed{B}",
        r"\boxed{A} 再確認 \boxed{A}",
        r"\boxed{A} 但也出現 \boxed{\text{B}}",
        r"\boxed{\text{A}}",
    ],
)
def test_strict_answer_parser_rejects_ambiguous_or_explained_output(text: str) -> None:
    assert parse_mcq_answer(text) is None


def _record(
    example_id: str,
    model: str,
    *,
    gold: str = "A",
    prediction: str | None = "A",
    subject: str = "medicine",
) -> PredictionRecord:
    raw = prediction or "無法解析"
    return prediction_record(
        example_id=example_id,
        model=model,
        source="synthetic",
        subject=subject,
        gold=gold,
        raw_output=raw,
        latency_seconds=0.25,
        prompt_tokens=10,
        completion_tokens=1,
    )


def test_prediction_record_keeps_digest_not_raw_output() -> None:
    record = _record("id-1", "base", prediction=None)
    public = record.as_public_dict()

    assert record.prediction is None
    assert len(record.raw_output_sha256) == 64
    assert "raw_output" not in public
    assert public["parsed"] is False
    assert public["correct"] is False


def test_accuracy_and_subject_summary_count_parse_failure_as_wrong() -> None:
    records = [
        _record("1", "base", subject="a"),
        _record("2", "base", subject="a", prediction=None),
        _record("3", "base", subject="b", gold="B", prediction="A"),
    ]

    assert accuracy_summary(records) == {
        "total": 3,
        "parsed": 2,
        "parse_failures": 1,
        "correct": 1,
        "accuracy": pytest.approx(1 / 3),
        "parse_rate": pytest.approx(2 / 3),
    }
    assert list(subject_accuracy(records)) == ["a", "b"]


def test_accuracy_rejects_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="duplicate prediction ID"):
        accuracy_summary([_record("1", "base"), _record("1", "base")])


def _paired_records() -> tuple[list[PredictionRecord], list[PredictionRecord]]:
    outcomes: Iterable[tuple[str, str | None, str | None, str]] = [
        ("1", "A", "A", "medical"),
        ("2", "B", "A", "medical"),
        ("3", "A", "B", "control"),
        ("4", None, "A", "control"),
    ]
    base: list[PredictionRecord] = []
    adapter: list[PredictionRecord] = []
    for example_id, base_prediction, adapter_prediction, subject in outcomes:
        base.append(
            _record(
                example_id,
                "base",
                prediction=base_prediction,
                subject=subject,
            )
        )
        adapter.append(
            _record(
                example_id,
                "adapter",
                prediction=adapter_prediction,
                subject=subject,
            )
        )
    return base, adapter


def test_paired_bootstrap_is_deterministic_and_reports_percentage_points() -> None:
    base, adapter = _paired_records()
    first = paired_bootstrap_accuracy_difference(base, adapter, iterations=500, seed=3407)
    second = paired_bootstrap_accuracy_difference(base, adapter, iterations=500, seed=3407)

    assert first == second
    assert first["observed_difference_percentage_points"] == pytest.approx(25.0)
    assert first["ci_lower_percentage_points"] <= 25
    assert first["ci_upper_percentage_points"] >= 25


def test_mcnemar_exact_test_uses_paired_discordant_counts() -> None:
    base, adapter = _paired_records()
    result = mcnemar_exact_test(base, adapter)

    assert result["base_wrong_adapter_correct"] == 2
    assert result["base_correct_adapter_wrong"] == 1
    assert result["discordant_pairs"] == 3
    assert result["two_sided_exact_p_value"] == pytest.approx(1.0)


def test_noninferiority_reports_only_predeclared_conclusions() -> None:
    records = [_record(str(index), "base", subject=f"control-{index % 2}") for index in range(8)]
    adapter = [
        _record(str(index), "adapter", subject=f"control-{index % 2}")
        for index in range(8)
    ]
    result = forgetting_noninferiority(records, adapter, iterations=200, seed=3407)

    assert result["conclusion"] == "no_material_forgetting"
    assert result["required_ci_lower_bound_above"] == -2.0


def test_representative_cases_publish_ids_and_labels_only() -> None:
    base, adapter = _paired_records()
    cases = representative_case_ids(base, adapter, limit=4)

    assert len(cases) == 4
    assert {case["category"] for case in cases} >= {"improved", "regressed"}
    assert all("question" not in case and "raw_output" not in case for case in cases)


def test_paired_statistics_reject_different_question_sets() -> None:
    with pytest.raises(ValueError, match="paired prediction IDs differ"):
        mcnemar_exact_test([_record("1", "base")], [_record("2", "adapter")])

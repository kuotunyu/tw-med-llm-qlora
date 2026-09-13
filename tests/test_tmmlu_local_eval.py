from __future__ import annotations

import hashlib

import pytest

from tw_med_qlora.local_inference import GenerationResult
from tw_med_qlora.tmmlu import shuffle_options, tmmlu_row_to_example
from tw_med_qlora.tmmlu_local_eval import (
    PUBLIC_FORBIDDEN_KEYS,
    SYSTEM_PROMPT,
    analyze_probe,
    build_probe_plan,
    in_backend_ceiling,
    private_record,
    public_record,
    token_limit_hit,
    visible_prompt,
    wilson_interval,
)
from tw_med_qlora.tmmlu_revision import RevisionRow, build_revision_mapping

OLD_REVISION = "a" * 40
NEW_REVISION = "b" * 40
SEED = 3407
SUBJECTS = ["medicine", "law"]


def _row(index: int, *, answer: str = "B") -> dict[str, str]:
    return {
        "question": f"合成題目 {index}",
        "A": f"選項甲 {index}",
        "B": f"選項乙 {index}",
        "C": f"選項丙 {index}",
        "D": f"選項丁 {index}",
        "answer": answer,
    }


def _rows(version: str, revision: str, subject: str, source_rows):
    return [
        RevisionRow(
            version=version,
            subject=subject,
            row_index=index,
            item=tmmlu_row_to_example(
                row,
                subject=subject,
                split="test",
                source="ikala/tmmluplus",
                revision=revision,
                row_index=index,
            ),
        )
        for index, row in enumerate(source_rows)
    ]


def _fixture():
    old_rows = []
    new_rows = []
    for offset, subject in enumerate(SUBJECTS):
        base = offset * 100
        old_source = [_row(base + i, answer="ABCD"[i % 4]) for i in range(8)]
        new_source = [dict(row) for row in old_source[:7]]
        new_source[0]["answer"] = "D" if old_source[0]["answer"] != "D" else "A"
        if subject == "medicine":
            new_source.append(_row(base + 99, answer="C"))  # uncovered v1.1 question
        old_rows += _rows("old", OLD_REVISION, subject, old_source)
        new_rows += _rows("new", NEW_REVISION, subject, new_source)
    return old_rows, new_rows, build_revision_mapping(old_rows, new_rows)


def test_system_prompt_matches_phase4_contract() -> None:
    expected = (
        "請選擇唯一最佳答案。不要解釋或重述題目；只輸出單一大寫 A–D 字母，"
        "或一個 LaTeX 答案框，例如 \\boxed{A}。"
    )
    assert SYSTEM_PROMPT == expected
    item = tmmlu_row_to_example(
        _row(1), subject="medicine", split="test", source="ikala/tmmluplus", revision=OLD_REVISION
    )
    expected_prompt = "合成題目 1\n\nA. 選項甲 1\nB. 選項乙 1\nC. 選項丙 1\nD. 選項丁 1"
    assert visible_prompt(item) == expected_prompt


def test_probe_plan_is_deterministic_and_reuses_historical_shuffle() -> None:
    old_rows, new_rows, mapping = _fixture()

    plan = build_probe_plan(
        new_rows, mapping, subjects=SUBJECTS, per_subject=3, seed=SEED, option_seed=SEED
    )
    again = build_probe_plan(
        new_rows, mapping, subjects=SUBJECTS, per_subject=3, seed=SEED, option_seed=SEED
    )

    assert [item.new_example_id for item in plan] == [item.new_example_id for item in again]
    uncovered = [item for item in plan if item.historical_example_id is None]
    covered = [item for item in plan if item.historical_example_id is not None]
    assert len(uncovered) == 1 and uncovered[0].shuffle_id_space == "v1.1"
    assert uncovered[0].reason == "uncovered_v1_1_question"
    assert len(covered) == 6
    assert {item.subject for item in covered} == set(SUBJECTS)

    old_by_id = {row.example_id: row for row in old_rows}
    for item in covered:
        historical = old_by_id[item.historical_example_id]
        expected = shuffle_options(historical.item, seed=SEED)
        assert item.prompt == visible_prompt(expected)
        assert item.shuffle_id_space == "v1.0"
        # Gold follows the v1.1 answer key inside the historical letter space.
        new_entry = next(
            row
            for row in mapping["rows"]
            if row["version"] == "new" and row["example_id"] == item.new_example_id
        )
        if new_entry["category"] == "unchanged":
            assert item.gold == expected.example.answer
        else:
            assert item.gold != expected.example.answer


def test_public_record_is_content_free_and_applies_token_limit_rule() -> None:
    _old_rows, new_rows, mapping = _fixture()
    plan = build_probe_plan(
        new_rows, mapping, subjects=SUBJECTS, per_subject=1, seed=SEED, option_seed=SEED
    )
    item = plan[0]

    record = public_record(
        item,
        model="localized-base",
        backend_label="test-backend",
        raw_text=f"{item.gold}\n",
        stripped_text=item.gold,
        prompt_tokens=120,
        completion_tokens=2,
        max_new_tokens=256,
        latency_seconds=0.4,
    )
    assert record["prediction"] == item.gold and record["correct"] is True
    assert record["prompt_sha256"] == hashlib.sha256(item.prompt.encode()).hexdigest()
    assert not PUBLIC_FORBIDDEN_KEYS.intersection(record)
    assert "合成題目" not in str(record)

    truncated = public_record(
        item,
        model="localized-base",
        backend_label="test-backend",
        raw_text="A",
        stripped_text="A",
        prompt_tokens=120,
        completion_tokens=256,
        max_new_tokens=256,
        latency_seconds=9.0,
    )
    assert truncated["max_token_limit_hit"] is True
    assert truncated["prediction"] is None and truncated["correct"] is False
    assert token_limit_hit(completion_tokens=255, max_new_tokens=256) is False

    private = private_record(item, model="localized-base", raw_text="A")
    assert private["prompt"] == item.prompt and private["question"].startswith("合成題目")


def test_generation_result_keeps_unstripped_raw_text() -> None:
    result = GenerationResult(
        text="A",
        parsed_answer="A",
        prompt_tokens=1,
        completion_tokens=1,
        first_token_seconds=None,
        total_generation_seconds=0.1,
        peak_allocated_gib=0.0,
        peak_reserved_gib=0.0,
        raw_text="A\n",
    )
    assert result.raw_text == "A\n" and result.text == "A"


def test_wilson_interval_and_ceiling() -> None:
    lower, upper = wilson_interval(95, 100)
    assert 0.88 < lower < 0.95 < upper < 0.99
    with pytest.raises(ValueError):
        wilson_interval(0, 0)

    historical = []
    for suite in ("tmmlu-full", "tmmlu-stability-3407"):
        for index in range(4):
            historical.append(
                {
                    "suite": suite,
                    "model": "localized-base",
                    "example_id": f"id{index}",
                    "prediction": "A" if suite == "tmmlu-full" or index else "B",
                    "raw_output_sha256": "x" * 64,
                    "prompt_tokens": 10,
                    "gold": "A",
                }
            )
    ceiling = in_backend_ceiling(historical)
    assert ceiling["localized-base"] == {
        "agree": 3,
        "total": 4,
        "raw_hash_agree": 4,
        "rate": 0.75,
    }


def test_analyze_probe_reports_agreement_per_model() -> None:
    _old_rows, new_rows, mapping = _fixture()
    plan = build_probe_plan(
        new_rows, mapping, subjects=SUBJECTS, per_subject=2, seed=SEED, option_seed=SEED
    )
    historical = []
    public_rows = []
    for item in plan:
        if item.historical_example_id is not None:
            historical.append(
                {
                    "suite": "tmmlu-full",
                    "model": "localized-medical-adapter",
                    "example_id": item.historical_example_id,
                    "prediction": item.gold,
                    "raw_output_sha256": hashlib.sha256(item.gold.encode()).hexdigest(),
                    "prompt_tokens": 100,
                    "gold": item.gold,
                }
            )
        public_rows.append(
            public_record(
                item,
                model="localized-medical-adapter",
                backend_label="test-backend",
                raw_text=item.gold,
                stripped_text=item.gold,
                prompt_tokens=100,
                completion_tokens=1,
                max_new_tokens=256,
                latency_seconds=0.1,
            )
        )

    analysis = analyze_probe(
        public_rows, historical, backend_label="test-backend", runtime={"gpu": "test"}
    )
    model = analysis["models"]["localized-medical-adapter"]
    assert model["probe_questions"] == 5
    assert model["probe_questions_with_historical_prompt"] == 4
    assert model["prediction_agreement_with_historical"]["agree"] == 4
    assert model["prompt_tokens_identical_to_historical"] == 4
    assert model["raw_output_hash_agreement_with_historical"] == 4
    assert model["uncovered_v1_1_questions"]["total"] == 1
    assert analysis["evidence_kind"] == "new_inference_separate_backend_probe"
    assert "original-instruct" not in analysis["models"]

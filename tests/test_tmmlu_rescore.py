from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from tw_med_qlora.tmmlu import shuffle_options, tmmlu_row_to_example
from tw_med_qlora.tmmlu_rescore import (
    ADAPTER_MODEL,
    BASE_MODEL,
    INSTRUCT_MODEL,
    MODEL_LABELS,
    VARIANT_RETAINED_NEW_GOLD,
    VARIANT_RETAINED_NEW_GOLD_WITH_DUPLICATES,
    VARIANT_RETAINED_OLD_GOLD,
    VARIANT_V1_0_FULL,
    analyze,
    load_historical_tmmlu_rows,
    rescored_records,
    shuffled_letter,
    verify_shuffle_contract,
)
from tw_med_qlora.tmmlu_revision import RevisionRow, build_revision_mapping

OLD_REVISION = "a" * 40
NEW_REVISION = "b" * 40
SEED = 3407
MEDICAL = ["medicine", "pharmacy"]
CONTROL = ["law", "logic"]


def _row(index: int, *, answer: str = "B") -> dict[str, str]:
    return {
        "question": f"合成題目 {index}",
        "A": f"選項甲 {index}",
        "B": f"選項乙 {index}",
        "C": f"選項丙 {index}",
        "D": f"選項丁 {index}",
        "answer": answer,
    }


def _revision_rows(version: str, revision: str, subject: str, source_rows):
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
    """Four subjects; each has one revised answer, one removed question, and unchanged rows."""

    old_rows: list[RevisionRow] = []
    new_rows: list[RevisionRow] = []
    for offset, subject in enumerate(MEDICAL + CONTROL):
        base = offset * 100
        old_source = [_row(base + i, answer="ABCD"[i % 4]) for i in range(6)]
        new_source = [dict(row) for row in old_source[:5]]  # row 5 removed in new
        new_source[0]["answer"] = "D" if old_source[0]["answer"] != "D" else "A"  # revised
        if subject == "law":
            duplicate = _row(base + 50, answer="A")
            old_source += [duplicate, duplicate]
            new_source += [duplicate, duplicate]
        old_rows += _revision_rows("old", OLD_REVISION, subject, old_source)
        new_rows += _revision_rows("new", NEW_REVISION, subject, new_source)
    mapping = build_revision_mapping(old_rows, new_rows)

    def prediction_for(model: str, row: RevisionRow) -> str | None:
        shuffled_gold = shuffle_options(row.item, seed=SEED).example.answer
        digest = int(hashlib.sha256(f"{model}:{row.example_id}".encode()).hexdigest(), 16)
        if model == ADAPTER_MODEL:
            return shuffled_gold if digest % 10 < 7 else "ABCD"[digest % 4]
        if model == BASE_MODEL:
            return None if digest % 10 < 3 else ("ABCD"[digest % 4])
        return shuffled_gold if digest % 2 else "ABCD"[digest % 4]

    public_rows = []
    for model in MODEL_LABELS:
        for row in old_rows:
            shuffled = shuffle_options(row.item, seed=SEED)
            prediction = prediction_for(model, row)
            public_rows.append(
                {
                    "request_id": hashlib.sha256(f"{model}{row.example_id}".encode()).hexdigest()[
                        :24
                    ],
                    "example_id": row.example_id,
                    "suite": "tmmlu-full",
                    "option_seed": SEED,
                    "model": model,
                    "source": f"ikala/tmmluplus/{row.subject}",
                    "subject": row.subject,
                    "gold": shuffled.example.answer,
                    "prediction": prediction,
                    "parsed": prediction is not None,
                    "correct": prediction == shuffled.example.answer,
                    "raw_output_sha256": hashlib.sha256(str(prediction).encode()).hexdigest(),
                    "latency_seconds": 0.5,
                    "prompt_tokens": 100,
                    "completion_tokens": 3,
                    "finish_reason": "length" if prediction is None else "stop",
                    "max_token_limit_hit": prediction is None,
                }
            )
    return old_rows, new_rows, mapping, public_rows


def test_shuffled_letter_matches_shuffle_options_for_every_gold() -> None:
    for answer in "ABCD":
        item = tmmlu_row_to_example(
            _row(7, answer=answer),
            subject="medicine",
            split="test",
            source="ikala/tmmluplus",
            revision=OLD_REVISION,
            row_index=7,
        )
        expected = shuffle_options(item, seed=SEED).example.answer
        assert shuffled_letter(item.example.id, seed=SEED, letter=answer) == expected
    with pytest.raises(ValueError, match="letter"):
        shuffled_letter("abc", seed=SEED, letter="E")


def test_variants_partition_questions_and_apply_revised_gold() -> None:
    old_rows, _new_rows, mapping, public_rows = _fixture()
    assert verify_shuffle_contract(
        public_rows,
        {row["example_id"]: row for row in mapping["rows"] if row["version"] == "old"},
        seed=SEED,
    ) == len(public_rows)

    full = rescored_records(public_rows, mapping, variant=VARIANT_V1_0_FULL, seed=SEED)
    retained_old = rescored_records(
        public_rows, mapping, variant=VARIANT_RETAINED_OLD_GOLD, seed=SEED
    )
    retained_new = rescored_records(
        public_rows, mapping, variant=VARIANT_RETAINED_NEW_GOLD, seed=SEED
    )
    with_duplicates = rescored_records(
        public_rows, mapping, variant=VARIANT_RETAINED_NEW_GOLD_WITH_DUPLICATES, seed=SEED
    )

    # 4 subjects x 6 rows + 2 duplicate rows in law = 26 old rows per model.
    assert {len(records) for records in full.values()} == {26}
    # Removed: 1 per subject (4); ambiguous duplicates: 2 -> 20 retained.
    assert {len(records) for records in retained_old.values()} == {20}
    assert {len(records) for records in retained_new.values()} == {20}
    assert {len(records) for records in with_duplicates.values()} == {22}

    historical = {(row["model"], row["example_id"]): row for row in public_rows}
    for model, records in full.items():
        for record in records:
            assert record.correct == historical[(model, record.example_id)]["correct"]

    revised_ids = {
        row["example_id"]
        for row in mapping["rows"]
        if row["version"] == "old" and row["category"] == "answer_revised"
    }
    assert len(revised_ids) == 4
    old_by_id = {record.example_id: record for record in retained_old[ADAPTER_MODEL]}
    new_by_id = {record.example_id: record for record in retained_new[ADAPTER_MODEL]}
    for example_id in old_by_id:
        if example_id in revised_ids:
            assert old_by_id[example_id].gold != new_by_id[example_id].gold
        else:
            assert old_by_id[example_id].gold == new_by_id[example_id].gold
    flipped = sum(
        old_by_id[example_id].correct != new_by_id[example_id].correct for example_id in revised_ids
    )
    assert flipped >= 1


def test_analyze_reports_coverage_paired_statistics_and_decomposition() -> None:
    _old_rows, _new_rows, mapping, public_rows = _fixture()
    analysis = analyze(
        public_rows,
        mapping,
        seed=SEED,
        medical_subjects=MEDICAL,
        control_subjects=CONTROL,
        iterations=200,
        margin_percentage_points=2.0,
    )

    coverage = analysis["coverage"]["all_13"]
    assert coverage["v1_0_total"] == 26
    assert coverage["v1_1_total"] == 22
    assert coverage["v1_1_covered_by_rescoring"] == 20
    assert coverage["v1_1_added_or_modified_uncovered"] == 0
    assert coverage["v1_1_duplicate_ambiguous"] == 2
    assert coverage["v1_0_answer_revised"] == 4

    control = analysis["paired"][VARIANT_RETAINED_NEW_GOLD][
        "control_noninferiority_adapter_vs_base"
    ]
    assert control["subjects"] == CONTROL
    assert control["questions"] == 10  # 2 control subjects x 5 retained rows
    assert control["required_ci_lower_bound_above"] == -2.0
    assert control["conclusion"] in {"no_material_forgetting", "forgetting_risk", "inconclusive"}

    medical = analysis["paired"][VARIANT_V1_0_FULL]["medical_macro_adapter_vs_instruct"]
    assert medical["questions"] == 12 and medical["stratified_by_subject"] is True
    assert "two_sided_exact_p_value" in medical["mcnemar_pooled"]

    adapter = analysis["decomposition"][ADAPTER_MODEL]["overall_micro"]
    assert adapter["removed_question_effect_pp"] == pytest.approx(
        (adapter["retained_old_gold"] - adapter["v1_0_full"]) * 100
    )
    assert adapter["total_change_pp"] == pytest.approx(
        adapter["removed_question_effect_pp"] + adapter["answer_revision_effect_pp"]
    )

    delta = analysis["version_delta_same_outputs"][INSTRUCT_MODEL]
    assert delta["questions"] == 20
    assert delta["answer_revision_flips"]["gold_changed"] == 4
    assert analysis["frozen_reproduction_check"] is None
    assert analysis["evidence_kind"] == "historical_outputs_rescored_under_v1_1"

    by_subject = analysis["variants"][VARIANT_RETAINED_NEW_GOLD]["models"][BASE_MODEL]["by_subject"]
    for summary in by_subject.values():
        assert (
            summary["parse_fail_length"] + summary["parse_fail_other"] == summary["parse_failures"]
        )

    forbidden = {"question", "choices", "prompt", "raw_output", "answer_text"}

    def nested_keys(value: object) -> set[str]:
        keys: set[str] = set()
        if isinstance(value, dict):
            for key, child in value.items():
                keys.add(str(key))
                keys.update(nested_keys(child))
        elif isinstance(value, list):
            for child in value:
                keys.update(nested_keys(child))
        return keys

    assert not nested_keys(analysis) & forbidden


def test_load_historical_rows_verifies_archive_and_counts(tmp_path: Path) -> None:
    _old_rows, _new_rows, _mapping, public_rows = _fixture()
    archive = tmp_path / "public.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr(
            "public/public-predictions.jsonl",
            "\n".join(json.dumps(row, ensure_ascii=False) for row in public_rows) + "\n",
        )
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()

    rows, seed = load_historical_tmmlu_rows(
        archive, expected_sha256=digest, expected_rows_per_model=26
    )
    assert len(rows) == 78 and seed == SEED
    with pytest.raises(ValueError, match="SHA-256"):
        load_historical_tmmlu_rows(archive, expected_sha256="0" * 64, expected_rows_per_model=26)
    with pytest.raises(ValueError, match="row counts"):
        load_historical_tmmlu_rows(archive, expected_sha256=digest, expected_rows_per_model=25)

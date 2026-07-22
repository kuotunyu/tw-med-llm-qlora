from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from tw_med_qlora.tmmlu import (
    SubjectExample,
    TMMLUValidationError,
    read_tmmlu_csv,
    shuffle_options,
    stability_sample,
    stratified_calibration_sample,
    tmmlu_row_to_example,
    write_twinkle_dataset,
)

REVISION = "a" * 40


def _row(index: int = 0) -> dict[str, str]:
    return {
        "question": f"合成題目 {index}",
        "A": f"選項甲 {index}",
        "B": f"選項乙 {index}",
        "C": f"選項丙 {index}",
        "D": f"選項丁 {index}",
        "answer": "B",
    }


def _example(subject: str, index: int) -> SubjectExample:
    return tmmlu_row_to_example(
        _row(index),
        subject=subject,
        split="validation",
        source="ikala/tmmluplus",
        revision=REVISION,
    )


def test_tmmlu_row_is_strict_and_stable() -> None:
    first = _example("medicine", 1)
    second = _example("medicine", 1)

    assert first == second
    assert len(first.example.id) == 20
    assert first.example.answer == "B"
    with pytest.raises(TMMLUValidationError, match="fields must be exactly"):
        tmmlu_row_to_example(
            {**_row(), "extra": "no"},
            subject="medicine",
            split="validation",
            source="ikala/tmmluplus",
            revision=REVISION,
        )


def test_read_tmmlu_csv_validates_header_and_utf8(tmp_path: Path) -> None:
    path = tmp_path / "medicine_validation.csv"
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=["question", "A", "B", "C", "D", "answer"])
        writer.writeheader()
        writer.writerow(_row())

    records = read_tmmlu_csv(
        path,
        subject="medicine",
        split="validation",
        source="ikala/tmmluplus",
        revision=REVISION,
    )
    assert len(records) == 1
    assert records[0].example.question == "合成題目 0"


def test_read_tmmlu_csv_preserves_duplicate_source_rows_with_unique_ids(
    tmp_path: Path,
) -> None:
    path = tmp_path / "control_test.csv"
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=["question", "A", "B", "C", "D", "answer"])
        writer.writeheader()
        writer.writerow(_row())
        writer.writerow(_row())

    records = read_tmmlu_csv(
        path,
        subject="control",
        split="test",
        source="ikala/tmmluplus",
        revision=REVISION,
    )
    assert len(records) == 2
    assert records[0].example.id != records[1].example.id
    assert records[0].example.question == records[1].example.question


def test_calibration_sample_covers_all_subjects_deterministically() -> None:
    examples = {
        subject: [_example(subject, index) for index in range(5)]
        for subject in ("a", "b", "c")
    }
    first = stratified_calibration_sample(examples, total=5, seed=3407)
    second = stratified_calibration_sample(examples, total=5, seed=3407)

    assert first == second
    assert sum(map(len, first.values())) == 5
    assert all(first[subject] for subject in examples)


def test_option_shuffle_is_reproducible_and_remaps_gold_text() -> None:
    item = _example("medicine", 2)
    shuffled = shuffle_options(item, seed=3408)

    assert shuffled == shuffle_options(item, seed=3408)
    assert shuffled.example.choices[shuffled.example.answer] == item.example.choices["B"]
    assert set(shuffled.example.choices.values()) == set(item.example.choices.values())


def test_stability_sample_keeps_question_ids_before_option_seeds() -> None:
    examples = {"medicine": [_example("medicine", index) for index in range(10)]}
    sampled = stability_sample(examples, per_subject=4, sample_seed=3407)

    ids = [item.example.id for item in sampled["medicine"]]
    for option_seed in (3407, 3408, 3409):
        shuffled_ids = [
            shuffle_options(item, seed=option_seed).example.id
            for item in sampled["medicine"]
        ]
        assert shuffled_ids == ids


def test_twinkle_materialization_keeps_private_content_out_of_manifest(tmp_path: Path) -> None:
    records = {"medicine": [_example("medicine", index) for index in range(2)]}
    manifest = write_twinkle_dataset(tmp_path, records, option_seed=3407)
    dataset_text = (tmp_path / "medicine.jsonl").read_text(encoding="utf-8")
    rows = [json.loads(line) for line in dataset_text.splitlines()]

    assert set(rows[0]) == {"question", "A", "B", "C", "D", "answer"}
    assert manifest["total"] == 2
    serialized_manifest = json.dumps(manifest, ensure_ascii=False)
    assert "合成題目" not in serialized_manifest
    assert "選項甲" not in serialized_manifest

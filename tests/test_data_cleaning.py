from __future__ import annotations

import json
from pathlib import Path

import pytest

from tw_med_qlora.medqa import (
    assert_split_isolation,
    deduplicate_splits,
    file_sha256,
    has_ambiguous_choice_text,
    normalized_question,
    write_jsonl_atomic,
)
from tw_med_qlora.types import MCQExample, stable_example_id

SOURCE = "example/source"
REVISION = "a" * 40
CHOICES = {"A": "甲", "B": "乙", "C": "丙", "D": "丁"}


def example(question: str, split: str, *, choices: dict[str, str] | None = None) -> MCQExample:
    selected_choices = choices or CHOICES
    return MCQExample(
        id=stable_example_id(
            source=SOURCE,
            revision=REVISION,
            split=split,
            question=question,
            choices=selected_choices,
        ),
        source=SOURCE,
        revision=REVISION,
        split=split,
        question=question,
        choices=selected_choices,
        answer="A",
    )


def test_deduplication_honors_test_validation_train_priority() -> None:
    test_examples = [example("最高優先題", "test")]
    validation_examples = [
        example(" 最高優先題 ", "validation"),
        example("驗證題", "validation"),
    ]
    train_examples = [
        example("驗證題", "train"),
        example("訓練題", "train"),
        example("訓練題", "train"),
    ]

    cleaned, removals = deduplicate_splits(
        {"test": test_examples, "validation": validation_examples, "train": train_examples}
    )

    assert cleaned["test"] == test_examples
    assert [item.question for item in cleaned["validation"]] == ["驗證題"]
    assert [item.question for item in cleaned["train"]] == ["訓練題"]
    assert [(item.removed_split, item.winner_split) for item in removals] == [
        ("validation", "test"),
        ("train", "validation"),
        ("train", "train"),
    ]
    assert_split_isolation(cleaned)


def test_test_duplicates_are_audited_but_not_removed() -> None:
    test_examples = [example("同題", "test"), example("同題", "test")]

    cleaned, _ = deduplicate_splits(
        {"test": test_examples, "validation": [], "train": []}
    )

    assert cleaned["test"] == test_examples


def test_ambiguous_visible_choices_are_detected() -> None:
    choices = {"A": "相同", "B": " 相同 ", "C": "不同一", "D": "不同二"}

    assert has_ambiguous_choice_text(example("題目", "train", choices=choices)) is True


def test_duplicate_normalization_does_not_apply_nfkc_symbol_folding() -> None:
    assert normalized_question("  MIXED   Case  ") == "mixed case"
    assert normalized_question("Ⅰ型") != normalized_question("I型")


def test_isolation_rejects_cross_split_question_overlap() -> None:
    splits = {
        "train": [example("重複題", "train")],
        "validation": [],
        "test": [example("重複題", "test")],
    }

    with pytest.raises(ValueError, match="normalized question overlap"):
        assert_split_isolation(splits)


def test_jsonl_output_is_deterministic_and_has_fixed_schema(tmp_path: Path) -> None:
    output = tmp_path / "train.jsonl"
    examples = [example("第一題", "train"), example("第二題", "train")]

    write_jsonl_atomic(output, examples)
    first_hash = file_sha256(output)
    write_jsonl_atomic(output, examples)

    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert file_sha256(output) == first_hash
    assert set(records[0]) == {
        "id",
        "source",
        "revision",
        "split",
        "question",
        "choices",
        "answer",
    }

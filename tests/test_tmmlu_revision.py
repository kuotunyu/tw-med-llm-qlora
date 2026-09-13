from __future__ import annotations

import csv
from pathlib import Path

import pytest

from tw_med_qlora.tmmlu import tmmlu_row_to_example
from tw_med_qlora.tmmlu_revision import (
    CATEGORY_ADDED_OR_MODIFIED,
    CATEGORY_ANSWER_REVISED,
    CATEGORY_DUPLICATE_AMBIGUOUS,
    CATEGORY_REMOVED_OR_MODIFIED,
    CATEGORY_UNCHANGED,
    DIAGNOSTIC_CHOICES_REORDERED,
    DIAGNOSTIC_WHITESPACE_ONLY,
    RevisionRow,
    build_revision_mapping,
    group_counts,
    load_revision_rows,
    mapping_sha256,
)

OLD_REVISION = "a" * 40
NEW_REVISION = "b" * 40
FIELDS = ["question", "A", "B", "C", "D", "answer"]
FORBIDDEN_KEYS = {"question", "choices", "prompt", "raw_output", "answer_text"}


def _row(index: int, *, answer: str = "B", question: str | None = None) -> dict[str, str]:
    return {
        "question": question if question is not None else f"合成題目 {index}",
        "A": f"選項甲 {index}",
        "B": f"選項乙 {index}",
        "C": f"選項丙 {index}",
        "D": f"選項丁 {index}",
        "answer": answer,
    }


def _rows(version: str, revision: str, subject: str, source_rows: list[dict[str, str]]):
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


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _synthetic_versions() -> tuple[list[RevisionRow], list[RevisionRow]]:
    duplicate = _row(90, answer="A")
    reordered_old = _row(80)
    reordered_new = {**reordered_old, "A": reordered_old["B"], "B": reordered_old["A"]}
    old_rows = [
        _row(0),  # unchanged
        _row(1, answer="C"),  # answer revised
        _row(2),  # removed
        duplicate,  # duplicate in old, single in new -> ambiguous
        duplicate,
        _row(3, question="題目 3"),  # whitespace-only variant in new
        reordered_old,
        _row(4),  # duplicate in both with equal counts -> sensitivity pairing
        _row(4),
    ]
    new_rows = [
        _row(1, answer="D"),
        _row(0),
        duplicate,
        _row(5),  # added
        _row(3, question="題目  3"),
        reordered_new,
        _row(4),
        _row(4),
    ]
    return (
        _rows("old", OLD_REVISION, "medicine", old_rows),
        _rows("new", NEW_REVISION, "medicine", new_rows),
    )


def test_mapping_classifies_every_row_by_exact_content() -> None:
    old, new = _synthetic_versions()
    mapping = build_revision_mapping(old, new)

    by_old = {entry["example_id"]: entry for entry in mapping["rows"] if entry["version"] == "old"}
    by_new = {entry["example_id"]: entry for entry in mapping["rows"] if entry["version"] == "new"}
    assert len(by_old) == 9 and len(by_new) == 8

    unchanged = by_old[old[0].example_id]
    assert unchanged["category"] == CATEGORY_UNCHANGED
    assert unchanged["paired_example_id"] == new[1].example_id
    assert unchanged["paired_gold"] == "B"

    revised = by_old[old[1].example_id]
    assert revised["category"] == CATEGORY_ANSWER_REVISED
    assert (revised["gold"], revised["paired_gold"]) == ("C", "D")
    assert by_new[new[0].example_id]["paired_example_id"] == old[1].example_id

    assert by_old[old[2].example_id]["category"] == CATEGORY_REMOVED_OR_MODIFIED
    assert by_new[new[3].example_id]["category"] == CATEGORY_ADDED_OR_MODIFIED

    for row in (old[3], old[4], new[2]):
        entry = (by_old if row.version == "old" else by_new)[row.example_id]
        assert entry["category"] == CATEGORY_DUPLICATE_AMBIGUOUS
        assert entry["paired_example_id"] is None
        assert entry["sensitivity_paired_example_id"] is None

    for left, right in ((old[7], new[6]), (old[8], new[7])):
        assert by_old[left.example_id]["category"] == CATEGORY_DUPLICATE_AMBIGUOUS
        assert by_old[left.example_id]["sensitivity_paired_example_id"] == right.example_id
        assert by_new[right.example_id]["sensitivity_paired_example_id"] == left.example_id

    groups = {tuple(group["old_example_ids"]): group for group in mapping["duplicate_groups"]}
    ambiguous = groups[(old[3].example_id, old[4].example_id)]
    sensitivity = groups[(old[7].example_id, old[8].example_id)]
    assert ambiguous["sensitivity_pairing_in_row_order"] is False
    assert sensitivity["sensitivity_pairing_in_row_order"] is True


def test_diagnostics_never_change_categories() -> None:
    old, new = _synthetic_versions()
    mapping = build_revision_mapping(old, new)
    by_old = {entry["example_id"]: entry for entry in mapping["rows"] if entry["version"] == "old"}

    whitespace = by_old[old[5].example_id]
    assert whitespace["category"] == CATEGORY_REMOVED_OR_MODIFIED
    assert whitespace["diagnostic"] == DIAGNOSTIC_WHITESPACE_ONLY
    assert whitespace["diagnostic_paired_example_id"] == new[4].example_id
    assert whitespace["paired_example_id"] is None

    reordered = by_old[old[6].example_id]
    assert reordered["category"] == CATEGORY_REMOVED_OR_MODIFIED
    assert reordered["diagnostic"] == DIAGNOSTIC_CHOICES_REORDERED
    assert reordered["diagnostic_paired_example_id"] == new[5].example_id

    counts = mapping["counts"]
    assert counts["old"]["by_subject"]["medicine"]["diagnostic_whitespace_only"] == 1
    assert counts["new"]["by_subject"]["medicine"]["diagnostic_choices_reordered"] == 1


def test_counts_partition_every_row_and_stay_content_free() -> None:
    old, new = _synthetic_versions()
    mapping = build_revision_mapping(old, new)
    counts = mapping["counts"]

    assert counts["old"]["total"] == 9
    assert counts["new"]["total"] == 8
    assert (
        counts["old"]["unchanged"]
        + counts["old"]["answer_revised"]
        + counts["old"]["removed_or_modified"]
        + counts["old"]["duplicate_ambiguous"]
        == 9
    )
    assert (
        counts["new"]["unchanged"]
        + counts["new"]["answer_revised"]
        + counts["new"]["added_or_modified"]
        + counts["new"]["duplicate_ambiguous"]
        == 8
    )
    assert group_counts(counts, version="new", subjects=["medicine"])["added_or_modified"] == 3

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

    assert not nested_keys(mapping) & FORBIDDEN_KEYS
    assert "合成題目" not in str(mapping)
    assert mapping_sha256(mapping) == mapping_sha256(build_revision_mapping(old, new))


def test_load_revision_rows_reads_only_mapped_subject_files(tmp_path: Path) -> None:
    root = tmp_path / "snapshot"
    _write_csv(root / "data" / "medicine_test.csv", [_row(0), _row(1)])
    _write_csv(root / "data" / "law_test.csv", [_row(2)])

    rows, files = load_revision_rows(
        root, version="old", revision=OLD_REVISION, subjects=["medicine", "law"]
    )

    assert [(row.subject, row.row_index) for row in rows] == [
        ("medicine", 0),
        ("medicine", 1),
        ("law", 0),
    ]
    assert files["medicine_test.csv"]["rows"] == 2
    assert len(files["law_test.csv"]["sha256"]) == 64

    _write_csv(root / "data" / "extra_test.csv", [_row(3)])
    with pytest.raises(FileNotFoundError, match="unexpected"):
        load_revision_rows(root, version="old", revision=OLD_REVISION, subjects=["medicine", "law"])
    with pytest.raises(FileNotFoundError, match="missing"):
        load_revision_rows(root, version="new", revision=NEW_REVISION, subjects=["medicine"])


def test_mapping_rejects_mislabelled_versions() -> None:
    old, new = _synthetic_versions()
    with pytest.raises(ValueError, match="version"):
        build_revision_mapping(new, old)

"""Content-based question mapping between two pinned TMMLU+ revisions.

Stable example IDs bake the dataset revision and the source row index into the hash, so
IDs from two revisions can never be joined directly. This module pairs questions by
exact content (subject, question, ordered A-D choices) and classifies every row of both
revisions without fuzzy matching. Ambiguous multiplicities are reported, never resolved
silently.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tw_med_qlora.medqa import file_sha256
from tw_med_qlora.phase4_full import canonical_json
from tw_med_qlora.tmmlu import SubjectExample, read_tmmlu_csv
from tw_med_qlora.types import CHOICE_KEYS

DATASET_ID = "ikala/tmmluplus"
SPLIT = "test"

CATEGORY_UNCHANGED = "unchanged"
CATEGORY_ANSWER_REVISED = "answer_revised"
CATEGORY_REMOVED_OR_MODIFIED = "removed_or_modified"
CATEGORY_ADDED_OR_MODIFIED = "added_or_modified"
CATEGORY_DUPLICATE_AMBIGUOUS = "duplicate_ambiguous"

OLD_CATEGORIES = (
    CATEGORY_UNCHANGED,
    CATEGORY_ANSWER_REVISED,
    CATEGORY_REMOVED_OR_MODIFIED,
    CATEGORY_DUPLICATE_AMBIGUOUS,
)
NEW_CATEGORIES = (
    CATEGORY_UNCHANGED,
    CATEGORY_ANSWER_REVISED,
    CATEGORY_ADDED_OR_MODIFIED,
    CATEGORY_DUPLICATE_AMBIGUOUS,
)
PAIRED_CATEGORIES = frozenset({CATEGORY_UNCHANGED, CATEGORY_ANSWER_REVISED})

DIAGNOSTIC_WHITESPACE_ONLY = "whitespace_only_match"
DIAGNOSTIC_CHOICES_REORDERED = "choices_reordered_match"

_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class RevisionRow:
    """One source row of one revision with its content-derived identity."""

    version: str
    subject: str
    row_index: int
    item: SubjectExample

    @property
    def example_id(self) -> str:
        return self.item.example.id

    @property
    def gold(self) -> str:
        return self.item.example.answer


def content_key(subject: str, item: SubjectExample) -> tuple[str, ...]:
    """Exact content identity: subject, question, then the choices in A-D order."""

    example = item.example
    return (subject, example.question, *(example.choices[key] for key in CHOICE_KEYS))


def _normalized_key(key: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(_WHITESPACE.sub(" ", part).strip() for part in key)


def _reordered_key(key: tuple[str, ...]) -> tuple[str, ...]:
    subject, question, *choices = key
    return (subject, question, *sorted(choices))


def expected_test_files(subjects: Sequence[str]) -> list[str]:
    """CSV names the evaluation protocol reads for the configured subjects."""

    return [f"{subject}_test.csv" for subject in subjects]


def load_revision_rows(
    snapshot_root: Path,
    *,
    version: str,
    revision: str,
    subjects: Sequence[str],
    source: str = DATASET_ID,
) -> tuple[list[RevisionRow], dict[str, dict[str, Any]]]:
    """Read the configured subject test CSVs of one pinned snapshot.

    Only the mapped ``<subject>_test.csv`` files are read; unexpected CSVs under the
    snapshot ``data`` directory abort instead of being silently ignored.
    """

    if version not in {"old", "new"}:
        raise ValueError("version must be 'old' or 'new'")
    data_dir = snapshot_root / "data"
    expected = set(expected_test_files(subjects))
    present = {path.name for path in data_dir.glob("*.csv")} if data_dir.is_dir() else set()
    missing = sorted(expected.difference(present))
    unexpected = sorted(present.difference(expected))
    if missing or unexpected:
        raise FileNotFoundError(
            f"snapshot {revision[:12]} CSV set mismatch: missing={missing} unexpected={unexpected}"
        )
    rows: list[RevisionRow] = []
    files: dict[str, dict[str, Any]] = {}
    for subject in subjects:
        path = data_dir / f"{subject}_test.csv"
        items = read_tmmlu_csv(
            path, subject=subject, split=SPLIT, source=source, revision=revision
        )
        for row_index, item in enumerate(items):
            rows.append(
                RevisionRow(version=version, subject=subject, row_index=row_index, item=item)
            )
        files[path.name] = {
            "rows": len(items),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
    return rows, files


def _row_record(row: RevisionRow) -> dict[str, Any]:
    return {
        "version": row.version,
        "subject": row.subject,
        "row_index": row.row_index,
        "example_id": row.example_id,
        "gold": row.gold,
        "category": None,
        "paired_example_id": None,
        "paired_row_index": None,
        "paired_gold": None,
        "sensitivity_paired_example_id": None,
        "diagnostic": None,
        "diagnostic_paired_example_id": None,
    }


def build_revision_mapping(
    old_rows: Iterable[RevisionRow],
    new_rows: Iterable[RevisionRow],
) -> dict[str, Any]:
    """Classify every old and new row by exact content multiset matching.

    Returns a content-free mapping: IDs, row indices, gold letters, categories, and
    diagnostics. Question text never enters the result.
    """

    old = list(old_rows)
    new = list(new_rows)
    if any(row.version != "old" for row in old) or any(row.version != "new" for row in new):
        raise ValueError("old_rows must have version 'old' and new_rows version 'new'")

    by_key: dict[tuple[str, ...], tuple[list[RevisionRow], list[RevisionRow]]] = defaultdict(
        lambda: ([], [])
    )
    for row in old:
        by_key[content_key(row.subject, row.item)][0].append(row)
    for row in new:
        by_key[content_key(row.subject, row.item)][1].append(row)

    records: dict[tuple[str, str], dict[str, Any]] = {}
    for row in old + new:
        if (row.version, row.example_id) in records:
            raise ValueError(f"duplicate example ID within {row.version}: {row.example_id}")
        records[(row.version, row.example_id)] = _row_record(row)

    def record(row: RevisionRow) -> dict[str, Any]:
        return records[(row.version, row.example_id)]

    def pair(left: RevisionRow, right: RevisionRow, category: str) -> None:
        for source, target in ((left, right), (right, left)):
            entry = record(source)
            entry["category"] = category
            entry["paired_example_id"] = target.example_id
            entry["paired_row_index"] = target.row_index
            entry["paired_gold"] = target.gold

    duplicate_groups: list[dict[str, Any]] = []
    for key in sorted(by_key):
        olds, news = by_key[key]
        olds.sort(key=lambda row: row.row_index)
        news.sort(key=lambda row: row.row_index)
        if len(olds) == 1 and len(news) == 1:
            category = (
                CATEGORY_UNCHANGED if olds[0].gold == news[0].gold else CATEGORY_ANSWER_REVISED
            )
            pair(olds[0], news[0], category)
        elif olds and not news:
            for row in olds:
                record(row)["category"] = CATEGORY_REMOVED_OR_MODIFIED
        elif news and not olds:
            for row in news:
                record(row)["category"] = CATEGORY_ADDED_OR_MODIFIED
        else:
            for row in olds + news:
                record(row)["category"] = CATEGORY_DUPLICATE_AMBIGUOUS
            old_golds = {row.gold for row in olds}
            new_golds = {row.gold for row in news}
            sensitivity = len(olds) == len(news) and len(old_golds) == 1 and len(new_golds) == 1
            if sensitivity:
                for left, right in zip(olds, news, strict=True):
                    record(left)["sensitivity_paired_example_id"] = right.example_id
                    record(right)["sensitivity_paired_example_id"] = left.example_id
            duplicate_groups.append(
                {
                    "subject": key[0],
                    "old_example_ids": [row.example_id for row in olds],
                    "new_example_ids": [row.example_id for row in news],
                    "old_golds": sorted(old_golds),
                    "new_golds": sorted(new_golds),
                    "sensitivity_pairing_in_row_order": sensitivity,
                }
            )

    # Diagnostics only: would an unmatched row match under whitespace normalization or a
    # choice reordering? These never change a category or feed any score.
    unmatched_old = [row for row in old if record(row)["category"] == CATEGORY_REMOVED_OR_MODIFIED]
    unmatched_new = [row for row in new if record(row)["category"] == CATEGORY_ADDED_OR_MODIFIED]
    for label, transform in (
        (DIAGNOSTIC_WHITESPACE_ONLY, _normalized_key),
        (DIAGNOSTIC_CHOICES_REORDERED, _reordered_key),
    ):
        new_index: dict[tuple[str, ...], list[RevisionRow]] = defaultdict(list)
        for row in unmatched_new:
            new_index[transform(content_key(row.subject, row.item))].append(row)
        for row in unmatched_old:
            entry = record(row)
            if entry["diagnostic"] is not None:
                continue
            candidates = new_index.get(transform(content_key(row.subject, row.item)), [])
            if len(candidates) == 1:
                target = candidates[0]
                entry["diagnostic"] = label
                entry["diagnostic_paired_example_id"] = target.example_id
                target_entry = record(target)
                if target_entry["diagnostic"] is None:
                    target_entry["diagnostic"] = label
                    target_entry["diagnostic_paired_example_id"] = row.example_id

    ordered_records = [record(row) for row in old] + [record(row) for row in new]
    if any(entry["category"] is None for entry in ordered_records):
        raise AssertionError("every row must receive a category")

    counts = _count_categories(ordered_records)
    if counts["old"]["total"] != len(old) or counts["new"]["total"] != len(new):
        raise AssertionError("category partition does not cover every row")

    return {
        "schema_version": 1,
        "matching": {
            "key": "exact (subject, question, A, B, C, D)",
            "multiset": True,
            "fuzzy_matching_used_for_pairing": False,
            "categories_old": list(OLD_CATEGORIES),
            "categories_new": list(NEW_CATEGORIES),
        },
        "counts": counts,
        "duplicate_groups": duplicate_groups,
        "rows": ordered_records,
    }


def _count_categories(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for version, categories in (("old", OLD_CATEGORIES), ("new", NEW_CATEGORIES)):
        version_rows = [entry for entry in records if entry["version"] == version]
        by_subject: dict[str, dict[str, int]] = {}
        for subject in sorted({entry["subject"] for entry in version_rows}):
            subject_rows = [entry for entry in version_rows if entry["subject"] == subject]
            counter = Counter(entry["category"] for entry in subject_rows)
            by_subject[subject] = {
                "total": len(subject_rows),
                **{category: counter.get(category, 0) for category in categories},
                "diagnostic_whitespace_only": sum(
                    entry["diagnostic"] == DIAGNOSTIC_WHITESPACE_ONLY for entry in subject_rows
                ),
                "diagnostic_choices_reordered": sum(
                    entry["diagnostic"] == DIAGNOSTIC_CHOICES_REORDERED for entry in subject_rows
                ),
            }
        total_counter = Counter(entry["category"] for entry in version_rows)
        summary[version] = {
            "total": len(version_rows),
            **{category: total_counter.get(category, 0) for category in categories},
            "by_subject": by_subject,
        }
    return summary


def group_counts(
    counts: Mapping[str, Any], *, version: str, subjects: Sequence[str]
) -> dict[str, int]:
    """Aggregate per-subject category counts for one subject group."""

    by_subject = counts[version]["by_subject"]
    categories = OLD_CATEGORIES if version == "old" else NEW_CATEGORIES
    aggregate = {"total": 0, **{category: 0 for category in categories}}
    for subject in subjects:
        subject_counts = by_subject.get(subject)
        if subject_counts is None:
            raise KeyError(f"subject missing from mapping counts: {subject}")
        for field in aggregate:
            aggregate[field] += int(subject_counts[field])
    return aggregate


def mapping_sha256(mapping: Mapping[str, Any]) -> str:
    """Fingerprint a mapping payload deterministically."""

    return hashlib.sha256(canonical_json(mapping).encode("utf-8")).hexdigest()

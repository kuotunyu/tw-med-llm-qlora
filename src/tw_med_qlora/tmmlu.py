"""Pinned TMMLU+ parsing and deterministic private evaluation materialization."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from tw_med_qlora.medqa import file_sha256
from tw_med_qlora.types import CHOICE_KEYS, MCQExample, stable_example_id

TMMLU_FIELDS = frozenset({"question", "A", "B", "C", "D", "answer"})


class TMMLUValidationError(ValueError):
    """Raised when a TMMLU+ row or materialized split is malformed."""


@dataclass(frozen=True)
class SubjectExample:
    """An MCQ with the subject kept out of the visible model prompt."""

    subject: str
    example: MCQExample

    def __post_init__(self) -> None:
        if not self.subject:
            raise ValueError("subject must not be empty")


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TMMLUValidationError(f"{field} must be a non-empty string")
    try:
        if value.encode("utf-8").decode("utf-8") != value:
            raise TMMLUValidationError(f"{field} failed UTF-8 round trip")
    except UnicodeError as error:
        raise TMMLUValidationError(f"{field} must be UTF-8") from error
    return value


def tmmlu_row_to_example(
    row: Mapping[str, Any],
    *,
    subject: str,
    split: str,
    source: str,
    revision: str,
    row_index: int | None = None,
) -> SubjectExample:
    """Convert a six-field source row without exposing subject metadata to the prompt."""

    if set(row) != TMMLU_FIELDS:
        raise TMMLUValidationError(
            f"fields must be exactly {sorted(TMMLU_FIELDS)}; got {sorted(row)}"
        )
    question = _text(row["question"], field="question")
    choices = {key: _text(row[key], field=key) for key in CHOICE_KEYS}
    answer = _text(row["answer"], field="answer").strip().upper()
    if answer not in CHOICE_KEYS:
        raise TMMLUValidationError("answer must be one of A, B, C, or D")
    if row_index is not None and row_index < 0:
        raise TMMLUValidationError("row_index must be non-negative")
    qualified_source = f"{source}/{subject}"
    id_source = (
        qualified_source if row_index is None else f"{qualified_source}#source-row={row_index}"
    )
    example = MCQExample(
        id=stable_example_id(
            source=id_source,
            revision=revision,
            split=split,
            question=question,
            choices=choices,
        ),
        source=qualified_source,
        revision=revision,
        split=split,
        question=question,
        choices=choices,
        answer=answer,
    )
    return SubjectExample(subject=subject, example=example)


def read_tmmlu_csv(
    path: Path,
    *,
    subject: str,
    split: str,
    source: str,
    revision: str,
) -> list[SubjectExample]:
    """Read one pinned subject CSV and validate every row."""

    examples: list[SubjectExample] = []
    with path.open("r", encoding="utf-8-sig", newline="") as source_file:
        reader = csv.DictReader(source_file)
        if set(reader.fieldnames or []) != TMMLU_FIELDS:
            raise TMMLUValidationError(f"unexpected CSV header: {reader.fieldnames}")
        for row_index, row in enumerate(reader):
            examples.append(
                tmmlu_row_to_example(
                    row,
                    subject=subject,
                    split=split,
                    source=source,
                    revision=revision,
                    row_index=row_index,
                )
            )
    if not examples:
        raise TMMLUValidationError(f"empty TMMLU+ file: {path.name}")
    if len({item.example.id for item in examples}) != len(examples):
        raise AssertionError(f"source-row stable IDs collided: {path.name}")
    return examples


def deterministic_order(
    examples: Iterable[SubjectExample],
    *,
    seed: int,
    purpose: str,
) -> list[SubjectExample]:
    """Order examples by a version-independent SHA-256 key."""

    values = list(examples)
    return sorted(
        values,
        key=lambda item: hashlib.sha256(
            f"{purpose}:{seed}:{item.example.id}".encode()
        ).hexdigest(),
    )


def stratified_calibration_sample(
    examples_by_subject: Mapping[str, Sequence[SubjectExample]],
    *,
    total: int,
    seed: int,
) -> dict[str, list[SubjectExample]]:
    """Round-robin a deterministic sample so every configured subject is represented."""

    if total <= 0:
        raise ValueError("total must be positive")
    subjects = list(examples_by_subject)
    if not subjects:
        raise ValueError("at least one subject is required")
    if total < len(subjects):
        raise ValueError("calibration total must be at least the number of subjects")
    ordered = {
        subject: deterministic_order(
            examples_by_subject[subject], seed=seed, purpose="calibration-sample"
        )
        for subject in subjects
    }
    if any(not rows for rows in ordered.values()):
        raise ValueError("every subject must contain at least one example")

    selected = {subject: [] for subject in subjects}
    offsets = {subject: 0 for subject in subjects}
    while sum(len(rows) for rows in selected.values()) < total:
        added = False
        for subject in subjects:
            offset = offsets[subject]
            if offset >= len(ordered[subject]):
                continue
            selected[subject].append(ordered[subject][offset])
            offsets[subject] += 1
            added = True
            if sum(len(rows) for rows in selected.values()) == total:
                break
        if not added:
            raise ValueError("calibration total exceeds the available examples")
    return selected


def stability_sample(
    examples_by_subject: Mapping[str, Sequence[SubjectExample]],
    *,
    per_subject: int,
    sample_seed: int,
) -> dict[str, list[SubjectExample]]:
    """Select the same at-most-N questions before applying each option seed."""

    if per_subject <= 0:
        raise ValueError("per_subject must be positive")
    return {
        subject: deterministic_order(
            examples,
            seed=sample_seed,
            purpose="stability-sample",
        )[:per_subject]
        for subject, examples in examples_by_subject.items()
    }


def shuffle_options(item: SubjectExample, *, seed: int) -> SubjectExample:
    """Apply a stable per-question choice permutation and remap the gold label."""

    example = item.example
    original_keys = sorted(
        CHOICE_KEYS,
        key=lambda key: hashlib.sha256(
            f"option-order:{seed}:{example.id}:{key}".encode()
        ).hexdigest(),
    )
    new_choices: dict[str, str] = {}
    new_answer = ""
    for new_key, old_key in zip(CHOICE_KEYS, original_keys, strict=True):
        new_choices[new_key] = example.choices[old_key]
        if old_key == example.answer:
            new_answer = new_key
    if not new_answer:
        raise AssertionError("option shuffle failed to remap the gold answer")
    return replace(
        item,
        example=replace(example, choices=new_choices, answer=new_answer),
    )


def _ids_sha256(examples: Sequence[SubjectExample]) -> str:
    digest = hashlib.sha256()
    for item in examples:
        digest.update(item.example.id.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def write_twinkle_dataset(
    output_dir: Path,
    examples_by_subject: Mapping[str, Sequence[SubjectExample]],
    *,
    option_seed: int,
) -> dict[str, Any]:
    """Write private Twinkle Eval JSONL plus a content-safe mapping manifest."""

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "option_seed": option_seed,
        "subjects": {},
    }
    for subject, source_examples in examples_by_subject.items():
        shuffled = [shuffle_options(item, seed=option_seed) for item in source_examples]
        dataset_path = output_dir / f"{subject}.jsonl"
        temporary = dataset_path.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as output_file:
            for item in shuffled:
                example = item.example
                row = {
                    "question": example.question,
                    **{key: example.choices[key] for key in CHOICE_KEYS},
                    "answer": example.answer,
                }
                output_file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                output_file.write("\n")
        temporary.replace(dataset_path)
        manifest["subjects"][subject] = {
            "count": len(shuffled),
            "ordered_ids_sha256": _ids_sha256(shuffled),
            "private_file_sha256": file_sha256(dataset_path),
            "id_by_line": [item.example.id for item in shuffled],
        }
    manifest["total"] = sum(
        int(subject["count"]) for subject in manifest["subjects"].values()
    )
    return manifest

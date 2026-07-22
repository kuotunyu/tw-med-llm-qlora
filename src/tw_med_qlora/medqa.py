"""Strict MedQA parsing and content-safe validation helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TextIO

from tw_med_qlora.types import CHOICE_KEYS, MCQExample, stable_example_id

MEDQA_FIELDS = frozenset({"meta_info", "question", "answer_idx", "answer", "options"})


class MedQAValidationError(ValueError):
    """Raised when a source row cannot be converted without ambiguity."""


@dataclass(frozen=True)
class DuplicateRemoval:
    """Content-safe provenance for a lower-priority duplicate."""

    example_id: str
    removed_split: str
    winner_split: str
    question_key_sha256: str


def _utf8_text(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise MedQAValidationError(f"{field} must be a string")
    if not value.strip():
        raise MedQAValidationError(f"{field} must not be empty")
    try:
        if value.encode("utf-8").decode("utf-8") != value:
            raise MedQAValidationError(f"{field} failed UTF-8 round trip")
    except UnicodeError as error:
        raise MedQAValidationError(f"{field} is not valid UTF-8 text") from error
    return value


def medqa_row_to_example(
    row: Mapping[str, Any],
    *,
    split: str,
    source: str,
    revision: str,
) -> MCQExample:
    """Convert one source-format row and reject malformed or ambiguous answers."""

    missing = MEDQA_FIELDS.difference(row)
    if missing:
        raise MedQAValidationError(f"missing fields: {', '.join(sorted(missing))}")

    question = _utf8_text(row["question"], field="question")
    answer = _utf8_text(row["answer_idx"], field="answer_idx").strip().upper()
    answer_text = _utf8_text(row["answer"], field="answer")

    raw_options = row["options"]
    if not isinstance(raw_options, list):
        raise MedQAValidationError("options must be a list")

    choices: dict[str, str] = {}
    for index, option in enumerate(raw_options):
        if not isinstance(option, Mapping):
            raise MedQAValidationError(f"options[{index}] must be an object")
        key = _utf8_text(option.get("key"), field=f"options[{index}].key").strip().upper()
        value = _utf8_text(option.get("value"), field=f"options[{index}].value")
        if key in choices:
            raise MedQAValidationError(f"duplicate option key: {key}")
        choices[key] = value

    if tuple(sorted(choices)) != CHOICE_KEYS:
        raise MedQAValidationError("options must contain exactly A, B, C, and D")
    if answer not in choices:
        raise MedQAValidationError("answer_idx must be one of A, B, C, or D")
    if answer_text != choices[answer]:
        raise MedQAValidationError("answer text does not match the selected option")

    example_id = stable_example_id(
        source=source,
        revision=revision,
        split=split,
        question=question,
        choices=choices,
    )
    return MCQExample(
        id=example_id,
        source=source,
        revision=revision,
        split=split,
        question=question,
        choices=choices,
        answer=answer,
    )


def iter_parquet_rows(path: Path, *, limit: int | None = None) -> Iterator[dict[str, Any]]:
    """Stream source rows from Parquet without loading a full split into memory."""

    import pyarrow.parquet as parquet

    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")

    emitted = 0
    parquet_file = parquet.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=min(limit or 1024, 1024)):
        for row in batch.to_pylist():
            yield row
            emitted += 1
            if limit is not None and emitted >= limit:
                return


def content_fingerprint(examples: Iterable[MCQExample]) -> str:
    """Hash canonical private content without returning or persisting that content."""

    digest = hashlib.sha256()
    for example in examples:
        payload = {
            "id": example.id,
            "source": example.source,
            "revision": example.revision,
            "split": example.split,
            "question": example.question,
            "choices": dict(example.choices),
            "answer": example.answer,
        }
        digest.update(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        digest.update(b"\n")
    return digest.hexdigest()


def normalized_question(question: str) -> str:
    """Conservatively normalize whitespace and case without folding medical symbols."""

    return " ".join(question.split()).casefold()


def has_ambiguous_choice_text(example: MCQExample) -> bool:
    """Detect two option keys whose normalized visible text is identical."""

    normalized_values = [normalized_question(example.choices[key]) for key in CHOICE_KEYS]
    return len(set(normalized_values)) != len(normalized_values)


def deduplicate_splits(
    examples_by_split: Mapping[str, list[MCQExample]],
) -> tuple[dict[str, list[MCQExample]], list[DuplicateRemoval]]:
    """Apply test > validation > train priority while preserving test unchanged."""

    expected_splits = {"train", "validation", "test"}
    if set(examples_by_split) != expected_splits:
        raise ValueError(f"expected splits: {sorted(expected_splits)}")

    cleaned: dict[str, list[MCQExample]] = {
        "test": list(examples_by_split["test"]),
        "validation": [],
        "train": [],
    }
    removals: list[DuplicateRemoval] = []
    seen: dict[str, str] = {}

    for example in cleaned["test"]:
        seen.setdefault(normalized_question(example.question), "test")

    for split in ("validation", "train"):
        for example in examples_by_split[split]:
            key = normalized_question(example.question)
            winner_split = seen.get(key)
            if winner_split is not None:
                removals.append(
                    DuplicateRemoval(
                        example_id=example.id,
                        removed_split=split,
                        winner_split=winner_split,
                        question_key_sha256=hashlib.sha256(key.encode("utf-8")).hexdigest(),
                    )
                )
                continue
            cleaned[split].append(example)
            seen[key] = split

    return cleaned, removals


def count_within_split_duplicate_rows(examples: Iterable[MCQExample]) -> int:
    """Count repeat rows after the first occurrence without removing them."""

    seen: set[str] = set()
    duplicate_rows = 0
    for example in examples:
        key = normalized_question(example.question)
        if key in seen:
            duplicate_rows += 1
        else:
            seen.add(key)
    return duplicate_rows


def duplicate_removals_fingerprint(removals: Iterable[DuplicateRemoval]) -> str:
    """Fingerprint ordered removal provenance without exposing source content."""

    digest = hashlib.sha256()
    for removal in removals:
        digest.update(
            json.dumps(asdict(removal), sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def assert_split_isolation(examples_by_split: Mapping[str, list[MCQExample]]) -> None:
    """Assert normalized questions and stable IDs do not cross split boundaries."""

    question_keys: dict[str, set[str]] = {}
    ids: dict[str, set[str]] = {}
    for split, examples in examples_by_split.items():
        question_keys[split] = {normalized_question(item.question) for item in examples}
        ids[split] = {item.id for item in examples}

    splits = ("train", "validation", "test")
    for index, left in enumerate(splits):
        for right in splits[index + 1 :]:
            if question_keys[left].intersection(question_keys[right]):
                raise ValueError(f"normalized question overlap remains: {left}/{right}")
            if ids[left].intersection(ids[right]):
                raise ValueError(f"stable ID overlap remains: {left}/{right}")


def example_record(example: MCQExample) -> dict[str, Any]:
    """Return the fixed serialized MCQ schema."""

    record = asdict(example)
    record["choices"] = {key: example.choices[key] for key in CHOICE_KEYS}
    return record


def _write_jsonl_rows(output_file: TextIO, examples: Iterable[MCQExample]) -> None:
    for example in examples:
        output_file.write(
            json.dumps(
                example_record(example),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        output_file.write("\n")


def write_jsonl_atomic(path: Path, examples: Iterable[MCQExample]) -> None:
    """Write deterministic UTF-8 JSONL and replace only after a complete write."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as output_file:
        _write_jsonl_rows(output_file, examples)
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    """Hash a file in bounded chunks."""

    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

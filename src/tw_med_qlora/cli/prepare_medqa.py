"""Download, validate, deduplicate, and fingerprint the full MedQA Taiwan dataset."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pyarrow.parquet as parquet
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

from tw_med_qlora.config import load_project_config
from tw_med_qlora.medqa import (
    MEDQA_FIELDS,
    MedQAValidationError,
    assert_split_isolation,
    content_fingerprint,
    count_within_split_duplicate_rows,
    deduplicate_splits,
    duplicate_removals_fingerprint,
    file_sha256,
    has_ambiguous_choice_text,
    iter_parquet_rows,
    medqa_row_to_example,
    write_jsonl_atomic,
)
from tw_med_qlora.types import MCQExample

SPLITS = ("train", "validation", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/project.toml"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/medqa"))
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed/medqa"),
    )
    parser.add_argument("--output", type=Path, default=Path("reports/data_validation.json"))
    return parser.parse_args()


def _download_and_parse(
    *,
    dataset_id: str,
    config_name: str,
    revision: str,
    split: str,
    raw_dir: Path,
    token: str | None,
) -> tuple[list[MCQExample], dict[str, Any]]:
    filename = f"{config_name}/{split}/0000.parquet"
    path = Path(
        hf_hub_download(
            repo_id=dataset_id,
            filename=filename,
            repo_type="dataset",
            revision=revision,
            local_dir=raw_dir,
            token=token,
        )
    )
    parquet_file = parquet.ParquetFile(path)
    fields = frozenset(parquet_file.schema_arrow.names)
    if fields != MEDQA_FIELDS:
        raise ValueError(f"unexpected source schema for {split}: {sorted(fields)}")

    examples: list[MCQExample] = []
    validation_errors: Counter[str] = Counter()
    for row in iter_parquet_rows(path):
        try:
            examples.append(
                medqa_row_to_example(
                    row,
                    split=split,
                    source=dataset_id,
                    revision=revision,
                )
            )
        except MedQAValidationError as error:
            validation_errors[str(error)] += 1

    raw_rows = parquet_file.metadata.num_rows
    invalid_rows = sum(validation_errors.values())
    if split == "test" and validation_errors:
        summary = ", ".join(
            f"{reason}={count}" for reason, count in sorted(validation_errors.items())
        )
        raise ValueError(f"test contains invalid rows and must remain unchanged: {summary}")
    if len(examples) + invalid_rows != raw_rows:
        raise ValueError(f"{split} parsed row count differs from Parquet metadata")

    return examples, {
        "source_file": filename,
        "source_file_sha256": file_sha256(path),
        "raw_rows": raw_rows,
        "parsed_rows": len(examples),
        "invalid_rows": invalid_rows,
        "invalid_reasons": dict(sorted(validation_errors.items())),
        "schema_fields": sorted(fields),
        "ambiguous_choice_rows": sum(has_ambiguous_choice_text(item) for item in examples),
        "within_split_duplicate_rows": count_within_split_duplicate_rows(examples),
    }


def main() -> None:
    args = parse_args()
    load_dotenv()
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    config = load_project_config(args.config)
    medqa = config.raw["data"]["medqa"]
    dataset_id = str(medqa["dataset_id"])
    config_name = str(medqa["config"])
    revision = str(medqa["revision"])
    token = os.getenv("HF_TOKEN") or None

    raw_examples: dict[str, list[MCQExample]] = {}
    source_audit: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        raw_examples[split], source_audit[split] = _download_and_parse(
            dataset_id=dataset_id,
            config_name=config_name,
            revision=revision,
            split=split,
            raw_dir=args.raw_dir,
            token=token,
        )

    ambiguous_counts = {
        split: sum(has_ambiguous_choice_text(item) for item in raw_examples[split])
        for split in SPLITS
    }
    if ambiguous_counts["test"]:
        raise ValueError("test contains ambiguous choice text and must remain unchanged")
    quality_filtered = {
        "test": list(raw_examples["test"]),
        "validation": [
            item for item in raw_examples["validation"] if not has_ambiguous_choice_text(item)
        ],
        "train": [
            item for item in raw_examples["train"] if not has_ambiguous_choice_text(item)
        ],
    }

    cleaned, removals = deduplicate_splits(quality_filtered)
    if cleaned["test"] != raw_examples["test"]:
        raise AssertionError("test split must remain row-for-row ordered and unfiltered")
    assert_split_isolation(cleaned)

    expected_counts = {
        "train": int(medqa["expected_train_rows"]),
        "validation": int(medqa["expected_validation_rows"]),
        "test": int(medqa["expected_test_rows"]),
    }
    actual_counts = {split: len(cleaned[split]) for split in SPLITS}
    if actual_counts != expected_counts:
        raise ValueError(f"clean counts changed: expected {expected_counts}, got {actual_counts}")

    processed_files: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        output_path = args.processed_dir / f"{split}.jsonl"
        write_jsonl_atomic(output_path, cleaned[split])
        processed_files[split] = {
            "logical_path": output_path.as_posix(),
            "rows": len(cleaned[split]),
            "file_sha256": file_sha256(output_path),
            "content_sha256": content_fingerprint(cleaned[split]),
        }

    removal_pairs = Counter(
        f"{removal.removed_split}->{removal.winner_split}" for removal in removals
    )
    invalid_counts = {split: source_audit[split]["invalid_rows"] for split in SPLITS}
    report = {
        "schema_version": 1,
        "dataset": {
            "id": dataset_id,
            "config": config_name,
            "revision": revision,
        },
        "policy": {
            "priority": ["test", "validation", "train"],
            "duplicate_key": "whitespace-collapse + casefold(question)",
            "unicode_compatibility_folding": False,
            "within_split_duplicates": "removed from validation and train",
            "test_unchanged": True,
            "test_used_for_training": False,
            "invalid_rows_silently_removed": False,
        },
        "source": source_audit,
        "quality_filtering": {
            "removed_rows": sum(invalid_counts.values()) + sum(ambiguous_counts.values()),
            "removed_by_split": {
                split: invalid_counts[split] + ambiguous_counts[split] for split in SPLITS
            },
            "reasons_by_split": {
                split: {
                    **source_audit[split]["invalid_reasons"],
                    "ambiguous normalized choice text": ambiguous_counts[split],
                }
                for split in SPLITS
            },
        },
        "deduplication": {
            "removed_rows": len(removals),
            "removed_by_split": dict(sorted(Counter(r.removed_split for r in removals).items())),
            "winner_pairs": dict(sorted(removal_pairs.items())),
            "removal_provenance_sha256": duplicate_removals_fingerprint(removals),
        },
        "processed": processed_files,
        "isolation": {
            "normalized_question_cross_split_overlaps": 0,
            "stable_id_cross_split_overlaps": 0,
            "train_referenced_splits": ["train"],
            "validation_referenced_splits": ["validation"],
            "test_referenced_splits": ["test"],
        },
        "packages": {
            "huggingface-hub": version("huggingface-hub"),
            "pyarrow": version("pyarrow"),
            "python-dotenv": version("python-dotenv"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "raw_rows": {split: source_audit[split]["raw_rows"] for split in SPLITS},
                "clean_rows": actual_counts,
                "quality_removed_rows": sum(
                    invalid_counts.values()
                )
                + sum(
                    ambiguous_counts.values()
                ),
                "duplicate_removed_rows": len(removals),
                "test_unchanged": True,
                "report": args.output.as_posix(),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

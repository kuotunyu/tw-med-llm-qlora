"""Run the mandatory five-row MedQA validation gate."""

from __future__ import annotations

import argparse
import json
import os
from importlib.metadata import version
from pathlib import Path

import pyarrow.parquet as parquet
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from tw_med_qlora.chat import validate_chat_template_round_trip
from tw_med_qlora.config import load_project_config
from tw_med_qlora.medqa import (
    MEDQA_FIELDS,
    content_fingerprint,
    file_sha256,
    iter_parquet_rows,
    medqa_row_to_example,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/project.toml"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/medqa"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/data_sample_validation.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv()
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    config = load_project_config(args.config)
    medqa = config.raw["data"]["medqa"]
    sample_size = int(medqa["sample_size"])
    split = "train"
    filename = f"{medqa['config']}/{split}/0000.parquet"
    token = os.getenv("HF_TOKEN") or None

    parquet_path = Path(
        hf_hub_download(
            repo_id=str(medqa["dataset_id"]),
            filename=filename,
            repo_type="dataset",
            revision=str(medqa["revision"]),
            local_dir=args.raw_dir,
            token=token,
        )
    )
    parquet_file = parquet.ParquetFile(parquet_path)
    schema_fields = frozenset(parquet_file.schema_arrow.names)
    if schema_fields != MEDQA_FIELDS:
        raise ValueError(f"unexpected source schema: {sorted(schema_fields)}")

    rows = list(iter_parquet_rows(parquet_path, limit=sample_size))
    if len(rows) != sample_size:
        raise ValueError(f"expected {sample_size} sample rows, found {len(rows)}")
    examples = [
        medqa_row_to_example(
            row,
            split=split,
            source=str(medqa["dataset_id"]),
            revision=str(medqa["revision"]),
        )
        for row in rows
    ]

    tokenizer = AutoTokenizer.from_pretrained(
        config.primary.model_id,
        revision=config.primary.revision,
        token=token,
    )
    template_checks = [validate_chat_template_round_trip(tokenizer, item) for item in examples]

    report = {
        "schema_version": 1,
        "gate": "medqa_five_row_sample",
        "dataset": {
            "id": medqa["dataset_id"],
            "config": medqa["config"],
            "revision": medqa["revision"],
            "split": split,
            "source_file": filename,
            "source_file_sha256": file_sha256(parquet_path),
            "source_rows": parquet_file.metadata.num_rows,
            "schema_fields": sorted(schema_fields),
        },
        "sample": {
            "requested_rows": sample_size,
            "valid_rows": len(examples),
            "example_ids": [item.id for item in examples],
            "content_sha256": content_fingerprint(examples),
            "checks": {
                "utf8_round_trip": True,
                "exactly_four_choices": True,
                "answer_key_valid": True,
                "answer_text_matches": True,
                "chat_template_round_trip": all(
                    bool(check["round_trip_equal"]) for check in template_checks
                ),
                "single_bos": all(check["bos_count"] == 1 for check in template_checks),
            },
        },
        "tokenizer": {
            "model_id": config.primary.model_id,
            "revision": config.primary.revision,
            "class": type(tokenizer).__name__,
            "template_checks": template_checks,
        },
        "packages": {
            "huggingface-hub": version("huggingface-hub"),
            "pyarrow": version("pyarrow"),
            "python-dotenv": version("python-dotenv"),
            "transformers": version("transformers"),
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
                "gate": report["gate"],
                "valid_rows": len(examples),
                "chat_template_round_trip": report["sample"]["checks"][
                    "chat_template_round_trip"
                ],
                "report": args.output.as_posix(),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

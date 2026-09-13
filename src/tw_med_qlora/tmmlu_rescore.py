"""Rescore the frozen Phase 4 TMMLU+ outputs under a newer dataset revision.

The Phase 4 prompt was zero-shot, so a v1.1 question whose subject, stem, and ordered
choices are byte-identical to a v1.0 question received exactly the historical prompt.
For those questions the historical prediction can be rescored against the revised gold
answer. Public rows store gold and prediction in the *shuffled* letter space produced by
``tmmlu.shuffle_options`` with the v1.0 example ID, so the revised gold must be mapped
through the same permutation before comparison. Nothing here is a new v1.1 run.
"""

from __future__ import annotations

import hashlib
import json
import random
import statistics
import zipfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from tw_med_qlora.evaluation import (
    PredictionRecord,
    _percentile,
    accuracy_summary,
    forgetting_noninferiority,
    mcnemar_exact_test,
    paired_bootstrap_accuracy_difference,
    subject_accuracy,
)
from tw_med_qlora.medqa import file_sha256
from tw_med_qlora.tmmlu_revision import (
    CATEGORY_ADDED_OR_MODIFIED,
    CATEGORY_ANSWER_REVISED,
    CATEGORY_DUPLICATE_AMBIGUOUS,
    CATEGORY_REMOVED_OR_MODIFIED,
    CATEGORY_UNCHANGED,
    PAIRED_CATEGORIES,
    group_counts,
)
from tw_med_qlora.types import CHOICE_KEYS

SUITE = "tmmlu-full"
INSTRUCT_MODEL = "original-instruct"
BASE_MODEL = "localized-base"
ADAPTER_MODEL = "localized-medical-adapter"
MODEL_LABELS = (INSTRUCT_MODEL, BASE_MODEL, ADAPTER_MODEL)
EVIDENCE_KIND = "historical_outputs_rescored_under_v1_1"

VARIANT_V1_0_FULL = "v1_0_full"
VARIANT_RETAINED_OLD_GOLD = "retained_old_gold"
VARIANT_RETAINED_NEW_GOLD = "retained_new_gold"
VARIANT_RETAINED_NEW_GOLD_WITH_DUPLICATES = "retained_new_gold_with_duplicate_row_order_pairing"
VARIANT_DESCRIPTIONS = {
    VARIANT_V1_0_FULL: "All 5,573 v1.0 questions scored with the v1.0 answer key "
    "(must reproduce the frozen Phase 4 numbers).",
    VARIANT_RETAINED_OLD_GOLD: "Only questions whose content is byte-identical in v1.1, "
    "scored with the v1.0 answer key (isolates the removed-question effect).",
    VARIANT_RETAINED_NEW_GOLD: "Only questions whose content is byte-identical in v1.1, "
    "scored with the v1.1 answer key (primary rescored result).",
    VARIANT_RETAINED_NEW_GOLD_WITH_DUPLICATES: "Sensitivity only: additionally pairs "
    "duplicate-content rows in source-row order when both revisions carry the same "
    "multiplicity and a uniform gold letter.",
}


def shuffled_letter(example_id: str, *, seed: int, letter: str) -> str:
    """Map a source-order choice letter into the shuffled prompt letter space.

    Mirrors ``tmmlu.shuffle_options``: the shuffled position ``k`` shows the original
    choice ``perm[k]`` where ``perm`` sorts A-D by a stable SHA-256 key. The gold letter
    therefore moves to the index of its original key inside ``perm``.
    """

    if letter not in CHOICE_KEYS:
        raise ValueError("letter must be one of A, B, C, or D")
    perm = sorted(
        CHOICE_KEYS,
        key=lambda key: hashlib.sha256(
            f"option-order:{seed}:{example_id}:{key}".encode()
        ).hexdigest(),
    )
    return CHOICE_KEYS[perm.index(letter)]


def load_historical_tmmlu_rows(
    archive_path,
    *,
    expected_sha256: str,
    expected_rows_per_model: int,
    suite: str = SUITE,
) -> tuple[list[dict[str, Any]], int]:
    """Read the frozen public per-question rows for one suite after verifying the ZIP."""

    actual = file_sha256(archive_path)
    if actual != expected_sha256:
        raise ValueError(f"historical archive SHA-256 mismatch: {actual}")
    with zipfile.ZipFile(archive_path) as archive:
        lines = archive.read("public/public-predictions.jsonl").decode("utf-8").splitlines()
    rows = [json.loads(line) for line in lines]
    suite_rows = [row for row in rows if row.get("suite") == suite]
    per_model: dict[str, int] = defaultdict(int)
    for row in suite_rows:
        per_model[row["model"]] += 1
    if set(per_model) != set(MODEL_LABELS):
        raise ValueError(f"unexpected model labels in {suite}: {sorted(per_model)}")
    if any(count != expected_rows_per_model for count in per_model.values()):
        raise ValueError(f"unexpected {suite} row counts: {dict(per_model)}")
    seeds = {row["option_seed"] for row in suite_rows}
    if len(seeds) != 1:
        raise ValueError(f"{suite} rows mix option seeds: {sorted(seeds)}")
    return suite_rows, int(seeds.pop())


def index_mapping_rows(mapping: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Index the v1.0 rows of a mapping by their historical example ID."""

    indexed = {row["example_id"]: row for row in mapping["rows"] if row["version"] == "old"}
    if not indexed:
        raise ValueError("mapping contains no old rows")
    return indexed


def verify_shuffle_contract(
    rows: Iterable[Mapping[str, Any]],
    old_rows: Mapping[str, Mapping[str, Any]],
    *,
    seed: int,
) -> int:
    """Fail unless every public gold equals the shuffled v1.0 source gold."""

    checked = 0
    for row in rows:
        entry = old_rows.get(row["example_id"])
        if entry is None:
            raise ValueError(f"historical example missing from mapping: {row['example_id']}")
        expected = shuffled_letter(row["example_id"], seed=seed, letter=entry["gold"])
        if row["gold"] != expected:
            raise ValueError(
                f"public gold does not match shuffled source gold: {row['example_id']}"
            )
        checked += 1
    if checked == 0:
        raise ValueError("no historical rows to verify")
    return checked


def _record(row: Mapping[str, Any], *, gold: str) -> PredictionRecord:
    return PredictionRecord(
        example_id=row["example_id"],
        model=row["model"],
        source=row["source"],
        subject=row["subject"],
        gold=gold,
        prediction=row["prediction"],
        raw_output_sha256=row["raw_output_sha256"],
        latency_seconds=float(row["latency_seconds"]),
        prompt_tokens=row["prompt_tokens"],
        completion_tokens=row["completion_tokens"],
    )


def rescored_records(
    rows: Iterable[Mapping[str, Any]],
    mapping: Mapping[str, Any],
    *,
    variant: str,
    seed: int,
) -> dict[str, list[PredictionRecord]]:
    """Build per-model prediction records for one scoring variant.

    Records keep the v1.0 example ID so the three models stay aligned for the paired
    statistics in :mod:`tw_med_qlora.evaluation`.
    """

    if variant not in VARIANT_DESCRIPTIONS:
        raise ValueError(f"unknown variant: {variant}")
    old_rows = index_mapping_rows(mapping)
    duplicate_new_gold = _duplicate_group_new_gold(mapping)
    records: dict[str, list[PredictionRecord]] = {label: [] for label in MODEL_LABELS}
    for row in rows:
        entry = old_rows[row["example_id"]]
        category = entry["category"]
        if variant == VARIANT_V1_0_FULL:
            gold = row["gold"]
        elif variant == VARIANT_RETAINED_OLD_GOLD:
            if category not in PAIRED_CATEGORIES:
                continue
            gold = shuffled_letter(row["example_id"], seed=seed, letter=entry["gold"])
        else:
            if category in PAIRED_CATEGORIES:
                source_gold = entry["paired_gold"]
            elif (
                variant == VARIANT_RETAINED_NEW_GOLD_WITH_DUPLICATES
                and category == CATEGORY_DUPLICATE_AMBIGUOUS
                and entry.get("sensitivity_paired_example_id")
            ):
                source_gold = duplicate_new_gold[entry["sensitivity_paired_example_id"]]
            else:
                continue
            gold = shuffled_letter(row["example_id"], seed=seed, letter=source_gold)
        records[row["model"]].append(_record(row, gold=gold))
    if any(not values for values in records.values()):
        raise ValueError(f"variant {variant} produced an empty model")
    return records


def _duplicate_group_new_gold(mapping: Mapping[str, Any]) -> dict[str, str]:
    return {
        row["example_id"]: row["gold"]
        for row in mapping["rows"]
        if row["version"] == "new" and row["category"] == CATEGORY_DUPLICATE_AMBIGUOUS
    }


def failure_breakdown(
    rows: Iterable[Mapping[str, Any]],
    example_ids: set[str] | None = None,
) -> dict[str, dict[str, dict[str, int]]]:
    """Count parse failures per model and subject, split into token-limit and other."""

    counts: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"parse_fail_length": 0, "parse_fail_other": 0})
    )
    for row in rows:
        if example_ids is not None and row["example_id"] not in example_ids:
            continue
        if row["prediction"] is not None:
            continue
        bucket = "parse_fail_length" if row.get("max_token_limit_hit") else "parse_fail_other"
        counts[row["model"]][row["subject"]][bucket] += 1
    return {model: dict(subjects) for model, subjects in counts.items()}


def model_summary(
    records: Sequence[PredictionRecord],
    *,
    medical_subjects: Sequence[str],
    control_subjects: Sequence[str],
    failures: Mapping[str, Mapping[str, int]] | None = None,
) -> dict[str, Any]:
    """Overall micro accuracy, per-subject table, and the two predeclared macros."""

    by_subject = subject_accuracy(records)
    for subject, summary in by_subject.items():
        breakdown = (failures or {}).get(subject, {})
        summary["parse_fail_length"] = int(breakdown.get("parse_fail_length", 0))
        summary["parse_fail_other"] = int(breakdown.get("parse_fail_other", 0))
        if summary["parse_fail_length"] + summary["parse_fail_other"] != summary["parse_failures"]:
            raise ValueError(f"parse failure breakdown mismatch for {subject}")
    missing = [s for s in (*medical_subjects, *control_subjects) if s not in by_subject]
    if missing:
        raise ValueError(f"subjects missing from records: {missing}")
    return {
        "overall": accuracy_summary(records),
        "by_subject": by_subject,
        "medical_subject_macro_accuracy": statistics.mean(
            by_subject[subject]["accuracy"] for subject in medical_subjects
        ),
        "control_subject_macro_accuracy": statistics.mean(
            by_subject[subject]["accuracy"] for subject in control_subjects
        ),
    }


def _filter(records: Iterable[PredictionRecord], subjects: Sequence[str]) -> list[PredictionRecord]:
    allowed = set(subjects)
    return [record for record in records if record.subject in allowed]


def paired_statistics(
    records: Mapping[str, Sequence[PredictionRecord]],
    *,
    medical_subjects: Sequence[str],
    control_subjects: Sequence[str],
    iterations: int,
    seed: int,
    margin_percentage_points: float,
) -> dict[str, Any]:
    """Predeclared control non-inferiority plus medical/overall paired comparisons."""

    base = records[BASE_MODEL]
    adapter = records[ADAPTER_MODEL]
    instruct = records[INSTRUCT_MODEL]
    result: dict[str, Any] = {
        "control_noninferiority_adapter_vs_base": {
            "subjects": list(control_subjects),
            "questions": len(_filter(adapter, control_subjects)),
            **forgetting_noninferiority(
                _filter(base, control_subjects),
                _filter(adapter, control_subjects),
                margin_percentage_points=margin_percentage_points,
                iterations=iterations,
                seed=seed,
            ),
        }
    }
    for label, reference in (("base", base), ("instruct", instruct)):
        medical_reference = _filter(reference, medical_subjects)
        medical_adapter = _filter(adapter, medical_subjects)
        result[f"medical_macro_adapter_vs_{label}"] = {
            "subjects": list(medical_subjects),
            "questions": len(medical_adapter),
            **paired_bootstrap_accuracy_difference(
                medical_reference,
                medical_adapter,
                iterations=iterations,
                seed=seed,
                stratify_by_subject=True,
            ),
            "mcnemar_pooled": mcnemar_exact_test(medical_reference, medical_adapter),
        }
        result[f"overall_micro_adapter_vs_{label}"] = {
            "questions": len(adapter),
            **paired_bootstrap_accuracy_difference(
                reference, adapter, iterations=iterations, seed=seed, stratify_by_subject=False
            ),
            "mcnemar_pooled": mcnemar_exact_test(reference, adapter),
        }
    return result


def _bootstrap_mean_difference(
    grouped: Mapping[str, Sequence[int]], *, iterations: int, seed: int
) -> dict[str, float | int]:
    if not grouped or any(not values for values in grouped.values()):
        raise ValueError("bootstrap requires non-empty groups")
    observed = sum(sum(values) / len(values) for values in grouped.values()) / len(grouped)
    rng = random.Random(seed)
    replicates: list[float] = []
    for _ in range(iterations):
        means = []
        for values in grouped.values():
            sampled = sum(values[rng.randrange(len(values))] for _ in values)
            means.append(sampled / len(values))
        replicates.append(sum(means) / len(means))
    replicates.sort()
    return {
        "iterations": iterations,
        "seed": seed,
        "observed_difference_percentage_points": observed * 100,
        "ci_lower_percentage_points": _percentile(replicates, 0.025) * 100,
        "ci_upper_percentage_points": _percentile(replicates, 0.975) * 100,
    }


def version_delta(
    old_gold: Mapping[str, Sequence[PredictionRecord]],
    new_gold: Mapping[str, Sequence[PredictionRecord]],
    *,
    medical_subjects: Sequence[str],
    control_subjects: Sequence[str],
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    """Same model, same questions, same outputs: v1.1 minus v1.0 answer-key effect."""

    result: dict[str, Any] = {}
    for model in MODEL_LABELS:
        old_by_id = {record.example_id: record for record in old_gold[model]}
        new_by_id = {record.example_id: record for record in new_gold[model]}
        if set(old_by_id) != set(new_by_id):
            raise ValueError(f"version delta requires identical question sets: {model}")
        diffs_all: list[int] = []
        diffs_by_subject: dict[str, list[int]] = defaultdict(list)
        flips = {"to_correct": 0, "to_wrong": 0, "gold_changed": 0}
        for example_id in sorted(old_by_id):
            old_record = old_by_id[example_id]
            new_record = new_by_id[example_id]
            diff = int(new_record.correct) - int(old_record.correct)
            diffs_all.append(diff)
            diffs_by_subject[old_record.subject].append(diff)
            if old_record.gold != new_record.gold:
                flips["gold_changed"] += 1
                if diff > 0:
                    flips["to_correct"] += 1
                elif diff < 0:
                    flips["to_wrong"] += 1
        result[model] = {
            "questions": len(diffs_all),
            "answer_revision_flips": flips,
            "overall_micro": _bootstrap_mean_difference(
                {"__all__": diffs_all}, iterations=iterations, seed=seed
            ),
            "medical_macro": _bootstrap_mean_difference(
                {subject: diffs_by_subject[subject] for subject in medical_subjects},
                iterations=iterations,
                seed=seed,
            ),
            "control_macro": _bootstrap_mean_difference(
                {subject: diffs_by_subject[subject] for subject in control_subjects},
                iterations=iterations,
                seed=seed,
            ),
        }
    return result


def all_parsed_slice(
    records: Mapping[str, Sequence[PredictionRecord]],
    *,
    medical_subjects: Sequence[str],
    control_subjects: Sequence[str],
) -> dict[str, Any]:
    """Descriptive accuracy on questions every model parsed; selection-biased by design."""

    parsed_ids: set[str] | None = None
    for model_records in records.values():
        ids = {record.example_id for record in model_records if record.parsed}
        parsed_ids = ids if parsed_ids is None else parsed_ids & ids
    assert parsed_ids is not None
    result: dict[str, Any] = {
        "questions": len(parsed_ids),
        "note": "conditional on all three models producing a parseable answer; this slice "
        "does not remove the format-compliance confound",
        "models": {},
    }
    for model, model_records in records.items():
        subset = [record for record in model_records if record.example_id in parsed_ids]
        result["models"][model] = model_summary(
            subset, medical_subjects=medical_subjects, control_subjects=control_subjects
        )
    return result


def coverage(
    mapping: Mapping[str, Any],
    *,
    medical_subjects: Sequence[str],
    control_subjects: Sequence[str],
) -> dict[str, Any]:
    """Question coverage of the rescoring per subject group."""

    counts = mapping["counts"]
    groups = {
        "all_13": [*medical_subjects, *control_subjects],
        "medical_8": list(medical_subjects),
        "control_5": list(control_subjects),
    }
    result: dict[str, Any] = {}
    for name, subjects in groups.items():
        old = group_counts(counts, version="old", subjects=subjects)
        new = group_counts(counts, version="new", subjects=subjects)
        result[name] = {
            "v1_0_total": old["total"],
            "v1_0_retained_with_identical_content": old[CATEGORY_UNCHANGED]
            + old[CATEGORY_ANSWER_REVISED],
            "v1_0_answer_revised": old[CATEGORY_ANSWER_REVISED],
            "v1_0_removed_or_modified": old[CATEGORY_REMOVED_OR_MODIFIED],
            "v1_0_duplicate_ambiguous": old[CATEGORY_DUPLICATE_AMBIGUOUS],
            "v1_1_total": new["total"],
            "v1_1_covered_by_rescoring": new[CATEGORY_UNCHANGED] + new[CATEGORY_ANSWER_REVISED],
            "v1_1_added_or_modified_uncovered": new[CATEGORY_ADDED_OR_MODIFIED],
            "v1_1_duplicate_ambiguous": new[CATEGORY_DUPLICATE_AMBIGUOUS],
        }
    return result


def analyze(
    rows: Sequence[Mapping[str, Any]],
    mapping: Mapping[str, Any],
    *,
    seed: int,
    medical_subjects: Sequence[str],
    control_subjects: Sequence[str],
    iterations: int,
    margin_percentage_points: float,
    frozen_tmmlu_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Produce the complete content-free rescoring analysis."""

    old_rows = index_mapping_rows(mapping)
    verified = verify_shuffle_contract(rows, old_rows, seed=seed)

    variants: dict[str, dict[str, Any]] = {}
    records_by_variant: dict[str, dict[str, list[PredictionRecord]]] = {}
    for variant in VARIANT_DESCRIPTIONS:
        records = rescored_records(rows, mapping, variant=variant, seed=seed)
        records_by_variant[variant] = records
        ids = {record.example_id for record in records[ADAPTER_MODEL]}
        failures = failure_breakdown(rows, ids)
        variants[variant] = {
            "description": VARIANT_DESCRIPTIONS[variant],
            "questions_per_model": len(ids),
            "models": {
                model: model_summary(
                    records[model],
                    medical_subjects=medical_subjects,
                    control_subjects=control_subjects,
                    failures=failures.get(model, {}),
                )
                for model in MODEL_LABELS
            },
        }

    reproduction = None
    if frozen_tmmlu_summary is not None:
        reproduction = _reproduction_check(variants[VARIANT_V1_0_FULL], frozen_tmmlu_summary)

    paired = {
        variant: paired_statistics(
            records_by_variant[variant],
            medical_subjects=medical_subjects,
            control_subjects=control_subjects,
            iterations=iterations,
            seed=seed,
            margin_percentage_points=margin_percentage_points,
        )
        for variant in (VARIANT_V1_0_FULL, VARIANT_RETAINED_NEW_GOLD)
    }

    decomposition = {}
    for model in MODEL_LABELS:
        views = {
            variant: _metric_view(variants[variant]["models"][model])
            for variant in (
                VARIANT_V1_0_FULL,
                VARIANT_RETAINED_OLD_GOLD,
                VARIANT_RETAINED_NEW_GOLD,
            )
        }
        decomposition[model] = {}
        for metric, key in (
            ("overall_micro", "overall"),
            ("medical_macro", "medical_subject_macro_accuracy"),
            ("control_macro", "control_subject_macro_accuracy"),
        ):
            full = views[VARIANT_V1_0_FULL][key]
            old_gold = views[VARIANT_RETAINED_OLD_GOLD][key]
            new_gold = views[VARIANT_RETAINED_NEW_GOLD][key]
            decomposition[model][metric] = {
                "v1_0_full": full,
                "retained_old_gold": old_gold,
                "retained_new_gold": new_gold,
                "removed_question_effect_pp": (old_gold - full) * 100,
                "answer_revision_effect_pp": (new_gold - old_gold) * 100,
                "total_change_pp": (new_gold - full) * 100,
            }

    return {
        "schema_version": 1,
        "evidence_kind": EVIDENCE_KIND,
        "suite": SUITE,
        "option_seed": seed,
        "shuffle_contract_rows_verified": verified,
        "models": list(MODEL_LABELS),
        "statistics": {
            "bootstrap_iterations": iterations,
            "bootstrap_seed": seed,
            "noninferiority_margin_percentage_points": margin_percentage_points,
            "macro_definition": "unweighted mean of per-subject accuracy",
            "parse_failures_count_as_incorrect": True,
        },
        "coverage": coverage(
            mapping, medical_subjects=medical_subjects, control_subjects=control_subjects
        ),
        "variants": variants,
        "frozen_reproduction_check": reproduction,
        "paired": paired,
        "decomposition": decomposition,
        "version_delta_same_outputs": version_delta(
            records_by_variant[VARIANT_RETAINED_OLD_GOLD],
            records_by_variant[VARIANT_RETAINED_NEW_GOLD],
            medical_subjects=medical_subjects,
            control_subjects=control_subjects,
            iterations=iterations,
            seed=seed,
        ),
        "all_parsed_slice": {
            variant: all_parsed_slice(
                records_by_variant[variant],
                medical_subjects=medical_subjects,
                control_subjects=control_subjects,
            )
            for variant in (VARIANT_V1_0_FULL, VARIANT_RETAINED_NEW_GOLD)
        },
    }


def _metric_view(summary: Mapping[str, Any]) -> dict[str, float]:
    return {
        "overall": float(summary["overall"]["accuracy"]),
        "medical_subject_macro_accuracy": float(summary["medical_subject_macro_accuracy"]),
        "control_subject_macro_accuracy": float(summary["control_subject_macro_accuracy"]),
    }


def _reproduction_check(
    variant: Mapping[str, Any], frozen: Mapping[str, Any]
) -> dict[str, Any]:
    mismatches: list[str] = []
    for model in MODEL_LABELS:
        ours = variant["models"][model]
        theirs = frozen["models"][model]
        for key, value in ours["overall"].items():
            if theirs["overall"].get(key) != value:
                mismatches.append(f"{model}/overall/{key}")
        for subject, summary in ours["by_subject"].items():
            frozen_subject = theirs["by_subject"].get(subject, {})
            for key in ("total", "parsed", "parse_failures", "correct", "accuracy", "parse_rate"):
                if frozen_subject.get(key) != summary[key]:
                    mismatches.append(f"{model}/{subject}/{key}")
        for key in ("medical_subject_macro_accuracy", "control_subject_macro_accuracy"):
            if abs(theirs[key] - ours[key]) > 1e-12:
                mismatches.append(f"{model}/{key}")
    return {"status": "passed" if not mismatches else "failed", "mismatches": mismatches}

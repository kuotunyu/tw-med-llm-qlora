"""Dependency-free helpers shared by smoke tests and formal evaluation."""

from __future__ import annotations

import hashlib
import math
import random
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

_STRICT_DIRECT_ANSWER = re.compile(r"\A\s*([A-D])\s*[。.]?\s*\Z")
_BOXED_ANSWER = re.compile(r"\\boxed\s*\{\s*([A-D])\s*\}")
_BOX_MARKER = re.compile(r"\\boxed\b")
_SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")


def parse_mcq_answer(text: str) -> str | None:
    """Return one unambiguous A-D answer under the evaluation contract.

    A response is valid when it is either a standalone uppercase choice or contains
    exactly one simple LaTeX box whose entire payload is an uppercase choice. The
    latter supports benchmark models that reason before emitting ``\\boxed{A}``.
    Missing, malformed, nested, or multiple boxes are rejected.
    """

    direct = _STRICT_DIRECT_ANSWER.fullmatch(text)
    if direct is not None:
        return direct.group(1)

    box_markers = _BOX_MARKER.findall(text)
    valid_boxes = _BOXED_ANSWER.findall(text)
    if len(box_markers) == 1 and len(valid_boxes) == 1:
        return valid_boxes[0]
    return None


@dataclass(frozen=True)
class PredictionRecord:
    """Content-safe per-question result used for public statistical summaries."""

    example_id: str
    model: str
    source: str
    subject: str
    gold: str
    prediction: str | None
    raw_output_sha256: str
    latency_seconds: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    def __post_init__(self) -> None:
        if not self.example_id:
            raise ValueError("example_id must not be empty")
        if not self.model:
            raise ValueError("model must not be empty")
        if not self.source:
            raise ValueError("source must not be empty")
        if self.gold not in {"A", "B", "C", "D"}:
            raise ValueError("gold must be one of A, B, C, or D")
        if self.prediction is not None and self.prediction not in {"A", "B", "C", "D"}:
            raise ValueError("prediction must be None or one of A, B, C, or D")
        if not _SHA256.fullmatch(self.raw_output_sha256):
            raise ValueError("raw_output_sha256 must be a lowercase SHA-256 digest")
        if not math.isfinite(self.latency_seconds) or self.latency_seconds < 0:
            raise ValueError("latency_seconds must be finite and non-negative")
        for name, count in (
            ("prompt_tokens", self.prompt_tokens),
            ("completion_tokens", self.completion_tokens),
        ):
            if count is not None and count < 0:
                raise ValueError(f"{name} must be non-negative")

    @property
    def parsed(self) -> bool:
        return self.prediction is not None

    @property
    def correct(self) -> bool:
        return self.prediction == self.gold

    def as_public_dict(self) -> dict[str, Any]:
        """Serialize without question text or raw model output."""

        record = asdict(self)
        record["parsed"] = self.parsed
        record["correct"] = self.correct
        return record


def prediction_record(
    *,
    example_id: str,
    model: str,
    source: str,
    subject: str,
    gold: str,
    raw_output: str,
    latency_seconds: float,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
) -> PredictionRecord:
    """Parse one strict A-D response while retaining only its digest publicly."""

    return PredictionRecord(
        example_id=example_id,
        model=model,
        source=source,
        subject=subject,
        gold=gold,
        prediction=parse_mcq_answer(raw_output),
        raw_output_sha256=hashlib.sha256(raw_output.encode("utf-8")).hexdigest(),
        latency_seconds=latency_seconds,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


def _unique_by_id(records: Iterable[PredictionRecord]) -> dict[str, PredictionRecord]:
    indexed: dict[str, PredictionRecord] = {}
    for record in records:
        if record.example_id in indexed:
            raise ValueError(f"duplicate prediction ID: {record.example_id}")
        indexed[record.example_id] = record
    if not indexed:
        raise ValueError("at least one prediction record is required")
    return indexed


def accuracy_summary(records: Iterable[PredictionRecord]) -> dict[str, int | float]:
    """Report accuracy with parse failures counted as incorrect."""

    indexed = _unique_by_id(records)
    values = list(indexed.values())
    parsed = sum(record.parsed for record in values)
    correct = sum(record.correct for record in values)
    total = len(values)
    return {
        "total": total,
        "parsed": parsed,
        "parse_failures": total - parsed,
        "correct": correct,
        "accuracy": correct / total,
        "parse_rate": parsed / total,
    }


def subject_accuracy(
    records: Iterable[PredictionRecord],
) -> dict[str, dict[str, int | float]]:
    """Return stable alphabetical subject summaries."""

    grouped: dict[str, list[PredictionRecord]] = defaultdict(list)
    for record in records:
        if not record.subject:
            raise ValueError("subject_accuracy requires non-empty subjects")
        grouped[record.subject].append(record)
    if not grouped:
        raise ValueError("at least one prediction record is required")
    return {subject: accuracy_summary(grouped[subject]) for subject in sorted(grouped)}


def _aligned_pairs(
    base_records: Iterable[PredictionRecord],
    adapter_records: Iterable[PredictionRecord],
) -> list[tuple[PredictionRecord, PredictionRecord]]:
    base = _unique_by_id(base_records)
    adapter = _unique_by_id(adapter_records)
    if set(base) != set(adapter):
        missing_adapter = sorted(set(base).difference(adapter))
        missing_base = sorted(set(adapter).difference(base))
        raise ValueError(
            "paired prediction IDs differ: "
            f"missing_adapter={missing_adapter[:3]}, missing_base={missing_base[:3]}"
        )

    pairs: list[tuple[PredictionRecord, PredictionRecord]] = []
    for example_id in sorted(base):
        left = base[example_id]
        right = adapter[example_id]
        if (left.gold, left.source, left.subject) != (
            right.gold,
            right.source,
            right.subject,
        ):
            raise ValueError(f"paired prediction metadata differs: {example_id}")
        pairs.append((left, right))
    return pairs


def _percentile(sorted_values: Sequence[float], quantile: float) -> float:
    if not sorted_values:
        raise ValueError("percentile requires values")
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be between zero and one")
    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight)


def paired_bootstrap_accuracy_difference(
    base_records: Iterable[PredictionRecord],
    adapter_records: Iterable[PredictionRecord],
    *,
    iterations: int = 10_000,
    seed: int = 3407,
    stratify_by_subject: bool = False,
) -> dict[str, int | float | bool]:
    """Bootstrap adapter-minus-base accuracy using paired question outcomes.

    When ``stratify_by_subject`` is true, questions are resampled within each subject and
    the replicate statistic is the unweighted mean of subject-level differences.
    """

    if iterations <= 0:
        raise ValueError("iterations must be positive")
    pairs = _aligned_pairs(base_records, adapter_records)
    grouped: dict[str, list[int]] = defaultdict(list)
    for base, adapter in pairs:
        subject = base.subject if stratify_by_subject else "__all__"
        if stratify_by_subject and not subject:
            raise ValueError("stratified bootstrap requires non-empty subjects")
        grouped[subject].append(int(adapter.correct) - int(base.correct))

    observed = sum(sum(values) / len(values) for values in grouped.values()) / len(grouped)
    rng = random.Random(seed)
    replicates: list[float] = []
    for _ in range(iterations):
        subject_differences = []
        for values in grouped.values():
            sampled_sum = sum(values[rng.randrange(len(values))] for _ in values)
            subject_differences.append(sampled_sum / len(values))
        replicates.append(sum(subject_differences) / len(subject_differences))
    replicates.sort()

    return {
        "iterations": iterations,
        "seed": seed,
        "stratified_by_subject": stratify_by_subject,
        "observed_difference_percentage_points": observed * 100,
        "ci_lower_percentage_points": _percentile(replicates, 0.025) * 100,
        "ci_upper_percentage_points": _percentile(replicates, 0.975) * 100,
    }


def mcnemar_exact_test(
    base_records: Iterable[PredictionRecord],
    adapter_records: Iterable[PredictionRecord],
) -> dict[str, int | float]:
    """Return the two-sided exact McNemar test for paired correctness outcomes."""

    pairs = _aligned_pairs(base_records, adapter_records)
    improved = sum((not base.correct) and adapter.correct for base, adapter in pairs)
    regressed = sum(base.correct and (not adapter.correct) for base, adapter in pairs)
    discordant = improved + regressed
    if discordant == 0:
        p_value = 1.0
    else:
        tail = min(improved, regressed)
        log_probabilities = [
            math.lgamma(discordant + 1)
            - math.lgamma(k + 1)
            - math.lgamma(discordant - k + 1)
            - discordant * math.log(2)
            for k in range(tail + 1)
        ]
        maximum = max(log_probabilities)
        cdf = math.exp(maximum) * sum(
            math.exp(value - maximum) for value in log_probabilities
        )
        p_value = min(1.0, 2 * cdf)
    return {
        "base_wrong_adapter_correct": improved,
        "base_correct_adapter_wrong": regressed,
        "discordant_pairs": discordant,
        "two_sided_exact_p_value": p_value,
    }


def forgetting_noninferiority(
    base_records: Iterable[PredictionRecord],
    adapter_records: Iterable[PredictionRecord],
    *,
    margin_percentage_points: float = 2.0,
    iterations: int = 10_000,
    seed: int = 3407,
) -> dict[str, Any]:
    """Apply the predeclared subject-macro catastrophic-forgetting rule."""

    if margin_percentage_points <= 0:
        raise ValueError("margin_percentage_points must be positive")
    bootstrap = paired_bootstrap_accuracy_difference(
        base_records,
        adapter_records,
        iterations=iterations,
        seed=seed,
        stratify_by_subject=True,
    )
    lower = float(bootstrap["ci_lower_percentage_points"])
    upper = float(bootstrap["ci_upper_percentage_points"])
    threshold = -margin_percentage_points
    if lower > threshold:
        conclusion = "no_material_forgetting"
    elif upper < threshold:
        conclusion = "forgetting_risk"
    else:
        conclusion = "inconclusive"
    return {
        **bootstrap,
        "noninferiority_margin_percentage_points": margin_percentage_points,
        "required_ci_lower_bound_above": threshold,
        "conclusion": conclusion,
    }


def representative_case_ids(
    base_records: Iterable[PredictionRecord],
    adapter_records: Iterable[PredictionRecord],
    *,
    limit: int = 10,
    seed: int = 3407,
) -> list[dict[str, str | None]]:
    """Select content-safe IDs across improvement, regression, error, and parse categories."""

    if limit <= 0:
        raise ValueError("limit must be positive")
    categories: dict[str, list[tuple[PredictionRecord, PredictionRecord]]] = defaultdict(list)
    for base, adapter in _aligned_pairs(base_records, adapter_records):
        if not base.parsed or not adapter.parsed:
            category = "parse_failure"
        elif not base.correct and adapter.correct:
            category = "improved"
        elif base.correct and not adapter.correct:
            category = "regressed"
        elif base.correct and adapter.correct:
            category = "both_correct"
        else:
            category = "both_wrong"
        categories[category].append((base, adapter))

    order = ["improved", "regressed", "both_wrong", "parse_failure", "both_correct"]
    for category, pairs in categories.items():
        pairs.sort(
            key=lambda pair: hashlib.sha256(
                f"{seed}:{category}:{pair[0].example_id}".encode()
            ).hexdigest()
        )

    selected: list[dict[str, str | None]] = []
    while len(selected) < limit:
        added = False
        for category in order:
            pairs = categories.get(category, [])
            if not pairs:
                continue
            base, adapter = pairs.pop(0)
            selected.append(
                {
                    "example_id": base.example_id,
                    "subject": base.subject or None,
                    "gold": base.gold,
                    "base_prediction": base.prediction,
                    "adapter_prediction": adapter.prediction,
                    "category": category,
                }
            )
            added = True
            if len(selected) == limit:
                break
        if not added:
            break
    return selected

"""Windows RTX 4090 probe for the TMMLU+ v1.1 limited evaluation.

This is a *separate experiment*, not a continuation of the Phase 4 run: the formal
evaluation used vLLM 0.25.1 with bitsandbytes on an A100, while this module uses
Transformers + bitsandbytes NF4 + SDPA on Windows. Prompts, parser, decoding target, and
token-limit rule replicate the Phase 4 contract so that agreement with the historical
predictions can be measured, but the outputs are stored under their own backend label and
must never be merged into the historical rows.

Heavy CUDA imports happen only inside the functions that need them, so planning and
record building stay testable on CPU-only machines.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tw_med_qlora.evaluation import parse_mcq_answer
from tw_med_qlora.tmmlu import SubjectExample, deterministic_order, shuffle_options
from tw_med_qlora.tmmlu_rescore import ADAPTER_MODEL, BASE_MODEL, INSTRUCT_MODEL, MODEL_LABELS
from tw_med_qlora.tmmlu_revision import (
    CATEGORY_ADDED_OR_MODIFIED,
    PAIRED_CATEGORIES,
    RevisionRow,
)
from tw_med_qlora.types import CHOICE_KEYS

# Byte-identical to scripts/build_full_eval_notebook.py (Phase 4 contract).
SYSTEM_PROMPT = (
    "請選擇唯一最佳答案。不要解釋或重述題目；只輸出單一大寫 "
    r"A–D 字母，或一個 LaTeX 答案框，例如 \boxed{A}。"
)
EVIDENCE_KIND = "new_inference_separate_backend_probe"
PROBE_PURPOSE = "tmmlu-v1.1-local-probe"
SHUFFLE_SPACE_HISTORICAL = "v1.0"
SHUFFLE_SPACE_NEW = "v1.1"
PUBLIC_FORBIDDEN_KEYS = frozenset({"question", "choices", "prompt", "raw_output", "system_prompt"})


def visible_prompt_template() -> str:
    return "{question}\\n\\nA. {A}\\nB. {B}\\nC. {C}\\nD. {D}"


def visible_prompt(item: SubjectExample) -> str:
    """Render the user turn exactly as the Phase 4 notebook did."""

    example = item.example
    choices = "\n".join(f"{key}. {example.choices[key]}" for key in CHOICE_KEYS)
    return f"{example.question}\n\n{choices}"


@dataclass(frozen=True)
class ProbeItem:
    """One planned generation: which content, which shuffle identity, which reason."""

    subject: str
    new_example_id: str
    historical_example_id: str | None
    shuffle_id_space: str
    reason: str
    shuffled: SubjectExample

    @property
    def prompt(self) -> str:
        return visible_prompt(self.shuffled)

    @property
    def gold(self) -> str:
        return self.shuffled.example.answer


def build_probe_plan(
    new_rows: Sequence[RevisionRow],
    mapping: Mapping[str, Any],
    *,
    subjects: Sequence[str],
    per_subject: int,
    seed: int,
    option_seed: int,
) -> list[ProbeItem]:
    """Select uncovered v1.1 rows plus a deterministic per-subject sample of covered rows.

    Covered rows reuse the historical (v1.0) example ID for the option shuffle, so their
    prompt is byte-identical to what Phase 4 sent. Uncovered rows have no historical ID
    and use their own v1.1 ID, exactly as a fresh v1.1 run would.
    """

    if per_subject < 0:
        raise ValueError("per_subject must be non-negative")
    new_entries = {row["example_id"]: row for row in mapping["rows"] if row["version"] == "new"}
    by_subject: dict[str, list[RevisionRow]] = defaultdict(list)
    plan: list[ProbeItem] = []
    for row in new_rows:
        entry = new_entries[row.example_id]
        if entry["category"] == CATEGORY_ADDED_OR_MODIFIED:
            plan.append(
                ProbeItem(
                    subject=row.subject,
                    new_example_id=row.example_id,
                    historical_example_id=None,
                    shuffle_id_space=SHUFFLE_SPACE_NEW,
                    reason="uncovered_v1_1_question",
                    shuffled=shuffle_options(row.item, seed=option_seed),
                )
            )
        elif entry["category"] in PAIRED_CATEGORIES:
            by_subject[row.subject].append(row)
    for subject in subjects:
        ordered = deterministic_order(
            (row.item for row in by_subject.get(subject, [])), seed=seed, purpose=PROBE_PURPOSE
        )
        rows_by_id = {row.example_id: row for row in by_subject.get(subject, [])}
        for item in ordered[:per_subject]:
            row = rows_by_id[item.example.id]
            entry = new_entries[row.example_id]
            historical_id = entry["paired_example_id"]
            # Rebuild the SubjectExample with the historical ID so the shuffle matches.
            historical_item = SubjectExample(
                subject=row.subject,
                example=_with_id(row.item, historical_id),
            )
            plan.append(
                ProbeItem(
                    subject=subject,
                    new_example_id=row.example_id,
                    historical_example_id=historical_id,
                    shuffle_id_space=SHUFFLE_SPACE_HISTORICAL,
                    reason=f"agreement_probe_{entry['category']}",
                    shuffled=shuffle_options(historical_item, seed=option_seed),
                )
            )
    return plan


def _with_id(item: SubjectExample, example_id: str):
    from dataclasses import replace

    return replace(item.example, id=example_id)


def token_limit_hit(*, completion_tokens: int, max_new_tokens: int) -> bool:
    """Phase 4 forced ``finish_reason == 'length'`` to a parse failure; mirror it."""

    return completion_tokens >= max_new_tokens


def public_record(
    item: ProbeItem,
    *,
    model: str,
    backend_label: str,
    raw_text: str,
    stripped_text: str,
    prompt_tokens: int,
    completion_tokens: int,
    max_new_tokens: int,
    latency_seconds: float,
) -> dict[str, Any]:
    """Content-safe row: hashes and letters only."""

    limit_hit = token_limit_hit(completion_tokens=completion_tokens, max_new_tokens=max_new_tokens)
    prediction = None if limit_hit else parse_mcq_answer(stripped_text)
    record = {
        "evidence_kind": EVIDENCE_KIND,
        "backend_label": backend_label,
        "model": model,
        "subject": item.subject,
        "source": f"ikala/tmmluplus/{item.subject}",
        "new_example_id": item.new_example_id,
        "historical_example_id": item.historical_example_id,
        "shuffle_id_space": item.shuffle_id_space,
        "reason": item.reason,
        "gold": item.gold,
        "prediction": prediction,
        "parsed": prediction is not None,
        "correct": prediction == item.gold,
        "max_token_limit_hit": limit_hit,
        "prompt_sha256": hashlib.sha256(item.prompt.encode("utf-8")).hexdigest(),
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "raw_output_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
        "raw_output_stripped_sha256": hashlib.sha256(stripped_text.encode("utf-8")).hexdigest(),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "latency_seconds": latency_seconds,
    }
    leaked = PUBLIC_FORBIDDEN_KEYS.intersection(record)
    if leaked:
        raise ValueError(f"public probe record leaks private keys: {sorted(leaked)}")
    return record


def private_record(item: ProbeItem, *, model: str, raw_text: str) -> dict[str, Any]:
    return {
        "model": model,
        "subject": item.subject,
        "new_example_id": item.new_example_id,
        "historical_example_id": item.historical_example_id,
        "shuffle_id_space": item.shuffle_id_space,
        "gold": item.gold,
        "question": item.shuffled.example.question,
        "choices": dict(item.shuffled.example.choices),
        "system_prompt": SYSTEM_PROMPT,
        "prompt": item.prompt,
        "raw_output": raw_text,
    }


def wilson_interval(
    successes: int, total: int, *, z: float = 1.959963984540054
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""

    if total <= 0:
        raise ValueError("total must be positive")
    if not 0 <= successes <= total:
        raise ValueError("successes must be between 0 and total")
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    half = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
    half /= denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def in_backend_ceiling(historical_rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Prediction agreement between ``tmmlu-full`` and ``tmmlu-stability-3407``.

    Both suites sent byte-identical prompts to the same vLLM server, so their agreement is
    the nondeterminism ceiling a different backend cannot be expected to exceed.
    """

    full = {
        (row["model"], row["example_id"]): row
        for row in historical_rows
        if row["suite"] == "tmmlu-full"
    }
    result: dict[str, dict[str, Any]] = {}
    for row in historical_rows:
        if row["suite"] != "tmmlu-stability-3407":
            continue
        reference = full.get((row["model"], row["example_id"]))
        if reference is None:
            continue
        bucket = result.setdefault(row["model"], {"agree": 0, "total": 0, "raw_hash_agree": 0})
        bucket["total"] += 1
        bucket["agree"] += int(reference["prediction"] == row["prediction"])
        bucket["raw_hash_agree"] += int(reference["raw_output_sha256"] == row["raw_output_sha256"])
    for bucket in result.values():
        bucket["rate"] = bucket["agree"] / bucket["total"] if bucket["total"] else None
    return result


def analyze_probe(
    public_rows: Sequence[Mapping[str, Any]],
    historical_rows: Sequence[Mapping[str, Any]],
    *,
    backend_label: str,
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    """Agreement with historical predictions plus descriptive v1.1 accuracy."""

    historical = {
        (row["model"], row["example_id"]): row
        for row in historical_rows
        if row["suite"] == "tmmlu-full"
    }
    ceiling = in_backend_ceiling(historical_rows)
    models: dict[str, Any] = {}
    for model in MODEL_LABELS:
        rows = [row for row in public_rows if row["model"] == model]
        if not rows:
            continue
        with_history = [row for row in rows if row["historical_example_id"] is not None]
        agree = 0
        prompt_tokens_same = 0
        raw_hash_same = 0
        historical_correct = 0
        for row in with_history:
            reference = historical[(model, row["historical_example_id"])]
            agree += int(reference["prediction"] == row["prediction"])
            prompt_tokens_same += int(reference["prompt_tokens"] == row["prompt_tokens"])
            raw_hash_same += int(
                reference["raw_output_sha256"]
                in {row["raw_output_sha256"], row["raw_output_stripped_sha256"]}
            )
            historical_correct += int(reference["prediction"] == row["gold"])
        lower, upper = wilson_interval(agree, len(with_history)) if with_history else (None, None)
        uncovered = [row for row in rows if row["historical_example_id"] is None]
        models[model] = {
            "probe_questions": len(rows),
            "probe_questions_with_historical_prompt": len(with_history),
            "prediction_agreement_with_historical": {
                "agree": agree,
                "total": len(with_history),
                "rate": agree / len(with_history) if with_history else None,
                "wilson_ci_lower": lower,
                "wilson_ci_upper": upper,
            },
            "raw_output_hash_agreement_with_historical": raw_hash_same,
            "prompt_tokens_identical_to_historical": prompt_tokens_same,
            "in_backend_ceiling_full_vs_stability_3407": ceiling.get(model),
            "accuracy_under_v1_1_gold": {
                "total": len(with_history),
                "correct": sum(int(row["correct"]) for row in with_history),
                "accuracy": (
                    sum(int(row["correct"]) for row in with_history) / len(with_history)
                    if with_history
                    else None
                ),
                "historical_prediction_correct_under_v1_1_gold": historical_correct,
            },
            "parse_failures": {
                "length": sum(int(row["max_token_limit_hit"]) for row in rows),
                "other": sum(
                    int((not row["parsed"]) and not row["max_token_limit_hit"]) for row in rows
                ),
            },
            "uncovered_v1_1_questions": {
                "total": len(uncovered),
                "correct": sum(int(row["correct"]) for row in uncovered),
                "predictions": [
                    {
                        "new_example_id": row["new_example_id"],
                        "subject": row["subject"],
                        "gold": row["gold"],
                        "prediction": row["prediction"],
                        "correct": row["correct"],
                    }
                    for row in uncovered
                ],
            },
        }
    return {
        "schema_version": 1,
        "evidence_kind": EVIDENCE_KIND,
        "backend_label": backend_label,
        "note": "Different backend from the Phase 4 run (vLLM 0.25.1 bitsandbytes, A100). "
        "Never merge these rows into the historical results.",
        "runtime": dict(runtime),
        "models": models,
    }


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def model_specs(config_raw: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """The three fixed Phase 4 models, from config rather than from names."""

    primary = config_raw["models"]["primary"]
    publication = config_raw["publication"]
    return {
        INSTRUCT_MODEL: {
            "base_model": primary["baseline_id"],
            "base_revision": primary["baseline_revision"],
            "adapter": None,
        },
        BASE_MODEL: {
            "base_model": primary["model_id"],
            "base_revision": primary["revision"],
            "adapter": None,
        },
        ADAPTER_MODEL: {
            "base_model": primary["model_id"],
            "base_revision": primary["revision"],
            "adapter": publication["adapter_repo_id"],
        },
    }


def run_model_probe(
    *,
    model_label: str,
    spec: Mapping[str, Any],
    adapter_revision: str | None,
    items: Sequence[ProbeItem],
    max_new_tokens: int,
    token: str | None,
    backend_label: str,
    progress: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Load one model stack, generate every planned item, and return rows + load info."""

    from tw_med_qlora.local_inference import (
        generate_one,
        load_adapter_contract,
        load_model_stack,
        validate_adapter_contract,
    )

    adapter_info = None
    adapter = spec["adapter"]
    load_revision = None
    if adapter is not None:
        contract = load_adapter_contract(adapter, token=token, revision=adapter_revision)
        validate_adapter_contract(
            contract,
            expected_base_model=spec["base_model"],
            expected_base_revision=spec["base_revision"],
        )
        load_revision = contract.resolved_revision or adapter_revision
        adapter_info = {
            "source": adapter,
            "resolved_revision": contract.resolved_revision,
            "config_sha256": contract.config_sha256,
            "weights_sha256": contract.weights_sha256,
            "peft_type": contract.peft_type,
        }
    model, tokenizer, load_seconds = load_model_stack(
        base_model=spec["base_model"],
        base_revision=spec["base_revision"],
        adapter=adapter,
        adapter_revision=load_revision,
        token=token,
    )
    public_rows: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, item in enumerate(items, start=1):
        result = generate_one(
            model,
            tokenizer,
            prompt=item.prompt,
            system_prompt=SYSTEM_PROMPT,
            max_new_tokens=max_new_tokens,
            stream_to_console=False,
        )
        public_rows.append(
            public_record(
                item,
                model=model_label,
                backend_label=backend_label,
                raw_text=result.raw_text,
                stripped_text=result.text,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                max_new_tokens=max_new_tokens,
                latency_seconds=result.total_generation_seconds,
            )
        )
        private_rows.append(private_record(item, model=model_label, raw_text=result.raw_text))
        if progress and (index % 10 == 0 or index == len(items)):
            elapsed = time.perf_counter() - started
            print(f"[{model_label}] {index}/{len(items)} generated in {elapsed:.0f}s", flush=True)
    del model
    try:
        import torch

        torch.cuda.empty_cache()
    except ImportError:  # pragma: no cover
        pass
    return (
        public_rows,
        private_rows,
        {
            "model_load_seconds": load_seconds,
            "generation_seconds": time.perf_counter() - started,
            "adapter": adapter_info,
        },
    )

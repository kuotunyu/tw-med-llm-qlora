"""Stable, dependency-free project data types."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass

CHOICE_KEYS = ("A", "B", "C", "D")
SPLITS = frozenset({"train", "validation", "test"})


def stable_example_id(
    *,
    source: str,
    revision: str,
    split: str,
    question: str,
    choices: Mapping[str, str],
) -> str:
    """Create a deterministic ID without exposing the question text."""

    payload = {
        "source": source,
        "revision": revision,
        "split": split,
        "question": question,
        "choices": {key: choices[key] for key in CHOICE_KEYS},
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]


@dataclass(frozen=True)
class MCQExample:
    """Canonical four-choice example shared by preparation and evaluation."""

    id: str
    source: str
    revision: str
    split: str
    question: str
    choices: Mapping[str, str]
    answer: str

    def __post_init__(self) -> None:
        if self.split not in SPLITS:
            raise ValueError(f"Unsupported split: {self.split}")
        if not self.question.strip():
            raise ValueError("question must not be empty")
        if tuple(sorted(self.choices)) != CHOICE_KEYS:
            raise ValueError("choices must contain exactly A, B, C, and D")
        if any(not self.choices[key].strip() for key in CHOICE_KEYS):
            raise ValueError("choice text must not be empty")
        if self.answer not in CHOICE_KEYS:
            raise ValueError("answer must be one of A, B, C, or D")

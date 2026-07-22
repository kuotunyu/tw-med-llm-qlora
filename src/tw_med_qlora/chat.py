"""Model-owned chat rendering for medical multiple-choice examples."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from tw_med_qlora.types import CHOICE_KEYS, MCQExample


class ChatTokenizer(Protocol):
    """Minimal tokenizer surface used by the data validation gate."""

    bos_token_id: int | None

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> Any: ...

    def __call__(self, text: str, **kwargs: Any) -> Mapping[str, Sequence[int]]: ...


def build_user_prompt(example: MCQExample) -> str:
    """Build only semantic content; special tokens belong to the model template."""

    options = "\n".join(f"{key}. {example.choices[key]}" for key in CHOICE_KEYS)
    return (
        "請閱讀以下台灣醫療多選題，選出唯一正確答案。\n\n"
        f"{example.question}\n\n{options}\n\n"
        "請只回答 A、B、C 或 D 中的一個字母。"
    )


def build_training_messages(example: MCQExample) -> list[dict[str, str]]:
    """Return a user turn plus the single-letter assistant training target."""

    return [
        {"role": "user", "content": build_user_prompt(example)},
        {"role": "assistant", "content": example.answer},
    ]


def _input_ids(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        value = value["input_ids"]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("tokenizer output must contain a sequence of input_ids")
    return [int(token_id) for token_id in value]


def validate_chat_template_round_trip(
    tokenizer: ChatTokenizer,
    example: MCQExample,
) -> dict[str, int | str | bool]:
    """Verify template rendering and re-tokenization agree without a second BOS."""

    messages = build_training_messages(example)
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    if not isinstance(rendered, str) or not rendered:
        raise ValueError("chat template did not return non-empty text")

    direct_ids = _input_ids(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
    )
    retokenized_ids = _input_ids(tokenizer(rendered, add_special_tokens=False))
    if direct_ids != retokenized_ids:
        raise ValueError("chat template tokenization round trip failed")
    if not direct_ids:
        raise ValueError("chat template produced no tokens")

    bos_token_id = tokenizer.bos_token_id
    bos_count = direct_ids.count(bos_token_id) if bos_token_id is not None else 0
    if bos_token_id is not None and bos_count != 1:
        raise ValueError(f"expected exactly one BOS token, found {bos_count}")

    return {
        "round_trip_equal": True,
        "token_count": len(direct_ids),
        "bos_count": bos_count,
        "rendered_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
    }

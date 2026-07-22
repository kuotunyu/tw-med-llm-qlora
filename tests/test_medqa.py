from __future__ import annotations

from copy import deepcopy

import pytest

from tw_med_qlora.chat import build_training_messages, validate_chat_template_round_trip
from tw_med_qlora.medqa import MedQAValidationError, medqa_row_to_example

SOURCE = "bigbio/med_qa"
REVISION = "e04abdc0672c54547fa1dbe36cfefc000e4f2657"
ROW = {
    "meta_info": "台灣",
    "question": "下列何者正確？",
    "answer_idx": "C",
    "answer": "丙",
    "options": [
        {"key": "A", "value": "甲"},
        {"key": "B", "value": "乙"},
        {"key": "C", "value": "丙"},
        {"key": "D", "value": "丁"},
    ],
}


class FakeTokenizer:
    bos_token_id = 2

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str | dict[str, list[int]]:
        assert add_generation_prompt is False
        rendered = "<bos>" + "".join(
            f"<{message['role']}>{message['content']}" for message in messages
        )
        return {"input_ids": [2, *rendered.encode("utf-8")]} if tokenize else rendered

    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        assert add_special_tokens is False
        return {"input_ids": [2, *text.encode("utf-8")]}


def parse(row: dict[str, object] = ROW):
    return medqa_row_to_example(row, split="train", source=SOURCE, revision=REVISION)


def test_medqa_row_converts_to_canonical_example() -> None:
    example = parse()

    assert example.answer == "C"
    assert tuple(sorted(example.choices)) == ("A", "B", "C", "D")
    assert len(example.id) == 20


def test_training_target_is_only_the_answer_letter() -> None:
    messages = build_training_messages(parse())

    assert messages[-1] == {"role": "assistant", "content": "C"}
    assert "A. 甲" in messages[0]["content"]


def test_chat_template_round_trip_uses_one_bos() -> None:
    result = validate_chat_template_round_trip(FakeTokenizer(), parse())

    assert result["round_trip_equal"] is True
    assert result["bos_count"] == 1


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"answer": "乙"}, "answer text does not match"),
        ({"answer_idx": "E"}, "answer_idx must be one of"),
        ({"question": "\ud800"}, "not valid UTF-8"),
    ],
)
def test_invalid_medqa_rows_are_rejected(mutation: dict[str, object], message: str) -> None:
    row = deepcopy(ROW)
    row.update(mutation)

    with pytest.raises(MedQAValidationError, match=message):
        parse(row)


def test_duplicate_option_keys_are_rejected() -> None:
    row = deepcopy(ROW)
    row["options"][3]["key"] = "A"

    with pytest.raises(MedQAValidationError, match="duplicate option key"):
        parse(row)

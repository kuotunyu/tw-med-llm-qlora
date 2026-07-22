import pytest

from tw_med_qlora.types import MCQExample, stable_example_id

CHOICES = {"A": "甲", "B": "乙", "C": "丙", "D": "丁"}


def test_stable_id_is_deterministic_and_content_sensitive() -> None:
    values = {
        "source": "example/source",
        "revision": "abc123",
        "split": "train",
        "question": "下列何者正確？",
        "choices": CHOICES,
    }

    first = stable_example_id(**values)
    second = stable_example_id(**values)
    changed = stable_example_id(**{**values, "question": "下列何者錯誤？"})

    assert first == second
    assert first != changed
    assert len(first) == 20


def test_mcq_example_accepts_a_valid_record() -> None:
    example = MCQExample(
        id="record-id",
        source="example/source",
        revision="abc123",
        split="test",
        question="下列何者正確？",
        choices=CHOICES,
        answer="C",
    )

    assert example.answer == "C"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("split", "dev", "Unsupported split"),
        ("question", " ", "question must not be empty"),
        ("answer", "E", "answer must be one of"),
        ("choices", {"A": "甲", "B": "乙", "C": "丙"}, "exactly A, B, C, and D"),
    ],
)
def test_mcq_example_rejects_invalid_records(field: str, value: object, message: str) -> None:
    values = {
        "id": "record-id",
        "source": "example/source",
        "revision": "abc123",
        "split": "train",
        "question": "下列何者正確？",
        "choices": CHOICES,
        "answer": "A",
    }
    values[field] = value

    with pytest.raises(ValueError, match=message):
        MCQExample(**values)  # type: ignore[arg-type]


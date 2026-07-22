import pytest

from tw_med_qlora.cost import estimate_training_cost


def test_cost_projection_uses_measured_throughput_and_optional_rates() -> None:
    estimate = estimate_training_cost(
        smoke_wall_seconds=100.0,
        smoke_steps=10,
        full_train_examples=11_248,
        effective_batch_size=16,
        epochs=1,
        compute_units_per_hour=2.0,
        price_per_compute_unit=3.0,
    )

    assert estimate.full_steps == 703
    assert estimate.seconds_per_step == pytest.approx(10.0)
    assert estimate.projected_hours == pytest.approx(7030 / 3600)
    assert estimate.compute_units == pytest.approx(estimate.projected_hours * 2)
    assert estimate.estimated_cost == pytest.approx(estimate.compute_units * 3)


def test_cost_projection_does_not_invent_colab_rates() -> None:
    estimate = estimate_training_cost(
        smoke_wall_seconds=50.0,
        smoke_steps=10,
        full_train_examples=100,
        effective_batch_size=16,
        epochs=1,
    )

    assert estimate.compute_units is None
    assert estimate.estimated_cost is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"smoke_wall_seconds": 0}, "smoke_wall_seconds"),
        ({"smoke_steps": 0}, "smoke_steps"),
        ({"effective_batch_size": 0}, "effective_batch_size"),
        ({"price_per_compute_unit": 1.0}, "requires compute_units_per_hour"),
    ],
)
def test_cost_projection_rejects_invalid_inputs(
    kwargs: dict[str, float | int],
    message: str,
) -> None:
    defaults: dict[str, float | int | None] = {
        "smoke_wall_seconds": 10.0,
        "smoke_steps": 1,
        "full_train_examples": 100,
        "effective_batch_size": 10,
        "epochs": 1,
        "compute_units_per_hour": None,
        "price_per_compute_unit": None,
    }
    defaults.update(kwargs)

    with pytest.raises(ValueError, match=message):
        estimate_training_cost(**defaults)  # type: ignore[arg-type]

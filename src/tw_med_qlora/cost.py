"""Project full-training time and Colab cost from a measured smoke run."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingCostEstimate:
    """Measured-throughput projection with optional billing inputs."""

    full_steps: int
    seconds_per_step: float
    projected_hours: float
    compute_units: float | None
    estimated_cost: float | None


def estimate_training_cost(
    *,
    smoke_wall_seconds: float,
    smoke_steps: int,
    full_train_examples: int,
    effective_batch_size: int,
    epochs: float,
    compute_units_per_hour: float | None = None,
    price_per_compute_unit: float | None = None,
) -> TrainingCostEstimate:
    """Scale observed step time to a full run without assuming a Colab price."""

    if smoke_wall_seconds <= 0:
        raise ValueError("smoke_wall_seconds must be positive")
    if smoke_steps <= 0:
        raise ValueError("smoke_steps must be positive")
    if full_train_examples <= 0:
        raise ValueError("full_train_examples must be positive")
    if effective_batch_size <= 0:
        raise ValueError("effective_batch_size must be positive")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if compute_units_per_hour is not None and compute_units_per_hour <= 0:
        raise ValueError("compute_units_per_hour must be positive when provided")
    if price_per_compute_unit is not None and price_per_compute_unit <= 0:
        raise ValueError("price_per_compute_unit must be positive when provided")
    if price_per_compute_unit is not None and compute_units_per_hour is None:
        raise ValueError("price_per_compute_unit requires compute_units_per_hour")

    full_steps = math.ceil(full_train_examples * epochs / effective_batch_size)
    seconds_per_step = smoke_wall_seconds / smoke_steps
    projected_hours = seconds_per_step * full_steps / 3600
    compute_units = (
        projected_hours * compute_units_per_hour
        if compute_units_per_hour is not None
        else None
    )
    estimated_cost = (
        compute_units * price_per_compute_unit
        if compute_units is not None and price_per_compute_unit is not None
        else None
    )
    return TrainingCostEstimate(
        full_steps=full_steps,
        seconds_per_step=seconds_per_step,
        projected_hours=projected_hours,
        compute_units=compute_units,
        estimated_cost=estimated_cost,
    )

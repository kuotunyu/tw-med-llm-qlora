"""Load and validate the project configuration without GPU dependencies."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrainingProfile:
    """A hardware-selected model and training profile."""

    name: str
    hardware_profile: str
    model_id: str
    revision: str
    baseline_id: str
    baseline_revision: str
    requires_bf16: bool
    batch_size: int
    gradient_accumulation_steps: int
    max_sequence_length: int
    allow_tf32: bool

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps


@dataclass(frozen=True)
class ProjectConfig:
    """Validated subset of configuration needed before training starts."""

    seed: int
    effective_batch_size: int
    primary: TrainingProfile
    fallback: TrainingProfile
    hardware_profiles: tuple[HardwareProfile, ...]
    raw: dict[str, Any]


@dataclass(frozen=True)
class HardwareProfile:
    """A VRAM/precision tier that changes throughput without changing effective batch."""

    name: str
    model_profile: str
    min_vram_gib: float
    min_compute_capability: tuple[int, int]
    requires_bf16: bool
    batch_size: int
    gradient_accumulation_steps: int
    max_sequence_length: int
    allow_tf32: bool

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps


def _profile(name: str, values: dict[str, Any]) -> TrainingProfile:
    return TrainingProfile(
        name=name,
        hardware_profile=name,
        model_id=str(values["model_id"]),
        revision=str(values["revision"]),
        baseline_id=str(values["baseline_id"]),
        baseline_revision=str(values["baseline_revision"]),
        requires_bf16=bool(values["requires_bf16"]),
        batch_size=int(values["batch_size"]),
        gradient_accumulation_steps=int(values["gradient_accumulation_steps"]),
        max_sequence_length=int(values["max_sequence_length"]),
        allow_tf32=False,
    )


def _hardware_profile(values: dict[str, Any]) -> HardwareProfile:
    capability = values["min_compute_capability"]
    if not isinstance(capability, list) or len(capability) != 2:
        raise ValueError("min_compute_capability must contain [major, minor]")
    return HardwareProfile(
        name=str(values["name"]),
        model_profile=str(values["model_profile"]),
        min_vram_gib=float(values["min_vram_gib"]),
        min_compute_capability=(int(capability[0]), int(capability[1])),
        requires_bf16=bool(values["requires_bf16"]),
        batch_size=int(values["batch_size"]),
        gradient_accumulation_steps=int(values["gradient_accumulation_steps"]),
        max_sequence_length=int(values["max_sequence_length"]),
        allow_tf32=bool(values["allow_tf32"]),
    )


def load_project_config(path: Path) -> ProjectConfig:
    """Read the TOML config and enforce cross-profile invariants."""

    with path.open("rb") as config_file:
        raw = tomllib.load(config_file)

    expected_batch = int(raw["project"]["effective_batch_size"])
    primary = _profile("primary", raw["models"]["primary"])
    fallback = _profile("fallback", raw["models"]["fallback"])

    for profile in (primary, fallback):
        for revision in (profile.revision, profile.baseline_revision):
            if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
                raise ValueError(f"{profile.name} revisions must be full lowercase commit hashes")
        if profile.effective_batch_size != expected_batch:
            raise ValueError(
                f"{profile.name} effective batch size is {profile.effective_batch_size}; "
                f"expected {expected_batch}"
            )
        if profile.max_sequence_length <= 0:
            raise ValueError(f"{profile.name} max_sequence_length must be positive")

    hardware_profiles = tuple(
        sorted(
            (_hardware_profile(item) for item in raw["hardware_profiles"]),
            key=lambda item: item.min_vram_gib,
            reverse=True,
        )
    )
    names = [profile.name for profile in hardware_profiles]
    if len(names) != len(set(names)):
        raise ValueError("hardware profile names must be unique")
    for profile in hardware_profiles:
        if profile.model_profile not in {"primary", "fallback"}:
            raise ValueError(f"unsupported model_profile: {profile.model_profile}")
        if profile.min_vram_gib <= 0:
            raise ValueError(f"{profile.name} min_vram_gib must be positive")
        if profile.effective_batch_size != expected_batch:
            raise ValueError(
                f"{profile.name} effective batch size is {profile.effective_batch_size}; "
                f"expected {expected_batch}"
            )
        if profile.max_sequence_length <= 0:
            raise ValueError(f"{profile.name} max_sequence_length must be positive")
    configured_models = {profile.model_profile for profile in hardware_profiles}
    if configured_models != {"primary", "fallback"}:
        raise ValueError("hardware profiles must cover primary and fallback models")

    return ProjectConfig(
        seed=int(raw["project"]["seed"]),
        effective_batch_size=expected_batch,
        primary=primary,
        fallback=fallback,
        hardware_profiles=hardware_profiles,
        raw=raw,
    )


def select_training_profile(
    config: ProjectConfig,
    *,
    total_vram_gib: float,
    bf16_supported: bool,
    compute_capability: tuple[int, int] = (9, 0),
) -> TrainingProfile:
    """Choose the approved profile or fail instead of silently changing models."""

    for hardware in config.hardware_profiles:
        if total_vram_gib < hardware.min_vram_gib:
            continue
        if compute_capability < hardware.min_compute_capability:
            continue
        if hardware.requires_bf16 and not bf16_supported:
            continue
        model = config.primary if hardware.model_profile == "primary" else config.fallback
        return replace(
            model,
            hardware_profile=hardware.name,
            requires_bf16=hardware.requires_bf16,
            batch_size=hardware.batch_size,
            gradient_accumulation_steps=hardware.gradient_accumulation_steps,
            max_sequence_length=hardware.max_sequence_length,
            allow_tf32=hardware.allow_tf32,
        )
    raise RuntimeError(
        f"No approved training profile for {total_vram_gib:.1f} GiB VRAM "
        f"(bf16_supported={bf16_supported}, compute_capability={compute_capability})"
    )

from pathlib import Path

import pytest

from tw_med_qlora.config import load_project_config, select_training_profile

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "project.toml"


def test_profiles_share_the_declared_effective_batch_size() -> None:
    config = load_project_config(CONFIG_PATH)

    assert config.primary.effective_batch_size == 16
    assert config.fallback.effective_batch_size == 16
    assert {profile.effective_batch_size for profile in config.hardware_profiles} == {16}
    assert len(config.primary.revision) == 40
    assert len(config.fallback.baseline_revision) == 40


@pytest.mark.parametrize(
    ("total_vram_gib", "expected_hardware", "batch_size", "gradient_accumulation"),
    [
        (80.0, "primary_80g", 8, 2),
        (40.0, "primary_40g", 4, 4),
        (24.0, "primary_24g", 1, 16),
    ],
)
def test_bf16_gpu_selects_fastest_safe_primary_tier(
    total_vram_gib: float,
    expected_hardware: str,
    batch_size: int,
    gradient_accumulation: int,
) -> None:
    config = load_project_config(CONFIG_PATH)

    profile = select_training_profile(
        config,
        total_vram_gib=total_vram_gib,
        bf16_supported=True,
        compute_capability=(8, 0),
    )

    assert profile.name == "primary"
    assert profile.hardware_profile == expected_hardware
    assert profile.batch_size == batch_size
    assert profile.gradient_accumulation_steps == gradient_accumulation
    assert profile.effective_batch_size == 16
    assert profile.allow_tf32 is True


def test_t4_like_gpu_selects_fallback() -> None:
    config = load_project_config(CONFIG_PATH)

    profile = select_training_profile(
        config,
        total_vram_gib=15.0,
        bf16_supported=False,
        compute_capability=(7, 5),
    )

    assert profile.name == "fallback"
    assert profile.hardware_profile == "fallback_16g"
    assert profile.effective_batch_size == 16
    assert profile.allow_tf32 is False


def test_large_pre_ampere_gpu_does_not_select_bf16_primary() -> None:
    config = load_project_config(CONFIG_PATH)

    profile = select_training_profile(
        config,
        total_vram_gib=40.0,
        bf16_supported=False,
        compute_capability=(7, 5),
    )

    assert profile.name == "fallback"
    assert profile.hardware_profile == "fallback_16g"


def test_small_gpu_fails_instead_of_switching_to_unapproved_model() -> None:
    config = load_project_config(CONFIG_PATH)

    with pytest.raises(RuntimeError, match="No approved training profile"):
        select_training_profile(
            config,
            total_vram_gib=4.0,
            bf16_supported=False,
            compute_capability=(7, 5),
        )

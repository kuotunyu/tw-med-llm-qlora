from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from tw_med_qlora.phase4_full import (
    atomic_copy_verified,
    evaluation_request_id,
    plan_result_shards,
    read_verified_result_shard,
    write_result_shard,
)


def _plans():
    request_ids = [
        evaluation_request_id(suite="medqa-full", example_id=f"q-{index}", option_seed=None)
        for index in range(5)
    ]
    return plan_result_shards(
        suite="medqa-full",
        model="localized-base",
        request_ids=request_ids,
        shard_size=2,
        contract_fingerprint="a" * 64,
    )


def _rows(plan):
    public = []
    private = []
    for index, request_id in enumerate(plan.request_ids):
        raw_output = "A" if index == 0 else r"\boxed{B}"
        import hashlib

        public.append(
            {
                "request_id": request_id,
                "example_id": f"q-{index}",
                "suite": plan.suite,
                "model": plan.model,
                "raw_output_sha256": hashlib.sha256(raw_output.encode()).hexdigest(),
            }
        )
        private.append(
            {
                "request_id": request_id,
                "suite": plan.suite,
                "model": plan.model,
                "raw_output": raw_output,
                "question": "private",
            }
        )
    return public, private


def test_request_ids_and_shards_are_stable_and_bounded() -> None:
    plans = _plans()

    assert [len(plan.request_ids) for plan in plans] == [2, 2, 1]
    assert len({plan.filename for plan in plans}) == 3
    assert all(len(plan.fingerprint) == 64 for plan in plans)
    assert evaluation_request_id(
        suite="medqa-full", example_id="q-0", option_seed=None
    ) == evaluation_request_id(suite="medqa-full", example_id="q-0", option_seed=None)
    assert evaluation_request_id(
        suite="tmmlu-stability", example_id="q-0", option_seed=3407
    ) != evaluation_request_id(suite="tmmlu-stability", example_id="q-0", option_seed=3408)


def test_result_shard_round_trip_and_drive_copy(tmp_path: Path) -> None:
    plan = _plans()[0]
    public, private = _rows(plan)
    shard = tmp_path / plan.filename

    receipt = write_result_shard(
        shard,
        plan=plan,
        public_rows=public,
        private_rows=private,
    )
    loaded_public, loaded_private, manifest = read_verified_result_shard(
        shard, expected_plan=plan
    )
    copied = atomic_copy_verified(shard, tmp_path / "drive" / shard.name)

    assert loaded_public == public
    assert loaded_private == private
    assert manifest["rows"] == 2
    assert copied["sha256"] == receipt["sha256"]
    assert not list((tmp_path / "drive").glob("*.partial"))


def test_result_shard_rejects_public_content_leak(tmp_path: Path) -> None:
    plan = _plans()[0]
    public, private = _rows(plan)
    public[0]["question"] = "must not be public"

    with pytest.raises(ValueError, match="private keys"):
        write_result_shard(
            tmp_path / "bad.zip",
            plan=plan,
            public_rows=public,
            private_rows=private,
        )


def test_result_shard_rejects_tampering_and_wrong_plan(tmp_path: Path) -> None:
    plan = _plans()[0]
    public, private = _rows(plan)
    shard = tmp_path / plan.filename
    write_result_shard(shard, plan=plan, public_rows=public, private_rows=private)

    with zipfile.ZipFile(shard, "a") as archive:
        archive.writestr("unexpected.json", json.dumps({"bad": True}))
    with pytest.raises(ValueError, match="unexpected shard members"):
        read_verified_result_shard(shard, expected_plan=plan)

    other_plan = _plans()[1]
    clean = tmp_path / "clean.zip"
    write_result_shard(clean, plan=plan, public_rows=public, private_rows=private)
    with pytest.raises(ValueError, match="plan mismatch"):
        read_verified_result_shard(clean, expected_plan=other_plan)

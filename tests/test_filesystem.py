from pathlib import Path

import pytest

from tw_med_qlora.filesystem import rename_with_retry


def test_rename_with_retry_recovers_from_transient_permission_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "large.zip.partial"
    destination = tmp_path / "large.zip"
    source.write_bytes(b"verified")
    original_rename = Path.rename
    calls = 0

    def flaky_rename(path: Path, target: Path) -> Path:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise PermissionError("transient Windows lock")
        return original_rename(path, target)

    monkeypatch.setattr(Path, "rename", flaky_rename)

    rename_with_retry(source, destination, attempts=2, delay_seconds=0)

    assert calls == 2
    assert destination.read_bytes() == b"verified"
    assert not source.exists()


def test_rename_with_retry_preserves_source_after_final_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "adapter.partial"
    destination = tmp_path / "adapter"
    source.mkdir()

    def always_locked(path: Path, target: Path) -> Path:
        raise PermissionError("still locked")

    monkeypatch.setattr(Path, "rename", always_locked)

    with pytest.raises(PermissionError, match="still locked"):
        rename_with_retry(source, destination, attempts=2, delay_seconds=0)

    assert source.is_dir()
    assert not destination.exists()


def test_rename_with_retry_refuses_existing_destination(tmp_path: Path) -> None:
    source = tmp_path / "new.zip.partial"
    destination = tmp_path / "new.zip"
    source.write_bytes(b"new")
    destination.write_bytes(b"existing")

    with pytest.raises(FileExistsError, match="destination already exists"):
        rename_with_retry(source, destination, delay_seconds=0)

    assert source.read_bytes() == b"new"
    assert destination.read_bytes() == b"existing"

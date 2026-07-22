"""Small cross-platform filesystem helpers."""

from __future__ import annotations

import time
from pathlib import Path


def rename_with_retry(
    source: Path,
    destination: Path,
    *,
    attempts: int = 6,
    delay_seconds: float = 0.25,
) -> None:
    """Atomically rename a path, retrying transient Windows file locks.

    Antivirus and indexing services can briefly retain a handle after a large
    archive or safetensors file is closed.  Windows reports that race as
    ``PermissionError`` even when both paths are writable.  Other failures are
    surfaced immediately, and the final permission error is never hidden.  The
    destination must be new: using ``Path.rename`` instead of ``Path.replace``
    avoids Windows' replace-existing permission semantics and matches the
    callers' no-overwrite contract.
    """

    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    if delay_seconds < 0:
        raise ValueError("delay_seconds cannot be negative")

    for attempt in range(1, attempts + 1):
        if destination.exists():
            raise FileExistsError(f"destination already exists: {destination}")
        try:
            source.rename(destination)
            return
        except PermissionError:
            if attempt == attempts:
                raise
            time.sleep(delay_seconds)

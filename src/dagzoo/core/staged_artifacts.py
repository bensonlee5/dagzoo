"""Shared staged-output helpers for atomic artifact publication."""

from __future__ import annotations

import shutil
from pathlib import Path


def staged_output_path(*, parent_dir: Path, final_name: str, staging_token: str) -> Path:
    """Return one hidden temp path used before promotion into final visibility."""

    return parent_dir / f".{final_name}.{staging_token}.tmp"


def cleanup_path(path: Path | None) -> None:
    """Best-effort cleanup for one staged or promoted artifact path."""

    if path is None or not path.exists():
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
        return
    path.unlink(missing_ok=True)


def promote_staged_path(*, staged_path: Path, final_path: Path) -> None:
    """Promote one staged file or directory into its final visible location."""

    if final_path.exists():
        raise RuntimeError(
            f"Staged artifact promotion target already exists: {final_path}. "
            "Remove the existing artifact and retry."
        )
    staged_path.replace(final_path)


__all__ = [
    "cleanup_path",
    "promote_staged_path",
    "staged_output_path",
]

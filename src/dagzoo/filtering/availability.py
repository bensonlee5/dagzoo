"""Shared availability guards for temporarily unsupported filtering workflows."""

from __future__ import annotations

from typing import NoReturn

FILTERING_UNSUPPORTED_MESSAGE = (
    "Deferred filtering is temporarily disabled; generated outputs are the only "
    "supported corpus artifact for now."
)


def raise_filtering_unsupported() -> NoReturn:
    """Raise the shared unsupported-feature error for filtering workflows."""

    raise NotImplementedError(FILTERING_UNSUPPORTED_MESSAGE)

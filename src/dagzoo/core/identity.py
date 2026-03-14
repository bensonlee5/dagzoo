"""Stable identity helpers for persisted request, corpus, and dataset artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from dagzoo.math import sanitize_json


def stable_blake2s_hex(payload: Mapping[str, Any], *, digest_size: int = 16) -> str:
    """Return a stable BLAKE2s hex digest for sanitized JSON-compatible payloads."""

    encoded = json.dumps(
        sanitize_json(dict(payload)),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.blake2s(encoded, digest_size=int(digest_size)).hexdigest()


def canonical_request_run_split_group(
    *,
    seed: int,
    run_num_datasets: int,
    layout_signature: str,
    layout_plan_signature: str,
) -> str:
    """Return a stable split-group key for one canonical request run."""

    return stable_blake2s_hex(
        {
            "seed": int(seed),
            "run_num_datasets": int(run_num_datasets),
            "layout_signature": str(layout_signature),
            "layout_plan_signature": str(layout_plan_signature),
        }
    )


def canonical_layout_plan_split_group(
    *,
    layout_signature: str,
    layout_plan_signature: str,
    layout_execution_contract: str,
) -> str:
    """Return a stable split-group key for one fixed-layout execution plan."""

    return stable_blake2s_hex(
        {
            "layout_signature": str(layout_signature),
            "layout_plan_signature": str(layout_plan_signature),
            "layout_execution_contract": str(layout_execution_contract),
        }
    )


def canonical_dataset_id(
    *,
    request_run_split_group: str,
    layout_plan_split_group: str,
    dataset_index: int,
    dataset_seed: int,
) -> str:
    """Return a stable dataset identifier for one canonical dataset bundle."""

    return stable_blake2s_hex(
        {
            "request_run_split_group": str(request_run_split_group),
            "layout_plan_split_group": str(layout_plan_split_group),
            "dataset_index": int(dataset_index),
            "dataset_seed": int(dataset_seed),
        }
    )


__all__ = [
    "canonical_dataset_id",
    "canonical_layout_plan_split_group",
    "canonical_request_run_split_group",
    "stable_blake2s_hex",
]

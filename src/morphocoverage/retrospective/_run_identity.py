"""The identity of one replayed run: its typed group, split, order, stream and reservoir.

A frontier point is valid only when the decisions scored against a run's outcomes came from
that exact run (issue 230). Equal seeds, sizes or settings do not prove it: another table can
hold other fields under the same names. The fingerprint is a SHA-256 over the typed identity
(``_identity``) of the run's labels, its stream's fields in acquisition order and its
reservoir's fields as a set, so row order of the reservoir and ``1`` against ``"1"`` are told
apart exactly as everywhere else.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from .._identity import identity_key, row_keys

RUN_FINGERPRINT = "run_fingerprint"


def _field_keys(fovs: pd.DataFrame, fov_key: Sequence[str]) -> list[list[list[str]]]:
    """The typed key of each field over the FOV key columns, as JSON-ready lists."""
    return [[list(part) for part in key] for key in row_keys(fovs, list(fov_key))]


def run_fingerprint(
    where: Mapping[str, Any],
    stream: pd.DataFrame,
    reservoir: pd.DataFrame,
    fov_key: Sequence[str],
) -> str:
    """The fingerprint of one run: ``where`` (its labels), the ordered stream, the reservoir."""
    labels = [[name, list(identity_key(value))] for name, value in where.items()]
    held = sorted(_field_keys(reservoir, fov_key))
    payload = {"where": labels, "stream": _field_keys(stream, fov_key), "reservoir": held}
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

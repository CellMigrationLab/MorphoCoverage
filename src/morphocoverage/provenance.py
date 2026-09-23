"""Stable configuration identifiers for estimators, policies and strategies."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


def class_identity(cls: type) -> str:
    """Module-qualified class name; moving or renaming a class changes its ids."""
    return f"{cls.__module__}.{cls.__qualname__}"


def config_hash(identity: str, parameters: Mapping[str, Any]) -> str:
    """First 16 hex digits of the SHA-256 of ``identity`` plus canonical-JSON parameters.

    Two configurations get the same id exactly when they have the same identity
    and the same parameters, independent of key order. Parameters must be
    JSON-serializable, without NaN or infinity, so that the id is reproducible.
    """
    try:
        payload = json.dumps(
            {"identity": identity, "parameters": parameters},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"parameters of {identity} must be JSON-serializable for provenance: {error}"
        ) from None
    return hashlib.sha256(payload.encode()).hexdigest()[:16]

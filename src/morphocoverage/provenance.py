"""Stable configuration identifiers for estimators, policies and strategies."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

# Keys that tag what JSON cannot tell apart (see ``_canonical``); a parameter
# mapping may not use them itself, so a tagged value never equals a real one.
_TUPLE, _MAPPING = "__tuple__", "__mapping__"


def class_identity(cls: type) -> str:
    """Module-qualified class name; moving or renaming a class changes its ids."""
    return f"{cls.__module__}.{cls.__qualname__}"


def _key_text(value: Any) -> str:
    """Canonical JSON of an encoded key, to order the entries of a mapping."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _canonical(value: Any) -> Any:
    """``value`` with what JSON would merge made explicit.

    JSON writes a tuple as a list and every mapping key as text, so ``(1, 2)``
    and ``[1, 2]``, or ``{1: x}`` and ``{"1": x}``, would get one id. A tuple
    becomes ``{"__tuple__": [...]}`` and a mapping with a key that is not text
    becomes ``{"__mapping__": [[key, value], ...]}`` (entries in a fixed
    order). Everything JSON already keeps apart is left as it is, so the id of
    such parameters is the one earlier versions gave.
    """
    if isinstance(value, tuple):
        return {_TUPLE: [_canonical(item) for item in value]}
    if isinstance(value, list):
        return [_canonical(item) for item in value]
    if isinstance(value, Mapping):
        reserved = [key for key in value if key in (_TUPLE, _MAPPING)]
        if reserved:
            raise TypeError(f"the key {reserved[0]!r} is reserved")
        if all(isinstance(key, str) for key in value):
            return {key: _canonical(item) for key, item in value.items()}
        entries = [[_canonical(key), _canonical(item)] for key, item in value.items()]
        return {_MAPPING: sorted(entries, key=lambda entry: _key_text(entry[0]))}
    return value


def config_hash(identity: str, parameters: Mapping[str, Any]) -> str:
    """First 16 hex digits of the SHA-256 of ``identity`` plus canonical-JSON parameters.

    Two configurations get the same id exactly when they have the same identity
    and the same parameters, independent of key order; a tuple differs from a
    list and a key ``1`` from a key ``"1"`` (``_canonical``). Parameters must be
    JSON-serializable, without NaN or infinity, so that the id is reproducible.
    """
    try:
        payload = json.dumps(
            {"identity": identity, "parameters": _canonical(parameters)},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"parameters of {identity} must be JSON-serializable for provenance: {error}"
        ) from None
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def describe(component: Any) -> dict[str, Any]:
    """Provenance entry for an estimator, policy or transform: its name, type, config_id
    and parameters."""
    return {
        "name": component.name,
        "type": class_identity(type(component)),
        "config_id": component.config_id,
        "parameters": dict(component.parameters),
    }


__all__ = ["class_identity", "config_hash", "describe"]

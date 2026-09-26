"""Stable per-group seeding (private): a group's random draws never depend on other groups.

Every analysis that draws at random seeds each group from the caller's
``random_state`` and the group's own identity, so adding, removing or
re-sorting other groups never changes a group's result.
"""

from __future__ import annotations

import json
import zlib
from collections.abc import Mapping
from typing import Any

import numpy as np


def group_identity(group: Mapping[str, Any]) -> int:
    """A stable integer for a group, so its random draws never depend on other groups."""
    return zlib.crc32(json.dumps(sorted(group.items()), default=str).encode())


def group_seed(random_state: int, group: Mapping[str, Any], *more: int) -> list[int]:
    """The seed sequence ``[random_state, group_identity(group), *more]``."""
    return [random_state, group_identity(group), *more]


def group_rng(random_state: int, group: Mapping[str, Any], *more: int) -> np.random.Generator:
    """A generator seeded from ``group_seed``."""
    return np.random.default_rng(group_seed(random_state, group, *more))


__all__ = ["group_identity", "group_rng", "group_seed"]

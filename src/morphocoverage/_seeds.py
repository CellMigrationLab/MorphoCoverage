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


def _plain(value: Any) -> Any:
    """A numpy scalar as the Python value it holds (``np.int64(1)`` is ``1``)."""
    return value.item() if isinstance(value, np.generic) else value


def group_key(group: Mapping[str, Any]) -> str:
    """A group's identity as text, for looking groups up.

    Unlike ``group_identity`` (a 32-bit hash, fine for seeding), two different
    groups never share a key: ``{'repeat': 1}`` and ``{'repeat': '1'}`` differ,
    and a numpy value keys like the Python value it holds.
    """
    items = sorted(((str(k), _plain(v)) for k, v in group.items()), key=lambda kv: kv[0])
    return json.dumps(items, default=str)


def group_identity(group: Mapping[str, Any]) -> int:
    """A stable integer for a group, so its random draws never depend on other groups.

    A 32-bit hash of ``group_key``: two groups can collide, so it seeds draws
    but never identifies a group (look groups up with ``group_key``).
    """
    return zlib.crc32(group_key(group).encode())


def group_seed(random_state: int, group: Mapping[str, Any], *more: int) -> list[int]:
    """The seed sequence ``[random_state, group_identity(group), *more]``."""
    return [random_state, group_identity(group), *more]


def split_seed(random_state: int, split: int) -> int:
    """The seed of one split's reservoir, from ``random_state`` and the split number.

    Seeding split ``s`` with ``random_state + s`` made ``random_state=1`` reuse
    the reservoirs of ``random_state=0`` shifted by one split; this mixes both
    through ``numpy.random.SeedSequence``, so neighbouring seeds share nothing.
    """
    return int(np.random.SeedSequence([int(random_state), int(split)]).generate_state(1)[0])


def group_rng(random_state: int, group: Mapping[str, Any], *more: int) -> np.random.Generator:
    """A generator seeded from ``group_seed``."""
    return np.random.default_rng(group_seed(random_state, group, *more))


__all__ = ["group_identity", "group_key", "group_rng", "group_seed", "split_seed"]

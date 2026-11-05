"""Stable per-group seeding (private): a group's random draws never depend on other groups.

Every analysis that draws at random seeds each group from the caller's
``random_state`` (and the analysis's own extra numbers, such as the split)
and from the group's labels written canonically (``seed_labels``), and draws
over the group's fields in a canonical order (the FOV key). So adding,
removing or re-sorting other groups never changes a group's result, and
retyping a label (repeat ``1`` read as ``"1"`` or ``1.0``) is the same group,
with the same draws. Renaming a group is a new Monte Carlo draw of its
answers; the summaries state their Monte Carlo errors (audit 4, P9-2). The
biological repeat seeds under its former column name, ``repeat``, so the
renaming of that column to ``biological_repeat`` left every draw as it was.

``group_key`` tells groups apart for lookups (``1`` and ``"1"`` differ there);
``group_identity`` is a 32-bit hash for seeding only.
"""

from __future__ import annotations

import json
import zlib
from collections.abc import Mapping
from typing import Any

import numpy as np

from ._checks import is_integer, is_number
from ._former_repeat import FORMER_REPEAT_COLUMN
from .metadata import REPEAT_COLUMN

# Column names as seeding writes them: the biological repeat under its former name.
_SEED_NAMES = {REPEAT_COLUMN: FORMER_REPEAT_COLUMN}


def plain(value: Any) -> Any:
    """A numpy scalar as the Python value it holds (``np.int64(1)`` is ``1``)."""
    return value.item() if isinstance(value, np.generic) else value


def _typed(value: Any) -> dict[str, str]:
    """A label JSON cannot hold (a date, a timestamp, bytes) as its type and its text, so
    it never reads as the text it prints as (#164); ``json.dumps`` calls this for those only."""
    kind = type(value)
    return {"type": f"{kind.__module__}.{kind.__qualname__}", "text": str(value)}


def group_key(group: Mapping[str, Any]) -> str:
    """A group's identity as text, for looking groups up.

    Unlike ``group_identity`` (a 32-bit hash, fine for seeding), two different
    groups never share a key: ``{'plate': 1}`` and ``{'plate': '1'}`` differ,
    a date and the text it prints as differ (#164), and a numpy value keys like
    the Python value it holds.
    """
    items = sorted(((str(k), plain(v)) for k, v in group.items()), key=lambda kv: kv[0])
    return json.dumps(items, default=_typed)


def _seed_label(value: Any) -> str:
    """A label as seeding reads it: a number, or text that reads as one, in one spelling.

    ``1``, ``np.int64(1)``, ``1.0`` and ``"1"`` are the same label; so are
    ``0.5`` and ``"0.5"``. Other text is kept as it is.
    """
    value = plain(value)
    if isinstance(value, str):
        try:
            number = float(value)
        except ValueError:  # fallback: text that is not a number stays text
            return value
        value = number if np.isfinite(number) else value
    if is_integer(value):
        return str(int(value))
    if not is_number(value):
        return str(value)
    value = float(value)
    return str(int(value)) if value.is_integer() else repr(value)


def seed_labels(group: Mapping[str, Any]) -> str:
    """A group's labels written canonically, for seeding (``_seed_label``)."""
    items = sorted((_SEED_NAMES.get(str(k), str(k)), _seed_label(v)) for k, v in group.items())
    return json.dumps(items)


def group_identity(group: Mapping[str, Any]) -> int:
    """A stable integer for a group, so its random draws never depend on other groups.

    A 32-bit hash of the group's canonical labels (``seed_labels``): two
    groups can collide, so it seeds draws but never identifies a group (look
    groups up with ``group_key``).
    """
    return zlib.crc32(seed_labels(group).encode())


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


__all__ = ["group_identity", "group_key", "group_rng", "group_seed", "plain", "split_seed"]

"""The observation unit: what one row of the observation table is (private; schema re-exports).

Rows are generic phenotype observations (cells, nuclei, patches, slices, embeddings, FOV
summaries, ...). The unit is recorded, normalised to one lowercase singular label, and says how a
report words its observations; an undeclared unit is assumed to be objects and called cells.
"""

from __future__ import annotations

import re
from typing import Any

from ._wording import plural

# Units whose rows are individual detected objects, as the cell-number
# baselines (JCB, Marcotti) assume.
OBJECT_LIKE_UNITS = frozenset({"object", "cell", "nucleus"})

# Plural and spelling variants mapped to one canonical label, so that "cells"
# and "Cell" do not silently count as a different unit from "cell".
_UNIT_ALIASES = {
    "objects": "object",
    "cells": "cell",
    "nuclei": "nucleus",
    "nucleuses": "nucleus",
    "patches": "patch",
    "slices": "slice",
    "planes": "plane",
    "embeddings": "embedding",
    "fov_summaries": "fov_summary",
}


# Assumed when the caller does not say what a row is. Results record that the
# unit was assumed (``observation_unit_declared=False``), never silently.
DEFAULT_OBSERVATION_UNIT = "object"


# The words for observations whose unit was not declared (owner, 2026-10-02: "cells could
# be the default"): the report calls them cells, while the analysis still treats them as
# generic objects (``DEFAULT_OBSERVATION_UNIT``, ``observation_unit_declared=False``).
DEFAULT_UNIT_WORD = "cell"


def unit_words(observation_unit: str | None, declared: bool = True) -> tuple[str, str]:
    """The observations in words, singular and plural (``nucleus``, ``nuclei``): the
    declared unit's, or ``cell`` and ``cells`` when none was declared (``None``, or
    ``declared=False`` for a validated input's assumed unit)."""
    if observation_unit is None or not declared:
        return DEFAULT_UNIT_WORD, unit_plural(DEFAULT_UNIT_WORD)
    unit = normalize_observation_unit(observation_unit)
    return unit, unit_plural(unit)


def observation_words(settings: Any) -> tuple[str, str]:
    """``unit_words`` of a result's ``settings``, or of an object with those attributes."""
    if not isinstance(settings, dict):
        settings = {
            "observation_unit": getattr(settings, "observation_unit", None),
            "observation_unit_declared": getattr(settings, "observation_unit_declared", True),
        }
    return unit_words(
        settings.get("observation_unit"), bool(settings.get("observation_unit_declared", True))
    )


def unit_plural(observation_unit: str) -> str:
    """Plural of an observation unit for sentences, e.g. ``nuclei`` (``_wording.plural``)."""
    return plural(observation_unit)


def resolve_observation_unit(unit: str | None) -> tuple[str, bool]:
    """The canonical unit and whether the caller declared it (``None`` means assumed)."""
    if unit is None:
        return DEFAULT_OBSERVATION_UNIT, False
    return normalize_observation_unit(unit), True


def normalize_observation_unit(unit: object) -> str:
    """Canonical observation-unit label: lowercase, ``_`` for spaces/hyphens, singular.

    Any non-empty label is accepted, since rows may be of a kind this package
    does not know; known plurals are mapped to their singular.
    """
    if not isinstance(unit, str) or not unit.strip():
        raise ValueError(
            "observation_unit must be a non-empty description such as 'cell' or 'slice'"
        )
    label = re.sub(r"[\s-]+", "_", unit.strip().lower())
    return _UNIT_ALIASES.get(label, label)

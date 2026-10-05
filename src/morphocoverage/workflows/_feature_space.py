"""The other feature tables of a ``Setup`` (private; ``setup``, the workflows and the page).

``Setup.feature_tables`` names each other feature table and its columns (owner, 2026-10-05;
``morphocoverage.feature_tables``). The workflows take the tables themselves with their
``feature_tables=`` keyword and attach them at the door (``attached``), which records their
columns in the ``Setup`` they return. Here:

- ``check_feature_tables``: the declaration, refused when a name is empty, reserved or
  repeated, or a column is declared twice or is the readout, a feature or the rare label;
- ``space``: the weighted combined space of the distance-based analyses (the maps, the
  populations, unusual observations), the explainable columns at weight 1 and every other
  table at ``table_weights``'s; without other tables, the explainable columns alone, not
  weighted, as before;
- ``novelty_space``: the definition of new on that space: its transform followed by the
  weights (``ColumnWeights``), its radius times ``sqrt`` of the number of tables, as every
  table adds the explainable table's total variance (the default radius ``sqrt(d)`` of
  ``d`` explainable features becomes the square root of the space's total variance);
- ``attached``: the observation table with the tables' columns and the ``Setup`` naming them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .._wording import count
from ..feature_tables import (
    COMBINED,
    EXPLAINABLE,
    RESERVED_NAMES,
    FeatureTables,
    attach_feature_tables,
    table_weights,
)
from ..metadata import CONDITION_COLUMN
from ..separation import Separation, separation
from ..strategies import SupportNoveltyEstimator, SupportNoveltyStrategy
from ..transforms import Chain, ColumnWeights

if TYPE_CHECKING:
    from .setup import Setup


def check_feature_tables(setup: Setup) -> None:
    """``feature_tables`` as a dict of names and column tuples, checked (see the module)."""
    declared = setup.feature_tables
    if declared is None:
        return
    if not isinstance(declared, Mapping):
        raise TypeError("feature_tables must map each other table's name to its columns")
    tables: dict[str, tuple[str, ...]] = {}
    seen: dict[str, str] = {}
    explainable = {*setup.columns, *([setup.rare_label] if setup.rare_label else [])}
    for name, columns in declared.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"a feature table's name must be a non-empty text, not {name!r}")
        if name.strip().casefold() in RESERVED_NAMES or name.strip().casefold() in tables:
            raise ValueError(f"{name!r} is reserved or names two tables: give each its own name")
        if isinstance(columns, str) or not columns:
            raise TypeError(f"the feature table {name!r} must list its columns")
        columns = tuple(columns)
        for column in columns:
            if column in explainable:
                raise ValueError(f"{column!r} of {name!r} is also a declared feature or label")
            if column in seen:
                raise ValueError(f"{column!r} is a column of both {seen[column]!r} and {name!r}")
            seen[column] = name
        tables[name] = columns
    if tables and any(isinstance(rule, SupportNoveltyStrategy) for rule in setup.rules or ()):
        # The rules read every column the report gives them, here the tables' unweighted.
        raise ValueError(
            "a novelty rule among rules would read the other feature tables unweighted: leave "
            "it out of rules, and the report judges the novelty bound on the combined space"
        )
    object.__setattr__(setup, "feature_tables", tables or None)


def table_columns(setup: Setup) -> list[str]:
    """Every other table's columns, in the order declared."""
    return [c for columns in (setup.feature_tables or {}).values() for c in columns]


def space(
    setup: Setup, explainable: Sequence[str] | None = None
) -> tuple[list[str], dict[str, float] | None]:
    """The columns of the weighted combined space and each column's weight (``None``, not
    weighted, without other tables). ``explainable`` defaults to ``Setup.columns``."""
    columns = list(setup.columns if explainable is None else explainable)
    if not setup.feature_tables:
        return columns, None
    weights = {c: 1.0 for c in columns}
    weights.update(table_weights(setup.feature_tables, len(columns)))
    return list(weights), weights


def novelty_space(setup: Setup) -> tuple[SupportNoveltyEstimator | None, list[str]]:
    """The definition of new and the columns it reads: ``Setup.novelty`` on the declared
    features, or with other tables on the weighted combined space (see the module)."""
    novelty = setup.novelty
    if novelty is None:  # Setup declares features with novelty: none needed without it
        return None, list(setup.features)
    columns, weights = space(setup, setup.features)
    if weights is None:
        return novelty, columns
    weighting = ColumnWeights(tuple(weights[c] for c in columns))
    transform = weighting if novelty.transform is None else Chain([novelty.transform, weighting])
    radius = novelty.epsilon * float(np.sqrt(1 + len(setup.feature_tables or {})))
    return replace(novelty, epsilon=radius, transform=transform), columns


def attached(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    setup: Setup,
    feature_tables: Mapping[str, pd.DataFrame] | None,
) -> tuple[pd.DataFrame, Setup, FeatureTables | None]:
    """``objects`` with the other tables' columns, the ``Setup`` naming them, and what
    ``attach_feature_tables`` checked (``None`` without tables given). Without tables, the
    columns a ``Setup`` names must already be in ``objects``."""
    if feature_tables is None:
        absent = [c for c in table_columns(setup) if c not in objects.columns]
        if absent:
            raise ValueError(
                f"Setup.feature_tables names columns that are not in objects ({absent[:3]}): "
                "pass the tables themselves, feature_tables={name: table}, to report, "
                "html_report, scan_design or guide, or attach them first: objects = "
                "attach_feature_tables(objects, fovs, tables).objects"
            )
        return objects, setup, None
    tables = attach_feature_tables(
        objects, fovs, feature_tables, fov_col=setup.fov_col, group_cols=setup.group_cols
    )
    declared = setup.feature_tables
    if declared is not None and declared != tables.columns:
        raise ValueError(
            "the Setup names other feature tables or columns than those given: "
            f"{sorted(declared)} against {sorted(tables.columns)}"
        )
    return tables.objects, replace(setup, feature_tables=tables.columns), tables


def table_lines(setup: Setup, tables: FeatureTables | None = None) -> list[str]:
    """The overview's sentences on the other tables (what ``tables`` checked, when given),
    none without them."""
    if not setup.feature_tables:
        return []
    declared = setup.feature_tables
    weights = table_weights(declared, len(setup.columns))
    lines = (
        tables.lines()
        if tables is not None
        else [f"{name}: {count(len(c), 'feature column')}." for name, c in declared.items()]
    )
    shares = ", ".join(f"{name} {weights[c[0]]:.2f}" for name, c in declared.items())
    lines.append(
        "Other feature tables are not tested feature by feature; with the explainable features "
        "they make the maps, the populations, unusual observations, novelty, the copy checks "
        "and a fingerprint declared on them, in a combined space where every table weighs as "
        f"much as the explainable one (weight per column: {shares})."
    )
    return lines


def space_names(setup: Setup) -> dict[str, tuple[list[str], dict[str, float] | None]]:
    """Each feature space of the maps and the separation score, by the name the page shows:
    with other tables the combined space first, then the explainable features and each
    table; without them, the declared features alone."""
    if not setup.feature_tables:
        return {"Features": (list(setup.columns), None)}
    columns, weights = space(setup)
    spaces = {COMBINED.capitalize(): (columns, weights)}
    spaces[EXPLAINABLE.capitalize()] = (list(setup.columns), None)
    spaces.update({name: (list(c), None) for name, c in setup.feature_tables.items()})
    return spaces


def separation_in_report(
    objects: pd.DataFrame, fovs: pd.DataFrame, setup: Setup
) -> Separation | None:
    """The separation of the conditions in each space (``space_names``), on the values the
    figures read; ``None`` with one condition or no condition column."""
    if CONDITION_COLUMN not in fovs or fovs[CONDITION_COLUMN].nunique() < 2:
        return None
    spaces = space_names(setup)
    return separation(
        objects,
        fovs,
        spaces={name: columns for name, (columns, _) in spaces.items()},
        weights={name: weights for name, (_, weights) in spaces.items()},
        random_state=setup.random_state,
        **setup.common,
    )


__all__ = [
    "attached",
    "check_feature_tables",
    "novelty_space",
    "separation_in_report",
    "space",
    "space_names",
    "table_columns",
    "table_lines",
]

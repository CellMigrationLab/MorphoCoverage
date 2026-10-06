"""The other feature tables of a ``Setup`` (private; ``setup``, the workflows and the page).

``Setup.feature_tables`` names each other feature table and its columns (owner, 2026-10-05;
``morphocoverage.feature_tables``). The workflows take the tables themselves with their
``feature_tables=`` keyword and attach them at the door (``attached``), which records their
columns in the ``Setup`` they return. The tables join the maps, the populations, the copy
checks and their own fingerprints, never novelty or unusual observations, which read the
explainable features only (owner, 2026-10-06). Here:

- ``check_feature_tables``: the declaration, refused when a name is empty, reserved or
  repeated, or a column is declared twice or is the readout, a feature or the rare label;
- ``space``: the weighted combined space of the maps and the populations, its columns and
  its tables (``blocks``: the explainable features first, each table then divided by its
  first singular value, ``feature_tables.mfa_weights``); without other tables, the
  explainable columns alone, not weighted, as before;
- ``novelty_space``: the definition of new and the columns it reads, the declared features
  alone whatever the tables (owner, 2026-10-06);
- ``attached``: the observation table with the tables' columns and the ``Setup`` naming them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .._wording import count, listing
from ..embedding import robust_scale
from ..feature_tables import (
    COMBINED,
    EXPLAINABLE,
    RESERVED_NAMES,
    FeatureTables,
    attach_feature_tables,
    mfa_weights,
)
from ..metadata import CONDITION_COLUMN, REPEAT_COLUMN
from ..schema import validate_input_tables
from ..separation import Separation, separation
from ..strategies import SupportNoveltyEstimator

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
    object.__setattr__(setup, "feature_tables", tables or None)


def table_columns(setup: Setup) -> list[str]:
    """Every other table's columns, in the order declared."""
    return [c for columns in (setup.feature_tables or {}).values() for c in columns]


# The tables of a space, by name: the explainable features first, then each other table.
Blocks = dict[str, tuple[str, ...]]


def space(
    setup: Setup, explainable: Sequence[str] | None = None
) -> tuple[list[str], Blocks | None]:
    """The columns of the weighted combined space and its tables (``None``, not weighted,
    without other tables). ``explainable`` defaults to ``Setup.columns``."""
    columns = list(setup.columns if explainable is None else explainable)
    if not setup.feature_tables:
        return columns, None
    blocks = {EXPLAINABLE: tuple(columns), **dict(setup.feature_tables)}
    return [c for block in blocks.values() for c in block], blocks


def novelty_space(setup: Setup) -> tuple[SupportNoveltyEstimator | None, list[str]]:
    """The definition of new and the columns it reads: ``Setup.novelty`` on the declared
    features, whatever the other tables (owner, 2026-10-06: they never enter novelty)."""
    return setup.novelty, list(setup.features)


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


def table_lines(setup: Setup, tables: FeatureTables | None = None, centring: str = "") -> list[str]:
    """The overview's sentences on the other tables (what ``tables`` checked, when given),
    none without them."""
    if not setup.feature_tables:
        return []
    declared = setup.feature_tables
    lines = (
        tables.lines()
        if tables is not None
        else [f"{name}: {count(len(c), 'feature column')}." for name, c in declared.items()]
    )
    if not setup.columns:  # an embedding-only report (owner, 2026-10-06)
        lines.append(
            f"{features_words(setup)}. With no explainable features, each table is tested "
            "whole (the whole-table test of question 9) and read by the maps, the populations, "
            "the copy checks, unusual observations and its own fingerprint; never feature by "
            "feature."
        )
        return [*lines, *([centring] if centring else [])]
    lines.append(
        f"{features_words(setup)}. The other tables are never tested feature by feature: they "
        "are tested whole, beside the features, and join the maps, the populations, the copy "
        "checks and their own fingerprints, not novelty or unusual observations, which read "
        "the explainable features only; in the combined space each table counts as much as "
        "the explainable features, its first axis weighing as much as theirs."
    )
    return [*lines, *([centring] if centring else [])]


def features_words(setup: Setup) -> str:
    """``17 explainable features and 768 columns of 2 other tables`` (one table: ``of the
    other table DINOv2``); the explainable features alone without tables."""
    explainable = count(len(setup.columns), "explainable feature")
    declared = setup.feature_tables or {}
    if not declared:
        return explainable
    columns = count(sum(len(c) for c in declared.values()), "column")
    if not setup.columns:  # an embedding-only report
        names = next(iter(declared)) if len(declared) == 1 else f"{len(declared)} tables"
        return f"{columns} of {names}"
    if len(declared) == 1:
        return f"{explainable} and {columns} of the other table {next(iter(declared))}"
    return f"{explainable} and {columns} of {len(declared)} other tables"


def space_names(setup: Setup) -> dict[str, tuple[list[str], Blocks | None]]:
    """Each feature space of the maps and the separation score, by the name the page shows:
    with other tables the combined space first, then the explainable features and each
    table; without them, the declared features alone."""
    if not setup.feature_tables:
        return {"Features": (list(setup.columns), None)}
    columns, blocks = space(setup)
    spaces = {}
    # Without explainable features (owner, 2026-10-06), one table is its own combined space.
    if setup.columns or len(setup.feature_tables) > 1:
        spaces[COMBINED.capitalize()] = (columns, blocks)
    if setup.columns:
        spaces[EXPLAINABLE.capitalize()] = (list(setup.columns), None)
    spaces.update({name: (list(c), None) for name, c in setup.feature_tables.items()})
    return spaces


def unusual_space(
    objects: pd.DataFrame, fovs: pd.DataFrame, setup: Setup
) -> tuple[list[str], dict[str, float] | None, str]:
    """The columns unusual observations read, their weights and the space in words: the
    explainable features when there are any; in an embedding-only report, the MFA-weighted
    space of the tables, each table divided by its first singular value (owner, 2026-10-06)."""
    if setup.features or not setup.feature_tables:
        words = f"the {count(len(setup.features), 'explainable feature')}"
        return list(setup.features), None, words
    columns, blocks = space(setup)
    data = validate_input_tables(objects, fovs=fovs, feature_cols=columns, **setup.common)
    matrix = data.feature_matrix()
    complete = matrix[~np.isnan(matrix).any(axis=1)]
    centre, scale = robust_scale(complete)
    z = (complete - centre) / np.where(scale > 0, scale, 1.0)
    weights = mfa_weights(z, columns, blocks or {})
    names = listing(list(setup.feature_tables), 4)
    words = (
        f"the weighted combined space of {names} (each table divided by its first singular "
        "value; there are no explainable features)"
        if len(setup.feature_tables) > 1
        else f"the {count(len(columns), 'column')} of {names} (there are no explainable features)"
    )
    return columns, dict(zip(columns, map(float, weights), strict=True)), words


def separation_in_report(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    setup: Setup,
    names: Sequence[str] | None = None,
) -> Separation | None:
    """The separation of the conditions and of the repeats in each space (``space_names``,
    or those of ``names``), on the values the figures read; ``None`` with one condition or
    no condition column."""
    if CONDITION_COLUMN not in fovs or fovs[CONDITION_COLUMN].nunique() < 2:
        return None
    spaces = space_names(setup)
    if names is not None:
        spaces = {name: spaces[name] for name in names if name in spaces}
    return separation(
        objects,
        fovs,
        spaces={name: columns for name, (columns, _) in spaces.items()},
        blocks={name: blocks for name, (_, blocks) in spaces.items()},
        random_state=setup.random_state,
        **setup.common,
    )


# A repeat is centred on its control only when the control holds at least this many
# observations there: the control's median is subtracted from every observation of the
# repeat, so its noise is shared by all of them (owner, 2026-10-06).
CENTRED_MIN_CONTROL = 5
# How a repeat is centred (owner, 2026-10-06): on all its observations in a balanced design,
# else on its control.
ON_ALL, ON_CONTROL = "all", "control"


def balanced(fovs: pd.DataFrame) -> bool:
    """Whether every biological repeat holds the same set of conditions (fields of the FOV
    table); true with one repeat or no condition column."""
    if CONDITION_COLUMN not in fovs or REPEAT_COLUMN not in fovs:
        return True
    held = fovs.groupby(REPEAT_COLUMN)[CONDITION_COLUMN].agg(frozenset)
    return held.nunique() <= 1


def _centring(
    objects: pd.DataFrame, fovs: pd.DataFrame, setup: Setup
) -> tuple[pd.DataFrame, list[tuple[Any, Any, int]], str] | None:
    """The centred columns (see ``centred_tables``), the repeats left out, each as
    ``(control, repeat, observations)``, and the rule (``ON_ALL``, ``ON_CONTROL``); ``None``
    without other tables, or in an unbalanced design without a control."""
    control = (setup.single_control or {}).get(CONDITION_COLUMN)
    own = setup.own_references
    columns = table_columns(setup)
    on_all = balanced(fovs)
    if not columns or (not on_all and control is None and own is None):
        return None
    data = validate_input_tables(objects, fovs=fovs, feature_cols=columns, **setup.common)
    frame = data.objects
    if CONDITION_COLUMN not in frame and not on_all:
        return None
    repeat = frame[REPEAT_COLUMN] if REPEAT_COLUMN in frame else pd.Series("all", frame.index)
    if on_all:  # every repeat holds the same conditions: its median over all of them
        shift = frame[columns].groupby(repeat).transform("median").to_numpy(dtype=float)
        centred = frame[columns].to_numpy(dtype=float) - shift
        return pd.DataFrame(centred, index=frame.index, columns=columns), [], ON_ALL
    condition = frame[CONDITION_COLUMN]
    grouped = frame[columns].groupby([condition, repeat])
    medians, sizes = grouped.median(), grouped.size()
    reference = condition.map(own) if own is not None else pd.Series(control, frame.index)
    pairs = pd.MultiIndex.from_arrays([reference, repeat])
    held = sizes.reindex(pairs).fillna(0).to_numpy()
    shift = medians.reindex(pairs).to_numpy(dtype=float, copy=True)
    # A control of few observations in a repeat: that repeat is left out of the view.
    shift[held < CENTRED_MIN_CONTROL] = np.nan
    centred = frame[columns].to_numpy(dtype=float) - shift
    few = {(c, r, int(n)) for (c, r), n in zip(pairs, held, strict=True) if 0 < n}
    small = sorted(
        (item for item in few if item[2] < CENTRED_MIN_CONTROL),
        key=lambda item: (str(item[1]), str(item[0])),
    )
    return pd.DataFrame(centred, index=frame.index, columns=columns), small, ON_CONTROL


def centred_tables(objects: pd.DataFrame, fovs: pd.DataFrame, setup: Setup) -> pd.DataFrame | None:
    """``objects`` with every other table's column centred per biological repeat (owner,
    2026-10-06): when every repeat holds the same conditions, minus the median of all the
    repeat's observations; otherwise minus the median of the control's observations in the
    repeat, each condition's own control with several (``Setup.own_references``), a repeat
    whose control holds fewer than ``CENTRED_MIN_CONTROL`` observations there left out
    (missing), as is a condition without a control in its repeat (``centred_left_out``). The
    centred map, and with ``normalize`` the populations, unusual observations and separation
    of the tables (``analysed_tables``); never a test. ``None`` without other tables, or in
    an unbalanced design without a control."""
    found = _centring(objects, fovs, setup)
    if found is None:
        return None
    shifted = found[0]
    return objects.assign(**{c: shifted[c] for c in shifted.columns})


def analysed_tables(values: pd.DataFrame, fovs: pd.DataFrame, setup: Setup) -> pd.DataFrame:
    """The values the tables' populations, unusual observations and separation read: with
    ``normalize`` declared, the tables centred per repeat (``centred_tables``), as the
    explainable features are divided by their control; ``values`` otherwise. Never a test:
    the whole-table test and the fingerprint read the raw values."""
    if setup.normalize is None or not setup.feature_tables:
        return values
    centred = centred_tables(values, fovs, setup)
    return values if centred is None else centred


def centred_left_out(objects: pd.DataFrame, fovs: pd.DataFrame, setup: Setup) -> str:
    """The repeats the centred view leaves out, their control holding fewer than
    ``CENTRED_MIN_CONTROL`` observations there, in one sentence; empty when none is."""
    found = _centring(objects, fovs, setup)
    if found is None or not found[1]:
        return ""
    units = setup.unit_words
    named = "; ".join(f"{r} of {c} ({count(n, setup.unit_word, units)})" for c, r, n in found[1])
    return (
        f"Left out of the centred view, their control holding fewer than "
        f"{CENTRED_MIN_CONTROL} {units} in the repeat: {named}."
    )


def centring_line(objects: pd.DataFrame, fovs: pd.DataFrame, setup: Setup) -> str:
    """Which centring the tables had, in one line (owner, 2026-10-06), and what reads it;
    empty without one."""
    found = _centring(objects, fovs, setup)
    if found is None:
        return ""
    units = setup.unit_words
    if found[2] == ON_ALL:
        how = (
            f"on the median of all the {units} of their biological repeat, as every repeat "
            "holds the same conditions"
        )
    else:
        own = " (each condition's own)" if setup.own_references is not None else ""
        how = (
            f"on the median of the control{own} in their biological repeat, as the repeats do "
            f"not all hold the same conditions, a control of fewer than {CENTRED_MIN_CONTROL} "
            f"{units} leaving its repeat out"
        )
    reads = (
        "the centred map, the populations, unusual observations and the separation read them"
        if setup.normalize is not None
        else "the centred map reads them"
    )
    return f"Batch: the other tables centred {how}; {reads}, never a test."


__all__ = [
    "CENTRED_MIN_CONTROL",
    "ON_ALL",
    "ON_CONTROL",
    "analysed_tables",
    "attached",
    "balanced",
    "centred_left_out",
    "centred_tables",
    "centring_line",
    "check_feature_tables",
    "features_words",
    "novelty_space",
    "separation_in_report",
    "space",
    "space_names",
    "table_columns",
    "table_lines",
    "unusual_space",
]

"""The feature tables of a ``Setup`` (private; ``setup``, the workflows and the page).

``Setup.feature_tables`` names each non-explainable feature table and its columns (owner,
2026-10-05), ``Setup.explainable_tables`` each explainable one (owner, 2026-10-06;
``morphocoverage.feature_tables``). The workflows take the tables themselves with their
``feature_tables=`` keyword and attach them at the door (``attached``), which records their
columns in the ``Setup`` they return. The explainable tables' columns are explainable
features, declared in ``Setup.features`` and tested one by one; the other tables join the
maps, the populations, the copy checks and their own fingerprints, never novelty or unusual
observations, which read the explainable features only (owner, 2026-10-06). Here:

- ``check_feature_tables``: the declaration, refused when a name is empty, reserved or
  repeated, or a column is declared twice or, for a non-explainable table, is the readout, a
  feature or the rare label; with explainable tables and no features, every column of them
  is a feature, and a Setup without a readout is refused;
- ``explainable_blocks``: the declared explainable features by table, the observation
  table's own first; with two or more, novelty, unusual observations and the populations
  weigh each by MFA (``explainable_space``, ``novelty_space``, ``unusual_space``), the maps
  and the separation score read each apart (``space_names``), and the result tables name
  each feature's table (``with_table``);
- ``space``: the weighted combined space of the maps and the populations, its columns and
  its tables (``blocks``: the explainable features first, each table then divided by its
  first singular value, ``feature_tables.mfa_weights``); without other tables, the
  explainable features alone, as before;
- ``attached``: the observation table with the tables' columns and the ``Setup`` naming them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .._wording import count, listing
from ..embedding import MIN_FEATURES, robust_scale
from ..feature_tables import (
    COMBINED,
    EXPLAINABLE,
    OBSERVATION_TABLE,
    RESERVED_NAMES,
    FeatureTables,
    attach_feature_tables,
    mfa_weights,
)
from ..metadata import CONDITION_COLUMN, REPEAT_COLUMN
from ..sampling_unit._experiments import experiments
from ..schema import validate_input_tables
from ..separation import Separation, separation
from ..strategies import SupportNoveltyEstimator
from ..transforms import BlockWeights, Chain

if TYPE_CHECKING:
    from .setup import Setup


def _declared(declared: Any, seen: dict[str, str], names: list[str]) -> dict[str, tuple[str, ...]]:
    """A declaration of tables as a dict of names and column tuples, checked against the
    names and columns ``seen`` before it (see the module)."""
    if declared is None:
        return {}
    if not isinstance(declared, Mapping):
        raise TypeError("feature_tables must map each other table's name to its columns")
    tables: dict[str, tuple[str, ...]] = {}
    for name, columns in declared.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"a feature table's name must be a non-empty text, not {name!r}")
        folded = name.strip().casefold()
        if folded in RESERVED_NAMES or folded in names:
            raise ValueError(f"{name!r} is reserved or names two tables: give each its own name")
        names.append(folded)
        if isinstance(columns, str) or not columns:
            raise TypeError(f"the feature table {name!r} must list its columns")
        columns = tuple(columns)
        for column in columns:
            if column in seen:
                raise ValueError(f"{column!r} is a column of both {seen[column]!r} and {name!r}")
            seen[column] = name
        tables[name] = columns
    return tables


def check_feature_tables(setup: Setup) -> None:
    """``feature_tables`` and ``explainable_tables`` as dicts of names and column tuples,
    checked (see the module). With explainable tables, the readout must be declared, and
    with no ``features`` every column of them is a feature (owner, 2026-10-06)."""
    seen: dict[str, str] = {}
    names: list[str] = []
    explainable = _declared(setup.explainable_tables, seen, names)
    tables = _declared(setup.feature_tables, seen, names)
    declared = {*setup.columns, *([setup.rare_label] if setup.rare_label else [])}
    for name, columns in tables.items():
        for column in columns:
            if column in declared:
                raise ValueError(f"{column!r} of {name!r} is also a declared feature or label")
    if explainable and setup.readout is None:
        raise ValueError(
            f"explainable feature tables were given ({listing(list(explainable), 4)}): declare "
            "the readout, readout=... among their columns or the observation table's, and the "
            "features to test, features=(...), or none for every column of these tables"
        )
    if explainable and not setup.features:
        every = tuple(c for columns in explainable.values() for c in columns)
        object.__setattr__(setup, "features", every)
    object.__setattr__(setup, "feature_tables", tables or None)
    object.__setattr__(setup, "explainable_tables", explainable or None)


def table_columns(setup: Setup) -> list[str]:
    """Every other (non-explainable) table's columns, in the order declared."""
    return [c for columns in (setup.feature_tables or {}).values() for c in columns]


# The tables of a space, by name: the explainable features first, then each other table.
Blocks = dict[str, tuple[str, ...]]


def explainable_blocks(setup: Setup) -> Blocks:
    """The declared explainable features by table (owner, 2026-10-06): the observation
    table's own first (``OBSERVATION_TABLE``), then each explainable table's, a table none
    of whose columns is declared left out. One block, or none, without explainable tables."""
    tables = setup.explainable_tables or {}
    declared = setup.columns
    inside = {c for columns in tables.values() for c in columns}
    blocks: Blocks = {}
    own = tuple(c for c in declared if c not in inside)
    if own:
        blocks[OBSERVATION_TABLE] = own
    for name, columns in tables.items():
        chosen = set(columns)
        mine = tuple(c for c in declared if c in chosen)
        if mine:
            blocks[name] = mine
    return blocks


def several_explainable(setup: Setup) -> bool:
    """Whether the declared explainable features come from two tables or more: they are then
    weighted by MFA across them, mapped and named by table (owner, 2026-10-06)."""
    return len(explainable_blocks(setup)) > 1


def table_of(setup: Setup) -> dict[str, str]:
    """Each declared explainable feature's table, with explainable tables; empty without."""
    if not setup.explainable_tables:
        return {}
    return {c: name for name, columns in explainable_blocks(setup).items() for c in columns}


def with_table(frame: pd.DataFrame, setup: Setup) -> pd.DataFrame:
    """``frame`` with a ``table`` column after its ``feature`` column, naming each feature's
    table, when explainable tables were given (owner, 2026-10-06); ``frame`` otherwise."""
    owner = table_of(setup)
    if not owner or "feature" not in frame.columns or "table" in frame.columns:
        return frame
    tables = [owner.get(str(f), owner.get(f, "")) for f in frame["feature"]]
    frame = frame.copy()
    frame.insert(frame.columns.get_loc("feature") + 1, "table", tables)
    return frame


def copy_tables(setup: Setup) -> dict[str, tuple[str, ...]]:
    """The tables the copy checks read apart (owner, 2026-10-06): each explainable table
    when the features come from two or more, else the explainable features as one; then
    each other table."""
    blocks = explainable_blocks(setup)
    first = blocks if len(blocks) > 1 else {EXPLAINABLE: tuple(setup.columns)}
    return {**first, **dict(setup.feature_tables or {})}


def explainable_space(
    setup: Setup, features: Sequence[str] | None = None
) -> tuple[list[str], Blocks | None]:
    """The explainable features (``features``, default ``Setup.columns``) and their tables,
    ``None`` (not weighted) unless they come from two tables or more (MFA)."""
    columns = list(setup.columns if features is None else features)
    chosen = set(columns)
    blocks = {
        name: kept
        for name, block in explainable_blocks(setup).items()
        if (kept := tuple(c for c in block if c in chosen))
    }
    return columns, (blocks if len(blocks) > 1 else None)


def space(
    setup: Setup, explainable: Sequence[str] | None = None
) -> tuple[list[str], Blocks | None]:
    """The columns of the weighted combined space and its tables (``None``, not weighted,
    without other tables): each explainable table, or the explainable features as one
    block, then each other table. ``explainable`` defaults to ``Setup.columns``."""
    columns, blocks = explainable_space(setup, explainable)
    if not setup.feature_tables:
        return columns, blocks
    first = blocks if blocks is not None else {EXPLAINABLE: tuple(columns)}
    every = {**first, **dict(setup.feature_tables)}
    return [c for block in every.values() for c in block], every


def novelty_space(setup: Setup) -> tuple[SupportNoveltyEstimator | None, list[str]]:
    """The definition of new and the columns it reads: ``Setup.novelty`` on the declared
    features, whatever the other tables (owner, 2026-10-06: they never enter novelty); with
    features of two explainable tables or more, each table weighted by MFA after the
    definition's own transform, fitted on the visible fields (``BlockWeights``), the radius
    declared for the ``d`` standardised features, ``sqrt(d)``, read as the square root of the
    weighted space's total inertia (owner, 2026-10-06, audit 8); with one table, as
    declared."""
    novelty, columns = setup.novelty, list(setup.features)
    blocks = explainable_space(setup, columns)[1]
    if novelty is None or blocks is None:
        return novelty, columns
    weights = BlockWeights(tuple(blocks.values()), inertia_radius=True)
    transform = weights if novelty.transform is None else Chain([novelty.transform, weights])
    return replace(novelty, transform=transform), columns


def attached(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    setup: Setup,
    feature_tables: Mapping[str, pd.DataFrame | tuple[pd.DataFrame, str]] | None,
) -> tuple[pd.DataFrame, Setup, FeatureTables | None]:
    """``objects`` with the feature tables' columns, the ``Setup`` naming them (the
    explainable ones in ``explainable_tables``, the others in ``feature_tables``), and what
    ``attach_feature_tables`` checked (``None`` without tables given). Without tables, the
    columns a ``Setup`` names must already be in ``objects``."""
    if feature_tables is None:
        named = [
            *table_columns(setup),
            *(c for t in (setup.explainable_tables or {}).values() for c in t),
        ]
        absent = [c for c in named if c not in objects.columns]
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
    for declared, given in (
        (setup.feature_tables, tables.other),
        (setup.explainable_tables, tables.explainable),
    ):
        if declared is not None and dict(declared) != given:
            raise ValueError(
                "the Setup names other feature tables or columns than those given: "
                f"{sorted(declared)} against {sorted(given)}"
            )
    kept = replace(
        setup, feature_tables=tables.other or None, explainable_tables=tables.explainable or None
    )
    return tables.objects, kept, tables


def explainable_line(setup: Setup) -> str:
    """With explainable tables, how their columns are read, in one sentence (owner,
    2026-10-06); empty without."""
    blocks = explainable_blocks(setup)
    if not setup.explainable_tables:
        return ""
    by_table = listing([f"{name} {len(c)}" for name, c in blocks.items()], 8)
    weighted = (
        "; novelty, unusual observations and the populations weigh each table by MFA, so a "
        "table of many columns does not outweigh one of few"
        if len(blocks) > 1
        else ""
    )
    return (
        f"Explainable features by table: {by_table}. The columns of an explainable table are "
        "explainable features, each tested one by one when declared in features, every column "
        f"of these tables when none is declared; one correction family over them all{weighted}."
    )


def table_lines(
    setup: Setup, tables: FeatureTables | None = None, centring: str = "", unusual: bool = True
) -> list[str]:
    """The overview's sentences on the feature tables (what ``tables`` checked, when given),
    none without them; ``unusual=False`` when unusual observations do not run (no control,
    one observation per field)."""
    if not setup.feature_tables and not setup.explainable_tables:
        return []
    declared = {**(setup.explainable_tables or {}), **(setup.feature_tables or {})}
    lines = (
        tables.lines()
        if tables is not None
        else [f"{name}: {count(len(c), 'feature column')}." for name, c in declared.items()]
    )
    explainable = explainable_line(setup)
    lines += [explainable] if explainable else []
    if not setup.feature_tables:
        return lines
    if not setup.columns:  # an embedding-only report (owner, 2026-10-06)
        read = "the copy checks, unusual observations" if unusual else "the copy checks"
        lines.append(
            f"{features_words(setup)}. With no explainable features, each table is tested "
            f"whole (the whole-table test of question 9) and read by the maps, the populations, "
            f"{read} and its own fingerprint; never feature by feature."
        )
        return [*lines, *([centring] if centring else [])]
    others = "non-explainable" if setup.explainable_tables else "other"
    lines.append(
        f"{features_words(setup)}. The {others} tables are never tested feature by feature: "
        "they are tested whole, beside the features, and join the maps, the populations, the "
        "copy checks and their own fingerprints, not novelty or unusual observations, which "
        "read the explainable features only; in the combined space each table counts as much "
        "as the explainable features, its first axis weighing as much as theirs."
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
    with other tables the combined space first, then the explainable features (weighted by
    MFA across their tables), each explainable table when there are two or more, and each
    other table; without them, the declared features alone."""
    blocks = explainable_blocks(setup)
    several = len(blocks) > 1
    if not setup.feature_tables and not several:
        return {"Features": (list(setup.columns), None)}
    spaces: dict[str, tuple[list[str], Blocks | None]] = {}
    # Without explainable features (owner, 2026-10-06), one table is its own combined space.
    if setup.feature_tables and (setup.columns or len(setup.feature_tables) > 1):
        spaces[COMBINED.capitalize()] = space(setup)
    if setup.columns:
        spaces[EXPLAINABLE.capitalize()] = explainable_space(setup)
    if several:  # each explainable table its own map (owner, 2026-10-06)
        spaces.update({name: (list(c), None) for name, c in blocks.items()})
    spaces.update({name: (list(c), None) for name, c in (setup.feature_tables or {}).items()})
    return spaces


def _mfa(
    objects: pd.DataFrame, fovs: pd.DataFrame, setup: Setup, columns: list[str], blocks: Blocks
) -> dict[str, float]:
    """Each column's MFA weight in the space of ``blocks``, on every complete observation
    robustly standardised (``feature_tables.mfa_weights``)."""
    data = validate_input_tables(objects, fovs=fovs, feature_cols=columns, **setup.common)
    matrix = data.feature_matrix()
    complete = matrix[~np.isnan(matrix).any(axis=1)]
    centre, scale = robust_scale(complete)
    z = (complete - centre) / np.where(scale > 0, scale, 1.0)
    weights = mfa_weights(z, columns, blocks)
    return dict(zip(columns, map(float, weights), strict=True))


def unusual_space(
    objects: pd.DataFrame, fovs: pd.DataFrame, setup: Setup
) -> tuple[list[str], dict[str, float] | None, str]:
    """The columns unusual observations read, their weights and the space in words: the
    explainable features when there are any, each table weighted by MFA when they come from
    two tables or more; with a readout and no features, the readout alone (owner,
    2026-10-06, audit 8, seed 2: the other tables never enter unusual observations beside
    explainable features); in an embedding-only report, the MFA-weighted space of the
    tables, each table divided by its first singular value (owner, 2026-10-06)."""
    if setup.features or not setup.feature_tables:
        columns, blocks = explainable_space(setup, setup.features)
        words = f"the {count(len(columns), 'explainable feature')}"
        if blocks is None:
            return columns, None, words
        words += f", each of their {len(blocks)} tables weighted by MFA"
        return columns, _mfa(objects, fovs, setup, columns, blocks), words
    if setup.columns:  # the readout, with no features declared
        return list(setup.columns), None, f"the readout {setup.readout} alone"
    columns, blocks = space(setup)
    names = listing(list(setup.feature_tables), 4)
    words = (
        f"the weighted combined space of {names} (each table divided by its first singular "
        "value; there are no explainable features)"
        if len(setup.feature_tables) > 1
        else f"the {count(len(columns), 'column')} of {names} (there are no explainable features)"
    )
    return columns, _mfa(objects, fovs, setup, columns, blocks or {}), words


def separation_in_report(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    setup: Setup,
    names: Sequence[str] | None = None,
    raw: pd.DataFrame | None = None,
) -> Separation | None:
    """The separation of the conditions and of the repeats in each space with a map
    (``space_names``, or those of ``names``): of the conditions on the values the figures
    read, of the repeats on the ``raw`` values (default ``objects``; owner, 2026-10-06, audit
    8); ``None`` with one condition, no condition column or no space of ``MIN_FEATURES``
    columns."""
    if CONDITION_COLUMN not in fovs or fovs[CONDITION_COLUMN].nunique() < 2:
        return None
    # Only the spaces with a map (``MIN_FEATURES`` columns), so every score has its map.
    spaces = {n: s for n, s in space_names(setup).items() if len(s[0]) >= MIN_FEATURES}
    if names is not None:
        spaces = {name: spaces[name] for name in names if name in spaces}
    if not spaces:
        return None
    reasons = {
        name: why
        for name, (columns, _) in spaces.items()
        if raw is not None and (why := _dropped_reason(objects, raw, columns, setup))
    }
    return separation(
        objects,
        fovs,
        spaces={name: columns for name, (columns, _) in spaces.items()},
        blocks={name: blocks for name, (_, blocks) in spaces.items()},
        random_state=setup.random_state,
        repeat_values=raw,
        dropped_reasons=reasons,
        **setup.common,
    )


def _dropped_reason(
    values: pd.DataFrame, raw: pd.DataFrame, columns: Sequence[str], setup: Setup
) -> str:
    """Why observations with every raw value of ``columns`` have none in ``values`` (audit 8):
    an explainable feature normalised without the own control, a table's repeat with no
    condition to centre on; empty when none is."""
    reasons = []
    for part, why in (
        ([c for c in columns if c in setup.columns], "no own control"),
        ([c for c in columns if c not in setup.columns], "no condition to centre on"),
    ):
        if part and (values[part].isna().any(axis=1) & raw[part].notna().all(axis=1)).any():
            reasons.append(why)
    return " or ".join(reasons)


@dataclass(frozen=True)
class Centring:
    """How the other tables were centred per biological repeat (``centred_tables``):
    ``values`` the centred columns; ``centres`` each sub-experiment's repeats and the
    conditions in all of them, whose per-condition medians set its repeats' centres;
    ``uncentred`` the repeats of a sub-experiment no condition is in every repeat of, left
    out (missing) of the centred values."""

    values: pd.DataFrame
    centres: tuple[tuple[tuple[Any, ...], tuple[Any, ...]], ...]
    uncentred: tuple[Any, ...]


def _centring(objects: pd.DataFrame, fovs: pd.DataFrame, setup: Setup) -> Centring | None:
    """The centring of the other tables (see ``centred_tables``); ``None`` without them."""
    columns = table_columns(setup)
    if not columns:
        return None
    data = validate_input_tables(objects, fovs=fovs, feature_cols=columns, **setup.common)
    frame = data.objects
    alone = pd.Series("all", index=frame.index)
    repeat = frame.get(REPEAT_COLUMN, alone)
    condition = frame.get(CONDITION_COLUMN, alone)
    order = list(pd.unique(condition))
    held = {r: set(condition[repeat == r]) for r in pd.unique(repeat)}
    # The sub-experiments: repeats linked by a condition they share (``_experiments``).
    family = experiments(held)
    medians = frame[columns].groupby([condition, repeat]).median()
    shift = np.zeros((len(frame), len(columns)))
    centres, uncentred = [], []
    for number in dict.fromkeys(family.values()):
        repeats = [r for r in held if family[r] == number]
        shared = set.intersection(*(held[r] for r in repeats))
        if not shared:  # left out: raw values in a centred space would read as a batch
            uncentred += repeats
            for r in repeats:
                shift[(repeat == r).to_numpy()] = np.nan
            continue
        common = tuple(c for c in order if c in shared)
        centres.append((tuple(repeats), common))
        for r in repeats:
            # The median of the per-condition medians: no condition's cell count weighs.
            rows = [medians.loc[(c, r)].to_numpy(dtype=float) for c in common]
            shift[(repeat == r).to_numpy()] = np.nanmedian(np.vstack(rows), axis=0)
    centred = frame[columns].to_numpy(dtype=float) - shift
    values = pd.DataFrame(centred, index=frame.index, columns=columns)
    return Centring(values, tuple(centres), tuple(uncentred))


def centred_tables(objects: pd.DataFrame, fovs: pd.DataFrame, setup: Setup) -> pd.DataFrame | None:
    """``objects`` with every other table's column centred per biological repeat (owner,
    2026-10-06, audit 8): minus the median of the per-condition medians of the conditions in
    every repeat, balanced or not, so no control and no cell count decides the centre. When
    no condition is in every repeat, within each sub-experiment (repeats linked by the
    conditions they share, ``_experiments``) on the conditions in all of its repeats; a
    repeat of a sub-experiment without such a condition is left out (missing) and named
    (``centring_line``). The centred map, and with ``normalize`` the populations, unusual
    observations and the separation of the conditions (``analysed_tables``); never a test.
    ``None`` without other tables."""
    found = _centring(objects, fovs, setup)
    if found is None or not found.centres:
        return None
    shifted = found.values
    return objects.assign(**{c: shifted[c] for c in shifted.columns})


def analysed_tables(values: pd.DataFrame, fovs: pd.DataFrame, setup: Setup) -> pd.DataFrame:
    """The values the tables' populations, unusual observations and separation of the
    conditions read (the repeats' separation reads the raw values): with ``normalize``
    declared, the tables centred per repeat (``centred_tables``), as the explainable features
    are divided by their control; ``values`` otherwise. Never a test: the whole-table test
    and the fingerprint read the raw values."""
    if setup.normalize is None or not setup.feature_tables:
        return values
    centred = centred_tables(values, fovs, setup)
    return values if centred is None else centred


def _centre_words(found: Centring) -> str:
    """Which conditions set the centre, in words (see ``centring_line``)."""
    if len(found.centres) == 1:
        common = found.centres[0][1]
        every = "the condition" if len(common) == 1 else "the conditions"
        every += " in every repeat"
        return f"of {listing([str(c) for c in common], 8)}, {every}"
    parts = [
        f"{listing([str(c) for c in common], 4)} in {listing([str(r) for r in repeats], 4)}"
        for repeats, common in found.centres
    ]
    return f"of the conditions in every repeat of its sub-experiment: {'; '.join(parts)}"


def centring_line(
    objects: pd.DataFrame, fovs: pd.DataFrame, setup: Setup, unusual: bool = True
) -> str:
    """Which conditions set the centre of the tables, in one line (owner, 2026-10-06, audit
    8), the repeats left out named, and what reads the centred values; empty without
    other tables. Unusual observations read the tables only without explainable features,
    and when they run (``unusual``)."""
    found = _centring(objects, fovs, setup)
    if found is None:
        return ""
    if not found.centres:  # nothing to centre on: no centred values (``centred_tables``)
        return (
            "Batch: the other tables are not centred, as no condition is in every repeat of a "
            "sub-experiment; they are read as given."
        )
    else:
        how = "centred per biological repeat on the median of the per-condition medians "
        how += _centre_words(found)
    if found.uncentred and found.centres:
        named = listing([str(r) for r in found.uncentred], 8)
        verb = "is" if len(found.uncentred) == 1 else "are"
        how += (
            f"; {named} {verb} left out of the centred values, as no condition is in every "
            "repeat of its sub-experiment"
        )
    unusual = unusual and not setup.columns
    readers = ["the centred map", "the populations", "unusual observations"]
    readers.append("the separation of the conditions")
    readers = [r for r in readers if unusual or r != "unusual observations"]
    reads = f"{listing(readers)} read them" if setup.normalize is not None else ""
    reads = reads or "the centred map reads them"
    return f"Batch: the other tables {how}; {reads}, never a test."


__all__ = [
    "Centring",
    "analysed_tables",
    "attached",
    "centred_tables",
    "centring_line",
    "check_feature_tables",
    "copy_tables",
    "explainable_blocks",
    "explainable_line",
    "explainable_space",
    "features_words",
    "novelty_space",
    "separation_in_report",
    "several_explainable",
    "space",
    "space_names",
    "table_columns",
    "table_lines",
    "table_of",
    "unusual_space",
    "with_table",
]

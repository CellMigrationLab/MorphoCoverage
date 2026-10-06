"""Question 10's groups, found in each feature space (private; ``_populations``).

With other feature tables (owner, 2026-10-06), the groups are searched in each space apart:
the explainable features, each other table, and the weighted combined space
(``_feature_space.space``: each table divided by its first singular value). A group living in
one table is found there, where the combined space can drown it. Each distinct group is
reported once, naming the space or spaces that found it: groups found in two spaces are one
when their cells overlap with a Jaccard index (the cells in both over the cells in either)
of ``SAME_GROUP`` (0.5) or more, the main populations likewise among themselves. A group
keeps the cells, fields and description of the first space that found it, in the order
explainable, each table, combined.

A group the explainable features found is described, and given a rule, with them. A group
found only in other spaces is not: "found in DINOv2 only; no rule in the explainable
features describes it", as the explainable features that differ most there are noise (d
about 0.25). The enrichment tests run on the merged list of groups, one Benjamini-Hochberg
family as before (``_populations``). The explainable features are clustered as before; each
other space on at most ``SPACE_MAX_POINTS`` observations (a seeded sample of the same share of
every condition, the others taking the cluster of their nearest sampled neighbour), so that
the runtime stays modest. Without other tables, the explainable features alone, as before.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd

from .._wording import count, listing
from ..feature_tables import COMBINED, EXPLAINABLE
from ..metadata import CONDITION_COLUMN, REPEAT_COLUMN
from ..sampling_unit.populations import MAX_POINTS, Populations, find_populations
from ..schema import validate_input_tables
from ._feature_space import explainable_space, features_words, space, table_columns, table_of

if TYPE_CHECKING:
    from .setup import Setup

# Two groups found in two spaces are one at this Jaccard index of their cells, or more.
SAME_GROUP = 0.5
# The observations clustered in each space other than the explainable features, at most.
SPACE_MAX_POINTS = 5000
_WORDS = {EXPLAINABLE: "the explainable features", COMBINED: "the combined space"}


@dataclass
class _Found:
    """One distinct group: its kind, cells, the cells its space assessed, its row in the
    table of the first space that found it, and every space that found it."""

    kind: str
    inside: pd.Index
    assessed: pd.Index
    row: pd.Series
    spaces: list[str] = field(default_factory=list)


def space_words(name: str) -> str:
    """A space as the text names it: ``the explainable features``, ``DINOv2``, ``the
    combined space``."""
    return _WORDS.get(name, name)


def _jaccard(first: pd.Index, second: pd.Index) -> float:
    both = len(first.intersection(second))
    either = len(first.union(second))
    return both / either if either else 0.0


def _spaces(setup: Setup) -> dict[str, tuple[list[str], Any, list[str] | None]]:
    """Each space searched: its columns, its tables (the combined space) and the features
    that may describe its groups (``None``: all of them; empty: none)."""
    # The explainable features, each of their tables weighted by MFA (owner, 2026-10-06).
    explainable, weighted = explainable_space(setup)
    if not setup.feature_tables:
        return {EXPLAINABLE: (explainable, weighted, explainable)}
    columns, blocks = space(setup)
    spaces: dict[str, tuple[list[str], Any, list[str] | None]] = {}
    if setup.columns:  # none in an embedding-only report (owner, 2026-10-06)
        spaces[EXPLAINABLE] = (explainable, weighted, explainable)
    spaces.update({name: (list(c), None, []) for name, c in setup.feature_tables.items()})
    if setup.columns or len(setup.feature_tables) > 1:  # one table is its own combined space
        spaces[COMBINED] = (columns, blocks, [])
    return spaces


def _merge(results: dict[str, Populations]) -> list[_Found]:
    """The distinct groups of every space (see the module), in the order found."""
    found: list[_Found] = []
    for name, populations in results.items():
        table = populations.table.set_index("population")
        for population in populations.names:
            inside, assessed = (populations.members or {})[population]
            kind = str(table.loc[population, "kind"])
            overlaps = [(_jaccard(inside, f.inside), f) for f in found if f.kind == kind]
            best = max(overlaps, key=lambda pair: pair[0], default=(0.0, None))
            if best[1] is not None and best[0] >= SAME_GROUP:
                best[1].spaces.append(name)
                continue
            found.append(_Found(kind, inside, assessed, table.loc[population], [name]))
    return found


def _merged(results: dict[str, Populations], index: pd.Index) -> Populations:
    """One ``Populations`` of the distinct groups: the main populations ``P1``, ``P2``, then
    the groups ``G1``, ``G2``, ..., the largest first, each with the spaces that found it
    (``found_in``); a cell in two groups is labelled by the later on the maps."""
    found = _merge(results)
    mains = [f for f in found if f.kind == "main"]
    groups = sorted((f for f in found if f.kind == "group"), key=lambda f: -len(f.inside))
    if not groups:
        mains = []  # as in one space: no group, nothing to list
    named = [(f"P{i + 1}", f) for i, f in enumerate(mains)]
    named += [(f"G{i + 1}", f) for i, f in enumerate(groups)]
    labels = pd.Series([None] * len(index), index=index, dtype=object)
    rows, members = [], {}
    for name, f in named:
        labels[labels.index.isin(f.inside)] = name
        members[name] = (f.inside, f.assessed)
        rows.append({**f.row.to_dict(), "population": name, "found_in": list(f.spaces)})
    first = next(iter(results.values()))
    columns = [*first.table.columns, "found_in"]
    table = pd.DataFrame(rows, columns=columns)
    settings = dict(results.get(EXPLAINABLE, first).settings)
    settings["spaces"] = {
        name: {
            "features": len(p.settings["features"]),
            "observations": p.settings["observations"],
            "incomplete_observations": p.settings["incomplete_observations"],
            "clustered": p.settings["clustered"],
            "sampled": p.settings["sampled"],
        }
        for name, p in results.items()
    }
    settings["same_group"] = SAME_GROUP
    settings["clustered"] = any(p.settings["clustered"] for p in results.values())
    return Populations(labels, table, settings, members)


def _left_out(objects: pd.DataFrame, raw: pd.DataFrame | None, columns: list[str]) -> pd.Index:
    """The observations with every raw value of ``columns`` and not every value in
    ``objects``: normalised without their own control, or in a repeat with no condition to
    centre on (audit 8)."""
    if raw is None or raw is objects or not columns:
        return pd.Index([])
    lost = raw[columns].notna().all(axis=1) & objects[columns].isna().any(axis=1)
    return objects.index[lost.reindex(objects.index, fill_value=False).to_numpy()]


def _left_without_control(
    objects: pd.DataFrame, fovs: pd.DataFrame, setup: Setup, left: pd.Index
) -> list[tuple[str, list[str]]]:
    """Each repeat of ``left`` with its conditions, in the order of the observation table."""
    data = validate_input_tables(
        objects.loc[left],
        fovs=fovs,
        feature_cols=(list(setup.columns) or table_columns(setup))[:1],
        **setup.common,
    )
    frame = data.objects
    alone = pd.Series("all", index=frame.index)
    repeat, condition = frame.get(REPEAT_COLUMN, alone), frame.get(CONDITION_COLUMN, alone)
    return [
        (str(r), [str(c) for c in pd.unique(condition[repeat == r])]) for r in pd.unique(repeat)
    ]


def no_own_control_words(settings: dict[str, Any], unit: str, units: str) -> str:
    """``36 cells of b1 and b2 in R4 left out: no own control``, said once for the spaces
    that read the values concerned (audit 8, seed 4: never a missing value); likewise the
    repeats with no condition to centre the tables on; empty when none was."""
    parts = []
    for found in settings.get("left_out_by") or []:
        where = "; ".join(f"{listing(c, 4)} in {r}" for r, c in found["where"])
        spaces = found.get("spaces") or []
        of = f" of {listing([space_words(s) for s in spaces], 4)}" if len(spaces) > 1 else ""
        number = count(found["observations"], unit, units)
        parts.append(f"{number} of {where} left out{of}: {found['reason']}")
    return "; ".join(parts)


def populations_in_spaces(
    objects: pd.DataFrame, fovs: pd.DataFrame, setup: Setup, raw: pd.DataFrame | None = None
) -> Populations | None:
    """The populations of every space, merged (see the module); ``None`` with fewer than two
    explainable features and no other table. With ``raw`` (the values before normalisation),
    the observations left out for want of their own control, or of a condition to centre
    the tables on, are counted apart from those missing a value (``no_own_control_words``)."""
    spaces = _spaces(setup)
    results = {}
    for name, (columns, blocks, named_by) in spaces.items():
        if len(columns) < 2:
            continue
        results[name] = find_populations(
            objects,
            fovs,
            features=columns,
            random_state=setup.random_state,
            max_points=MAX_POINTS if name == EXPLAINABLE else SPACE_MAX_POINTS,
            blocks=blocks,
            named_by=named_by,
            **setup.common,
        )
    if not results:
        return None
    explainable = list(setup.columns)
    kinds = (
        []
        if setup.normalize is None
        else [
            ("no own control", explainable),
            ("no condition to centre on", table_columns(setup)),
        ]
    )
    left_by = []
    lost: dict[str, pd.Index] = {name: pd.Index([]) for name in results}
    for reason, columns in kinds:
        left = _left_out(objects, raw, columns)
        reading = [n for n in results if set(spaces[n][0]) & set(columns)]
        if not len(left) or not reading:
            continue
        for name in reading:
            lost[name] = lost[name].union(left)
        where = _left_without_control(objects, fovs, setup, left)
        left_by.append(
            {"reason": reason, "observations": len(left), "where": where, "spaces": reading}
        )
    for name, index in lost.items():  # counted apart from the missing values
        results[name].settings["incomplete_observations"] -= len(index)
    found = results[EXPLAINABLE] if not setup.feature_tables else _merged(results, objects.index)
    if left_by:
        found.settings["left_out_by"] = left_by
    owner = table_of(setup)
    if len(set(owner.values())) > 1:  # each rule feature's table named (owner, 2026-10-06)
        found.table["rule_tables"] = [
            rule_tables(terms, owner) for terms in found.table["rule_terms"]
        ]
    return found


def rule_tables(terms: Any, owner: dict[str, str]) -> str:
    """The table of each feature of a rule: ``AR from fiberfeatures, MinFeret from
    CellProfiler``; empty without a rule."""
    features = dict.fromkeys(term[0] for term in (terms or ()))
    return ", ".join(f"{f} from {owner[f]}" for f in features if f in owner)


def found_words(row: Any, rule: bool = True) -> str:
    """Where a group was found, when other tables were searched: ``found in the explainable
    features and DINOv2``, or for a group the explainable features did not find, ``found in
    DINOv2 only; no rule in the explainable features describes it`` (``rule=False`` leaves
    the rule out, said elsewhere); empty otherwise."""
    spaces = row.get("found_in") if isinstance(row, dict | pd.Series) else None
    if not isinstance(spaces, list) or not spaces:
        return ""
    named = listing([space_words(s) for s in spaces], 8)
    if EXPLAINABLE in spaces:
        return f"found in {named}"
    if not rule:
        return f"found in {named} only"
    return f"found in {named} only; no rule in the explainable features describes it"


def undescribed_line(populations: Populations) -> str:
    """The groups no rule in the explainable features describes, said once: ``Not found in the
    explainable features, so no rule in them describes it: G1 (DINOv2)``; empty when there is
    none (owner, 2026-10-06)."""
    rows = [r for _, r in populations.table.iterrows() if r["kind"] == "group" and not described(r)]
    if not rows:
        return ""
    named = [
        f"{r['population']} ({listing([space_words(s) for s in r['found_in']], 8)})" for r in rows
    ]
    them = "them" if len(rows) > 1 else "it"
    return (
        f"Not found in the explainable features, so no rule in them describes {them}: "
        f"{listing(named, 8)}."
    )


def described(row: Any) -> bool:
    """Whether a group is described with the explainable features (found by them, or no
    other table searched)."""
    spaces = row.get("found_in") if isinstance(row, dict | pd.Series) else None
    return not isinstance(spaces, list) or EXPLAINABLE in spaces


def spaces_method(populations: Populations, setup: Setup) -> str:
    """How the spaces were clustered, in words: ``in each feature space apart: the 17
    explainable features (720 cells), DINOv2 (384 columns, 595 cells; 125 with a missing
    value left out) and the combined space of all of them (720 cells)``."""
    spaces = populations.settings.get("spaces") or {}
    unit = setup.unit_word
    parts = []
    for name, numbers in spaces.items():
        cells = count(numbers["observations"], unit)
        missing = numbers["incomplete_observations"]
        if missing:
            cells += f"; {count(missing, unit)} with a missing value left out"
        if name == EXPLAINABLE:
            what = f"the {count(numbers['features'], 'explainable feature')}"
        elif name == COMBINED:
            what = f"the combined space of {features_words(setup)}"
        else:
            what = f"{name}, {count(numbers['features'], 'column')}"
        parts.append(f"{what} ({cells})")
    return f"in each feature space apart: {listing(parts, 8)}"


__all__ = [
    "SAME_GROUP",
    "SPACE_MAX_POINTS",
    "described",
    "found_words",
    "no_own_control_words",
    "populations_in_spaces",
    "rule_tables",
    "space_words",
    "spaces_method",
    "undescribed_line",
]

"""Representation controls: could these features recognize the phenotype I care about?

Question R1 of ``docs/questions.md``. A "nothing new" answer is only as good as
the features it was computed on: a phenotype they cannot separate is never new.
A declared control tests this on the dataset itself:

- **positive**: FOVs of a declared phenotype (for example a knockdown well),
  judged as candidate next FOVs against ``fovs_seen`` random FOVs of the
  reference. The share judged new is the **detection rate** for whole fields.
  With ``planted_objects``, that many objects of a random positive FOV are also
  added to remaining reference FOVs: the detection rate for a small group of
  such objects in an otherwise ordinary field. Planted objects come from other
  fields, so this approximates a rare state rather than measuring one.
- **reference**: the declared negative population (for example MOCK). Its
  remaining FOVs, judged the same way, give the **false-alarm rate**. Without
  a reference, the undeclared rest of the stratum stands in for it, and the
  false-alarm rate is "not assessed".
- With neither, nothing can be said: "not assessed".

**Leave one field out.** The rates above judge a field against ``fovs_seen``
reference fields, as a stopping rule judges the next field. ``inspect_fovs``
judges each field against every other field of its analysis group, and the
false-alarm rate depends strongly on how many fields were seen (41% after 3
fields, 0% after 29, on one plate). So the same definition is also measured
that way, within the inspection's analysis groups (``lofo_group_cols``; by
default the analysis groups of the data, as ``inspect_fovs`` resolves them)
inside each stratum: each reference field against every other reference
field of its group (``reference_lofo``, the false-alarm rate of the
inspection, which the report uses for its "expected by chance" clause), and
each positive field against the reference fields of each reference group
(``positive_lofo``, one check per positive field and group: the detection
rate after a whole group was acquired, which the report uses to adjust the
unseen-state bound). Both are measured in every stratum that has the fields
they need, also where the rates after ``fovs_seen`` fields cannot be.

**Planted, leave one field out.** With ``planted_objects``, the same
planting is also judged as the unseen-state bound judges a field: each
reference field of an analysis group, with that many objects of a random
positive field added, against the other reference fields of its group
(``planted_lofo``). It is the detection rate of a small state in an
ordinary field, and the report adjusts the unseen-state bound by it
(``report_profile``): the whole-field rate describes whole-field phenotypes
only (97% whole-field against 19% for 5 planted objects on one dataset).

**Seen fields hold observations.** The ``fovs_seen`` reference fields of a
draw are drawn among the reference fields holding observations: an empty
field adds nothing to the reference, and with it among the seen fields the
definition of new often could not be judged at all (with ``fovs_seen`` equal
to the estimator's ``min_fovs``, any empty seen field left the whole draw
unjudged). Empty reference fields remain candidates, and are never new.
**Positive fields are judged only when they hold observations**: an empty
positive field cannot show the phenotype, and counting it as a miss made
the detection rate the share of positive fields with observations (67% at
30% empty fields, although every field with cells was recognized). Their
number is reported apart (``settings["positive_empty"]``).

"New" is the ``SupportNoveltyEstimator`` definition the report uses
(``novel_fovs_against``, transform fitted on the seen FOVs only). Reference and
positive FOVs are paired within ``pair_cols`` (for example the plate), so a
plate effect is not mistaken for a phenotype; pooling across plates must be
asked for with ``pair_cols=[]``. Rates are pooled over draws and candidates,
which are not independent; they describe these controls in this dataset.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .._checks import is_integer
from .._parallel import check_n_jobs, map_units, worker_pool
from .._seeds import group_rng
from ..data_check import validate_input_tables
from ..evaluator import FovSubsets, restrict_input
from ..metadata import MORPHOBRICKS_GROUP_COLUMNS
from ..schema import ValidatedInput, fov_positions
from ..strategies.support_novelty import SupportNoveltyEstimator
from ..sufficiency_text import novelty_definition
from ._text import unit_plural, where
from .controls_text import control_summary

CANDIDATES = (
    "reference",
    "positive",
    "planted",
    "reference_lofo",
    "positive_lofo",
    "planted_lofo",
)


def _selector(value: Mapping[str, Any] | None, label: str) -> dict[str, tuple] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or not value:
        raise TypeError(f"{label} must map FOV-table columns to accepted values")
    selector = {}
    for column, accepted in value.items():
        if isinstance(accepted, str) or not isinstance(accepted, Sequence):
            accepted = (accepted,)
        selector[str(column)] = tuple(accepted)
    return selector


@dataclass(frozen=True)
class RepresentationControl:
    """Declared control FOVs, selected by FOV-table values.

    ``positive`` and ``reference`` map FOV-table columns to accepted values,
    for example ``{"condition": ["YAP"]}`` and ``{"condition": ["MOCK"]}``.
    ``planted_objects`` (needs both) also plants that many positive objects into
    reference FOVs.
    """

    positive: Mapping[str, Any] | None = None
    reference: Mapping[str, Any] | None = None
    planted_objects: int | None = None
    # Deprecated spelling of ``planted_objects``, accepted with a DeprecationWarning.
    planted_cells: int | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.planted_cells is not None:
            warnings.warn(
                "RepresentationControl(planted_cells=...) is deprecated; use planted_objects",
                DeprecationWarning,
                stacklevel=3,
            )
            if self.planted_objects is not None:
                raise ValueError("give planted_objects only (planted_cells is its old name)")
            object.__setattr__(self, "planted_objects", self.planted_cells)
            object.__setattr__(self, "planted_cells", None)
        object.__setattr__(self, "positive", _selector(self.positive, "positive"))
        object.__setattr__(self, "reference", _selector(self.reference, "reference"))
        if self.planted_objects is not None:
            if not is_integer(self.planted_objects):
                raise TypeError("planted_objects must be a positive integer")
            if self.planted_objects < 1:
                raise ValueError("planted_objects must be a positive integer")
            object.__setattr__(self, "planted_objects", int(self.planted_objects))
            if self.positive is None or self.reference is None:
                raise ValueError("planting cells needs a positive and a reference control")

    @property
    def parameters(self) -> dict[str, Any]:
        """The declaration, for provenance."""
        return {
            "positive": None
            if self.positive is None
            else {k: list(v) for k, v in self.positive.items()},
            "reference": None
            if self.reference is None
            else {k: list(v) for k, v in self.reference.items()},
            "planted_objects": self.planted_objects,
        }


@dataclass(frozen=True)
class ControlResult:
    """Rates per stratum and candidate kind; ``status`` says what could be assessed.

    ``table`` has one row per stratum (``pair_cols`` values) and candidate
    (``reference``: false alarms; ``positive``: whole positive FOVs;
    ``planted``: reference FOVs with planted positive objects;
    ``reference_lofo``, ``positive_lofo`` and ``planted_lofo``: judged leave
    one field out, see the module), with ``judged_new``, ``candidates`` and
    ``rate``; ``pooled``
    the same over strata. ``status`` is ``"assessed"``, ``"positive only"``
    or ``"not assessed"`` (from the ``positive`` and ``reference`` rows).
    ``settings["planted_skipped"]`` lists the strata where no objects could
    be planted, with the reason; ``settings["positive_empty"]`` counts the
    positive fields without observations, which are not judged;
    ``settings["lofo_group_cols"]`` the analysis groups the
    leave-one-field-out rates were measured within, and
    ``settings["novelty_definition"]`` the definition of new in words, so a
    report combines the rates only with answers judged the same way.
    """

    table: pd.DataFrame
    pooled: pd.DataFrame
    status: str
    pair_cols: tuple[str, ...]
    settings: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """The control's answer in sentences."""
        return control_summary(self)


def _select(fovs: pd.DataFrame, selector: dict[str, tuple]) -> pd.Series:
    mask = pd.Series(True, index=fovs.index)
    for column, accepted in selector.items():
        if column not in fovs.columns:
            raise ValueError(f"control column {column!r} is not in the FOV table")
        mask &= fovs[column].isin(accepted)
    return mask


def _planted(
    subsets: FovSubsets,
    host_rows: np.ndarray,
    donors: list[pd.DataFrame],
    size: int,
    rng: np.random.Generator,
) -> ValidatedInput | None:
    """Each host FOV with ``size`` objects of a random donor FOV (with enough objects) added."""
    usable = [cells for cells in donors if len(cells) >= size]
    if not usable:
        return None
    data = subsets.data
    hosts_input = subsets.take(host_rows)
    hosts = hosts_input.fovs
    key = list(data.fov_key)
    added = []
    for _, host in hosts.iterrows():
        cells = usable[rng.integers(len(usable))]
        picked = cells.iloc[np.sort(rng.choice(len(cells), size=size, replace=False))].copy()
        for column in key:
            picked[column] = host[column]
        added.append(picked)
    objects = pd.concat([hosts_input.objects, *added], ignore_index=True)
    return restrict_input(data, objects, hosts)


def _judge(
    novelty: SupportNoveltyEstimator, seen: ValidatedInput, candidates: Mapping[str, ValidatedInput]
) -> dict[str, tuple[int, int, int]]:
    """Per candidate kind: FOVs judged new, FOVs judged, FOVs that could not be judged."""
    counts = {}
    for kind, table in candidates.items():
        flags = novelty.novel_fovs_against(seen, table)
        if flags is None:
            # Missing values or too few FOVs: this check could not be made.
            counts[kind] = (0, 0, len(table.fovs))
        else:
            counts[kind] = (int(np.sum(flags)), len(flags), 0)
    return counts


def _leave_one_out(
    novelty: SupportNoveltyEstimator,
    subsets: FovSubsets,
    groups: list[np.ndarray],
    positive: np.ndarray,
    with_reference: bool,
) -> dict[str, tuple[int, int, int]]:
    """Per kind: fields judged new, judged, not judged, within each reference group of ``groups``.

    ``groups`` are the reference fields (FOV rows) of each analysis group of
    the stratum. Reference fields are judged leave one field out within their
    group, as ``inspect_fovs`` judges a field (a field with a new state is
    new; empty fields count, as there); each positive field against the
    reference fields of each group, one check per field and group.
    """
    counts = {kind: [0, 0, 0] for kind in ("reference_lofo", "positive_lofo")}
    positive_input = subsets.take(positive) if len(positive) else None
    for rows in groups:
        reference = subsets.take(rows)
        if with_reference:
            members = novelty.novel_observation_flags(reference, new_states_only=True)
            if members is None:
                counts["reference_lofo"][2] += len(rows)
            else:
                positions = fov_positions(reference, reference.fovs)
                counts["reference_lofo"][0] += len(np.unique(positions[members]))
                counts["reference_lofo"][1] += len(rows)
        if positive_input is not None:
            flags = novelty.novel_fovs_against(reference, positive_input)
            if flags is None:
                counts["positive_lofo"][2] += len(positive)
            else:
                counts["positive_lofo"][0] += int(np.sum(flags))
                counts["positive_lofo"][1] += len(flags)
    return {kind: (new, total, unjudged) for kind, (new, total, unjudged) in counts.items()}


def _planted_leave_one_out(
    novelty: SupportNoveltyEstimator,
    subsets: FovSubsets,
    groups: list[np.ndarray],
    donors: list[pd.DataFrame],
    size: int,
    rng: np.random.Generator,
) -> list[tuple]:
    """The ``planted_lofo`` checks: each reference field of a group, with ``size`` objects
    of a random positive field planted, against the other reference fields of its group."""
    jobs = []
    for rows in groups:
        for host in rows:
            planted = _planted(subsets, np.array([host]), donors, size, rng)
            if planted is not None:
                seen = subsets.take(rows[rows != host])
                jobs.append((novelty, seen, {"planted_lofo": planted}))
    return jobs


def _planting_skipped(donors: list[pd.DataFrame], size: int) -> str | None:
    """Why no objects can be planted in this stratum (no donor field holds enough), or None."""
    largest = max((len(cells) for cells in donors), default=0)
    if largest >= size:
        return None
    return f"no positive-control field holds {size} objects to plant (the largest holds {largest})"


def _lofo_columns(data: ValidatedInput, lofo_group_cols: Sequence[str] | None) -> list[str]:
    """The analysis-group columns the leave-one-field-out rates are measured within.

    By default the analysis groups ``inspect_fovs`` resolves for these tables:
    the MorphoBricks group columns that identify a FOV here.
    """
    if lofo_group_cols is None:
        return [c for c in MORPHOBRICKS_GROUP_COLUMNS if c in data.fov_key and c != data.fov_col]
    if isinstance(lofo_group_cols, str):
        raise TypeError("lofo_group_cols must be a sequence of column names")
    columns = list(lofo_group_cols)
    missing = [c for c in columns if c not in data.fovs.columns]
    if missing:
        raise ValueError(f"lofo_group_cols missing from the FOV table: {missing}")
    return columns


def _groups_of(fovs: pd.DataFrame, rows: np.ndarray, columns: list[str]) -> list[np.ndarray]:
    """The FOV rows ``rows`` split by the values of ``columns`` (one part without columns)."""
    if not len(rows):
        return []
    if not columns:
        return [rows]
    parts = fovs.iloc[rows].groupby(columns, sort=False, dropna=False).indices
    return [rows[np.asarray(index)] for index in parts.values()]


def _stratum_rows(
    stratum: Mapping[str, Any],
    counts: Mapping[str, Sequence[int]],
) -> list[dict]:
    """One table row per candidate kind that was measured or could not be judged."""
    label = where(stratum) if stratum else "all FOVs"
    return [
        {
            **stratum,
            "stratum": label,
            "candidate": kind,
            "judged_new": new,
            "candidates": total,
            "not_judged": not_judged,
            "rate": new / total if total else None,
        }
        for kind, (new, total, not_judged) in counts.items()
        if total or not_judged
    ]


def representation_control(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    novelty: SupportNoveltyEstimator,
    control: RepresentationControl,
    pair_cols: Sequence[str],
    fovs_seen: int,
    draws: int = 20,
    random_state: int = 0,
    fov_col: str | None = None,
    observation_unit: str | None = None,
    n_jobs: int = 1,
    identity_cols: Sequence[str] = (),
    lofo_group_cols: Sequence[str] | None = None,
) -> ControlResult:
    """Detection and false-alarm rates of ``novelty`` for the declared controls.

    ``fovs_seen`` is how many reference FOVs holding observations are
    "already imaged" in each draw, ``draws`` how many random draws per
    stratum. See the module docstring.
    ``identity_cols`` are columns that, with condition, repeat and ``fov_col``,
    identify one FOV (for example ``well`` when position names repeat in every
    well). They only name FOVs: controls are still compared across them, within
    ``pair_cols``.
    ``lofo_group_cols`` are the analysis groups within which the
    leave-one-field-out rates are measured: give the ``group_cols`` of the
    ``inspect_fovs`` (and ``sufficiency_report``) whose answers they are read
    with; ``None`` (the default) takes the analysis groups these tables
    resolve to by default, and ``[]`` measures them over each whole stratum.
    ``n_jobs`` runs the novelty checks of the draws in that many worker
    processes (``-1``: every CPU); results are identical for every value
    (bit for bit when BLAS is single-threaded; see ``morphocoverage._parallel``).
    """
    check_n_jobs(n_jobs)
    if not isinstance(novelty, SupportNoveltyEstimator):
        raise TypeError("novelty must be a SupportNoveltyEstimator")
    if not isinstance(control, RepresentationControl):
        raise TypeError("control must be a RepresentationControl")
    for value, label in ((fovs_seen, "fovs_seen"), (draws, "draws")):
        if not is_integer(value) or value < 1:
            raise ValueError(f"{label} must be a positive integer")
    if not is_integer(random_state):
        raise TypeError("random_state must be an integer seed")
    fovs_seen, draws, random_state = int(fovs_seen), int(draws), int(random_state)
    pair_cols = list(pair_cols)
    settings = {
        "control": control.parameters,
        "novelty": dict(novelty.parameters),
        "novelty_definition": novelty_definition(novelty, list(feature_cols)),
        "pair_cols": pair_cols,
        "fovs_seen": fovs_seen,
        "draws": draws,
        "random_state": random_state,
    }
    columns = ["stratum", "candidate", "judged_new", "candidates", "not_judged", "rate"]
    if control.positive is None:
        empty = pd.DataFrame(columns=columns)
        return ControlResult(
            empty, empty.copy(), "not assessed", tuple(pair_cols), {**settings, "reason": "none"}
        )
    data = validate_input_tables(
        objects,
        fovs=fovs,
        feature_cols=feature_cols,
        fov_col=fov_col,
        # Identity only: the strata come from pair_cols, never from these columns.
        group_cols=list(identity_cols),
        observation_unit=observation_unit,
    )
    missing = [c for c in pair_cols if c not in data.fovs.columns]
    if missing:
        raise ValueError(f"pair_cols missing from the FOV table: {missing}")
    lofo_cols = _lofo_columns(data, lofo_group_cols)
    positive_mask = _select(data.fovs, control.positive)
    if control.reference is not None:
        base_mask = _select(data.fovs, control.reference)
        if (base_mask & positive_mask).any():
            raise ValueError("a FOV cannot be both a positive and a reference control")
    else:
        base_mask = ~positive_mask
    strata = data.fovs[pair_cols].drop_duplicates().to_dict("records") if pair_cols else [{}]
    subsets = FovSubsets(data)
    observed = np.bincount(subsets.positions[subsets.positions >= 0], minlength=len(data.fovs)) > 0
    rows, skipped, planted_skipped = [], [], []
    positive_empty = 0
    # Every random choice is made here, in stratum and draw order; only the
    # novelty checks, which use none, go to worker processes. One stratum's
    # draws at a time keeps memory to one stratum.
    with worker_pool(n_jobs) as pool:
        for stratum in strata:
            # One generator per stratum, so adding a plate never changes another plate's rates.
            rng = group_rng(random_state, stratum)
            inside = pd.Series(True, index=data.fovs.index)
            for column, value in stratum.items():
                inside &= data.fovs[column].eq(value)
            base = np.flatnonzero((inside & base_mask).to_numpy())
            declared = np.flatnonzero((inside & positive_mask).to_numpy())
            # An empty positive field cannot show the phenotype: it is counted apart,
            # never as a miss (P1-2).
            positive = declared[observed[declared]]
            positive_empty += len(declared) - len(positive)
            label = where(stratum) if stratum else "all FOVs"
            counts = {kind: [0, 0, 0] for kind in CANDIDATES}  # new, judged, not judged
            # Judged as inspect_fovs judges a field, within its analysis group; measured
            # wherever the fields exist, also in a stratum the draws below cannot use.
            reference_groups = _groups_of(data.fovs, base, lofo_cols)
            lofo = _leave_one_out(
                novelty, subsets, reference_groups, positive, control.reference is not None
            )
            planted_lofo = []
            if control.planted_objects is not None and len(positive):
                donors = [data.objects.loc[subsets.positions == row] for row in positive]
                if _planting_skipped(donors, control.planted_objects) is None:
                    # Its own generator, so the draws below keep their random choices.
                    jobs = _planted_leave_one_out(
                        novelty,
                        subsets,
                        reference_groups,
                        donors,
                        control.planted_objects,
                        group_rng(random_state, stratum, 1),
                    )
                    planted_lofo = map_units(_judge, jobs, n_jobs, pool)
            # The seen fields hold observations: an empty field adds nothing to the reference.
            seeing = base[observed[base]]
            if len(positive) == 0 or len(seeing) <= fovs_seen:
                reason = (
                    "no positive-control FOV with observations"
                    if len(positive) == 0
                    else f"{len(seeing)} reference FOVs with {_observations(data)}, not more "
                    f"than the {fovs_seen} seen"
                )
                skipped.append({**stratum, "stratum": label, "reason": reason})
                judged = []
            else:
                jobs, gap = _draws(
                    novelty, subsets, data, control, seeing, base, positive, fovs_seen, draws, rng
                )
                if gap is not None:
                    # Said, not silently absent: the planted rate was declared.
                    planted_skipped.append({**stratum, "stratum": label, "reason": gap})
                judged = map_units(_judge, jobs, n_jobs, pool)
            for draw in [*judged, lofo, *planted_lofo]:
                for kind, (new, total, not_judged) in draw.items():
                    counts[kind][0] += new
                    counts[kind][1] += total
                    counts[kind][2] += not_judged
            rows += _stratum_rows(stratum, counts)
    table = pd.DataFrame(rows, columns=[*pair_cols, *columns])
    pooled = (
        table.groupby("candidate", sort=False)[["judged_new", "candidates", "not_judged"]]
        .sum()
        .reset_index()
    )
    pooled["rate"] = [
        new / total if total else None
        for new, total in zip(pooled["judged_new"], pooled["candidates"], strict=True)
    ]
    measured = set(pooled.loc[pooled["candidates"] > 0, "candidate"])
    # The status says what was actually measured, not what was declared.
    status = "assessed" if {"positive", "reference"} <= measured else "positive only"
    if "positive" not in measured:
        status = "not assessed"
    settings = {
        **settings,
        "lofo_group_cols": lofo_cols,
        "skipped_strata": skipped,
        "planted_skipped": planted_skipped,
        "positive_empty": positive_empty,
        "reason": "unusable" if skipped else None,
        "observation_unit": data.observation_unit,
    }
    return ControlResult(table, pooled, status, tuple(pair_cols), settings)


def _observations(data: ValidatedInput) -> str:
    """``cells``, ``objects``: the plural of the observation unit."""
    return unit_plural(data.observation_unit)


def _draws(
    novelty: SupportNoveltyEstimator,
    subsets: FovSubsets,
    data: ValidatedInput,
    control: RepresentationControl,
    seeing: np.ndarray,
    base: np.ndarray,
    positive: np.ndarray,
    fovs_seen: int,
    draws: int,
    rng: np.random.Generator,
) -> tuple[list[tuple], str | None]:
    """The novelty checks of one stratum's draws, and why nothing was planted there (or None).

    Each draw sees ``fovs_seen`` reference fields holding observations; the
    other reference fields, empty ones included, are the reference candidates
    and the hosts of planted objects.
    """
    positive_input = subsets.take(positive)
    donors = [data.objects.loc[subsets.positions == row] for row in positive]
    plant = control.planted_objects is not None
    gap = None
    if plant:
        gap = _planting_skipped(donors, control.planted_objects)
        plant = gap is None
    jobs = []
    for _ in range(draws):
        order = rng.permutation(len(seeing))
        seen_rows = seeing[order[:fovs_seen]]
        seen = subsets.take(seen_rows)
        # The unseen fields holding observations, in the draw's order, then the empty ones.
        rest = np.concatenate([seeing[order[fovs_seen:]], base[~np.isin(base, seeing)]])
        candidates = {"positive": positive_input}
        if control.reference is not None:
            candidates["reference"] = subsets.take(rest)
        if plant:
            planted = _planted(subsets, rest, donors, control.planted_objects, rng)
            if planted is not None:
                candidates["planted"] = planted
        jobs.append((novelty, seen, candidates))
    return jobs, gap


__all__ = [
    "CANDIDATES",
    "ControlResult",
    "RepresentationControl",
    "control_summary",
    "representation_control",
]

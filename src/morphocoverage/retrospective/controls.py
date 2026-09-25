"""Representation controls: could these features recognize the phenotype I care about?

Roadmap Step 6. A "nothing new" answer is only as good as the features it was
computed on: a phenotype they cannot separate is never new. A declared control
tests this on the dataset itself:

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

from .._parallel import check_n_jobs, map_units, worker_pool
from ..evaluator import FovSubsets, restrict_input
from ..schema import validate_input_tables
from ..strategies.support_novelty import SupportNoveltyEstimator
from ._text import group_identity, percent, unit_plural, where

CANDIDATES = ("reference", "positive", "planted")


def _selector(value, label) -> dict[str, tuple] | None:
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
            if isinstance(self.planted_objects, bool) or not isinstance(self.planted_objects, int):
                raise TypeError("planted_objects must be a positive integer")
            if self.planted_objects < 1:
                raise ValueError("planted_objects must be a positive integer")
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
    ``planted``: reference FOVs with planted positive objects), with
    ``judged_new``, ``candidates`` and ``rate``; ``pooled`` the same over
    strata. ``status`` is ``"assessed"``, ``"positive only"`` or
    ``"not assessed"``.
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


def _planted(subsets: FovSubsets, host_rows, donors: list[pd.DataFrame], size, rng):
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


def _judge(novelty, seen, candidates):
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
) -> ControlResult:
    """Detection and false-alarm rates of ``novelty`` for the declared controls.

    ``fovs_seen`` is how many reference FOVs are "already imaged" in each draw,
    ``draws`` how many random draws per stratum. See the module docstring.
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
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{label} must be a positive integer")
    if isinstance(random_state, bool) or not isinstance(random_state, int):
        raise TypeError("random_state must be an integer seed")
    pair_cols = list(pair_cols)
    settings = {
        "control": control.parameters,
        "novelty": dict(novelty.parameters),
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
        group_cols=[],
        observation_unit=observation_unit,
    )
    missing = [c for c in pair_cols if c not in data.fovs.columns]
    if missing:
        raise ValueError(f"pair_cols missing from the FOV table: {missing}")
    positive_mask = _select(data.fovs, control.positive)
    if control.reference is not None:
        base_mask = _select(data.fovs, control.reference)
        if (base_mask & positive_mask).any():
            raise ValueError("a FOV cannot be both a positive and a reference control")
    else:
        base_mask = ~positive_mask
    strata = data.fovs[pair_cols].drop_duplicates().to_dict("records") if pair_cols else [{}]
    subsets = FovSubsets(data)
    rows, skipped = [], []
    # Every random choice is made here, in stratum and draw order; only the
    # novelty checks, which use none, go to worker processes. One stratum's
    # draws at a time keeps memory to one stratum.
    with worker_pool(n_jobs) as pool:
        for stratum in strata:
            # One generator per stratum, so adding a plate never changes another plate's rates.
            rng = np.random.default_rng([random_state, group_identity(stratum)])
            inside = pd.Series(True, index=data.fovs.index)
            for column, value in stratum.items():
                inside &= data.fovs[column].eq(value)
            base = np.flatnonzero((inside & base_mask).to_numpy())
            positive = np.flatnonzero((inside & positive_mask).to_numpy())
            label = where(stratum) if stratum else "all FOVs"
            if len(positive) == 0 or len(base) <= fovs_seen:
                reason = (
                    "no positive-control FOV"
                    if len(positive) == 0
                    else f"{len(base)} reference FOVs, not more than the {fovs_seen} seen"
                )
                skipped.append({**stratum, "stratum": label, "reason": reason})
                continue
            positive_input = subsets.take(positive)
            donors = [data.objects.loc[subsets.positions == row] for row in positive]
            jobs = []
            for _ in range(draws):
                order = rng.permutation(len(base))
                seen = subsets.take(base[order[:fovs_seen]])
                rest = base[order[fovs_seen:]]
                candidates = {"positive": positive_input}
                if control.reference is not None:
                    candidates["reference"] = subsets.take(rest)
                if control.planted_objects is not None:
                    planted = _planted(subsets, rest, donors, control.planted_objects, rng)
                    if planted is not None:
                        candidates["planted"] = planted
                jobs.append((novelty, seen, candidates))
            counts = {kind: [0, 0, 0] for kind in CANDIDATES}  # new, judged, not judged
            for draw in map_units(_judge, jobs, n_jobs, pool):
                for kind, (new, total, not_judged) in draw.items():
                    counts[kind][0] += new
                    counts[kind][1] += total
                    counts[kind][2] += not_judged
            for kind, (new, total, not_judged) in counts.items():
                if total or not_judged:
                    rows.append(
                        {
                            **stratum,
                            "stratum": label,
                            "candidate": kind,
                            "judged_new": new,
                            "candidates": total,
                            "not_judged": not_judged,
                            "rate": new / total if total else None,
                        }
                    )
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
        "skipped_strata": skipped,
        "reason": "unusable" if skipped else None,
        "observation_unit": data.observation_unit,
    }
    return ControlResult(table, pooled, status, tuple(pair_cols), settings)


def control_summary(result: ControlResult) -> str:
    """ "Can these features see the declared phenotype?" in sentences."""
    skipped = result.settings.get("skipped_strata", [])
    cells = unit_plural(result.settings.get("observation_unit", "object"))
    if result.status == "not assessed":
        declared = result.settings.get("reason") != "none"
        why = (
            "a control was declared, but no stratum could be judged ("
            + "; ".join(f"{s['stratum']}: {s['reason']}" for s in skipped)
            + ")"
            if declared and skipped
            else "a control was declared, but no check could be made (missing values?)"
            if declared
            else "no control was declared"
        )
        return (
            f"Representation sensitivity: not assessed: {why}. A 'nothing new' answer says "
            "nothing about phenotypes these features cannot see."
        )
    rates = result.pooled.set_index("candidate")
    seen = result.settings["fovs_seen"]
    lines = [f"Can these features see the declared phenotype? (after {seen} reference fields)"]
    positive = rates.loc["positive"]
    lines.append(
        f"  Fields of the declared positive control were recognized as different in "
        f"{percent(positive['rate'])} of {int(positive['candidates'])} checks."
    )
    if "planted" in rates.index:
        planted = rates.loc["planted"]
        planted_n = result.settings["control"]["planted_objects"]
        lines.append(
            f"  Ordinary reference fields with {planted_n} positive-control {cells} added were "
            f"recognized in {percent(planted['rate'])} of {int(planted['candidates'])} checks."
        )
    if "reference" in rates.index and rates.loc["reference", "candidates"] > 0:
        reference = rates.loc["reference"]
        lines.append(
            f"  Reference fields were called different in {percent(reference['rate'])} of "
            f"{int(reference['candidates'])} checks (false alarms)."
        )
    elif result.status == "positive only" and "reference" in rates.index:
        lines.append("  False-alarm rate: not assessed (no reference field could be judged).")
    else:
        lines.append("  False-alarm rate: not assessed (no reference control was declared).")
    if len(result.table["stratum"].unique()) > 1:
        by = result.table[result.table["candidate"] == "positive"]
        lines.append(
            "  Positive control per "
            + "/".join(result.pair_cols)
            + ": "
            + ", ".join(f"{r.stratum}: {percent(r.rate)}" for r in by.itertuples())
            + "."
        )
    if skipped:
        lines.append(
            "  Not judged: " + "; ".join(f"{s['stratum']} ({s['reason']})" for s in skipped) + "."
        )
    lines.append(
        "  This holds for this definition of new and these features, on these controls; "
        "a phenotype unlike the control may be seen better or worse."
    )
    return "\n".join(lines)


__all__ = [
    "CANDIDATES",
    "ControlResult",
    "RepresentationControl",
    "control_summary",
    "representation_control",
]

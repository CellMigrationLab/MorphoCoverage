"""The common design skeleton: biological repeats -> conditions -> wells -> fields -> cells.

Every outcome generator (``scalar``, ``embedding``, ``populations``, ``sequential``) draws its
values on a ``Layout`` built here, so the design factors of the protocol
(docs/common/validation.md, "Grid") are written once:

- repeats, the conditions each repeat holds (``presence``: complete, connected incomplete or
  disconnected designs), wells per condition and repeat, recorded in the tables or not
  (``wells_recorded``: an unrecorded well still adds its effect);
- fields per well (``fields``, with ``field_imbalance`` the coefficient of variation of a
  well's field count) and cells per field (negative binomial, mean ``cells``, extra
  dispersion ``cell_dispersion``; exactly ``cells`` with ``fixed_cells``, as one image per
  field), a treatment changing the cell number (``count_effect``),
  informative cluster size (``informative_size``: the log cell count rises with the field's
  latent effect), empty fields recorded or absent (``empty_fields``, ``empty_recorded``);
- missingness of whole fields: MCAR, MAR (on the acquisition order, which is observed),
  MNAR (on the field's own outcome, applied by the generator through ``drop_mnar``) and
  "condition" (QC exclusion correlated with condition: treated fields only);
- the acquisition: imaging sessions (``sessions``: a gap between two sets of repeats, a
  session confounded with the treatment, or one plate per well) and the order of the fields
  within a well, with an AR(1) autocorrelation of the field effects along it.

The latent effects are standard normal at every level (``z_repeat``, ``z_well``,
``z_field``); a generator scales them by its own variance shares. Every draw comes from the
generator passed in, so a dataset is a pure function of its seed.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import qmc

from morphocoverage.metadata import CONDITION_COLUMN, REPEAT_COLUMN, TECHNICAL_REPEAT_COLUMN

CONTROL = "control"
MISSING = ("none", "mcar", "mar", "mnar", "condition")
SESSIONS = ("none", "repeats", "confounded", "plates")
ORDER_COLUMN = "acquisition_order"
FIELD_COLUMN = "position_id"


@dataclass(frozen=True)
class Design:
    """One experimental design (see the module); every field has the reference design's
    default value (docs/common/validation.md, grid A, the centre point)."""

    repeats: int = 4
    conditions: tuple[str, ...] = (CONTROL, "treated")
    presence: tuple[tuple[str, ...], ...] | None = None
    wells: int = 1
    wells_recorded: bool = True
    fields: int = 15
    field_imbalance: float = 0.0
    cells: float = 20.0
    cell_dispersion: float = 0.3
    fixed_cells: bool = False
    count_effect: float = 1.0
    informative_size: float = 0.0
    empty_fields: float = 0.0
    empty_recorded: bool = True
    missing: str = "none"
    missing_rate: float = 0.0
    sessions: str = "none"
    autocorrelation: float = 0.0

    def __post_init__(self) -> None:
        if self.repeats < 1 or self.wells < 1 or self.fields < 1:
            raise ValueError("repeats, wells and fields must be at least 1")
        if self.missing not in MISSING:
            raise ValueError(f"missing must be one of {MISSING}")
        if self.sessions not in SESSIONS:
            raise ValueError(f"sessions must be one of {SESSIONS}")
        if self.presence is not None and len(self.presence) != self.repeats:
            raise ValueError("presence must list the conditions of every repeat")
        if not 0 <= self.autocorrelation < 1 or not 0 <= self.missing_rate < 1:
            raise ValueError("autocorrelation and missing_rate must be in [0, 1)")

    def held(self, repeat: int) -> tuple[str, ...]:
        """The conditions imaged in repeat ``repeat`` (0-based)."""
        return self.conditions if self.presence is None else self.presence[repeat]


def design_from(params: dict[str, Any]) -> Design:
    """A ``Design`` from the design keys of ``params`` (the other keys ignored), lists made
    tuples so the design stays hashable."""
    names = {f.name for f in fields(Design)}
    given = {k: v for k, v in params.items() if k in names}
    for key in ("conditions", "presence"):
        if isinstance(given.get(key), list):
            value = given[key]
            given[key] = tuple(tuple(v) if isinstance(v, list) else v for v in value)
    return Design(**given)


def connected_incomplete(conditions: tuple[str, ...], repeats: int) -> tuple[tuple[str, ...], ...]:
    """A connected incomplete design: repeat ``r`` holds the control and two consecutive
    other conditions (cyclically), so every pair is linked through shared repeats but most
    pairs never meet directly."""
    control, *others = conditions
    held = []
    for r in range(repeats):
        pair = {others[r % len(others)], others[(r + 1) % len(others)]}
        held.append(tuple(c for c in conditions if c == control or c in pair))
    return tuple(held)


def disconnected(conditions: tuple[str, ...], repeats: int) -> tuple[tuple[str, ...], ...]:
    """Two experiment sets sharing no condition: the first half of the repeats holds the first
    half of ``conditions``, the rest the others (so a contrast across the sets is not
    estimable)."""
    half_c, half_r = len(conditions) // 2, repeats // 2
    return tuple(conditions[:half_c] if r < half_r else conditions[half_c:] for r in range(repeats))


@dataclass(frozen=True)
class Layout:
    """The fields of one simulated experiment: one row per field acquired (also the empty
    ones), with its condition, repeat, well (``well`` always, ``technical_repeat`` as
    recorded), ``position_id``, acquisition order, session, cell count and latent effects
    ``z_repeat``, ``z_well``, ``z_field`` (standard normal, AR(1) along the order)."""

    design: Design
    fields: pd.DataFrame

    def cells(self) -> np.ndarray:
        """Each cell's row in ``fields`` (one entry per cell, fields in order)."""
        return np.repeat(np.arange(len(self.fields)), self.fields["n_cells"].to_numpy())


def _field_counts(design: Design, wells: int, rng: np.random.Generator) -> np.ndarray:
    """The fields of each of ``wells`` wells: ``fields``, or gamma-spread around it."""
    if design.field_imbalance <= 0:
        return np.full(wells, design.fields)
    shape = 1.0 / design.field_imbalance**2
    drawn = design.fields * rng.gamma(shape, 1.0 / shape, size=wells)
    return np.maximum(1, np.round(drawn)).astype(int)


def _ar1(counts: np.ndarray, rho: float, rng: np.random.Generator) -> np.ndarray:
    """Standard normal values for wells of ``counts`` fields, AR(1) with lag-one correlation
    ``rho`` along each well's order (independent between wells)."""
    z = rng.standard_normal(int(counts.sum()))
    if rho <= 0:
        return z
    out = z.copy()
    start = np.concatenate([[0], np.cumsum(counts)[:-1]])
    first = np.zeros(len(z), dtype=bool)
    first[start[counts > 0]] = True
    for k in range(1, len(z)):
        if not first[k]:
            out[k] = rho * out[k - 1] + np.sqrt(1 - rho**2) * z[k]
    return out


def _session(design: Design, repeat: int, condition: str, well: int) -> str:
    """The imaging session of a well (see ``SESSIONS``)."""
    if design.sessions == "repeats":
        return "s1" if repeat >= design.repeats // 2 else "s0"
    if design.sessions == "confounded":
        return "s0" if condition == design.conditions[0] else "s1"
    if design.sessions == "plates":
        return f"plate{repeat}_{condition}_{well}"
    return "s0"


def _wells(design: Design, rng: np.random.Generator) -> pd.DataFrame:
    """One row per well: its repeat, condition, latent effects and field count."""
    z_repeat = rng.standard_normal(design.repeats)
    rows = []
    for r in range(design.repeats):
        for condition in design.held(r):
            counts = _field_counts(design, design.wells, rng)
            for w in range(design.wells):
                rows.append(
                    {
                        CONDITION_COLUMN: condition,
                        REPEAT_COLUMN: f"R{r + 1}",
                        "well": f"R{r + 1}_{condition}_w{w + 1}",
                        TECHNICAL_REPEAT_COLUMN: f"w{w + 1}",
                        "session": _session(design, r, condition, w),
                        "z_repeat": z_repeat[r],
                        "z_well": rng.standard_normal(),
                        "count": int(counts[w]),
                    }
                )
    return pd.DataFrame(rows)


def layout(design: Design, rng: np.random.Generator) -> Layout:
    """Draw the fields and cell counts of one experiment (see the module); MNAR missingness
    is left to the outcome generator (``drop_mnar``)."""
    wells = _wells(design, rng)
    counts = wells["count"].to_numpy()
    frame = wells.loc[np.repeat(wells.index, counts)].drop(columns="count")
    frame = frame.reset_index(drop=True)
    order = np.arange(len(frame)) - np.repeat(np.cumsum(counts) - counts, counts)
    frame[ORDER_COLUMN] = order
    frame[FIELD_COLUMN] = frame["well"] + "_f" + (order + 1).astype(str)
    frame["z_field"] = _ar1(counts, design.autocorrelation, rng)
    frame["treated"] = frame[CONDITION_COLUMN] != design.conditions[0]
    mean = design.cells * np.where(frame["treated"], design.count_effect, 1.0)
    tilt = design.informative_size
    mean = mean * np.exp(tilt * frame["z_field"].to_numpy() - tilt**2 / 2)
    if design.cell_dispersion > 0 and not design.fixed_cells:
        shape = 1.0 / design.cell_dispersion
        mean = mean * rng.gamma(shape, 1.0 / shape, size=len(frame))
    n_cells = np.round(mean).astype(int) if design.fixed_cells else rng.poisson(mean)
    n_cells[rng.random(len(frame)) < design.empty_fields] = 0
    frame["n_cells"] = n_cells
    frame = _missing(design, frame, rng)
    if not design.empty_recorded:
        frame = frame[frame["n_cells"] > 0]
    return Layout(design, frame.reset_index(drop=True))


def _missing(design: Design, frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """``frame`` without the fields lost to MCAR, MAR or condition-related missingness."""
    rate = design.missing_rate
    if design.missing in ("none", "mnar") or rate == 0:
        return frame
    if design.missing == "mcar":
        chance = np.full(len(frame), rate)
    elif design.missing == "mar":
        # Later fields are lost more often (a run cut short): the order is observed.
        rank = frame.groupby("well")[ORDER_COLUMN].rank(pct=True).to_numpy()
        chance = np.clip(2 * rate * rank, 0, 1)
    else:  # condition: QC exclusion of treated fields only
        chance = np.where(frame["treated"], rate, 0.0)
    return frame[rng.random(len(frame)) >= chance]


def drop_mnar(
    frame: pd.DataFrame, score: np.ndarray, rate: float, rng: np.random.Generator
) -> np.ndarray:
    """Which fields of ``frame`` to keep under MNAR loss: a field is lost with a chance
    rising with its own outcome ``score`` (``2 rate`` times its rank, mean ``rate``)."""
    rank = pd.Series(score).rank(pct=True).to_numpy()
    return rng.random(len(frame)) >= np.clip(2 * rate * rank, 0, 1)


def tables(
    lay: Layout, values: pd.DataFrame, keep: np.ndarray | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The MorphoBricks observation and FOV tables of ``lay`` with one row of ``values`` per
    cell (``keep``: the cells kept, after segmentation loss). The well is written in
    ``technical_repeat`` only when the design records it; empty fields stay in the FOV
    table."""
    fovs = lay.fields
    columns = [CONDITION_COLUMN, REPEAT_COLUMN, FIELD_COLUMN, ORDER_COLUMN]
    if lay.design.wells_recorded:
        columns.insert(2, TECHNICAL_REPEAT_COLUMN)
    rows = lay.cells()
    objects = fovs.iloc[rows][[c for c in columns if c != ORDER_COLUMN]].reset_index(drop=True)
    objects["label_id"] = objects.groupby(FIELD_COLUMN).cumcount() + 1
    objects = pd.concat([objects, values.reset_index(drop=True)], axis=1)
    if keep is not None:
        objects = objects[keep].reset_index(drop=True)
    return objects, fovs[columns].reset_index(drop=True)


def latin_hypercube(
    ranges: dict[str, tuple[float, float]], points: int, seed: int
) -> list[dict[str, float]]:
    """``points`` settings of the factors of ``ranges`` (each ``(low, high)``) on a seeded
    Latin hypercube (grid B, screening): integers stay integers when both bounds are."""
    if not ranges:
        return []
    sampler = qmc.LatinHypercube(d=len(ranges), seed=seed)
    unit = sampler.random(points)
    out = []
    for row in unit:
        point: dict[str, float] = {}
        for value, (name, (low, high)) in zip(row, ranges.items(), strict=True):
            drawn = low + value * (high - low)
            whole = isinstance(low, int) and isinstance(high, int)
            point[name] = round(drawn) if whole else float(drawn)
        out.append(point)
    return out


__all__ = [
    "CONTROL",
    "Design",
    "Layout",
    "connected_incomplete",
    "design_from",
    "disconnected",
    "drop_mnar",
    "latin_hypercube",
    "layout",
    "tables",
]

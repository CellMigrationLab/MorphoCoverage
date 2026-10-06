"""Effect preservation: with ``k`` fields a side, is the difference from the control kept?

The first multi-group outcome (issue #81, phase 1; owner, 2026-10-05): for one
declared feature, each condition against the declared control, biological
repeat by biological repeat, judged on fields that the ``k`` drawn ones never
include. It is a standalone retrospective analysis: it is not an
``OutcomeMetric``, and ``optimise`` and the report do not use it yet.

Design, per biological repeat and condition (the sides as ``condition_contrast``
pairs them: the condition's analysis groups of the repeat against the control's):

1. **The effect** is the one the condition analysis reports per repeat:
   Cohen's d on the observations, condition minus control
   (``statistics.cohen_d``). Dividing by the control mean of the repeat (the
   normalised values) leaves it unchanged, so it needs no normalisation.
2. **The held-back reference.** For each of ``n_splits`` splits, a fixed share
   (``reservoir_fraction``, rounded up, at least one field and at most all but
   one) of each side's acquired fields is held back, at random, whatever their
   well. The reference d is Cohen's d of the held-back fields of both sides.
   It is the same for every ``k`` of a split, so the curve is not partly a
   curve of reference quality (the design of ``sampling_reliability``).
3. **Is the reference clear?** Its 95% interval comes from resampling the
   held-back fields (technical repeats, then their fields;
   ``contrast_sampling.sampling_row``, the interval question 2 uses). A split
   whose interval includes 0, or whose held-back fields hold observations in
   fewer than two fields on a side, has no reference: the difference is not
   established there, so a reduced acquisition cannot be judged against it.
   A ``k`` with no such split is **not evaluable**, never "not preserved".
4. **The reduced acquisitions.** For each ``k`` in ``fov_counts`` and each of
   ``n_draws`` draws, ``k`` fields a side are drawn at random, without
   replacement, from the fields not held back, as a representative
   acquisition of ``k`` fields would have them, and d_k is Cohen's d of their
   observations. A drawn field without observations is acquired effort.
5. **Preserved** when d_k has the sign of the reference and lies within
   ``tolerance`` of it, in units of d. A draw whose d is undefined (fewer than
   two observations on a side) is not preserved. The share is the mean over
   the splits with a reference of each split's share of draws.

The default tolerance, 0.2, is Cohen's "small" effect: a reduced acquisition
preserves the difference when it misreads it by less than a small effect. It
is a provisional preset, printed with the result; the right tolerance depends
on the effect the experiment must resolve, so it is a parameter.

Limits: draws and splits describe one repeat's fields, not independent
replicates; whether the difference holds across repeats is question 9. The
held-back reference is itself a finite sample, so the shares understate how
often ``k`` fields give the repeat's true difference. Draws treat the fields of
a side as exchangeable, as random-position sampling does. Every draw is seeded
from the contrast's labels, the split and ``k``, so adding a ``k``, a condition
or a repeat never changes another's draws.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .._checks import as_count, check_seed, is_integer, is_number
from .._control import check_comparisons
from .._former_repeat import one_warning, read_former
from .._seeds import group_rng
from .._style import DASH, table, wrap
from .._wording import column_word, count, number, percent, rate
from ..metadata import CONDITION_COLUMN, REPEAT_COLUMN, TECHNICAL_REPEAT_COLUMN
from ..sampling_unit._common import groups, validated
from ..sampling_unit._units import (
    contrasts_of,
    control_label,
    describe_groups,
    reference_of,
    resolve_columns,
    side_units,
    sides,
)
from ..sampling_unit.contrast_sampling import LEVEL, N_BOOT, sampling_row
from ..statistics import cohen_d

DEFAULT_FOV_COUNTS = (2, 3, 5, 8, 10, 15, 20)
# Cohen's "small" effect: a reduced acquisition may misread d by less than this.
DEFAULT_TOLERANCE = 0.2
DEFAULT_RESERVOIR_FRACTION = 0.5

OK = "ok"
UNCLEAR = "not evaluable: the held-back fields' interval of d includes 0"
SHALLOW = "not evaluable: fewer than two held-back fields with observations on a side"
BEYOND = "beyond the fields acquired: fewer than k fields a side outside the held-back ones"
NO_CONTROL = "not evaluable: no control fields in this repeat"

TABLE_COLUMNS = [
    REPEAT_COLUMN,
    "condition",
    "control_condition",
    "fovs",
    "share",
    "draws",
    "splits",
    "evaluable_splits",
    "reference_d",
    "d_all",
    "stream_fovs",
    "reference_fovs",
    "status",
]
REFERENCE_COLUMNS = [
    REPEAT_COLUMN,
    "condition",
    "split",
    "reference_d",
    "d_low",
    "d_high",
    "reference_fovs",
    "stream_fovs",
    "status",
]


@dataclass(frozen=True)
class EffectPreservation:
    """How often ``k`` fields a side preserve each repeat's difference from the control.

    - ``table``: one row per biological repeat, condition and ``k``
      (``fovs``, fields a side): ``share``, the share of draws that preserved
      the held-back difference (mean over the ``evaluable_splits`` of the
      ``splits``), ``draws`` judged, the mean ``reference_d`` over those
      splits, ``d_all`` (Cohen's d of every field, as the condition analysis
      reports it), the fields a side outside and inside the held-back share
      (``stream_fovs``, ``reference_fovs``, the smaller side) and ``status``
      (``"ok"``, or why the share is not given: ``UNCLEAR``, ``SHALLOW``,
      ``BEYOND``, ``NO_CONTROL``).
    - ``references``: one row per biological repeat, condition and split: the
      held-back ``reference_d``, its 95% interval (``d_low``, ``d_high``), the
      fields a side and whether the split has a reference (``status``).
    - ``settings``: the arguments, as used.
    """

    table: pd.DataFrame
    references: pd.DataFrame
    settings: dict[str, Any]

    def summary(self) -> str:
        """The shares in one table across contrasts, then what each status means, once."""
        return _summary(self)


def _from_setup(setup: Any, name: str, value: Any, attribute: str) -> Any:
    """``value``, or the setup's ``attribute`` when ``value`` is ``None`` and a setup is given."""
    if value is not None or setup is None:
        return value
    if not hasattr(setup, attribute):
        raise TypeError(f"setup must be a workflows.Setup; it has no {attribute!r} for {name}")
    return getattr(setup, attribute)


def _check_settings(
    fov_counts: Sequence[int],
    n_draws: object,
    n_splits: object,
    reservoir_fraction: object,
    tolerance: object,
) -> tuple[list[int], int, int, float, float]:
    """The settings, checked: sorted distinct counts, draws, splits, fraction, tolerance."""
    counts = [fov_counts] if is_integer(fov_counts) else list(fov_counts)
    if not counts or not all(is_integer(k) and int(k) >= 1 for k in counts):
        raise ValueError(f"fov_counts must be positive integers, not {fov_counts!r}")
    if not is_number(reservoir_fraction) or not 0 < float(reservoir_fraction) < 1:
        raise ValueError(f"reservoir_fraction must be between 0 and 1, not {reservoir_fraction!r}")
    if not is_number(tolerance) or not 0 <= float(tolerance) < math.inf:
        raise ValueError(f"tolerance must be a non-negative number of d units, not {tolerance!r}")
    return (
        sorted({int(k) for k in counts}),
        as_count(n_draws, "n_draws"),
        as_count(n_splits, "n_splits"),
        float(reservoir_fraction),
        float(tolerance),
    )


def _held_back(n: int, fraction: float) -> int:
    """Fields held back of ``n``: the share rounded up, at least 1 and at most ``n - 1``
    (``ReservoirSplit``'s rule); 0 when a side has fewer than two fields."""
    if n < 2:
        return 0
    return min(n - 1, max(1, math.ceil(fraction * n)))


def _split_side(
    units: list[list[np.ndarray]], fraction: float, rng: np.random.Generator
) -> tuple[list[list[np.ndarray]], list[np.ndarray]]:
    """A side's held-back fields, by unit (for the reference's interval), and its stream."""
    labelled = [(u, f) for u, unit in enumerate(units) for f in unit]
    chosen = set(rng.permutation(len(labelled))[: _held_back(len(labelled), fraction)].tolist())
    reference: list[list[np.ndarray]] = [[] for _ in units]
    stream = []
    for i, (u, values) in enumerate(labelled):
        if i in chosen:
            reference[u].append(values)
        else:
            stream.append(values)
    return [unit for unit in reference if unit], stream


def _pooled(fields_: Sequence[np.ndarray]) -> np.ndarray:
    """The observations of ``fields_`` together (an empty array without any)."""
    return np.concatenate(list(fields_)) if len(fields_) else np.array([], dtype=float)


def _reference(
    t_units: list[list[np.ndarray]],
    c_units: list[list[np.ndarray]],
    rng: np.random.Generator,
) -> dict[str, Any]:
    """The held-back d, its interval, and whether the split has a reference (``status``)."""
    t_fields = [f for u in t_units for f in u]
    c_fields = [f for u in c_units for f in u]
    d = cohen_d(_pooled(t_fields), _pooled(c_fields))
    filled = min(sum(len(f) > 0 for f in t_fields), sum(len(f) > 0 for f in c_fields))
    row = {"reference_d": d, "d_low": np.nan, "d_high": np.nan}
    if filled < 2:
        return {**row, "status": SHALLOW}
    sampled = sampling_row(t_units, c_units, d, rng, N_BOOT, LEVEL, min_fields=2)
    row.update(d_low=sampled["d_low"], d_high=sampled["d_high"])
    return {**row, "status": OK if sampled["resolved"] else UNCLEAR}


def _preserved_share(
    t_stream: list[np.ndarray],
    c_stream: list[np.ndarray],
    k: int,
    reference_d: float,
    tolerance: float,
    n_draws: int,
    rng: np.random.Generator,
) -> float:
    """The share of ``n_draws`` draws of ``k`` stream fields a side that preserve the d."""
    kept = 0
    for _ in range(n_draws):
        treated = _pooled([t_stream[i] for i in rng.choice(len(t_stream), k, replace=False)])
        control = _pooled([c_stream[i] for i in rng.choice(len(c_stream), k, replace=False)])
        d = cohen_d(treated, control)
        kept += bool(
            np.isfinite(d)
            and np.sign(d) == np.sign(reference_d)
            and abs(d - reference_d) <= tolerance
        )
    return kept / n_draws


def _contrast_rows(
    base: dict[str, Any],
    t_units: list[list[np.ndarray]],
    c_units: list[list[np.ndarray]],
    settings: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The rows of one repeat and condition: one per ``k``, and one per split."""
    seed = {**base, "question": "effect preservation"}
    random_state = settings["random_state"]
    t_all = [f for u in t_units for f in u]
    c_all = [f for u in c_units for f in u]
    d_all = cohen_d(_pooled(t_all), _pooled(c_all))
    splits, references = [], []
    for split in range(settings["n_splits"]):
        rng = group_rng(random_state, seed, split)
        t_reference, t_stream = _split_side(t_units, settings["reservoir_fraction"], rng)
        c_reference, c_stream = _split_side(c_units, settings["reservoir_fraction"], rng)
        reference = _reference(t_reference, c_reference, rng)
        held = min(sum(len(u) for u in t_reference), sum(len(u) for u in c_reference))
        stream = min(len(t_stream), len(c_stream))
        splits.append((reference, t_stream, c_stream))
        references.append(
            {
                **base,
                "split": split,
                **reference,
                "reference_fovs": held,
                "stream_fovs": stream,
            }
        )
    evaluable = [s for s in splits if s[0]["status"] == OK]
    rows = []
    for k in settings["fov_counts"]:
        stream = min(r["stream_fovs"] for r in references)
        row = {
            **base,
            "fovs": k,
            "share": np.nan,
            "draws": 0,
            "splits": settings["n_splits"],
            "evaluable_splits": len(evaluable),
            "reference_d": (
                float(np.mean([s[0]["reference_d"] for s in evaluable])) if evaluable else np.nan
            ),
            "d_all": d_all,
            "stream_fovs": stream,
            "reference_fovs": min(r["reference_fovs"] for r in references),
        }
        if k > stream:
            rows.append({**row, "status": BEYOND})
            continue
        if not evaluable:
            every_shallow = all(s[0]["status"] == SHALLOW for s in splits)
            rows.append({**row, "status": SHALLOW if every_shallow else UNCLEAR})
            continue
        shares = []
        for split, (reference, t_stream, c_stream) in enumerate(splits):
            if reference["status"] != OK:
                continue
            rng = group_rng(random_state, seed, split, k)
            shares.append(
                _preserved_share(
                    t_stream,
                    c_stream,
                    k,
                    reference["reference_d"],
                    settings["tolerance"],
                    settings["n_draws"],
                    rng,
                )
            )
        rows.append(
            {
                **row,
                "share": round(float(np.mean(shares)), 12),
                "draws": settings["n_draws"] * len(shares),
                "status": OK,
            }
        )
    return rows, references


@one_warning
def effect_preservation(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    *,
    feature: str | None = None,
    control: Mapping[str, Any] | None = None,
    fov_counts: Sequence[int] = DEFAULT_FOV_COUNTS,
    tolerance: float = DEFAULT_TOLERANCE,
    n_draws: int = 200,
    n_splits: int = 5,
    reservoir_fraction: float = DEFAULT_RESERVOIR_FRACTION,
    random_state: int | None = None,
    condition_col: str = CONDITION_COLUMN,
    repeat_col: str | None = REPEAT_COLUMN,
    technical_col: str | None = TECHNICAL_REPEAT_COLUMN,
    fov_col: str | None = None,
    group_cols: Sequence[str] | None = None,
    observation_unit: str | None = None,
    setup: Any = None,
    comparisons: Sequence[tuple[Any, Any]] | None = None,
) -> EffectPreservation:
    """How often ``k`` fields a side preserve each repeat's difference from the control.

    ``feature`` is the one feature judged and ``control`` names the control
    condition by FOV-table value (``{"condition": "DMSO"}``), as in
    ``sampling_unit.condition_contrast``, whose sides, repeat and
    technical-repeat columns this reads the same way. A ``workflows.Setup``
    passed as ``setup`` fills what is left out: its ``readout``, ``control``,
    ``fov_col``, ``group_cols``, ``observation_unit`` and ``random_state``.
    ``fov_counts`` are the fields a side ``k``; ``tolerance`` is how far, in
    units of d, a reduced d may lie from the held-back one and still preserve
    it. ``comparisons`` (or the setup's, when ``control`` is not passed) replace the control
    by ``(condition, reference)`` pairs, each a row label ``condition vs reference``. See
    the module for the design; nothing here feeds ``optimise``.
    """
    feature = _from_setup(setup, "feature", feature, "readout")
    if comparisons is None and control is None:
        comparisons = getattr(setup, "comparisons", None)
    control = _from_setup(setup, "control", control, "control")
    fov_col = _from_setup(setup, "fov_col", fov_col, "fov_col")
    group_cols = _from_setup(setup, "group_cols", group_cols, "group_cols")
    observation_unit = _from_setup(setup, "observation_unit", observation_unit, "observation_unit")
    random_state = _from_setup(setup, "random_state", random_state, "random_state")
    random_state = check_seed(0 if random_state is None else random_state)
    if not isinstance(feature, str) or not feature:
        raise TypeError("feature must be one column name")
    if comparisons is not None:
        comparisons = check_comparisons(comparisons)
    control, repeat_col = read_former(control=control, repeat_col=repeat_col)
    if comparisons is None and (not isinstance(control, Mapping) or not control):
        raise ValueError("control must be a non-empty mapping of FOV-table column to value")
    if comparisons is None and condition_col not in control:
        raise ValueError(f"control must name the condition column {condition_col!r}")
    counts, n_draws, n_splits, fraction, tolerance = _check_settings(
        fov_counts, n_draws, n_splits, reservoir_fraction, tolerance
    )
    data = validated(objects, fovs, [feature], fov_col, group_cols, observation_unit)
    repeat_col, technical_col = resolve_columns(
        data, condition_col, repeat_col, technical_col, control
    )
    all_groups = groups(data)
    described = describe_groups(all_groups, condition_col, repeat_col, control, comparisons)
    settings = {
        "feature": feature,
        "control": dict(control or {}),
        "comparisons": list(comparisons or []),
        "condition_col": condition_col,
        "repeat_col": repeat_col,
        "technical_col": technical_col,
        "fov_counts": counts,
        "tolerance": tolerance,
        "n_draws": n_draws,
        "n_splits": n_splits,
        "reservoir_fraction": fraction,
        "level": LEVEL,
        "random_state": random_state,
        "observation_unit": data.observation_unit,
    }
    rows, references = [], []
    for repeat, condition in contrasts_of(described):
        treated, controls = sides(described, repeat, condition)
        base = {
            REPEAT_COLUMN: repeat,
            "condition": condition,
            "control_condition": control_label(described, controls)
            or reference_of(described, condition, control, condition_col),
        }
        if not controls:
            rows += [{**base, "fovs": k, "status": NO_CONTROL} for k in counts]
            continue
        t_units, _ = side_units(all_groups, feature, treated, technical_col)
        c_units, _ = side_units(all_groups, feature, controls, technical_col)
        contrast_rows, contrast_references = _contrast_rows(base, t_units, c_units, settings)
        rows += contrast_rows
        references += contrast_references
    return EffectPreservation(
        pd.DataFrame(rows, columns=TABLE_COLUMNS),
        pd.DataFrame(references, columns=REFERENCE_COLUMNS),
        settings,
    )


# How a status reads in the summary's table, and the sentence that says it once below.
_CELLS = {UNCLEAR: "unclear", SHALLOW: "too few", NO_CONTROL: "no control", BEYOND: DASH}
_NOTES = {
    UNCLEAR: "unclear: the held-back fields do not establish the difference (its 95% "
    "interval includes 0 in every split that holds enough fields), so a reduced acquisition "
    "cannot be judged against it.",
    SHALLOW: "too few: the held-back fields hold observations in fewer than two fields on a side.",
    NO_CONTROL: "no control: this repeat has no control fields.",
    BEYOND: f"{DASH}: more fields a side than remain outside the held-back ones.",
}


def _summary(result: EffectPreservation) -> str:
    """The summary of ``EffectPreservation`` (see its ``summary``)."""
    settings, frame = result.settings, result.table
    feature = settings["feature"]
    title = f"Is the difference from the control preserved with fewer fields? Feature {feature!r}."
    if frame.empty:
        return f"{title}\nEvery analysis group is the control: no condition to compare."
    control = ", ".join(str(v) for v in settings["control"].values())
    against = "its own control" if settings.get("comparisons") else f"the control {control}"
    intro = (
        f"Per {column_word(REPEAT_COLUMN)}, each condition against {against}: k "
        f"fields a side drawn at random from the fields not held back "
        f"({count(settings['n_draws'], 'draw')} in each of "
        f"{count(settings['n_splits'], 'split')}), against Cohen's d of the held-back fields, "
        f"{rate(settings['reservoir_fraction'])} of each side's. A draw preserves the difference "
        f"when its d has the held-back d's sign and lies within {number(settings['tolerance'])} "
        "of it. The table gives the share of draws that did, by fields a side."
    )
    counts = settings["fov_counts"]
    columns = ["condition", "repeat", "d", *(str(k) for k in counts)]
    lines, seen = [], set()
    by_contrast = frame.groupby([REPEAT_COLUMN, "condition"], sort=False, dropna=False)
    for (repeat, condition), rows in by_contrast:
        by_k = rows.set_index("fovs")
        cells = []
        for k in counts:
            status = by_k.loc[k, "status"]
            seen.add(status)
            cells.append(percent(by_k.loc[k, "share"]) if status == OK else _CELLS[status])
        d_all = rows["d_all"].iloc[0]
        lines.append(
            [
                condition,
                DASH if pd.isna(repeat) else repeat,
                f"{d_all:.2f}" if np.isfinite(d_all) else DASH,
                *cells,
            ]
        )
    notes = [_NOTES[status] for status in (UNCLEAR, SHALLOW, NO_CONTROL, BEYOND) if status in seen]
    partly = frame[(frame["status"] == OK) & (frame["evaluable_splits"] < frame["splits"])]
    if len(partly):
        notes.append(
            "Where some splits did not establish the difference, a share is over the splits "
            "that did (evaluable_splits in the table)."
        )
    closing = (
        "d is Cohen's d of every field. The held-back fields are a finite sample, so these "
        "shares understate how often k fields give the repeat's true difference. Each share "
        "describes one repeat's fields; whether the difference holds across "
        f"{column_word(REPEAT_COLUMN)}s is question 9."
    )
    parts = [wrap(title), wrap(intro), "", table(lines, columns, align="llr" + "r" * len(counts))]
    parts += ["", *(wrap(note) for note in notes), wrap(closing)]
    return "\n".join(parts)


__all__ = [
    "BEYOND",
    "DEFAULT_FOV_COUNTS",
    "DEFAULT_TOLERANCE",
    "NO_CONTROL",
    "OK",
    "SHALLOW",
    "UNCLEAR",
    "EffectPreservation",
    "effect_preservation",
]

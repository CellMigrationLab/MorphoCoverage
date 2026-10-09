"""Protocol-level reliability: what FOV number should the protocol say for experiments like this?

Question P2 of ``docs/optimise/optimise.md``. It starts from each sample's reliability
curve (``sampling_reliability``, one analysis group per sample). A sample
counts as **adequately represented** at ``k`` FOVs when its reliability reaches
``threshold`` (for example ``preserved >= 0.9``: in at least 90% of random
acquisitions of ``k`` FOVs, what you wanted was preserved in held-back FOVs;
a reliability of exactly 90% reaches it, whatever the rounding of its average).

Reported, per stratum (for example per condition; pooling across strata is an
explicit choice), criterion and ``k``:

- how many samples were adequately represented, out of how many could be
  evaluated at that depth, and how many could not, apart: ``k`` beyond their
  stream (``beyond_stream``), or a criterion that could not be judged against
  any of their held-back fields (``undefined``);
- the same per independent unit (biological repeat when declared, otherwise
  sample), from the declared ``SamplingHierarchy``;
- each sample's difficulty: the fewest listed FOVs from which it stays
  adequately represented at every larger listed count, with the difficult
  samples named. The median over samples counts a sample that never got
  there within the counts it could be judged at as needing more than all of
  them, so it is not the median of the easy samples only;
- a leave-one-unit-out check of a protocol chosen on the other units: the
  fewest FOVs at which at least ``protocol_share`` of their samples were
  adequate, and how the left-out unit fared at that number.

**Uncertainty counts independent units.** Subset draws and reservoir splits
estimate each sample's own behaviour; they are not replicates. The interval on
each share comes from resampling the declared independent units with
replacement (``statistics.bootstrap_fields``, each unit's samples as one field,
widened for few units), so it does not narrow when more draws or splits are
run for the same samples. It is never narrower than the exact Clopper-Pearson
interval over the units, in whole counts: a unit is a success when the
protocol serves it, at least ``protocol_share`` of its samples adequate, and a
failure otherwise (``scipy.stats.binomtest``; owner, 2026-10-05). When every
sample (or none) is adequate, resampling can only return that share again,
and the interval is then that exact one (5 of 5 units is compatible with a
true share of about 48%); just below, 19 of 20 samples in 5 units, it is no
narrower, rather than jumping to a much narrower bootstrap range. More
samples inside one unit never narrow it as if they were more units. It is
clipped to 0-100%. With few units the interval is wide, and the summary says
so.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .._checks import as_count, check_seed
from .._former_repeat import read_former
from .._frozen import owned
from .._seeds import group_rng
from ..metadata import column_names
from ..statistics import bootstrap_fields
from .hierarchy import SamplingHierarchy
from .protocol_text import FEW_UNITS, protocol_summary
from .reliability import ReliabilityResult

_CONFIDENCE = 0.95
# Reliabilities are shares of draws averaged over splits; a share of exactly the
# threshold must count as reaching it whatever the rounding of that average.
_TOLERANCE = 1e-9


@owned("table", "per_unit", "difficulty", "validation", "settings")
@dataclass(frozen=True)
class ProtocolResult:
    """Protocol curves, per-unit shares, sample difficulty and a leave-one-unit-out check.

    - ``table``: per stratum, criterion and ``fovs``: ``eligible`` samples,
      ``adequate`` samples, ``not_evaluated`` samples (``beyond_stream``: fewer
      fields than ``fovs``; ``undefined``: the criterion could not be judged
      there), ``partly_judged`` samples, ``share`` adequate, and
      ``share_low`` / ``share_high``, its 95% interval over the ``units``
      independent units (module docstring); ``all_units`` counts the units
      of the stratum.
    - ``per_unit``: the same counts per independent unit.
    - ``difficulty``: per sample, ``fields_needed`` (``<NA>`` when not reached
      within the listed counts it could be evaluated at) and ``max_fovs``.
    - ``validation``: per left-out unit, the ``chosen_fovs`` from the other
      units and the left-out unit's adequate / eligible samples there.
    """

    table: pd.DataFrame
    per_unit: pd.DataFrame
    difficulty: pd.DataFrame
    validation: pd.DataFrame
    sample_cols: tuple[str, ...]
    unit_cols: tuple[str, ...]
    strata: tuple[str, ...]
    settings: dict[str, Any] = field(default_factory=dict)

    def summary(self, detail: str = "short") -> str:
        """The protocol answer: ``"short"`` one table across criteria, ``"full"`` every field
        count per stratum and criterion (``protocol_text``); each caveat once, at the end."""
        return protocol_summary(self, detail)


def _check(
    per_sample: ReliabilityResult,
    hierarchy: SamplingHierarchy,
    threshold: float,
    strata: Sequence[str],
    protocol_share: float,
    n_boot: int,
    random_state: int,
) -> tuple[list[str], list[str], list[str]]:
    """The sample, unit and stratum columns, after refusing any invalid argument."""
    if not isinstance(per_sample, ReliabilityResult):
        raise TypeError("per_sample must be a ReliabilityResult from sampling_reliability")
    if not isinstance(hierarchy, SamplingHierarchy):
        raise TypeError("hierarchy must be a SamplingHierarchy")
    for value, label in ((threshold, "threshold"), (protocol_share, "protocol_share")):
        if isinstance(value, bool) or not 0 < float(value) <= 1:
            raise ValueError(f"{label} must be a share in (0, 1]")
    as_count(n_boot, "n_boot")
    check_seed(random_state)
    if isinstance(strata, str):
        # A bare name would be iterated letter by letter.
        raise TypeError("strata must be a sequence of column names, for example ['condition']")
    (strata,) = read_former(strata=strata)  # the former name ``repeat``
    groups = set(per_sample.group_cols)
    sample_cols = hierarchy.unit_key(hierarchy.sample)
    unit_cols = hierarchy.unit_key(hierarchy.independent_unit)
    missing = [c for c in [*sample_cols, *strata] if c not in groups]
    if missing:
        raise ValueError(
            f"columns {missing} are not analysis-group columns of the reliability result; "
            "run sampling_reliability with one analysis group per sample"
        )
    strata = list(column_names(strata, "strata"))
    if any(c in sample_cols for c in strata):
        raise ValueError("strata must be columns above the sample, for example condition")
    return sample_cols, unit_cols, strata


def _samples(
    per_sample: ReliabilityResult, sample_cols: list[str], strata: list[str], threshold: float
) -> pd.DataFrame:
    """One row per sample, criterion and FOV count, with the adequacy flag.

    ``beyond_stream`` marks a sample that had fewer fields than the count
    (``splits`` 0 in the curve, or, for a curve without it, a count above its
    stream); a sample neither adequate-or-not nor beyond its stream could not
    be judged there.
    """
    curve = per_sample.curve
    identity = [*strata, *sample_cols]
    if curve.groupby(identity + ["criterion", "fovs"], dropna=False).size().max() > 1:
        raise ValueError(
            "several analysis groups map to one sample; declare the columns that identify "
            "a sample in the hierarchy, or add them as strata"
        )
    rows = curve[[*identity, "criterion", "fovs", "preserved", "stream_fovs"]].copy()
    rows["adequate"] = rows["preserved"].astype(float) >= float(threshold) - _TOLERANCE
    if "splits" in curve.columns:
        beyond = pd.to_numeric(curve["splits"], errors="coerce").eq(0)
    else:
        beyond = curve["fovs"].astype(float) > curve["stream_fovs"].astype(float)
    rows["beyond_stream"] = (beyond & rows["preserved"].isna()).to_numpy()
    # How many hold-outs could judge the sample at this depth: a reliability
    # judged on 3 of 10 hold-outs (a rare state absent from the others' reservoir)
    # is named, not silently counted like a fully judged one.
    if {"defined_splits", "splits"} <= set(curve.columns):
        judged = curve["defined_splits"].astype(float) / curve["splits"].replace(0, np.nan)
        rows["partly_judged"] = (judged < 1).to_numpy()
    else:
        rows["partly_judged"] = False
    return rows


def _unit_ids(frame: pd.DataFrame, unit_cols: list[str]) -> pd.Series:
    """Each row's independent unit as one label, for example ``P1 / 2024_01_05``."""
    return frame[unit_cols].astype(str).agg(" / ".join, axis=1)


def _exact(successes: int, units: int) -> tuple[float, float]:
    """The exact Clopper-Pearson interval of ``successes`` of ``units`` whole units.

    ``scipy.stats.binomtest``'s exact interval: integer counts only, each end
    a one-sided bound at ``(1 + confidence) / 2``, closed at 0 or 1.
    """
    interval = stats.binomtest(int(successes), int(units)).proportion_ci(
        confidence_level=_CONFIDENCE, method="exact"
    )
    return float(interval.low), float(interval.high)


def _bootstrap(
    block: pd.DataFrame, n_boot: int, rng: np.random.Generator, protocol_share: float
) -> tuple[float | None, float | None]:
    """95% interval of the pooled share of adequate samples, over whole units.

    The interval of ``statistics.bootstrap_fields`` over the units (each
    unit's samples as one field, widened for few units), never narrower than
    the exact interval of the units served (``_exact``: a unit counts when at
    least ``protocol_share`` of its samples are adequate) and clipped to
    [0, 1]. At a share of 0 or 1 every resample returns that share, a
    zero-width interval that would look exact, so the exact interval alone is
    given. One unit gives no interval.
    """
    adequacy = block.groupby("unit")["adequate"]
    units = [own.to_numpy(dtype=float) for _, own in adequacy]
    if len(units) < 2:
        # One unit gives a zero-width interval that would look exact.
        return None, None
    served = sum(bool(own.mean() >= protocol_share - _TOLERANCE) for own in units)
    exact_low, exact_high = _exact(served, len(units))
    share = float(block["adequate"].mean())
    if share in (0.0, 1.0):
        return exact_low, exact_high
    _, low, high = bootstrap_fields(units, np.mean, n_boot=n_boot, rng=rng, confidence=_CONFIDENCE)
    if np.isnan(low):
        return exact_low, exact_high
    return max(0.0, min(low, exact_low)), min(1.0, max(high, exact_high))


def _fields_needed(rows: pd.DataFrame) -> int | None:
    """The fewest listed FOVs from which the sample stays adequate at every larger count."""
    needed = None
    for k, adequate in rows.sort_values("fovs")[["fovs", "adequate"]].itertuples(index=False):
        if adequate:
            needed = int(k) if needed is None else needed
        else:
            needed = None
    return needed


def _chosen(block: pd.DataFrame, protocol_share: float) -> int | None:
    """The fewest listed FOVs from which ``protocol_share`` of eligible samples stay adequate.

    The same "from here on" definition as ``fields_needed``, so a noisy dip
    at a larger count is not skipped over.
    """
    shares = block.groupby("fovs")["adequate"].mean().sort_index()
    chosen = None
    for k, share in shares.items():
        if share >= protocol_share:
            chosen = int(k) if chosen is None else chosen
        else:
            chosen = None
    return chosen


_TABLE_COLUMNS = (
    "eligible",
    "adequate",
    "partly_judged",
    "not_evaluated",
    "beyond_stream",
    "undefined",
    "share",
    "share_low",
    "share_high",
    "units",
    "all_units",
)


def protocol_reliability(
    per_sample: ReliabilityResult,
    hierarchy: SamplingHierarchy,
    *,
    threshold: float,
    strata: Sequence[str],
    protocol_share: float = 0.9,
    n_boot: int = 2000,
    random_state: int = 0,
) -> ProtocolResult:
    """How many samples are adequately represented at each FOV count, counted by independent unit.

    ``threshold`` is the per-sample reliability that counts as adequate
    (provisional preset 0.9; a reliability of at least ``threshold`` is
    adequate). ``strata`` names analysis-group columns kept apart (for
    example ``["condition"]``); ``[]`` pools every sample, which must be asked
    for. ``protocol_share`` is the share of samples a protocol should serve:
    the leave-one-unit-out check chooses a number by it, and a unit with at
    least that share adequate counts as served in the exact floor of the
    interval (module docstring). ``n_boot`` and
    ``random_state`` set the resampling of independent units (numpy integers
    accepted); each stratum has its own generator.
    """
    sample_cols, unit_cols, strata = _check(
        per_sample, hierarchy, threshold, strata, protocol_share, n_boot, random_state
    )
    n_boot, random_state = int(n_boot), int(random_state)
    samples = _samples(per_sample, sample_cols, strata, threshold)
    samples["unit"] = _unit_ids(samples, unit_cols)
    table, per_unit, difficulty, validation = [], [], [], []
    stratum_keys = [*strata, "criterion"]
    for key, block in samples.groupby(stratum_keys, sort=False, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        where = dict(zip(stratum_keys, key, strict=True))
        # One generator per stratum, so adding a stratum never changes another's interval.
        rng = group_rng(random_state, where)
        n_samples = block[sample_cols].drop_duplicates().shape[0]
        n_units = int(block["unit"].nunique())
        for k, rows in block.groupby("fovs", sort=True):
            eligible = rows[rows["preserved"].notna()]
            low, high = _bootstrap(eligible, n_boot, rng, float(protocol_share))
            beyond = int(rows["beyond_stream"].sum())
            table.append(
                {
                    **where,
                    "fovs": int(k),
                    "eligible": len(eligible),
                    "adequate": int(eligible["adequate"].sum()),
                    "partly_judged": int(eligible["partly_judged"].sum()),
                    "not_evaluated": n_samples - len(eligible),
                    "beyond_stream": beyond,
                    "undefined": n_samples - len(eligible) - beyond,
                    "share": float(eligible["adequate"].mean()) if len(eligible) else None,
                    "share_low": low,
                    "share_high": high,
                    "units": int(eligible["unit"].nunique()),
                    "all_units": n_units,
                }
            )
            for unit, own in eligible.groupby("unit", sort=False):
                per_unit.append(
                    {
                        **where,
                        "fovs": int(k),
                        "unit": unit,
                        "eligible": len(own),
                        "adequate": int(own["adequate"].sum()),
                    }
                )
        for sample, own in block.groupby(sample_cols, sort=False, dropna=False):
            sample = sample if isinstance(sample, tuple) else (sample,)
            evaluated = own[own["preserved"].notna()]
            difficulty.append(
                {
                    **where,
                    **dict(zip(sample_cols, sample, strict=True)),
                    "unit": own["unit"].iloc[0],
                    "fields_needed": _fields_needed(evaluated),
                    "max_fovs": int(evaluated["fovs"].max()) if len(evaluated) else None,
                }
            )
        eligible = block[block["preserved"].notna()]
        for unit in eligible["unit"].unique():
            others = eligible[eligible["unit"] != unit]
            left_out = eligible[eligible["unit"] == unit]
            chosen = _chosen(others, float(protocol_share)) if len(others) else None
            at = left_out[left_out["fovs"] == chosen] if chosen is not None else left_out.iloc[:0]
            validation.append(
                {
                    **where,
                    "left_out": unit,
                    "chosen_fovs": chosen,
                    "eligible": len(at),
                    "adequate": int(at["adequate"].sum()),
                }
            )
    # Explicit columns, so a result with nothing eligible anywhere still has them.
    stratum_cols = [*strata, "criterion"]
    difficulty_table = pd.DataFrame(
        difficulty, columns=[*stratum_cols, *sample_cols, "unit", "fields_needed", "max_fovs"]
    )
    difficulty_table["fields_needed"] = difficulty_table["fields_needed"].astype("Int64")
    validation_table = pd.DataFrame(
        validation,
        columns=[*stratum_cols, "left_out", "chosen_fovs", "eligible", "adequate"],
    )
    validation_table["chosen_fovs"] = validation_table["chosen_fovs"].astype("Int64")
    settings = {
        **dict(per_sample.settings),
        "threshold": float(threshold),
        "protocol_share": float(protocol_share),
        "hierarchy": hierarchy.parameters,
        "independent_unit": hierarchy.independent_unit,
        "n_boot": n_boot,
        "random_state_protocol": random_state,
    }
    return ProtocolResult(
        table=pd.DataFrame(table, columns=[*stratum_cols, "fovs", *_TABLE_COLUMNS]),
        per_unit=pd.DataFrame(
            per_unit, columns=[*stratum_cols, "fovs", "unit", "eligible", "adequate"]
        ),
        difficulty=_with_ceiling(difficulty_table, per_sample, sample_cols, strata, threshold),
        validation=validation_table,
        sample_cols=tuple(sample_cols),
        unit_cols=tuple(unit_cols),
        strata=tuple(strata),
        settings=settings,
    )


def _with_ceiling(
    difficulty: pd.DataFrame,
    per_sample: ReliabilityResult,
    sample_cols: list[str],
    strata: list[str],
    threshold: float,
) -> pd.DataFrame:
    """Add each sample's reliability ceiling: how often its whole stream was enough.

    ``ceiling_below_threshold`` marks samples whose held-back fields did not
    reach the criterion even with every available field in at least
    ``threshold`` of the hold-outs. Such a sample is not shown to need more
    fields than it has: its reference could not judge the criterion at all.
    Without a ceiling table the columns are missing values.
    """
    if not len(difficulty):
        difficulty["ceiling"] = pd.Series(dtype=float)
        difficulty["ceiling_below_threshold"] = pd.Series(dtype="boolean")
        return difficulty
    identity = [*strata, *sample_cols, "criterion"]
    ceiling = per_sample.ceiling
    if ceiling.empty or not set(identity) <= set(ceiling.columns):
        difficulty["ceiling"] = np.nan
        difficulty["ceiling_below_threshold"] = pd.array([None] * len(difficulty), dtype="boolean")
        return difficulty
    values = ceiling[[*identity, "whole_stream_preserved"]].drop_duplicates(subset=identity)
    merged = difficulty.merge(values, on=identity, how="left", validate="one_to_one")
    difficulty["ceiling"] = merged["whole_stream_preserved"].to_numpy(dtype=float)
    below = difficulty["ceiling"] < float(threshold)
    pairs = zip(difficulty["ceiling"], below, strict=True)
    flags = [None if pd.isna(c) else bool(b) for c, b in pairs]
    difficulty["ceiling_below_threshold"] = pd.array(flags, dtype="boolean")
    return difficulty


__all__ = ["FEW_UNITS", "ProtocolResult", "protocol_reliability", "protocol_summary"]

"""Protocol-level reliability: what FOV number should the protocol say for experiments like this?

Roadmap Step 4. It starts from each sample's reliability curve
(``sampling_reliability``, one analysis group per sample). A sample counts as
**adequately represented** at ``k`` FOVs when its reliability reaches
``threshold`` (for example ``preserved >= 0.9``: in at least 90% of random
acquisitions of ``k`` FOVs, what you wanted was preserved in held-back FOVs).

Reported, per stratum (for example per condition; pooling across strata is an
explicit choice), criterion and ``k``:

- how many samples were adequately represented, out of how many could be
  evaluated at that depth, and how many could not (``k`` beyond their stream);
- the same per independent unit (biological repeat when declared, otherwise
  sample), from the declared ``SamplingHierarchy``;
- each sample's difficulty: the fewest listed FOVs from which it stays
  adequately represented at every larger listed count, with the difficult
  samples named;
- a leave-one-unit-out check of a protocol chosen on the other units: the
  fewest FOVs at which at least ``protocol_share`` of their samples were
  adequate, and how the left-out unit fared at that number.

**Uncertainty counts independent units.** Subset draws and reservoir splits
estimate each sample's own behaviour; they are not replicates. The interval on
each share comes from resampling the declared independent units with
replacement (``n`` equal to the number of units), so it does not narrow when
more draws or splits are run for the same samples. With few units the interval
is wide, and the summary says so.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .hierarchy import SamplingHierarchy
from .reliability import ReliabilityResult

FEW_UNITS = 5


@dataclass(frozen=True)
class ProtocolResult:
    """Protocol curves, per-unit shares, sample difficulty and a leave-one-unit-out check.

    - ``table``: per stratum, criterion and ``fovs``: ``eligible`` samples,
      ``adequate`` samples, ``not_evaluated`` samples, ``share`` adequate, and
      ``share_low`` / ``share_high``, a percentile interval from resampling
      ``units`` independent units.
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

    def summary(self) -> str:
        """The protocol answer in sentences, per stratum and criterion."""
        return protocol_summary(self)


def _check(per_sample, hierarchy, threshold, strata, protocol_share, n_boot, random_state):
    if not isinstance(per_sample, ReliabilityResult):
        raise TypeError("per_sample must be a ReliabilityResult from sampling_reliability")
    if not isinstance(hierarchy, SamplingHierarchy):
        raise TypeError("hierarchy must be a SamplingHierarchy")
    for value, label in ((threshold, "threshold"), (protocol_share, "protocol_share")):
        if isinstance(value, bool) or not 0 < float(value) <= 1:
            raise ValueError(f"{label} must be a share in (0, 1]")
    if isinstance(n_boot, bool) or not isinstance(n_boot, int) or n_boot < 1:
        raise ValueError("n_boot must be a positive integer")
    if isinstance(random_state, bool) or not isinstance(random_state, int):
        raise TypeError("random_state must be an integer seed")
    groups = set(per_sample.group_cols)
    sample_cols = hierarchy.unit_key(hierarchy.sample)
    unit_cols = hierarchy.unit_key(hierarchy.independent_unit)
    missing = [c for c in [*sample_cols, *strata] if c not in groups]
    if missing:
        raise ValueError(
            f"columns {missing} are not analysis-group columns of the reliability result; "
            "run sampling_reliability with one analysis group per sample"
        )
    strata = list(strata)
    if any(c in sample_cols for c in strata):
        raise ValueError("strata must be columns above the sample, for example condition")
    return sample_cols, unit_cols, strata


def _samples(per_sample: ReliabilityResult, sample_cols, strata, threshold) -> pd.DataFrame:
    """One row per sample, criterion and FOV count, with the adequacy flag."""
    curve = per_sample.curve
    identity = [*strata, *sample_cols]
    if curve.groupby(identity + ["criterion", "fovs"], dropna=False).size().max() > 1:
        raise ValueError(
            "several analysis groups map to one sample; declare the columns that identify "
            "a sample in the hierarchy, or add them as strata"
        )
    rows = curve[[*identity, "criterion", "fovs", "preserved", "stream_fovs"]].copy()
    rows["adequate"] = rows["preserved"].astype(float) >= float(threshold)
    return rows


def _unit_ids(frame: pd.DataFrame, unit_cols: list[str]) -> pd.Series:
    return frame[unit_cols].astype(str).agg(" / ".join, axis=1)


def _bootstrap(block: pd.DataFrame, n_boot: int, rng: np.random.Generator):
    """Percentile interval of the pooled share of adequate samples, resampling units."""
    per_unit = block.groupby("unit")["adequate"].agg(["sum", "count"])
    units = len(per_unit)
    if units == 0:
        return None, None
    picks = rng.integers(0, units, size=(n_boot, units))
    adequate = per_unit["sum"].to_numpy(float)[picks].sum(axis=1)
    eligible = per_unit["count"].to_numpy(float)[picks].sum(axis=1)
    shares = adequate / eligible
    return float(np.quantile(shares, 0.025)), float(np.quantile(shares, 0.975))


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
    """The fewest listed FOVs at which ``protocol_share`` of eligible samples were adequate."""
    shares = block.groupby("fovs")["adequate"].mean().sort_index()
    good = shares[shares >= protocol_share]
    return None if good.empty else int(good.index[0])


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
    (provisional preset 0.9). ``strata`` names analysis-group columns kept
    apart (for example ``["condition"]``); ``[]`` pools every sample, which
    must be asked for. ``protocol_share`` is the share of samples a protocol
    should serve, used only for the leave-one-unit-out check. ``n_boot`` and
    ``random_state`` set the resampling of independent units.
    """
    sample_cols, unit_cols, strata = _check(
        per_sample, hierarchy, threshold, strata, protocol_share, n_boot, random_state
    )
    samples = _samples(per_sample, sample_cols, strata, threshold)
    samples["unit"] = _unit_ids(samples, unit_cols)
    rng = np.random.default_rng(random_state)
    table, per_unit, difficulty, validation = [], [], [], []
    stratum_keys = [*strata, "criterion"]
    for key, block in samples.groupby(stratum_keys, sort=False, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        where = dict(zip(stratum_keys, key, strict=True))
        n_samples = block[sample_cols].drop_duplicates().shape[0]
        for k, rows in block.groupby("fovs", sort=True):
            eligible = rows[rows["preserved"].notna()]
            low, high = _bootstrap(eligible, n_boot, rng)
            table.append(
                {
                    **where,
                    "fovs": int(k),
                    "eligible": len(eligible),
                    "adequate": int(eligible["adequate"].sum()),
                    "not_evaluated": n_samples - len(eligible),
                    "share": float(eligible["adequate"].mean()) if len(eligible) else None,
                    "share_low": low,
                    "share_high": high,
                    "units": int(eligible["unit"].nunique()),
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
    difficulty_table = pd.DataFrame(difficulty)
    if len(difficulty_table):
        difficulty_table["fields_needed"] = difficulty_table["fields_needed"].astype("Int64")
    validation_table = pd.DataFrame(validation)
    if len(validation_table):
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
        table=pd.DataFrame(table),
        per_unit=pd.DataFrame(per_unit),
        difficulty=difficulty_table,
        validation=validation_table,
        sample_cols=tuple(sample_cols),
        unit_cols=tuple(unit_cols),
        strata=tuple(strata),
        settings=settings,
    )


def _percent(value: float) -> str:
    return f"{100 * value:.0f}%"


def _match(frame: pd.DataFrame, where: dict) -> pd.DataFrame:
    for column, value in where.items():
        frame = frame[frame[column] == value]
    return frame


def protocol_summary(result: ProtocolResult) -> str:
    """ "For experiments like this one" in sentences."""
    labels = result.settings.get("labels", {})
    unit_word = result.settings.get("independent_unit", "unit")
    threshold = _percent(result.settings.get("threshold", 0.9))
    blocks = []
    keys = [*result.strata, "criterion"]
    for key, rows in result.table.groupby(keys, sort=False, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        where = dict(zip(keys, key, strict=True))
        stratum = ", ".join(f"{k}={where[k]}" for k in result.strata) or "all samples"
        criterion = labels.get(where["criterion"], where["criterion"])
        n_samples = int(rows["eligible"].max() + rows["not_evaluated"].min())
        n_units = int(rows["units"].max())
        lines = [
            (
                f"{stratum}: fields per sample for {criterion} "
                f"({n_samples} samples in {n_units} {unit_word}s; a sample counts when "
                f"{threshold} of random acquisitions were enough)"
            )
        ]
        units = _match(result.per_unit, where)
        for row in rows.sort_values("fovs").itertuples(index=False):
            if row.eligible == 0:
                lines.append(f"  {row.fovs} fields: no sample had that many fields to judge")
                continue
            per = units[units["fovs"] == row.fovs]
            by_unit = ", ".join(f"{u.adequate} of {u.eligible}" for u in per.itertuples())
            missing = (
                f"; {row.not_evaluated} could not be judged at this depth"
                if row.not_evaluated
                else ""
            )
            lines.append(
                f"  {row.fovs} fields: enough for {row.adequate} of {row.eligible} samples "
                f"({unit_word}s: {by_unit}; plausible range {_percent(row.share_low)}–"
                f"{_percent(row.share_high)}){missing}"
            )
        hard = _match(result.difficulty, where)
        needs_more = hard[hard["fields_needed"].isna()]
        if len(hard):
            known = hard["fields_needed"].dropna().astype(int)
            if len(known):
                lines.append(
                    f"  Fields needed per sample: median {int(np.median(known))}, "
                    f"range {known.min()}–{known.max()}."
                )
            if len(needs_more):
                named = "; ".join(
                    ", ".join(str(r[c]) for c in result.sample_cols)
                    + f" (more than {r['max_fovs']})"
                    for _, r in needs_more.iterrows()
                )
                lines.append(f"  Needs more fields than could be judged: {named}.")
        check = _match(result.validation, where)
        if len(check):
            chosen = check.dropna(subset=["chosen_fovs"])
            if len(chosen):
                served = int(chosen["adequate"].sum())
                total = int(chosen["eligible"].sum())
                lines.append(
                    f"  Choosing the number on the other {unit_word}s and checking it on the one "
                    f"left out: enough for {served} of {total} samples."
                )
        if n_units < FEW_UNITS:
            lines.append(
                f"  Uncertainty is large with only {n_units} {unit_word}s: add more "
                "before fixing a protocol."
            )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


__all__ = ["FEW_UNITS", "ProtocolResult", "protocol_reliability", "protocol_summary"]

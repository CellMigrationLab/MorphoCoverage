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
more draws or splits are run for the same samples. When every sample (or
none) is adequate, resampling can only return that share again, so the
interval is instead the exact Clopper-Pearson interval over the units (5 of 5
units is compatible with a true share of about 48%). With few units the
interval is wide, and the summary says so.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .._seeds import group_rng
from .._wording import count
from ..bounds import interval
from ..statistics import bootstrap_fields
from ._text import fields, percent, plural, rows_of, sample_word
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


def _check(
    per_sample: ReliabilityResult,
    hierarchy: SamplingHierarchy,
    threshold: float,
    strata: Sequence[str],
    protocol_share: float,
    n_boot: int,
    random_state: int,
) -> tuple[list[str], list[str], list[str]]:
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
    if isinstance(strata, str):
        # A bare name would be iterated letter by letter.
        raise TypeError("strata must be a sequence of column names, for example ['condition']")
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


def _samples(
    per_sample: ReliabilityResult, sample_cols: list[str], strata: list[str], threshold: float
) -> pd.DataFrame:
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
    return frame[unit_cols].astype(str).agg(" / ".join, axis=1)


def _bootstrap(
    block: pd.DataFrame, n_boot: int, rng: np.random.Generator
) -> tuple[float | None, float | None]:
    """95% interval of the pooled share of adequate samples, resampling whole units.

    The percentile interval of ``statistics.bootstrap_fields`` over the units
    (each unit's samples as one field). At a share of 0 or 1 every resample
    returns that share, a zero-width interval that would look exact, so the
    exact Clopper-Pearson interval over the units is given instead. One unit
    gives no interval.
    """
    adequacy = block.groupby("unit")["adequate"]
    units = [own.to_numpy(dtype=float) for _, own in adequacy]
    if len(units) < 2:
        # One unit gives a zero-width interval that would look exact.
        return None, None
    share = float(block["adequate"].mean())
    if share in (0.0, 1.0):
        return interval(int(share) * len(units), len(units), 0.95)
    _, low, high = bootstrap_fields(units, np.mean, n_boot=n_boot, rng=rng, confidence=0.95)
    return (None, None) if np.isnan(low) else (low, high)


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
            low, high = _bootstrap(eligible, n_boot, rng)
            table.append(
                {
                    **where,
                    "fovs": int(k),
                    "eligible": len(eligible),
                    "adequate": int(eligible["adequate"].sum()),
                    "partly_judged": int(eligible["partly_judged"].sum()),
                    "not_evaluated": n_samples - len(eligible),
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


def protocol_summary(result: ProtocolResult) -> str:
    """ "For experiments like this one" in sentences."""
    labels = result.settings.get("labels", {})
    hierarchy = result.settings.get("hierarchy", {})
    # The column's own name, readable: "plate", "technical repeat".
    unit = str(result.settings.get("independent_unit", "unit")).replace("_", " ")
    units_word, sample = plural(unit), sample_word(hierarchy)
    samples_word = plural(sample)
    threshold = percent(result.settings.get("threshold", 0.9))
    blocks = []
    keys = [*result.strata, "criterion"]
    for key, rows in result.table.groupby(keys, sort=False, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        where = dict(zip(keys, key, strict=True))
        stratum = ", ".join(f"{k}={where[k]}" for k in result.strata) or "all samples"
        criterion = labels.get(where["criterion"], where["criterion"])
        n_samples = int(rows["eligible"].max() + rows["not_evaluated"].min())
        n_units = int(rows["all_units"].max())
        lines = [
            (
                f"{stratum}: fields per {sample} for {criterion} "
                f"({count(n_samples, sample, samples_word)} in "
                f"{count(n_units, unit, units_word)}; a {sample} "
                f"counts when {threshold} of random acquisitions were enough)"
            )
        ]
        units = rows_of(result.per_unit, where)
        for row in rows.sort_values("fovs").itertuples(index=False):
            if row.eligible == 0:
                lines.append(f"  {fields(row.fovs)}: no {sample} had that many fields to judge")
                continue
            per = units[units["fovs"] == row.fovs]
            by_unit = ", ".join(f"{u.adequate} of {u.eligible}" for u in per.itertuples())
            missing = (
                f"; {row.not_evaluated} could not be judged at this depth"
                if row.not_evaluated
                else ""
            )
            partly = getattr(row, "partly_judged", 0)
            if partly:
                missing += (
                    f"; {partly} judged on only some of the hold-outs (the others could not "
                    "judge the criterion)"
                )
            interval = (
                ""
                if row.share_low is None or pd.isna(row.share_low)
                else f"; plausible range {percent(row.share_low)}–{percent(row.share_high)}"
            )
            lines.append(
                f"  {fields(row.fovs)}: enough for {row.adequate} of {row.eligible} "
                f"{samples_word} ({units_word}: {by_unit}{interval}){missing}"
            )
        hard = rows_of(result.difficulty, where)
        never = hard[hard["max_fovs"].isna()]
        needs_more = hard[hard["fields_needed"].isna() & hard["max_fovs"].notna()]
        # A sample whose held-back fields were not enough even with every field is
        # capped by its reference, not short of fields; it is named apart.
        capped = hard["ceiling_below_threshold"].fillna(False).astype(bool) if len(hard) else None
        if len(hard):
            known = hard["fields_needed"].dropna().astype(int)
            if len(known):
                median = float(np.median(known))
                shown = f"{median:.0f}" if median.is_integer() else f"{median:.1f}"
                lines.append(
                    f"  Fields needed per {sample}: median {shown}, "
                    f"range {known.min()}–{known.max()}."
                )
            short = needs_more[~capped.loc[needs_more.index]]
            if len(short):
                named = "; ".join(
                    ", ".join(str(r[c]) for c in result.sample_cols)
                    + f" (more than {r['max_fovs']})"
                    for _, r in short.iterrows()
                )
                lines.append(f"  Needs more fields than could be judged: {named}.")
            reference_capped = hard[capped]
            if len(reference_capped):
                named = "; ".join(
                    ", ".join(str(r[c]) for c in result.sample_cols)
                    + f" (all fields enough in {percent(float(r['ceiling']))} of hold-outs)"
                    for _, r in reference_capped.iterrows()
                )
                lines.append(
                    f"  For {len(reference_capped)} of {len(hard)} {samples_word} even every "
                    f"available field together was enough in fewer than {threshold} of "
                    "hold-outs: the held-back fields are themselves a sample and cap what can "
                    "be shown, so these are not counted as needing more fields: "
                    f"{named}."
                )
            if len(never):
                named = "; ".join(
                    ", ".join(str(r[c]) for c in result.sample_cols) for _, r in never.iterrows()
                )
                lines.append(f"  Could not be judged at any depth: {named}.")
        check = rows_of(result.validation, where)
        if len(check):
            chosen = check.dropna(subset=["chosen_fovs"])
            # A left-out unit whose samples have fewer fields than the chosen number
            # was not checked: it is named, never counted as "0 of 0".
            unchecked = chosen[chosen["eligible"] == 0]
            checked = chosen[chosen["eligible"] > 0]
            if len(checked):
                served = int(checked["adequate"].sum())
                total = int(checked["eligible"].sum())
                lines.append(
                    f"  Choosing the number on the other {units_word} and checking it on the "
                    f"one left out: enough for {served} of {total} {samples_word}."
                )
            if len(unchecked):
                named = "; ".join(
                    f"{r['left_out']} (chosen {int(r['chosen_fovs'])})"
                    for _, r in unchecked.iterrows()
                )
                lines.append(
                    f"  Not checked on {count(len(unchecked), unit, units_word)} whose "
                    f"{samples_word} had fewer fields than the number chosen on the others: "
                    f"{named}."
                )
            unchosen = len(check) - len(chosen)
            if unchosen and n_units < 2:
                lines.append(
                    f"  With a single {unit}, no number can be chosen on other {unit}s and "
                    "checked on it."
                )
            elif unchosen:
                lines.append(
                    f"  For {count(unchosen, unit, units_word)}, the others reached no number "
                    f"that served enough {samples_word}, so nothing was checked there."
                )
        if n_units < FEW_UNITS:
            lines.append(
                f"  Uncertainty is large with only {count(n_units, unit, units_word)}: add "
                "more before fixing a protocol."
            )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


__all__ = ["FEW_UNITS", "ProtocolResult", "protocol_reliability", "protocol_summary"]

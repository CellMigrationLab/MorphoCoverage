"""Protocol-level reliability in sentences: "for experiments like this one".

The wording of ``protocol_reliability`` (``protocol.py`` holds the
computation). ``protocol_summary`` and ``FEW_UNITS`` are re-exported there.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from .._wording import count, number
from ._text import column_word, fields, median_beyond, percent, plural, rows_of, sample_word

if TYPE_CHECKING:
    from .protocol import ProtocolResult

FEW_UNITS = 5


def protocol_summary(result: ProtocolResult) -> str:
    """ "For experiments like this one" in sentences."""
    labels = result.settings.get("labels", {})
    hierarchy = result.settings.get("hierarchy", {})
    # The column's own name, readable: "plate", "technical repeat".
    unit = column_word(result.settings.get("independent_unit", "unit"))
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
                f"counts when at least {threshold} of random acquisitions were enough)"
            )
        ]
        units = rows_of(result.per_unit, where)
        for row in rows.sort_values("fovs").itertuples(index=False):
            beyond = int(getattr(row, "beyond_stream", 0) or 0)
            undefined = int(row.not_evaluated) - beyond
            if row.eligible == 0:
                lines.append(f"  {fields(row.fovs)}: {_none_judged(sample, beyond, undefined)}")
                continue
            per = units[units["fovs"] == row.fovs]
            by_unit = ", ".join(f"{u.adequate} of {u.eligible}" for u in per.itertuples())
            missing = f"; {beyond} had fewer fields than that" if beyond else ""
            if undefined:
                missing += f"; {undefined} could not be judged at this depth"
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
            short = needs_more[~capped.loc[needs_more.index]]
            lines += _needed_line(hard, short, sample, samples_word)
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
                    f"  With a single {unit}, no number can be chosen on other {units_word} and "
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


def _none_judged(sample: str, beyond: int, undefined: int) -> str:
    """Why no sample could be judged at one depth: too few fields, or an undefined criterion."""
    if not undefined:
        return f"no {sample} had that many fields to judge"
    if not beyond:
        return (
            f"no {sample} could be judged at this depth (the criterion could not be judged "
            "against their held-back fields)"
        )
    return (
        f"no {sample} could be judged at this depth ({beyond} had fewer fields than that; "
        f"for {undefined} the criterion could not be judged against their held-back fields)"
    )


def _needed_line(
    hard: pd.DataFrame, short: pd.DataFrame, sample: str, samples_word: str
) -> list[str]:
    """The median fields needed per sample, a sample that never got there counting as more.

    Over the samples that got there (``fields_needed``) and those that need
    more fields than could be judged (``short``): these count as beyond every
    listed count (``_text.median_beyond``), so the median is not that of the
    easy samples only. Samples capped by their reference or never judged are
    left out, as the summary says separately.
    """
    known = hard["fields_needed"].dropna().astype(int)
    if not len(known) and not len(short):
        return []
    pool = [*known.tolist(), *([None] * len(short))]
    median = median_beyond(pool)
    spread = f"range {known.min()}–{known.max()}" if len(known) else ""
    if median is None:
        reached = f"{len(known)} of {count(len(pool), sample, samples_word)}"
        line = (
            f"  Fields needed per {sample}: median not reached within the listed counts "
            f"(only {reached} got there{'; ' + spread if spread else ''})."
        )
        return [line]
    others = f"; {len(short)} needing more counted as more" if len(short) else ""
    return [f"  Fields needed per {sample}: median {number(median)}, {spread}{others}."]


__all__ = ["FEW_UNITS", "protocol_summary"]

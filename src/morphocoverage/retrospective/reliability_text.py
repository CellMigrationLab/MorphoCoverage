"""The sampling reliability in sentences: "how many fields do you need?" per group.

The wording of ``sampling_reliability`` (``reliability.py`` holds the
computation). ``reliability_summary`` is re-exported there.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import pandas as pd

from .._wording import count
from ._text import fields, percent, skipped_lines, where

if TYPE_CHECKING:
    from .reliability import ReliabilityResult


def _ceiling_of(
    result: ReliabilityResult, group: Mapping[str, Any], criterion: str
) -> pd.DataFrame:
    """The ceiling row of one group and criterion (empty when there is none)."""
    match = result.ceiling
    if match.empty:
        return match
    match = match[match["criterion"] == criterion]
    for column, value in group.items():
        match = match[match[column] == value]
    return match


def _ceiling_value(match: pd.DataFrame, column: str) -> int | None:
    """An integer column of the ceiling row, or ``None`` when absent (a rebuilt result)."""
    if match.empty or column not in match.columns or pd.isna(match[column].iloc[0]):
        return None
    return int(match[column].iloc[0])


def reliability_summary(result: ReliabilityResult) -> str:
    """ "How many fields do you need?" in sentences, per group and criterion.

    With fewer distinct hold-outs than splits (few blocks), shares and spreads
    are said over splits, not hold-outs, since several splits held back the
    same fields. With unequal streams, the ceiling says the range of fields
    the whole streams had.
    """
    blocks = []
    keys = list(result.group_cols)
    labels = result.settings.get("labels", {})
    n_splits = int(result.settings.get("n_splits", 0))
    for key, rows in result.curve.groupby([*keys, "criterion"], sort=False, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        group = dict(zip(keys, key[: len(keys)], strict=True))
        criterion = key[-1]
        rows = rows.sort_values("fovs")
        stream = int(rows["stream_fovs"].max())
        match = _ceiling_of(result, group, criterion)
        splits = n_splits or int(rows["splits"].max())
        distinct = _distinct_holdouts(match)
        repeated = distinct is not None and distinct < splits
        # A share over splits that repeat hold-outs is not a share of distinct hold-outs.
        per, over_word = ("split", "splits") if repeated else ("hold-out", "hold-outs")
        lines = [
            f"{where(group)}: how many fields do you need for {labels.get(criterion, criterion)}?"
        ]
        for row in rows.itertuples(index=False):
            if row.splits == 0:
                lines.append(f"  {fields(row.fovs)}: more than the {stream} fields available")
                continue
            if row.preserved is None or pd.isna(row.preserved):
                lines.append(
                    f"  {fields(row.fovs)}: could not be judged "
                    f"({count(row.undefined_draws, 'draw')} undefined)"
                )
                continue
            spread = (
                ""
                if row.preserved_split_sd is None or pd.isna(row.preserved_split_sd)
                else f" ({_mc_se(row)}±{percent(row.preserved_split_sd)} across {over_word})"
            )
            if row.fovs == stream:
                # Every draw is the same acquisition: this point is the ceiling itself.
                spread = f" (all available fields: one acquisition per {per})"
            partial = []
            if n_splits and row.defined_splits < n_splits:
                partial.append(f"judged on {row.defined_splits} of {n_splits} {over_word}")
            if row.undefined_draws:
                total = row.undefined_draws + row.defined_draws
                partial.append(f"{percent(row.undefined_draws / total)} of draws not judged")
            note = f" [{'; '.join(partial)}]" if partial else ""
            lines.append(
                f"  {fields(row.fovs)}: enough in {percent(row.preserved)} of random "
                f"acquisitions{spread}{note}"
            )
        shortest = _ceiling_value(match, "shortest_stream_fovs")
        unequal = shortest is not None and shortest < stream
        if len(match) and match["whole_stream_preserved"].notna().iloc[0]:
            ceiling = match.iloc[0]
            defined = int(ceiling["defined_splits"])
            judged = (
                f" of the {defined} that could judge it" if defined < int(ceiling["splits"]) else ""
            )
            longest = int(ceiling["stream_fovs"])
            span = f"{shortest} to {longest}" if unequal else f"{longest}"
            lines.append(
                f"  All {span} available fields together were enough in "
                f"{percent(float(ceiling['whole_stream_preserved']))} of {over_word}{judged}: "
                "the held-back fields are themselves a sample, so the curve cannot be read "
                "above this."
            )
        elif result.ceiling.empty:
            # A result rebuilt without its ceiling table (for example from a saved curve)
            # would otherwise read as if every field count could reach 100%.
            lines.append(
                f"  How often all {stream} available fields together were enough was not "
                "computed: the held-back fields are themselves a sample, so a flat curve may "
                "be capped by them rather than by the fields."
            )
        lines.append(_basis_line(rows, match, splits, distinct, stream, shortest))
        if repeated:
            lines.append(
                f"  Warning: the {splits} splits repeat hold-outs (too few fields or blocks to "
                "hold back a different set each time): splits of the same hold-out differ "
                "only by the random draws, so the spread across splits is partly draw "
                f"noise, and the curve rests on {count(distinct, 'reference')}, not {splits}."
            )
        blocks.append("\n".join(lines))
    blocks += skipped_lines(result.skipped, keys)
    return "\n\n".join(blocks)


def _basis_line(
    rows: pd.DataFrame,
    match: pd.DataFrame,
    splits: int,
    distinct: int | None,
    stream: int,
    shortest: int | None,
) -> str:
    """What the curve of one group and criterion rests on: held-back fields and splits."""
    judged = rows[rows["splits"] > 0]
    if len(judged):
        smallest = int(judged["reservoir_fovs"].min())
    else:
        # Every listed count was beyond the streams: the reservoirs still held fields.
        smallest = _ceiling_value(match, "reservoir_fovs")
    over = (
        f"{count(splits, 'hold-out')}"
        if distinct is None
        else f"{count(splits, 'split')} over {count(distinct, 'distinct hold-out')}"
    )
    beyond = f"{stream} fields" + (
        f" ({shortest} in some splits)" if shortest is not None and shortest < stream else ""
    )
    held = "" if smallest is None else f" {smallest} or more held-back fields in"
    anything = rows["preserved"].notna().any() or (
        len(match) and match["whole_stream_preserved"].notna().iloc[0]
    )
    if not anything:
        amount = "fields" if smallest is None else f"{smallest} or more fields"
        return (
            f"  Held back {amount} in {over}; the criterion could not be judged against any "
            "of them, so nothing is said at any field count."
        )
    return (
        f"  Judged on{held or ''} {over}; nothing is said beyond {beyond}, and fields are "
        "treated as random draws. The shares are conservative by construction: the "
        "held-back fields are a sample too, and their own error counts against the drawn "
        "fields, so a field count read from this curve errs high."
    )


def _mc_se(row: object) -> str:
    """``Monte Carlo SE 5%; `` before the spread, or '' when the curve has no such column."""
    value = getattr(row, "preserved_mc_se", None)
    if value is None or pd.isna(value):
        return ""
    return f"Monte Carlo SE {percent(value)}; "


def _distinct_holdouts(ceiling: pd.DataFrame) -> int | None:
    """How many distinct hold-outs the group's splits used, when the ceiling table says."""
    if ceiling.empty or "distinct_holdouts" not in ceiling.columns:
        return None
    value = ceiling["distinct_holdouts"].iloc[0]
    return None if pd.isna(value) else int(value)


__all__ = ["reliability_summary"]

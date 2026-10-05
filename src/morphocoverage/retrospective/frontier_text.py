"""The frontier score in words (``frontier_summary``), kept apart from its computation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from .._caveats import assumption, caveat_lines
from .._style import Status, check_detail, group_columns, headline, table, worst, wrap
from .._wording import number
from ._text import fields, percent
from .effort import MIN_BOOT_SUCCESS
from .protocol_text import FEW_UNITS

if TYPE_CHECKING:
    from .frontier import FrontierResult

BASELINE = "fixed fields"
# Units from which the widened interval of the fields saved came close to its level in
# simulation (0.64 at 5 units, 0.80 at 12, 0.95 at 20 for a nominal 95%).
_MANY_UNITS = 20
_NEVER = "never got there at any setting tried"


def _setting(value: object) -> str:
    """A family setting as given: ``0.05``, ``12`` or a label such as ``strict``."""
    if isinstance(value, int | float) and not isinstance(value, bool):
        return f"{value:g}"
    return str(value)


def _number(value: float) -> str:
    """One decimal, without a negative zero."""
    value = round(float(value), 1)
    return f"{0.0 if value == 0 else value:.1f}"


def _risk(row: object) -> str:
    """``4%``, or ``0% of the 4 of 10 runs judged`` when some runs could not be judged
    (outcome undefined): they are left out of the risk, so its denominator is said."""
    judged, runs = getattr(row, "judged", None), getattr(row, "runs", None)
    text = percent(row.risk)
    if judged is None or runs is None or pd.isna(judged) or int(judged) >= int(runs):
        return text
    return f"{text} of the {int(judged)} of {int(runs)} runs judged"


def _lapse(row: object) -> str | None:
    """``met this risk at 2 (risk 0%) but not from there on: ...``, or None if it never met it."""
    met = getattr(row, "met_setting", None)
    if met is None or pd.isna(met):
        return None
    return (
        f"met this risk at {_setting(met)} (risk {percent(row.met_risk)}) but "
        f"not from there on: at {_setting(row.exceeded_setting)}, with more fields, the risk "
        f"was {percent(row.exceeded_risk)}, so no setting is taken as safe"
    )


def _not_chosen(row: object) -> str:
    """A family with no safe setting: met the risk once but not from there on, or never."""
    lapse = _lapse(row)
    if lapse is None:
        return f"  {row.family}: never got there at any setting tried."
    return f"  {row.family}: {lapse}."


def _unscored(rows: pd.DataFrame, base: object) -> list[str]:
    """A risk level the fixed count never met lastingly: why, then each family on its own."""
    lapse = _lapse(base)
    if lapse is not None:
        first = f"  a fixed number of fields {lapse}; no rule is scored against it at this level."
    else:
        first = (
            "  a fixed number of fields never got there within the fields available, "
            "so no rule can be scored against it at this level."
        )
    lines = [first]
    for row in rows[rows["family"] != BASELINE].itertuples(index=False):
        if row.effort is None or pd.isna(row.effort):
            lines.append(_not_chosen(row))
        else:
            lines.append(
                f"  {row.family} at {_setting(row.setting)}: {fields(row.effort)} on average "
                f"(risk {_risk(row)}), with no fixed count to compare."
            )
    return lines


def _interval(row: object, saved: float, units: int | None = None) -> str:
    """The interval of the fields saved in the verdict's direction, with its notes, or ''.

    No interval when fewer than ``MIN_BOOT_SUCCESS`` of the resamples reached
    the risk for both (the others are a selected subset), nor with fewer than
    ``FEW_UNITS`` independent units, below which no resampling of the units
    gives an interval near its level. The interval is widened for the number
    of units and still approximate, and says so.
    """
    if units is not None and units < FEW_UNITS:
        return ""
    reached = getattr(row, "saved_reached", None)
    reached = None if reached is None or pd.isna(reached) else float(reached)
    if reached is not None and reached < MIN_BOOT_SUCCESS:
        return (
            f", no interval: only {percent(reached)} of the resamples reached this risk for "
            f"both (at least {percent(MIN_BOOT_SUCCESS)} are needed)"
        )
    if row.saved_low is None or pd.isna(row.saved_low):
        return ""
    # The interval reads in the verdict's direction: fields saved, or fields it costs.
    if saved < -0.05:
        interval = (
            f", approximate {percent(0.95)} interval {_number(-row.saved_high)} to "
            f"{_number(-row.saved_low)}"
        )
    else:
        of = "" if saved > 0.05 else " of the fields saved"
        interval = (
            f", approximate {percent(0.95)} interval{of} {_number(row.saved_low)} to "
            f"{_number(row.saved_high)}"
        )
    notes = []
    # A saving (or cost) whose interval includes no change is not resolved.
    if abs(saved) > 0.05 and row.saved_low <= 0 <= row.saved_high:
        notes.append("not resolved")
    if reached is not None and reached < 1:
        notes.append(
            f"from the {percent(reached)} of the resamples in which both reached this risk"
        )
    return interval + (f" ({'; '.join(notes)})" if notes else "")


def _resampled(result: FrontierResult) -> list[str]:
    """How the intervals were made (or why there are none), stated once per criterion: a
    caution when the units are few (a count of these data), else a standing assumption;
    and that hold-outs and orders are not independent experiments (an assumption)."""
    units = result.settings.get("units", 0)
    unit_cols = result.settings.get("unit_cols") or []
    named = f" ({group_columns(unit_cols)})" if unit_cols else ""
    if units >= FEW_UNITS:
        few = units < _MANY_UNITS
        resampled = (
            f"Intervals resample the {units} units{named}, treated as independent, "
            "widened for their number (approximate"
            + (
                f": with fewer than {_MANY_UNITS} units they cover the truth less often than stated"
                if few
                else ""
            )
            + ")"
        )
        resampled = resampled if few else assumption(resampled)
    elif units >= 2:
        resampled = (
            f"No interval: only {units} units{named} to resample, and at least "
            f"{FEW_UNITS} are needed for one near its level"
        )
    else:
        resampled = f"No interval: there is a single unit{named} to resample"
    return [resampled, assumption("hold-outs and orders are not independent experiments")]


def _scored(base: pd.Series) -> bool:
    return not (base["effort"] is None or pd.isna(base["effort"]))


def _full_lines(rows: pd.DataFrame, units: int) -> list[str]:
    """One risk level in sentences: the fixed count, then each family (today's wording)."""
    base = rows[rows["family"] == BASELINE].iloc[0]
    if not _scored(base):
        # The families are still listed, with their own effort and risk.
        baseline = next(rows[rows["family"] == BASELINE].itertuples(index=False))
        return _unscored(rows, baseline)
    # Fields saved subtract mean efforts; when streams shorter than the fixed count
    # stop early, the baseline's mean effort is below its setting, and the text says so.
    mean_effort = (
        ""
        if abs(float(base["effort"]) - float(base["setting"])) < 0.05
        else f", {_number(base['effort'])} on average as some acquisitions had fewer"
    )
    needed = fields(int(base["setting"]))
    lines = [f"  a fixed number of fields needed {needed}{mean_effort} (risk {_risk(base)})."]
    for row in rows[rows["family"] != BASELINE].itertuples(index=False):
        if row.effort is None or pd.isna(row.effort):
            lines.append(_not_chosen(row))
            continue
        saved = row.fields_saved
        verdict = (
            f"saves {_number(saved)} fields"
            if saved > 0.05
            else f"costs {_number(-saved)} more fields"
            if saved < -0.05
            else "does as well as the fixed count, no better"
        )
        interval = _interval(row, saved, units)
        adaptivity = getattr(row, "adaptivity", None)
        adaptive = (
            ""
            if adaptivity is None or pd.isna(adaptivity)
            else f"; its stops follow the fields each acquisition needed with Spearman "
            f"{adaptivity:.2f}"
        )
        lines.append(
            f"  {row.family} at {_setting(row.setting)}: {fields(row.effort)} on average "
            f"(risk {_risk(row)}); it {verdict}{interval}{adaptive}."
        )
    return lines


def _against(row: object, units: int) -> tuple[str, str | None]:
    """The short table's verdict cell (``saves 0.9 (0.2 to 1.0)``) and a note, or None."""
    saved = float(row.fields_saved)
    if saved > 0.05:
        verdict = f"saves {_number(saved)}"
    elif saved < -0.05:
        verdict = f"costs {_number(-saved)}"
    else:
        verdict = "as well, no better"
    interval = _interval(row, saved, units)
    if not interval.startswith(", approximate"):
        return verdict, interval.removeprefix(", ") or None
    # ", approximate 95% interval[ of the fields saved] a to b[ (notes)]"
    body = interval.split(" interval", 1)[1].removeprefix(" of the fields saved")
    span, _, notes = body.strip().partition(" (")
    notes = notes.rstrip(")")
    resolved = "not resolved" in notes
    rest = [n for n in notes.split("; ") if n and n != "not resolved"]
    cell = f"{verdict} ({span}{', not resolved' if resolved else ''})"
    return cell, ("; ".join(rest) or None)


def _short_rows(rows: pd.DataFrame, units: int) -> tuple[list[list[str]], list[str]]:
    """The families of one risk level as table rows, and the notes under the table."""
    table_rows, notes = [], []
    base = rows[rows["family"] == BASELINE].iloc[0]
    for row in rows[rows["family"] != BASELINE].itertuples(index=False):
        if row.effort is None or pd.isna(row.effort):
            lapse = _lapse(row)
            table_rows.append([row.family, "–", "–", "no lasting setting" if lapse else _NEVER])
            if lapse:
                notes.append(f"{row.family} {lapse}")
            continue
        name = f"{row.family} at {_setting(row.setting)}"
        if not _scored(base):
            cell, note = "no fixed count to compare", None
        else:
            cell, note = _against(row, units)
        table_rows.append([name, number(row.effort), _risk(row), cell])
        if note:
            notes.append(f"{row.family}: {note}")
    return table_rows, notes


def _baseline_text(rows: pd.DataFrame) -> str:
    """``a fixed count needs 2 fields (risk 4%)``, or why no rule is scored at this level."""
    base = rows[rows["family"] == BASELINE].iloc[0]
    if _scored(base):
        mean = (
            ""
            if abs(float(base["effort"]) - float(base["setting"])) < 0.05
            else f", {_number(base['effort'])} on average"
        )
        needed = fields(int(base["setting"]))
        return f"a fixed count needs {needed}{mean} (risk {_risk(base)})"
    lapse = _lapse(next(rows[rows["family"] == BASELINE].itertuples(index=False)))
    if lapse is not None:
        return f"a fixed count {lapse}; no rule is scored against it"
    return "a fixed count never got there within the fields available; no rule is scored"


def _best(rows: pd.DataFrame) -> object | None:
    """The family that saves most at this level, or None when none got there."""
    scored = rows[(rows["family"] != BASELINE) & rows["effort"].notna()]
    if scored.empty or "fields_saved" not in scored:
        return None
    ordered = scored.sort_values("fields_saved", ascending=False, kind="stable")
    return next(ordered.itertuples(index=False))


def level_status(rows: pd.DataFrame, units: int) -> tuple[Status, str]:
    """One risk level's status and phrase (see ``frontier_status``)."""
    base = rows[rows["family"] == BASELINE].iloc[0]
    if not _scored(base):
        return Status.NOT_ASSESSED, "not scored (a fixed count never met it lastingly)"
    fixed = fields(int(base["setting"]))
    if (rows["family"] == BASELINE).all():
        # No family given: no rule was tested, so none can be said not to get there.
        return Status.NOT_ASSESSED, f"no rule family given; a fixed {fixed} got there"
    best = _best(rows)
    if best is None:
        return Status.OK, f"no rule got there; a fixed {fixed} did"
    saved = float(best.fields_saved)
    low = getattr(best, "saved_low", None)
    has_interval = units >= FEW_UNITS and not (low is None or pd.isna(low))
    reached = getattr(best, "saved_reached", None)
    if reached is not None and not pd.isna(reached) and reached < MIN_BOOT_SUCCESS:
        has_interval = False
    if saved > 0.05:
        if has_interval and low > 0:
            return Status.OK, f"{best.family} saves {_number(saved)} of {fixed}"
        why = "not resolved" if has_interval else "no interval"
        return Status.CAUTION, f"{best.family} may save {_number(saved)} fields ({why})"
    if not has_interval:
        return Status.CAUTION, f"no rule beats a fixed {fixed} (no interval)"
    return Status.OK, f"no rule beats a fixed {fixed}"


def frontier_status(result: FrontierResult) -> tuple[Status, str]:
    """The criterion's status and one-line answer, over its risk levels.

    Per risk level: ➖ when a fixed count never met it lastingly (no rule is
    scored) or no rule family was given; ✅ when the family that saves most does so with an interval
    above 0, or when no family saves fields (the null result) with an
    interval; ⚠️ when the best saving is not resolved (its interval includes
    0) or has no interval (fewer than ``FEW_UNITS`` units, or too few
    resamples reached the level). The worst over the levels; levels with the
    same phrase are named once.
    """
    units = result.settings.get("units", 0)
    levels = list(result.scores.groupby("alpha", sort=True))
    parts = [level_status(rows, units) for _, rows in levels]
    if not parts:
        return Status.NOT_ASSESSED, "not scored"
    phrases: dict[str, list[str]] = {}
    for (alpha, _), (_, phrase) in zip(levels, parts, strict=True):
        phrases.setdefault(phrase, []).append(f"≤{percent(alpha)}")
    text = "; ".join(f"at {' and '.join(at)} risk, {what}" for what, at in phrases.items())
    return worst(status for status, _ in parts), text


def _label(result: FrontierResult) -> str:
    return str(result.settings.get("label", result.settings.get("criterion")))


def frontier_summary(result: FrontierResult, detail: str = "short") -> str:
    """The score per risk level: ``"short"``, one table of the families per level (levels
    with the same answer merged); ``"full"``, every family in sentences. Both open with the
    criterion's status and close with how the intervals were made, once."""
    short = check_detail(detail) == "short"
    label = _label(result)
    units = result.settings.get("units", 0)
    status, answer = frontier_status(result)
    lines = [wrap(headline(status, f"For {label}", answer), 0, hang=3)]
    shown: list[list] = []
    for alpha, rows in result.scores.groupby("alpha", sort=True):
        if not short:
            lines.append(f"  Stopping too early in at most {percent(alpha)} of acquisitions:")
            lines += [wrap(line, 2, hang=4) for line in _full_lines(rows, units)]
            continue
        table_rows, notes = _short_rows(rows, units)
        body = [_baseline_text(rows), table_rows, notes]
        if shown and shown[-1][1:] == [body]:
            shown[-1][0].append(percent(alpha))
        else:
            shown.append([[percent(alpha)], body])
    for levels, (baseline, table_rows, notes) in shown:
        levels_text = " or ".join(levels)
        lines.append(
            wrap(
                f"Stopping too early in at most {levels_text} of acquisitions: {baseline}.",
                2,
                hang=4,
            )
        )
        if table_rows and all(cells[-1] == _NEVER for cells in table_rows):
            lines.append("    No rule got there at any setting tried.")
        elif table_rows:
            header = ["rule", "fields", "risk", "against the fixed count (95% interval)"]
            lines.append(table(table_rows, header, indent=4, align="lrrl"))
        lines += [wrap(f"{note}.", 4, hang=6) for note in notes]
    # How the intervals were made, where some level was scored against a fixed count.
    scored = [
        _scored(rows[rows["family"] == BASELINE].iloc[0])
        for _, rows in result.scores.groupby("alpha", sort=True)
    ]
    if any(scored):
        lines += caveat_lines(_resampled(result), indent=2, detail=detail)
    lines.append("  Risk is judged on held-back fields.")
    return "\n".join(lines)


__all__ = ["BASELINE", "frontier_status", "frontier_summary", "level_status"]

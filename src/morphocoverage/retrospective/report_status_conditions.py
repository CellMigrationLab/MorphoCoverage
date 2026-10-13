"""Question 9's status in "At a glance" (``conditions_status``; the rule is in
``report_status``'s module docstring, which lists every question's rule)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

from .._style import Status, span
from .._wording import column_word, count, listing
from ..sampling_unit._units import opposite_signs
from ..sampling_unit.across_text import secondary_headline
from ..sampling_unit.contrast_repeats import MIN_REPEATS, TREND_REPEATS
from ..sampling_unit.contrast_text import (
    P_ALPHA,
    across_tested,
    design_limited,
)
from ..sampling_unit.reading_text import (
    CONFIRMED,
    NO_CONTROL,
    condition_readings,
    readings_headline,
)
from ..sampling_unit.reagents_text import reagents_headline
from ..schema import observation_words

Answer = tuple[Status, str]


def across_headline(across: pd.DataFrame, repeat_col: str | None) -> str:
    """``12 conditions across 3 biological repeats: 12 on one side of the control in every
    repeat (8 lower, 4 higher); 0 of 12 significant on the repeat means (paired t, ...)``."""
    held = across["repeats"].astype(int)
    word = column_word(repeat_col)
    # Each condition's own repeats: one count when they share it, else the range.
    each = (
        count(int(held.max()), word)
        if held.min() == held.max()
        else f"{held.min()}-{count(int(held.max()), word)}"
    )
    lower = int((across["lower"] == across["repeats"]).sum())
    higher = int((across["higher"] == across["repeats"]).sum())
    text = (
        f"{count(len(across), 'condition')} across {each}: "
        f"{lower + higher} in the same direction in every repeat"
    )
    if lower + higher:
        text += f" ({lower} lower, {higher} higher than the control)"
    tested = across[across["p_paired_t"].notna()]
    if len(tested):
        differ = int((tested["q_paired_t"] <= 0.05).sum())
        normalised = int((tested["q_paired_t_normalised"] <= 0.05).sum())
        text += (
            f"; paired t on the repeat means, raw / normalised: {differ} / {normalised} of "
            f"{len(tested)} at q ≤ 0.05"
        )
    return text


_UNTESTED = {
    "only one technical repeat": "one well on a side",
    "too few technical repeats": "too few technical repeats on a side",
    "fewer than 2 fields": "too few fields on a side",
    "fields of one dish or well": "the fields' wells unknown",
    "no control group": "no control fields",
}


def _untested_words(statuses: Sequence[str]) -> str:
    """Why contrasts have no test, in a few words."""
    words = {
        next((short for start, short in _UNTESTED.items() if s.startswith(start)), s)
        for s in statuses
    }
    return words.pop() if len(words) == 1 else "too few units on a side"


# The answer when no repeat holds a condition and its own control.
NOTHING_COMPARED = "nothing compared: no biological repeat holds both a condition and its control"


def _none_compared(table: Any) -> bool:
    """Whether every contrast lacks its control in its repeat (``NO_CONTROL``)."""
    return bool(table["status"].astype(str).str.startswith(NO_CONTROL).all())


def _trends(tests: pd.DataFrame) -> str:
    """``trend: b higher in every repeat, not significant (area and length)``: the conditions
    on one side of the control in every one of 6 repeats or more without q ≤ 0.05, which the
    glance names (audit 10, C15); empty without."""
    needed = {"repeats", "lower", "higher", "q_paired", "condition", "feature"}
    if not len(tests) or not needed <= set(tests.columns):
        return ""
    held = tests[(tests["repeats"] >= TREND_REPEATS) & ~(tests["q_paired"] <= 0.05)]
    parts = []
    for side in ("higher", "lower"):
        rows = held[held[side] == held["repeats"]]
        for condition, mine in rows.groupby("condition", sort=False):
            features = listing([str(f) for f in mine["feature"]], 3)
            parts.append(f"{condition} {side} in every repeat ({features})")
    return f"trend: {'; '.join(parts)}, not significant" if parts else ""


def _reading_parts(contrast: Any, reading: str, several: bool) -> list[str]:
    """The contrast's opening words: the verdict per condition with the reagents that agree
    (none when several features are ranked), else the across-repeat counts, else the
    contrasts and their sizes."""
    across = getattr(contrast, "repeats", None)
    if reading:
        if several:
            return []
        # The verdict per condition, read from every test (``reading_text``).
        parts = [reading]
        agreeing = reagents_headline(contrast)
        if agreeing:
            parts.append(agreeing)
        return parts
    if across is not None and len(across):
        return [across_headline(across, contrast.settings["repeat_col"])]
    table = contrast.table
    sizes = table["d_fields"].abs().dropna()
    size = f", |d| on field means {span(sizes)}" if len(sizes) else ""
    return [f"{count(len(table), 'contrast')}{size}"]


def _untested_part(
    tested: pd.DataFrame, untested: pd.DataFrame, within: str, limited: bool
) -> tuple[Status, str]:
    """The contrasts the design or the data left untested: the status they give and why."""
    status = Status.OK
    # A test the design cannot run is left out, not flagged (see the module).
    if not limited:
        status = Status.CAUTION
    elif not len(tested):
        status = Status.NOT_ASSESSED
    which = "no test" if not len(tested) else f"{len(untested)} not tested"
    return status, f"{within}{which} ({_untested_words(list(untested['status']))})"


def _contrast_parts(contrast: Any, discovery: Any) -> tuple[Status, list[str], bool, bool]:
    """Question 9 from the contrast table: its status and words, whether every untested
    contrast is limited by the design, and whether the readout is a trend at most."""
    table = contrast.table
    across = getattr(contrast, "repeats", None)
    answered = across_tested(contrast)
    reading = readings_headline(contrast)
    verdicts = condition_readings(contrast)["verdict"]
    # Nothing significant across repeats for the readout: a trend at most, never a green
    # answer, unless another feature is (below).
    trend = bool(reading) and answered and not (verdicts == CONFIRMED).any()
    # Several features: the report is neutral, every feature alike (below); the declared
    # readout is what optimise and live target, not the report's answer.
    several = discovery is not None and len(discovery.ranking) > 0
    status, parts = Status.OK, _reading_parts(contrast, reading, several)
    tested = table[table["status"] == "ok"]
    untested = table[table["status"] != "ok"]
    within = "within repeats: " if across is not None and len(across) else ""
    if len(tested):
        differ = int((tested["p_bonferroni"] <= 0.05).sum())
        parts.append(f"{within}{differ} of {len(tested)} tested differ (Bonferroni p ≤ 0.05)")
    limited = bool(len(untested)) and bool(untested["status"].map(design_limited).all())
    if len(untested) and not answered:
        status, part = _untested_part(tested, untested, within, limited)
        parts.append(part)
    passing = tested[tested["p_value"] <= P_ALPHA]
    if opposite_signs(passing, ["condition"], contrast.settings["repeat_col"], str):
        status = Status.CAUTION
        parts.append("opposite signs between repeats")
    return status, parts, limited, trend


def _discovery_headline(contrast: Any, discovery: Any) -> tuple[str, bool, list[str]]:
    """The features significant across repeats and the screen's candidates, in words, with
    whether any feature was tested across repeats and the features significant."""
    passing = int(discovery.ranking["score"].notna().sum())
    n_features = len(discovery.ranking)
    significant = discovery.confirmed()
    across = discovery.tests["p_paired"].notna().any()
    held = int(discovery.tests["repeats"].max()) if "repeats" in discovery.tests else 1
    named = f" ({listing(significant, 4)})" if significant else ""
    share = discovery.settings.get("share")
    # A labelled state's share is compared like a feature, and named apart from them.
    what = (
        f"{n_features} ({n_features - 1} features and the {share})"
        if share in set(discovery.ranking["feature"])
        else f"{n_features} features"
    )
    # The screen named for the observations (owner, 2026-10-02: ``patch-level``).
    one = observation_words(contrast.settings if contrast is not None else {})[0]
    # Every feature alike: any of them significant across repeats is a difference
    # (owner, 2026-10-01: the report is neutral; the readout is optimise's and live's).
    headline = (
        f"{len(significant)} of {what} significant across repeats{named}; "
        f"{count(passing, 'candidate')} in the exploratory {one}-level screen"
        if across
        else f"{passing} of {what} candidates in the exploratory {one}-level "
        f"screen, hypotheses for another repeat (no test across repeats: n = {held})"
    )
    return headline, across, significant


def _discovery_answer(contrast: Any, discovery: Any, status: Status, parts: list[str]) -> Answer:
    """Question 9 when several features were ranked: every feature alike, the contrast's
    words (``parts``, with its ``status``) after them."""
    headline, across, significant = _discovery_headline(contrast, discovery)
    parts.insert(0, headline)
    parts[1:1] = [trend] if (trend := _trends(discovery.tests)) else []
    # The secondary rows, when one is significant (owner, 2026-10-05).
    secondary = secondary_headline(contrast)
    parts[1:1] = [secondary] if secondary else []
    # A verdict on 3 near-identical repeats (audit 10, owner, 2026-10-06): said, and ⚠️
    # when every verdict rests on such repeats.
    tests = discovery.tests
    verdicts = tests[tests["confirmed"].fillna(False).astype(bool)]
    flags = verdicts.get("near_identical", pd.Series(False, index=verdicts.index))
    resting = verdicts[flags.fillna(False).astype(bool)]
    if len(resting):
        named = listing([f"{r.feature} for {r.condition}" for r in resting.itertuples()], 3)
        rest = "rests" if len(resting) == 1 else "rest"
        parts.insert(1, f"{named} {rest} on {MIN_REPEATS} near-identical repeats")
    every = bool(len(verdicts)) and len(resting) == len(verdicts)
    if across and (not significant or every):
        status = Status.CAUTION
    elif not across and contrast is None:
        # The cell-level screen's candidates alone are hypotheses, no answer yet.
        status = Status.NOT_ASSESSED
    return status, "; ".join(parts)


def _ranking_answer(
    contrast: Any, ranking: Any, status: Status, parts: list[str], limited: bool, trend: bool
) -> Answer:
    """Question 9 without several features ranked: the contrast's words (``parts``, with its
    ``status``) and the ranking's gate."""
    if trend:
        status = Status.CAUTION
    if ranking is not None and len(ranking.ranking):
        passing = int(ranking.ranking["score"].notna().sum())
        n_features = count(len(ranking.ranking), "feature")
        # Untested is not failed: say so rather than "0 of n pass".
        parts.append(
            f"{passing} of {n_features} pass the gate"
            if ranking.ranking["tests"].any()
            else f"{n_features} not ranked (no contrast could be tested)"
        )
        if not ranking.ranking["tests"].any() and not (
            limited or (contrast is not None and across_tested(contrast))
        ):
            status = Status.CAUTION
    if not parts:
        return Status.NOT_ASSESSED, "not assessed"
    secondary = secondary_headline(contrast)
    parts[1:1] = [secondary] if secondary else []
    return status, "; ".join(parts)


def conditions_status(contrast: Any, ranking: Any, discovery: Any = None) -> Answer:
    """9. Do the conditions differ, and how confidently? (see the module)."""
    if contrast is None and ranking is None and discovery is None:
        return Status.NOT_ASSESSED, "not assessed"
    status, parts, limited, trend = Status.OK, [], False, False
    if contrast is not None and len(contrast.table) and _none_compared(contrast.table):
        # No repeat holds a condition and its control: nothing was compared, no difference
        # sought (audit 10, F12): ➖, never "no consistent difference".
        return Status.NOT_ASSESSED, NOTHING_COMPARED
    if contrast is not None and len(contrast.table):
        status, parts, limited, trend = _contrast_parts(contrast, discovery)
    elif contrast is not None:
        status = Status.NOT_ASSESSED
        parts.append("no condition to compare with the control")
    if discovery is not None and len(discovery.ranking):
        return _discovery_answer(contrast, discovery, status, parts)
    return _ranking_answer(contrast, ranking, status, parts, limited, trend)


__all__ = ["NOTHING_COMPARED", "conditions_status"]

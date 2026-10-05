"""Question 9's status in "At a glance" (``conditions_status``; the rule is in
``report_status``'s module docstring, which lists every question's rule)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

from .._style import Status, span
from .._wording import column_word, count, listing
from ..sampling_unit._units import opposite_signs
from ..sampling_unit.contrast_text import (
    P_ALPHA,
    across_tested,
    design_limited,
)
from ..sampling_unit.reading_text import CONFIRMED, condition_readings, readings_headline
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
    "only one technical repeat": "one well per condition in a repeat",
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


def conditions_status(contrast: Any, ranking: Any, discovery: Any = None) -> Answer:
    """9. Do the conditions differ, and how confidently? (see the module)."""
    if contrast is None and ranking is None and discovery is None:
        return Status.NOT_ASSESSED, "not assessed"
    status, parts, limited, trend = Status.OK, [], False, False
    if contrast is not None and len(contrast.table):
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
        if reading:
            if not several:
                # The verdict per condition, read from every test (``reading_text``).
                parts.append(reading)
                agreeing = reagents_headline(contrast)
                if agreeing:
                    parts.append(agreeing)
        elif across is not None and len(across):
            parts.append(across_headline(across, contrast.settings["repeat_col"]))
        else:
            sizes = table["d_fields"].abs().dropna()
            size = f", |d| on field means {span(sizes)}" if len(sizes) else ""
            parts.append(f"{count(len(table), 'contrast')}{size}")
        tested = table[table["status"] == "ok"]
        untested = table[table["status"] != "ok"]
        within = "within repeats: " if across is not None and len(across) else ""
        if len(tested):
            differ = int((tested["p_bonferroni"] <= 0.05).sum())
            parts.append(f"{within}{differ} of {len(tested)} tested differ (Bonferroni p ≤ 0.05)")
        limited = bool(len(untested)) and bool(untested["status"].map(design_limited).all())
        if len(untested) and not answered:
            # A test the design cannot run is left out, not flagged (see the module).
            if not limited:
                status = Status.CAUTION
            elif not len(tested):
                status = Status.NOT_ASSESSED
            which = "no test" if not len(tested) else f"{len(untested)} not tested"
            parts.append(f"{within}{which} ({_untested_words(list(untested['status']))})")
        passing = tested[tested["p_value"] <= P_ALPHA]
        if opposite_signs(passing, ["condition"], contrast.settings["repeat_col"], str):
            status = Status.CAUTION
            parts.append("opposite signs between repeats")
    elif contrast is not None:
        status = Status.NOT_ASSESSED
        parts.append("no condition to compare with the control")
    if discovery is not None and len(discovery.ranking):
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
        parts.insert(
            0,
            f"{len(significant)} of {what} significant across repeats{named}; "
            f"{passing} candidates in the exploratory {one}-level screen"
            if across
            else f"{passing} of {what} candidates in the exploratory {one}-level "
            f"screen, hypotheses for another repeat (no test across repeats: n = {held})",
        )
        if across and not significant:
            status = Status.CAUTION
        elif not across and contrast is None:
            # The cell-level screen's candidates alone are hypotheses, no answer yet.
            status = Status.NOT_ASSESSED
        return status, "; ".join(parts)
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
    return status, "; ".join(parts)


__all__ = ["conditions_status"]

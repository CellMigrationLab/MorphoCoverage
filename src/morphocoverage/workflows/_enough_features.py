"""Did I image enough? for every feature alike (private; ``retrospective``, ``_document``).

The report is neutral (owner, 2026-10-01): the readout is what optimise and live target, not
the report's answer, so with several features question 2 asks each of them the same two
questions and says, for each, when it is enough as well as when it is not:

- **enough to compare with the control?** per repeat and condition, whether the 95% interval
  of d excludes 0 with the fields acquired (``contrast_sampling``), from a contrast per
  feature without its cell-level test (``condition_contrast(cell_test=False)``, the slow
  part, read in question 9);
- **enough to pin down each well's own mean?** the mean-precision rules of each feature, run
  in one replay with the rules of every feature (``sufficiency_report``), as the report runs
  the readout's.

``enough_by_feature`` returns a ``FeatureEnough``: one row per feature (``table``, the order
of the feature discovery: significant across repeats first), the contrast of each feature
(for its detail, one feature at a time) and the precision tally of each. ``with_features``
puts it at the head of question 2 of the text report: its answer and status, a table across
the features, and in the full report every feature's resolution. The stopping rules on the
readout stay below it, as optimise and live run them. The HTML page reads the same results
its own way (``_html_enough``: one sentence and a table, then a tab per feature, the
stopping rules left to the plan of the next experiment); ``features_status`` is the headline
of both, and of "At a glance".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .._parallel import check_n_jobs, map_units
from .._style import Status, table, wrap
from .._wording import count, listing
from ..retrospective.report_resolved import has_resolution, resolved_full
from ..sampling_unit.contrast import condition_contrast
from ..sampling_unit.contrast_sampling import PRACTICAL
from ..sampling_unit.repeat_plan_text import field_words
from ..sufficiency import sufficiency_report
from ..sufficiency_short import rule_tallies

if TYPE_CHECKING:
    from ..retrospective import MicroscopistReport
    from .setup import Setup

ENOUGH, PARTLY, NOT_YET = "enough", "partly", "not yet"
# Where the readout's own resolution ends in the short form and its rules begin.
_RULES_HEAD = "The precision of each group's own mean:"
# Fewest features a worker replays on its own: below it, one replay over them all.
_PER_SHARE = 8


@dataclass(frozen=True)
class FeatureEnough:
    """Question 2 per feature (see the module)."""

    table: pd.DataFrame
    contrasts: dict[str, Any]
    precision: dict[str, dict[str, Any]]


def _order(features: list[str], discovery: Any) -> list[str]:
    """The features in the discovery's order (significant across repeats first, then the
    candidates by |d|), the rest after them as declared."""
    if discovery is None or not len(getattr(discovery, "ranking", ())):
        return features
    significant = [f for f in discovery.confirmed() if f in features]
    ranked = [f for f in discovery.ranking["feature"] if f in features]
    first = list(dict.fromkeys([*significant, *ranked]))
    return first + [f for f in features if f not in first]


def _replay(
    objects: pd.DataFrame, fovs: pd.DataFrame, setup: Setup, features: list[str], n_jobs: int
) -> dict[str, dict[str, Any]]:
    """The mean-precision tally of ``features``: one replay with each feature's rules."""
    from ._shared import orders

    # The readout's own mean-precision rules, the same rule on each feature in turn.
    template = [r for r in setup.stopping_rules() if getattr(r, "feature", None) == setup.readout]
    rules, owner = [], {}
    for feature in features:
        for rule in template:
            own = replace(rule, feature=feature)
            rules.append(own)
            owner[own.config_id] = feature
    if not rules:
        return {}
    report = sufficiency_report(
        objects,
        fovs,
        feature_cols=features,
        strategies=rules,
        objective=setup.objective(),
        # "Met with the fields acquired" does not depend on the order: one random order (or
        # the recorded one) is enough, and five times faster than the report's five.
        orders=orders(
            setup, permuted=0 if setup.acquisition_order_col else 1, dispersed=0, seed_permuted=True
        ),
        acquisition_order_col=setup.acquisition_order_col,
        n_jobs=n_jobs,
        **setup.common,
    )
    tallies: dict[str, dict[str, Any]] = {}
    for tally in rule_tallies(report):
        feature = owner.get(tally["config_id"])
        defined = tally["data_rule"] and tally["undefined"] < tally["groups"]
        if feature is not None and defined and feature not in tallies:
            tallies[feature] = tally
    return tallies


def _precision(
    objects: pd.DataFrame, fovs: pd.DataFrame, setup: Setup, features: list[str]
) -> dict[str, dict[str, Any]]:
    """The mean-precision tally of every feature. With many features and several workers,
    one replay per worker over its share of the features (the orders are drawn per group, the
    same in every share): about three times faster than one replay over them all."""
    workers = min(check_n_jobs(setup.n_jobs), len(features) // _PER_SHARE)
    if workers <= 1:
        return _replay(objects, fovs, setup, features, setup.n_jobs)
    shares = [list(share) for share in np.array_split(np.array(features, dtype=object), workers)]
    jobs = [(objects, fovs, setup, share, 1) for share in shares]
    tallies: dict[str, dict[str, Any]] = {}
    for share in map_units(_replay, jobs, workers):
        tallies.update(share)
    return {f: tallies[f] for f in features if f in tallies}


def _row(feature: str, contrast: Any, tally: dict[str, Any] | None) -> dict[str, Any]:
    rows = contrast.table[contrast.table["d_low"].notna()] if has_resolution(contrast) else []
    shown = int(rows["resolved"].sum()) if len(rows) else 0
    small = int(rows["too_small"].sum()) if len(rows) else 0
    needing = rows[~rows["resolved"] & ~rows["too_small"]] if len(rows) else rows
    counts = needing["fields_to_resolve"].astype(float) if len(needing) else pd.Series([])
    verdict = ENOUGH if len(rows) and shown == len(rows) else PARTLY if shown else NOT_YET
    return {
        "feature": feature,
        "comparisons": len(rows),
        "shown": shown,
        "verdict": verdict,
        "needing": len(needing),
        "fields_low": float(counts.min()) if len(counts) else np.nan,
        "fields_high": float(counts.max()) if len(counts) else np.nan,
        "too_small": small,
        "precision_rule": tally["label"] if tally else None,
        "precision_met": tally["met"] if tally else np.nan,
        "precision_groups": tally["groups"] if tally else np.nan,
    }


def _contrast(prepared: pd.DataFrame, fovs: pd.DataFrame, feature: str, setup: Setup) -> Any:
    """One feature against the control, its sampling only (no cell-level test)."""
    return condition_contrast(
        prepared,
        fovs,
        feature=feature,
        control=setup.control,
        random_state=setup.random_state,
        fields_independent=setup.fields_independent,
        object_counts=(),
        cell_test=False,
        **setup.common,
    )


def enough_by_feature(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    prepared: pd.DataFrame,
    setup: Setup,
    discovery: Any = None,
) -> FeatureEnough | None:
    """Question 2 for every declared feature (see the module), or ``None`` with one feature or
    no control; ``prepared`` holds the values the contrasts read (normalised as declared),
    ``objects`` the raw ones the rules read."""
    features = list(setup.columns)
    if len(features) < 2 or setup.control is None:
        return None
    ordered = _order(features, discovery)
    # One light contrast per feature, in worker processes when the setup allows them.
    jobs = [(prepared, fovs, feature, setup) for feature in ordered]
    contrasts = dict(zip(ordered, map_units(_contrast, jobs, setup.n_jobs), strict=True))
    contrasts = {f: c for f, c in contrasts.items() if has_resolution(c)}
    if not contrasts:
        return None
    # One correction family on the page: each feature's verdicts are the discovery's
    # (corrected over every feature and condition), never their own conditions' alone.
    from ._families import shared_family

    contrasts = {f: shared_family(c, discovery, f) for f, c in contrasts.items()}
    precision = _precision(objects, fovs, setup, list(contrasts))
    rows = [_row(f, c, precision.get(f)) for f, c in contrasts.items()]
    return FeatureEnough(pd.DataFrame(rows), contrasts, precision)


def _fields(features: FeatureEnough) -> str:
    return field_words(next(iter(features.contrasts.values())))[1]


def _needed(row: dict[str, Any], fields: str) -> str:
    """``about 42-56 cells a side`` (and those too small), ``–`` when nothing is needed."""
    parts = []
    if row["needing"]:
        low, high = int(row["fields_low"]), int(row["fields_high"])
        about = f"{low}" if low == high else f"{low}-{high}"
        parts.append(f"about {about} {fields} a side")
    if row["too_small"]:
        parts.append(f"{row['too_small']} too small to show")
    return "; ".join(parts) or "–"


def _precision_cell(row: dict[str, Any], word: str) -> str:
    if pd.isna(row["precision_met"]):
        return "–"
    return f"{int(row['precision_met'])} of {count(int(row['precision_groups']), word)}"


def features_table(features: FeatureEnough, word: str = "well", indent: int = 2) -> str:
    """One row per feature: the comparisons shown within single repeats, what more would
    show the others, and the wells whose own mean meets the precision asked."""
    fields = _fields(features)
    rule = next((r for r in features.table["precision_rule"] if r), None)
    rows = [
        [
            r["feature"],
            f"{r['verdict']}: shown in {r['shown']} of {r['comparisons']}",
            _needed(r, fields),
            *([_precision_cell(r, word)] if rule else []),
        ]
        for r in features.table.to_dict("records")
    ]
    columns = ["feature", "compared with the control", "still needed", "own mean"]
    return table(rows, columns if rule else columns[:3], indent=indent)


def default_precision(setup: Setup) -> bool:
    """Whether the mean-precision rules are the workflows' default: no ``rules`` and the
    default first tolerance (``Setup.stopping_rules``)."""
    from dataclasses import fields

    from .setup import Setup

    tolerance = next(f.default for f in fields(Setup) if f.name == "tolerances")[0]
    return setup.rules is None and setup.tolerances[0] == tolerance


def _for(names: list[str], total: int) -> str:
    """``for all 13``, ``for both``, or ``for 2 (a and b)``: the features of one verdict."""
    if len(names) == total and total > 2:
        return f"for all {total}"
    if len(names) == total == 2:
        return "for both"
    return f"for {len(names)} ({listing(names, 3)})"


def features_status(
    features: FeatureEnough, word: str = "well", default: bool = False
) -> tuple[Status, str]:
    """Question 2's status and headline over the features: for which it is enough, partly
    or not yet, the feature needing the most, and the precision of each well's mean (by the
    rule named; ``default``: the workflows' default rule, said so)."""
    rows = features.table
    fields = _fields(features)
    parts = []
    for verdict in (ENOUGH, PARTLY, NOT_YET):
        names = list(rows.loc[rows["verdict"] == verdict, "feature"].astype(str))
        if names:
            parts.append(f"{verdict} {_for(names, len(rows))}")
    head = "to compare with the control, within single repeats, of " + count(len(rows), "feature")
    text = f"{head}: {'; '.join(parts)}"
    small = rows[rows["too_small"] > 0]
    open_ = rows[rows["needing"] > 0]
    if len(small):
        # Beyond PRACTICAL times the fields: these need the most, said before any projection.
        many = count(int(small["too_small"].sum()), "comparison")
        beyond = f"even with {PRACTICAL} times the {fields}"
        text += (
            f"; {many} too small to show {beyond}, over all {len(rows)} features"
            if len(small) == len(rows) and len(rows) > 2
            else f"; {listing(list(small['feature'].astype(str)), 3)}: too small to show in "
            f"{many} {beyond}"
        )
    if len(open_):
        worst = open_.loc[open_["fields_high"].idxmax()]
        most = int(worst["fields_high"])
        which = "of the rest, " if len(small) else ""
        text += f"; {which}{worst['feature']} needs the most (about {most} {fields} a side)"
    precise = rows[rows["precision_met"].notna()]
    if len(precise):
        every = int((precise["precision_met"] == precise["precision_groups"]).sum())
        rule = precise["precision_rule"].iloc[0]
        share = (
            f"all {len(precise)}"
            if every == len(precise)
            else f"none of the {len(precise)}"
            if not every
            else f"{every} of {len(precise)}"
        )
        text += (
            f"; each {word}'s own mean meets {'the default' if default else 'the'} {rule} in "
            f"every {word} for {share} features, a different goal from the comparison"
        )
    status = Status.OK if (rows["verdict"] == ENOUGH).all() else Status.CAUTION
    return status, text


def _indent(text: str) -> str:
    return "\n".join(f"  {line}" if line else line for line in text.splitlines())


def with_features(
    document: MicroscopistReport,
    features: FeatureEnough | None,
    contrast: Any,
    word: str,
    default: bool = False,
) -> MicroscopistReport:
    """The report with question 2 answered over the features (see the module): its status and
    headline, the table across the features first in the short form (the readout's own
    resolution, now one row of it, left out), and every feature's resolution in the full
    form. ``default``: the mean-precision rules are the default ones, and the headline says
    so (with one feature, in its precision words)."""
    if features is None:
        return _default_named(document, default)
    status, headline = features_status(features, word, default)
    rule = next((r for r in features.table["precision_rule"] if r), None)
    own = f"; own mean: the {word}s whose mean meets the {rule}" if rule else ""
    intro = wrap(
        "Each feature alike, " + count(len(features.table), "feature") + ", ordered as in "
        f"“Do the conditions differ?” (significant across repeats first){own}:",
        2,
        hang=4,
    )
    summary = f"{intro}\n{features_table(features, word, indent=4)}"
    short, sections = dict(document.short), dict(document.sections)
    rest = short.get("enough", "")
    if _RULES_HEAD in rest:
        rest = "  The stopping rules, as optimise and live run them (on the readout):\n" + (
            rest[rest.index(_RULES_HEAD) + len(_RULES_HEAD) :].lstrip("\n")
        )
    elif has_resolution(contrast):
        rest = ""
    short["enough"] = "\n\n".join(p for p in (summary, rest) if p.strip())
    full = sections.get("enough", "")
    if has_resolution(contrast):
        full = full.replace(_indent(resolved_full(contrast)), "").lstrip("\n")
    every = "\n\n".join(_indent(resolved_full(c)) for c in features.contrasts.values())
    sections["enough"] = "\n\n".join(p for p in (summary, every, full) if p.strip())
    statuses = {**document.statuses, "enough": status}
    headlines = {**document.headlines, "enough": headline}
    return replace(document, short=short, sections=sections, statuses=statuses, headlines=headlines)


def _default_named(document: MicroscopistReport, default: bool) -> MicroscopistReport:
    """The one-feature headline with its precision rule named the default one."""
    headline = document.headlines.get("enough", "")
    match = re.search(r"own mean by the (?=\S)", headline)
    if not default or match is None:
        return document
    named = headline[: match.end()] + "default " + headline[match.end() :]
    return replace(document, headlines={**document.headlines, "enough": named})


__all__ = [
    "FeatureEnough",
    "default_precision",
    "enough_by_feature",
    "features_status",
    "features_table",
    "with_features",
]

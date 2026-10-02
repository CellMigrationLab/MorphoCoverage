"""Question 10, "Are there distinct populations?" (private; ``retrospective``, ``_document``).

The populations are found on the cells of every condition pooled, on all the declared
features (``sampling_unit.populations``), and then each condition is asked whether it is
enriched in one (owner, 2026-10-02): each population becomes a 0/1 column per observation,
whose mean per repeat is the share of its cells there, and ``feature_discovery`` tests it
against the control as it tests a feature (the paired t-test on the repeat shares,
Benjamini-Hochberg within each sub-experiment, a family of its own; the cell-level screen
beside it). A declared rare label is matched to the population that holds it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd

from .._style import Status, table, wrap
from .._wording import count, listing, p_value, percent
from ..metadata import CONDITION_COLUMN
from ..sampling_unit.discovery import FeatureDiscovery, feature_discovery
from ..sampling_unit.populations import Populations, defined_by, find_populations

if TYPE_CHECKING:
    from .setup import Setup

ALPHA = 0.05


@dataclass(frozen=True)
class PopulationsResult:
    """The populations, the enrichment tests (``None`` without a control or a second
    condition), each population's share per condition, and the label's overlap."""

    populations: Populations
    enrichment: FeatureDiscovery | None
    shares: pd.DataFrame  # population x condition: share of that condition's cells
    label: pd.DataFrame | None  # population, held (of the labelled), purity
    control: Any


def populations_in_report(
    objects: pd.DataFrame, fovs: pd.DataFrame, setup: Setup
) -> PopulationsResult | None:
    """The populations and their enrichment (see the module), or ``None`` with fewer than
    two features."""
    features = list(setup.columns)
    if len(features) < 2:
        return None
    populations = find_populations(
        objects, fovs, features=features, random_state=setup.random_state, **setup.common
    )
    if not populations.names:
        return PopulationsResult(populations, None, pd.DataFrame(), None, None)
    complete = objects.loc[~objects[features].isna().any(axis=1)]
    labels = populations.labels.reindex(complete.index)
    marked = complete.assign(**{name: (labels == name).astype(float) for name in populations.names})
    shares = (
        marked.groupby(CONDITION_COLUMN)[populations.names].mean().T
        if CONDITION_COLUMN in marked
        else pd.DataFrame()
    )
    control = (setup.control or {}).get(CONDITION_COLUMN)
    enrichment = None
    several = CONDITION_COLUMN in marked and marked[CONDITION_COLUMN].nunique() > 1
    if control is not None and several:
        enrichment = feature_discovery(
            marked,
            fovs,
            features=populations.names,
            control=setup.control,
            random_state=setup.random_state,
            **setup.common,
        )
    label = None
    if setup.rare_label is not None and setup.rare_label in complete:
        flag = complete[setup.rare_label].fillna(False).astype(bool)
        rows = []
        for name in populations.names:
            inside = labels == name
            if flag.sum() and inside.sum():
                rows.append(
                    {
                        "population": name,
                        "held": float((flag & inside).sum() / flag.sum()),
                        "purity": float((flag & inside).sum() / inside.sum()),
                    }
                )
        label = pd.DataFrame(rows, columns=["population", "held", "purity"])
    return PopulationsResult(populations, enrichment, shares, label, control)


def _tests(result: PopulationsResult, name: str) -> pd.DataFrame:
    if result.enrichment is None or not len(result.enrichment.tests):
        return pd.DataFrame()
    tests = result.enrichment.tests
    return tests[tests["feature"] == name]


def _enriched(result: PopulationsResult, name: str) -> str:
    """``drugB: higher in 3 of 3 repeats, 7.1% vs 1.8%, q 0.040`` for its strongest
    condition, or ``–``."""
    tests = _tests(result, name)
    tests = tests[tests["p_paired"].notna() | tests["d"].notna()]
    if not len(tests):
        return "–"
    best = tests.sort_values(["q_paired", "q_cells"], na_position="last").iloc[0]
    condition = best["condition"]
    share = result.shares.loc[name] if name in result.shares.index else pd.Series(dtype=float)
    shares = ""
    if condition in share and result.control in share:
        shares = f", {percent(share[condition])} vs {percent(share[result.control])}"
    side = "higher" if best["higher"] >= best["lower"] else "lower"
    repeats = int(best["repeats"])
    q = f", q {p_value(best['q_paired'])}" if pd.notna(best["q_paired"]) else ""
    return f"{condition}: {side} in {max(best['higher'], best['lower'])} of {repeats}{shares}{q}"


def _verdict(result: PopulationsResult) -> tuple[list[str], list[str]]:
    """The populations a condition is enriched in across repeats, and those with a trend."""
    significant, trend = [], []
    for name in result.populations.names:
        tests = _tests(result, name)
        if not len(tests):
            continue
        if (tests["q_paired"] <= ALPHA).any():
            significant.append(name)
        elif ((tests["higher"] == tests["repeats"]) | (tests["lower"] == tests["repeats"])).any():
            trend.append(name)
    return significant, trend


def _rarest(result: PopulationsResult, names: list[str]) -> list[str]:
    shares = result.populations.table.set_index("population")["share"]
    return sorted(names, key=lambda name: shares[name])


def _finding(result: PopulationsResult, name: str, units: str) -> str:
    """``P2 (3% of cells; high dna_intensity): more common in drugB across repeats (7% vs
    2%, q 0.048)``, for its significant conditions."""
    row = result.populations.table.set_index("population").loc[name]
    tests = _tests(result, name)
    tests = tests[tests["q_paired"] <= ALPHA]
    share = result.shares.loc[name]
    parts = []
    for t in tests.itertuples():
        more = "more" if t.higher >= t.lower else "less"
        versus = ""
        if t.condition in share and result.control in share:
            versus = f"{percent(share[t.condition])} vs {percent(share[result.control])}, "
        parts.append(f"{more} common in {t.condition} ({versus}q {p_value(t.q_paired)})")
    feature = defined_by(row["defining"][:1]).split(" (d")[0]
    return (
        f"{name} ({percent(row['share'])} of {units}; {feature}): {listing(parts, 3)} "
        "across repeats"
    )


def _label_line(result: PopulationsResult, label: str) -> str | None:
    if result.label is None or not len(result.label):
        return None
    best = result.label.sort_values("held", ascending=False).iloc[0]
    return (
        f"The cells labelled {label}: {percent(best['held'])} of them in {best['population']}, "
        f"which they make {percent(best['purity'])} of."
    )


def populations_text(result: PopulationsResult, setup: Setup) -> tuple[Status, str, str]:
    """Question 10's status, headline and text (indented for the report)."""
    pops = result.populations
    settings = pops.settings
    units = setup.unit_words
    method = (
        f"{count(settings['observations'], setup.observation_unit or 'observation')} of every "
        f"condition pooled, clustered on the {count(len(settings['features']), 'declared feature')}"
        f" (HDBSCAN on the {count(settings['components'], 'principal component')} holding "
        f"{percent(settings['variance'])} of their robustly standardised variance; a "
        f"population holds at least {settings['min_cells']} {units})"
    )
    if not pops.names:
        text = wrap(
            f"{method}: no distinct population; the {units} form one continuum on these features.",
            2,
            hang=4,
        )
        return Status.OK, "no distinct population: one continuum on these features", text
    lines = [
        wrap(
            f"{method}: {count(len(pops.names), 'population')}, {percent(settings['assigned'])} "
            f"of the {units} in one, the rest between them.",
            2,
            hang=4,
        )
    ]
    control = result.control
    heading = f"enriched in (vs {control})" if control is not None else "enriched in"
    rows = [
        [
            r.population,
            f"{r.observations} ({percent(r.share)})",
            r.defined_by or "–",
            r.fields,
            r.repeats,
            _enriched(result, r.population),
        ]
        for r in pops.table.itertuples()
    ]
    columns = ["population", units, "defined by", "fields", "repeats", heading]
    lines.append(table(rows, columns, indent=4))
    significant, trend = _verdict(result)
    for name in pops.names:
        row = pops.table.set_index("population").loc[name]
        tests = _tests(result, name)
        if not len(tests):
            continue
        parts = []
        for t in tests.itertuples():
            enriched = pd.notna(t.q_paired) and t.q_paired <= ALPHA
            side = "higher" if t.higher >= t.lower else "lower"
            consistent = max(t.higher, t.lower) == t.repeats and t.repeats > 1
            if enriched:
                parts.append(
                    f"{t.condition} {side}, significant across repeats (q {p_value(t.q_paired)})"
                )
            elif consistent:
                q = f", q {p_value(t.q_paired)}" if pd.notna(t.q_paired) else ""
                parts.append(f"{t.condition} {side} in every repeat, not significant{q}")
        if parts:
            lines.append(
                wrap(
                    f"- {name} ({percent(row['share'])} of {units}; "
                    f"{defined_by(row['defining'][:2])}): {'; '.join(parts)}.",
                    4,
                    hang=6,
                )
            )
    if setup.rare_label is not None:
        line = _label_line(result, setup.rare_label)
        if line:
            lines.append(wrap(line, 4, hang=6))
    lines.append(
        wrap(
            "Assumptions: The populations are found on every condition pooled, then each "
            "condition is compared with the control on its share of cells in each, per "
            "biological repeat (paired t-test on the repeat shares, Benjamini-Hochberg within "
            "each sub-experiment over every population and condition). The clusters depend on "
            "the declared features and their standardisation; a population held by few fields "
            "may be an artefact of those images (Fields to check lists them).",
            2,
            hang=4,
        )
    )
    if significant:
        headline = f"{count(len(pops.names), 'population')}; " + "; ".join(
            _finding(result, name, units) for name in _rarest(result, significant)[:3]
        )
        status = Status.OK
    elif result.enrichment is None:
        headline = f"{count(len(pops.names), 'population')}; no control to compare their shares"
        status = Status.NOT_ASSESSED
    elif trend:
        headline = (
            f"{count(len(pops.names), 'population')}; the same direction in every repeat for "
            f"{listing(trend, 4)}, not significant"
        )
        status = Status.CAUTION
    else:
        headline = f"{count(len(pops.names), 'population')}; no condition consistently enriched"
        status = Status.CAUTION
    held = int(result.enrichment.tests["repeats"].max()) if result.enrichment is not None else 0
    if result.enrichment is not None and held < 3:
        headline += f" (n = {held}: no test across repeats)"
        status = Status.NOT_ASSESSED
    return status, headline, "\n".join(lines)


def population_of(result: PopulationsResult | None, objects: pd.DataFrame) -> pd.Series | None:
    """Each observation's population (``None`` unassigned), for the maps."""
    if result is None or not result.populations.names:
        return None
    return result.populations.labels.reindex(objects.index)


__all__ = [
    "PopulationsResult",
    "population_of",
    "populations_in_report",
    "populations_text",
    "with_populations",
]


def with_populations(document: Any, result: PopulationsResult | None, setup: Setup):
    """The report with question 10 answered (``populations_text``)."""
    if result is None:
        return document
    from ._document import with_text  # the assembly imports this module

    status, headline, text = populations_text(result, setup)
    plain = "\n".join(line.removeprefix("  ") for line in text.splitlines())
    return with_text(
        document, "populations", plain, status=status, headline=headline, short_text=text
    )

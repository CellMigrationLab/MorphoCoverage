"""Question 10, "Are there populations or groups of cells?" (private; ``retrospective``,
``_document``).

The groups are found on the cells of every condition pooled, on all the declared
features (``sampling_unit.populations``), and then each condition is asked whether it is
enriched in one (owner, 2026-10-02): per biological repeat, the share of its cells in the
population against the control's, on the log-odds scale (``enrichment_tests``: a small
share is as steady there as a large one, and a population reads like its complement),
tested across the repeats by the paired t-test, Benjamini-Hochberg within each
sub-experiment over every population and condition (a family of its own); a main
population, holding most cells, is listed but not tested, its share being all but the
complement of the others'; a repeat where the group is absent from both sides counts as no
change (owner, 2026-10-02). A declared rare label is matched to the group that holds it,
and a group most of whose cells carry it (``LABELLED``) is the label's own: its share is
compared in question 9, so it is not tested again here (owner, 2026-10-02). A cell whose
label is missing is unknown: out of both shares of the overlap (owner, 2026-10-05).
Each group is a candidate to validate on the images, given with the rule on the features
that picks out its cells (``sampling_unit._population_rule``). With other feature tables the
groups are searched in each space apart and merged (``_population_spaces``, owner
2026-10-06); the enrichment tests then run on the merged list, one family as before.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd

from .._style import Status, table, wrap
from .._wording import count, listing, p_value, percent, percent_fine
from ..metadata import CONDITION_COLUMN, REPEAT_COLUMN
from ..sampling_unit.contrast_repeats import MIN_REPEATS, TREND_REPEATS
from ..sampling_unit.populations import Populations, defined_by, traits
from ..schema import observation_words, validate_input_tables
from ..statistics import ALPHA
from ._population_context import (
    AGAINST_WORDS,
    EXPERIMENTS_DIFFER,
    against_controls,
    by_evidence,
    compared_values,
    composition,
    few_valued,
    few_words,
    mostly_words,
)
from ._population_rule_words import rule_cell, rule_words
from ._population_spaces import (
    described,
    found_words,
    no_own_control_words,
    populations_in_spaces,
    space_words,
    spaces_method,
    undescribed_line,
)
from ._population_tests import enrichment_tests, one_sided

if TYPE_CHECKING:
    from ..retrospective import MicroscopistReport
    from .setup import Setup

# A group this labelled (the share of its cells carrying the declared rare label) is the
# label's own: compared as the label's share, never tested as a group too.
LABELLED = 0.9
# Where the label's share is compared, as the page titles it.
COMPARED_IN = "“Do the conditions differ?”"


@dataclass(frozen=True)
class PopulationsResult:
    """The populations, the enrichment tests (``None`` without a control or a second
    condition), the label's overlap and, by group, what it is against the conditions
    (``context``: ``mostly`` one condition, ``few`` its rule's features with few values;
    ``_population_context``), with ``against`` when the groups are described against their
    compared controls and ``normalised`` when found on normalised values as the experiments
    differ (owner, 2026-10-06, audit 10)."""

    populations: Populations
    enrichment: pd.DataFrame | None  # one row per population and condition (enrichment_tests)
    label: pd.DataFrame | None  # population, held (of the labelled), purity
    control: Any
    context: dict[str, dict[str, Any]] = field(default_factory=dict)
    against: bool = False
    normalised: bool = False


def populations_in_report(
    objects: pd.DataFrame, fovs: pd.DataFrame, setup: Setup, raw: pd.DataFrame | None = None
) -> PopulationsResult | None:
    """The populations and their enrichment (see the module), or ``None`` with fewer than
    two features. ``objects`` holds the values read (normalised with ``normalize``), ``raw``
    the values before, to name the observations left out for want of their own control
    (audit 8, seed 4)."""
    # Several experiments and no normalize: the values normalised to each own control.
    objects, read, normalised = compared_values(objects, fovs, setup, raw)
    # With other feature tables, in each space apart and merged (owner, 2026-10-06).
    populations = populations_in_spaces(objects, fovs, read, raw)
    if populations is None:
        return None
    if not populations.names:
        return PopulationsResult(populations, None, None, None, normalised=normalised)
    # The validated observations: group columns only in the FOV table are attached to them.
    data = validate_input_tables(objects, fovs=fovs, feature_cols=setup.columns, **setup.common)
    # Each group known where its space assessed the cell: 1 in it, 0 not, missing unknown.
    shares = {name: populations.membership(name) for name in populations.names}
    marks = pd.DataFrame(shares).reindex(data.objects.index)
    complete = data.objects.loc[marks.notna().any(axis=1)]
    marked = complete.assign(**marks.loc[complete.index])
    control = (setup.single_control or {}).get(CONDITION_COLUMN)
    if setup.comparisons is not None:  # several controls: each condition against its own
        control = setup.comparisons
    enrichment = None
    several = CONDITION_COLUMN in marked and marked[CONDITION_COLUMN].nunique() > 1
    from ._shared import pooled_columns  # _shared imports the assembly, which imports this

    pooled = pooled_columns(marked, fovs, setup, [CONDITION_COLUMN, REPEAT_COLUMN])
    label = None
    if setup.rare_label is not None and setup.rare_label in complete:
        # A missing label is unknown, out of both shares (owner, 2026-10-05).
        known = complete[setup.rare_label].notna()
        flag = complete[setup.rare_label].where(known, False).astype(bool)
        rows = []
        for name in populations.names:
            inside = marked[name].eq(1) & known
            if flag.sum() and inside.sum():
                rows.append(
                    {
                        "population": name,
                        "held": float((flag & inside).sum() / flag.sum()),
                        "purity": float((flag & inside).sum() / inside.sum()),
                    }
                )
        label = pd.DataFrame(rows, columns=["population", "held", "purity"])
    if control is not None and several and not pooled:
        # The declared label's group is compared once, as the label's share (question 9).
        own = set(labelled_populations(label))
        names = [name for name in tested_populations(populations) if name not in own]
        enrichment = enrichment_tests(marked, names, control)
    # Each group read against the conditions (audit 10): described against its compared
    # control, mostly one condition, a rule on few values.
    against = against_controls(populations.table, marked, setup)
    context = {
        row["population"]: {"mostly": composition(marked, row["population"])}
        | {"few": few_valued(row, marked)}
        for row in populations.table.to_dict("records")
        if row["kind"] == "group"
    }
    return PopulationsResult(populations, enrichment, label, control, context, against, normalised)


def labelled_populations(label: pd.DataFrame | None) -> list[str]:
    """The groups most of whose cells carry the declared rare label (``LABELLED``): the
    label's own, whose share question 9 compares, so not tested again (owner, 2026-10-02)."""
    if label is None or not len(label):
        return []
    return [str(r.population) for r in label.itertuples() if r.purity >= LABELLED]


def tested_populations(populations: Populations) -> list[str]:
    """The populations and groups whose enrichment is tested: all but the main population,
    whose share is all but the complement of the others' (its test would repeat theirs)."""
    return populations.of_kind("group")


def _tests(result: PopulationsResult, name: str) -> pd.DataFrame:
    if result.enrichment is None or not len(result.enrichment):
        return pd.DataFrame()
    tests = result.enrichment
    return tests[tests["feature"] == name]


def _side(row: Any) -> str:
    """``higher in 3 of 3``, ``lower in 2 of 3``, ``no change in 3 of 3``, or ``mixed (1
    higher, 1 lower, of 3)``."""
    n, higher, lower = int(row["repeats"]), int(row["higher"]), int(row["lower"])
    if higher == lower:
        if not higher:
            return f"no change in {n} of {n}"
        return f"mixed ({higher} higher, {lower} lower, of {n})"
    side = "higher" if row["higher"] > row["lower"] else "lower"
    return f"{side} in {max(row['higher'], row['lower'])} of {int(row['repeats'])}"


def _versus(row: Any) -> str:
    """``6.6% vs 1.6%``: the condition's share and its own control's, over its repeats."""
    if pd.isna(row["share"]) or pd.isna(row["control_share"]):
        return ""
    return f"{percent_fine(row['share'])} vs {percent_fine(row['control_share'])}"


def _enriched(result: PopulationsResult, name: str) -> str:
    """``drugB: higher in 3 of 3, 7.1% vs 1.8%, q 0.040`` for its most different condition
    (the smallest q, then the largest effect), ``, not tested`` below 3 repeats, or ``–``."""
    tests = _tests(result, name)
    if not len(tests):
        return "–"
    best = by_evidence(tests).iloc[0]  # the smallest q, then the largest effect (audit 10)
    versus = _versus(best)
    q = f", q {p_value(best['q_paired'])}" if pd.notna(best["q_paired"]) else ""
    untested = ", not tested" if int(best["repeats"]) < MIN_REPEATS else ""
    return f"{best['condition']}: {_side(best)}{', ' + versus if versus else ''}{q}{untested}"


def _verdict(result: PopulationsResult) -> tuple[list[str], list[str]]:
    """The populations a condition is enriched in across repeats, and those with a trend."""
    significant, trend = [], []
    for name in result.populations.names:
        tests = _tests(result, name)
        if not len(tests):
            continue
        if (tests["q_paired"] <= ALPHA).any():
            significant.append(name)
        elif trends(tests).any():
            trend.append(name)
    return significant, trend


def trends(tests: pd.DataFrame) -> pd.Series:
    """The tests on one side of the control in every repeat, from ``TREND_REPEATS`` repeats:
    with fewer, every repeat goes one way without an effect too often (owner, 2026-10-06,
    audit 10, as the features' trend rule)."""
    return one_sided(tests) & (tests["repeats"] >= TREND_REPEATS)


def _group_parts(result: PopulationsResult, name: str, units: str) -> list[str]:
    """What question 10 says of a group's conditions: mostly one condition (its shift); each
    condition more or less common across repeats (exploratory); a trend from 6 repeats; "not
    tested" below 3 (owner, 2026-10-06, audit 10)."""
    mostly = result.context.get(name, {}).get("mostly")
    parts = [mostly_words(mostly, units)] if mostly else []
    tests = _tests(result, name)
    if not len(tests):
        return parts
    trend = trends(tests)
    for index, t in tests.iterrows():
        side = "higher" if t["d"] > 0 else "lower"
        versus = _versus(t)
        q = f"q {p_value(t['q_paired'])}" if pd.notna(t["q_paired"]) else ""
        inside = ", ".join(x for x in (versus, q) if x)
        if pd.notna(t["q_paired"]) and t["q_paired"] <= ALPHA:
            more = "more" if t["d"] > 0 else "less"
            parts.append(f"{more} common in {t['condition']} (exploratory; {inside})")
        elif bool(trend.loc[index]):
            parts.append(f"{t['condition']} {side} in every repeat, not significant ({inside})")
        elif int(t["repeats"]) < MIN_REPEATS and bool(one_sided(tests).loc[index]):
            both = "both repeats" if int(t["repeats"]) == 2 else f"{int(t['repeats'])} repeats"
            versus = f" ({versus})" if versus else ""
            parts.append(f"{t['condition']} {side} in {both}{versus}, not tested")
    return parts


def _rarest(result: PopulationsResult, names: list[str]) -> list[str]:
    shares = result.populations.table.set_index("population")["share"]
    return sorted(names, key=lambda name: shares[name])


def _finding(result: PopulationsResult, name: str, units: str, trend: bool = False) -> str:
    """``G2 (3% of cells; high dna_intensity): more common in drugB (exploratory; 7% vs 2%,
    q 0.048)``, for its significant conditions; with ``trend``, those on one side of the
    control in every one of 6 or more repeats, ``higher in every repeat, not significant``;
    a group mostly of one condition, ``mostly a1, 94% of its cells: its shift, not a
    subpopulation`` (owner, 2026-10-06, audit 10: never "significant across repeats")."""
    row = result.populations.table.set_index("population").loc[name]
    tests = _tests(result, name)
    tests = tests[trends(tests)] if trend else tests[tests["q_paired"] <= ALPHA]
    parts = []
    for _, t in tests.iterrows():
        versus = _versus(t)
        versus = f"{versus}, " if versus else ""
        q = f"q {p_value(t['q_paired'])}" if pd.notna(t["q_paired"]) else ""
        if trend:
            side = "higher" if t["d"] > 0 else "lower"
            inside = ", ".join(x for x in (versus.rstrip(", "), q) if x)
            parts.append(f"{side} in {t['condition']} in every repeat, not significant ({inside})")
        else:
            more = "more" if t["d"] > 0 else "less"
            parts.append(f"{more} common in {t['condition']} (exploratory; {versus}{q})")
    feature = traits(row["defining"][:1]) if described(row) else found_words(row, rule=False)
    share = percent_fine(row["share"])
    mostly = result.context.get(name, {}).get("mostly")
    found = mostly_words(mostly, units) if mostly else listing(parts, 3)
    return f"{name} ({share} of {units}; {feature}): {found}"


def _label_line(
    result: PopulationsResult, label: str, units: str, said: bool = False
) -> str | None:
    """Where the declared label's observations are; ``said`` when the headline already
    says the group is the label's (``label_words``), which is then not said again."""
    if result.label is None or not len(result.label):
        return None
    best = result.label.sort_values("held", ascending=False).iloc[0]
    own = best["population"] in labelled_populations(result.label)
    if own and said:
        return (
            f"{best['population']}: {percent(best['purity'])} of its {units} carry the label "
            f"{label}, and it holds {percent(best['held'])} of the labelled ones; not tested "
            "again here."
        )
    if own:
        return (
            f"{label_words(best['population'], label, units)} ({percent(best['purity'])} of "
            f"them; it holds {percent(best['held'])} of the labelled ones); its share is "
            f"compared in {COMPARED_IN}, so it is not tested again here."
        )
    return (
        f"The {units} labelled {label}: {percent(best['held'])} of them in "
        f"{best['population']}, which they make {percent(best['purity'])} of."
    )


def label_words(name: str, label: str, units: str) -> str:
    """``G1 is the declared rare phenotype, mitotic: most of its cells carry the label``
    (``units``: the observations' plural word)."""
    return f"{name} is the declared rare phenotype, {label}: most of its {units} carry the label"


def only_labelled(result: PopulationsResult | None) -> list[str]:
    """The groups found when every one is the declared label's own (empty otherwise)."""
    if result is None:
        return []
    groups = tested_populations(result.populations)
    own = labelled_populations(result.label)
    return groups if groups and all(name in own for name in groups) else []


_KIND_WORDS = {"main": "main (the rest)", "group": "group"}


def _found_words(pops: Populations) -> str:
    """``1 group of cells to validate``, in the report's words for its observations."""
    groups = len(pops.of_kind("group"))
    units = observation_words(pops.settings)[1]
    return f"{count(groups, 'group')} of {units} to validate on the images"


def _shares_line(pops: Populations, units: str) -> str:
    """``P1 91%, P2 3.4%, G1 20%; 6% in none``: each one's share and the unassigned."""
    shares = ", ".join(f"{r.population} {percent_fine(r.share)}" for r in pops.table.itertuples())
    none = 1 - float(pops.table["share"].sum())
    return f"{shares}; {percent_fine(max(none, 0.0))} of the {units} in none"


def populations_text(result: PopulationsResult, setup: Setup) -> tuple[Status, str, str]:
    """Question 10's status, headline and text (indented for the report)."""
    pops = result.populations
    settings = pops.settings
    units = setup.unit_words
    method = (
        f"{count(settings['observations'], setup.unit_word)} in all, every condition pooled, "
        f"clustered on the {count(len(settings['features']), 'declared feature')}"
        f" (HDBSCAN on a {settings['map_components']}-dimensional UMAP of their robust "
        f"standardisation; a cluster holds at least {settings['min_cells']} {units}), each "
        "cluster then judged on all the features"
    )
    if settings.get("spaces"):  # other feature tables: each space apart (owner, 2026-10-06)
        method = (
            f"Every condition pooled, clustered {spaces_method(pops, setup)}; in each, HDBSCAN "
            "on a UMAP of the robust standardisation, each cluster then judged on all its "
            f"features; groups found in two spaces are one when their {units} overlap with a "
            f"Jaccard index of {settings['same_group']:g} or more"
        )
    # Observations without their own control, named once, never a missing value (audit 8).
    left = no_own_control_words(settings, setup.unit_word, units)
    method += f" ({left})" if left else ""
    method += f", {EXPERIMENTS_DIFFER}" if result.normalised else ""
    if not settings.get("observations") and not settings.get("spaces"):
        text = wrap(
            f"{method}: not assessed: no {setup.unit_word} has a value of every feature "
            f"({count(settings['incomplete_observations'], setup.unit_word)} with a missing "
            "value).",
            2,
            hang=4,
        )
        return Status.NOT_ASSESSED, f"not assessed: every {setup.unit_word} misses a value", text
    if not settings.get("clustered", True):
        text = wrap(
            f"{method}: not clustered: too few {units} (or no feature that varies) for two "
            f"clusters of {settings['min_cells']}.",
            2,
            hang=4,
        )
        return Status.NOT_ASSESSED, f"not clustered: too few {units}", text
    if not pops.names:
        # Not finding a group says nothing of a continuum, nor of rare states (#94; owner,
        # 2026-10-05): ➖, a neutral answer.
        text = wrap(
            f"{method}: no candidate group passed the neighbours-inside check on these "
            f"features ({percent(settings['purity'])} or more of a cluster's {units}' "
            f"{settings['neighbours']} nearest neighbours in it).",
            2,
            hang=4,
        )
        headline = "no group: no candidate group passed the check on these features"
        return Status.NOT_ASSESSED, headline, text
    lines = [wrap(f"{method}: {_found_words(pops)} ({_shares_line(pops, units)}).", 2, hang=4)]
    control = result.control
    if isinstance(control, tuple):  # several controls: each condition against its own
        control = "its control"
    heading = f"most different (vs {control})" if control is not None else "most different"
    tested = set(tested_populations(pops))
    own = set(labelled_populations(result.label))
    rows = [
        [
            r.population,
            _KIND_WORDS.get(r.kind, r.kind),
            f"{r.observations} ({percent_fine(r.share)})",
            defined_by(r.defining[:2])
            or ("no rule" if r.kind == "group" and not described(r._asdict()) else "–"),
            percent(r.purity) if pd.notna(r.purity) else "–",
            rule_cell(r),
            r.fields,
            percent(r.largest_field),
            r.repeats,
            _enriched(result, r.population)
            if r.population in tested and r.population not in own
            else "the declared label (not tested again)"
            if r.population in own
            else "the rest (not tested)",
        ]
        for r in pops.table.itertuples()
    ]
    columns = [
        "population",
        "kind",
        units,
        "defined by",
        "neighbours inside",
        "rule (held-out fields: found, right)",
        "fields",
        "in largest field",
        "repeats",
        heading,
    ]
    if settings.get("spaces"):  # where each was found (owner, 2026-10-06)
        for row, r in zip(rows, pops.table.itertuples(), strict=True):
            row.insert(3, listing([space_words(x) for x in r.found_in], 8))
        columns.insert(3, "found in")
    lines.append(table(rows, columns, indent=4))
    if result.against:
        lines.append(wrap(AGAINST_WORDS.format(units=units, unit=setup.unit_word), 4, hang=6))
    if undescribed_line(pops):
        lines.append(wrap(undescribed_line(pops), 4, hang=6))
    significant, trend = _verdict(result)
    for name in pops.names:
        row = pops.table.set_index("population").loc[name]
        parts = _group_parts(result, name, units)
        context = result.context.get(name, {})
        if context.get("few"):
            parts.append(few_words(context["few"], units))
        rule = rule_words(row, units) if row["rule"] else ""
        if parts or rule:  # every group with its rule, its conditions when there is a finding
            what = (
                " and ".join(defined_by(row["defining"][:2]).split(", "))
                if described(row)
                else found_words(row, rule=False)
            )
            said = f": {'; '.join(parts)}." if parts else "."
            lines.append(
                wrap(
                    f"- {name}, {percent_fine(row['share'])} of {units}, {what}{said}"
                    + (f" {rule[:1].upper()}{rule[1:]}." if rule else ""),
                    4,
                    hang=6,
                )
            )
    unpaired = result.enrichment.attrs.get("unpaired", []) if result.enrichment is not None else []
    if unpaired:
        lines.append(
            wrap(
                f"Not compared: {listing([str(c) for c in unpaired], 4)} (no biological repeat "
                "shared with the control).",
                4,
                hang=6,
            )
        )
    labelled = only_labelled(result)
    if setup.rare_label is not None:
        # When the label's group is the only one, the headline says so (In short).
        line = _label_line(result, setup.rare_label, units, said=bool(labelled))
        if line:
            lines.append(wrap(line, 4, hang=6))
    lines.append(
        wrap(
            f"Assumptions: The {units} of every condition are clustered together"
            + (  # compared only where a control is declared (audit 10, C24)
                f", then each condition is compared with its own control on its share of "
                f"{units} in each, per biological repeat (paired t-test on the log odds of the "
                "repeat shares, Benjamini-Hochberg within each sub-experiment over every "
                "population and condition)"
                if setup.reference_control is not None
                else ", and no condition is compared, as no control is declared"
            )
            + "; the main population, holding most of them, is the rest, listed and "
            "not tested. A cluster on the UMAP is kept as a group when "
            f"{percent(settings['purity'])} or more of its {units}' "
            f"{settings['neighbours']} nearest neighbours on all the features are in it too "
            "(neighbours inside). Whether a group is a separate state or the end of a "
            f"continuum (the most extreme {units} of a gradient) is not decided from the "
            "numbers: no test of a gap told the two apart reliably at these sizes, so open "
            "the images of its fields to validate it. Its rule is the simplest one on the "
            "features (at most two thresholds, a decision tree) that picks out its "
            f"{units}, fitted on all of them; the share of them it finds and the share right "
            "of those it picks describe that procedure refitted with whole fields held out "
            f"({units} held out for a group in one field), not the exact thresholds shown. "
            "Groups that overlap strongly on every feature "
            "are not told apart. The clusters depend on the declared features and their "
            "standardisation; a group held by few fields, or most of whose "
            f"{units} are in one (in largest field), may be an artefact of those images.",
            2,
            hang=4,
        )
    )
    found = _found_words(pops)
    if labelled and setup.rare_label is not None:
        # The label's group is compared once, as the label's share: no q of its own here.
        headline = (
            f"{found}; {label_words(labelled[0], setup.rare_label, units)}; its share is compared "
            f"in {COMPARED_IN}"
        )
        return Status.OK, headline, "\n".join(lines)
    if significant:
        # Exploratory until validated: ⚠️, never ✅ (owner, 2026-10-06, audit 10).
        headline = f"{found}; " + "; ".join(
            _finding(result, name, units) for name in _rarest(result, significant)[:3]
        )
        status = Status.CAUTION
    elif result.enrichment is None:
        headline = f"{found}; no control to compare their shares"
        status = Status.NOT_ASSESSED
    elif trend:
        headline = f"{found}; " + "; ".join(
            _finding(result, name, units, trend=True) for name in _rarest(result, trend)[:3]
        )
        status = Status.CAUTION
    else:
        headline = f"{found}; no condition consistently enriched"
        status = Status.CAUTION
    held = (
        int(result.enrichment["repeats"].max())
        if result.enrichment is not None and len(result.enrichment)
        else 0
    )
    if result.enrichment is not None and held < MIN_REPEATS:
        headline += f" (n = {held}: no test across repeats)"
        status = Status.NOT_ASSESSED
    return status, headline, "\n".join(lines)


def population_of(result: PopulationsResult | None, objects: pd.DataFrame) -> pd.Series | None:
    """Each observation's population (``None`` unassigned), for the maps."""
    if result is None or not result.populations.names:
        return None
    return result.populations.labels.reindex(objects.index)


__all__ = [
    "LABELLED",
    "PopulationsResult",
    "enrichment_tests",
    "label_words",
    "labelled_populations",
    "one_sided",
    "only_labelled",
    "population_of",
    "populations_in_report",
    "populations_text",
    "with_populations",
]


def with_populations(
    document: MicroscopistReport, result: PopulationsResult | None, setup: Setup
) -> MicroscopistReport:
    """The report with question 10 answered (``populations_text``)."""
    if result is None:
        return document
    from ._document import with_text  # the assembly imports this module

    status, headline, text = populations_text(result, setup)
    plain = "\n".join(line.removeprefix("  ") for line in text.splitlines())
    return with_text(
        document, "populations", plain, status=status, headline=headline, short_text=text
    )

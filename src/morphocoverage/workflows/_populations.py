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
complement of the others'. A declared rare label is matched to the group that holds it.
Each group is a candidate to validate on the images, given with the rule on the features
that picks out its cells (``sampling_unit._population_rule``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .._style import Status, table, wrap
from .._wording import count, listing, p_value, percent, percent_fine
from ..metadata import CONDITION_COLUMN, REPEAT_COLUMN
from ..sampling_unit._experiments import corrected, experiments
from ..sampling_unit.contrast_repeats import MIN_REPEATS
from ..sampling_unit.populations import Populations, defined_by, find_populations, traits
from ..schema import unit_plural
from ..statistics import paired_t

if TYPE_CHECKING:
    from ..retrospective import MicroscopistReport
    from .setup import Setup

ALPHA = 0.05


@dataclass(frozen=True)
class PopulationsResult:
    """The populations, the enrichment tests (``None`` without a control or a second
    condition) and the label's overlap."""

    populations: Populations
    enrichment: pd.DataFrame | None  # one row per population and condition (enrichment_tests)
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
        return PopulationsResult(populations, None, None, None)
    complete = objects.loc[~objects[features].isna().any(axis=1)]
    labels = populations.labels.reindex(complete.index)
    marked = complete.assign(**{name: (labels == name).astype(float) for name in populations.names})
    control = (setup.control or {}).get(CONDITION_COLUMN)
    enrichment = None
    several = CONDITION_COLUMN in marked and marked[CONDITION_COLUMN].nunique() > 1
    from ._shared import pooled_columns  # _shared imports the assembly, which imports this

    pooled = pooled_columns(marked, fovs, setup, [CONDITION_COLUMN, REPEAT_COLUMN])
    if control is not None and several and not pooled:
        enrichment = enrichment_tests(marked, tested_populations(populations), control)
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
    return PopulationsResult(populations, enrichment, label, control)


def tested_populations(populations: Populations) -> list[str]:
    """The populations and groups whose enrichment is tested: all but the main population,
    whose share is all but the complement of the others' (its test would repeat theirs)."""
    return populations.of_kind("group")


def _log_odds_ratio(t: Any, c: Any) -> float:
    """The log odds ratio of the condition's share against its control's in one repeat
    (``t`` and ``c`` the cells in the population, ``sum``, and the cells, ``count``). One
    pseudo-cell is added to each side, split by the population's pooled share of that repeat,
    so a share of 0 has log odds and uneven numbers of cells do not tip the ratio (a fixed
    half cell would make 1 of 5000 read lower than 0 of 300); NaN when the population is
    absent from both sides."""
    pooled = (t["sum"] + c["sum"]) / (t["count"] + c["count"])
    if not pooled > 0:
        return np.nan

    def odds(k: float, n: float) -> float:
        return float(np.log((k + pooled) / (n - k + 1 - pooled)))

    return odds(t["sum"], t["count"]) - odds(c["sum"], c["count"])


def one_sided(tests: pd.DataFrame) -> pd.Series:
    """The tests on one side of the control in every repeat, over two or more repeats."""
    every = (tests["higher"] == tests["repeats"]) | (tests["lower"] == tests["repeats"])
    return every & (tests["repeats"] >= 2) & (tests["higher"] != tests["lower"])


def enrichment_tests(marked: pd.DataFrame, names: list[str], control: Any) -> pd.DataFrame:
    """Per population and condition, its share against the control's in each biological
    repeat, on the log-odds scale (a population and its complement alike, a small share as
    steady as a large one), tested across the repeats: the one-sample t-test on the log odds
    ratios, Benjamini-Hochberg within each sub-experiment over every population and
    condition (``_experiments``)."""
    repeat = REPEAT_COLUMN if REPEAT_COLUMN in marked else None
    keys = [CONDITION_COLUMN, *([repeat] if repeat else [])]
    counts = marked.groupby(keys, dropna=False)[names].agg(["sum", "count"])
    conditions = [c for c in pd.unique(marked[CONDITION_COLUMN]) if c != control]
    rows, held, unpaired = [], {}, []
    for condition in conditions:
        repeats = (
            [
                r
                for r in pd.unique(marked[repeat])
                if (condition, r) in counts.index and (control, r) in counts.index
            ]
            if repeat
            else [None]
        )
        if not repeat and not ((condition,) in counts.index or condition in counts.index):
            continue
        if not repeats:
            unpaired.append(condition)
            continue
        held[condition] = set(repeats)
        for name in names:
            ratios = []
            sides = {"t": [0.0, 0.0], "c": [0.0, 0.0]}  # cells in it, cells, on each side
            for r in repeats:
                t = counts.loc[(condition, r) if repeat else condition, name]
                c = counts.loc[(control, r) if repeat else control, name]
                ratios.append(_log_odds_ratio(t, c))
                sides["t"] = [sides["t"][0] + t["sum"], sides["t"][1] + t["count"]]
                sides["c"] = [sides["c"][0] + c["sum"], sides["c"][1] + c["count"]]
            # Absent from the condition and from its own control: nothing to compare.
            ratios = np.array([r for r in ratios if np.isfinite(r)])
            if not len(ratios):
                continue
            tested = len(ratios) >= MIN_REPEATS
            rows.append(
                {
                    "feature": name,
                    "condition": condition,
                    "repeats": len(ratios),
                    # The sides as the test reads them (the sign of each repeat's ratio).
                    "higher": int((ratios > 0).sum()),
                    "lower": int((ratios < 0).sum()),
                    "d": float(ratios.mean()),
                    "p_paired": paired_t(ratios) if tested else np.nan,
                    # Over the condition's repeats: its share and its own control's.
                    "share": sides["t"][0] / sides["t"][1] if sides["t"][1] else np.nan,
                    "control_share": sides["c"][0] / sides["c"][1] if sides["c"][1] else np.nan,
                }
            )
    columns = ["feature", "condition", "repeats", "higher", "lower", "d", "p_paired"]
    tests = pd.DataFrame(rows, columns=[*columns, "share", "control_share"])
    if len(tests):
        tests["experiment"] = tests["condition"].map(experiments(held))
        tests["q_paired"] = corrected(tests["p_paired"], tests["experiment"])
    else:
        tests["q_paired"] = pd.Series(dtype=float)
    # The conditions imaged in no biological repeat of the control: not compared, and said.
    tests.attrs["unpaired"] = unpaired
    return tests


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
    (the smallest q), or ``–``."""
    tests = _tests(result, name)
    if not len(tests):
        return "–"
    best = tests.sort_values(["q_paired", "p_paired"], na_position="last").iloc[0]
    versus = _versus(best)
    q = f", q {p_value(best['q_paired'])}" if pd.notna(best["q_paired"]) else ""
    return f"{best['condition']}: {_side(best)}{', ' + versus if versus else ''}{q}"


def _verdict(result: PopulationsResult) -> tuple[list[str], list[str]]:
    """The populations a condition is enriched in across repeats, and those with a trend."""
    significant, trend = [], []
    for name in result.populations.names:
        tests = _tests(result, name)
        if not len(tests):
            continue
        if (tests["q_paired"] <= ALPHA).any():
            significant.append(name)
        elif one_sided(tests).any():
            trend.append(name)
    return significant, trend


def _rarest(result: PopulationsResult, names: list[str]) -> list[str]:
    shares = result.populations.table.set_index("population")["share"]
    return sorted(names, key=lambda name: shares[name])


def _finding(result: PopulationsResult, name: str, units: str, trend: bool = False) -> str:
    """``P2 (3% of cells; high dna_intensity): more common in drugB across repeats (7% vs
    2%, q 0.048)``, for its significant conditions; with ``trend``, those on one side of the
    control in every repeat, ``in every repeat, not significant``."""
    row = result.populations.table.set_index("population").loc[name]
    tests = _tests(result, name)
    if trend:
        tests = tests[one_sided(tests)]
    else:
        tests = tests[tests["q_paired"] <= ALPHA]
    parts = []
    for _, t in tests.iterrows():
        more = "more" if t["d"] > 0 else "less"
        versus = _versus(t)
        versus = f"{versus}, " if versus else ""
        parts.append(f"{more} common in {t['condition']} ({versus}q {p_value(t['q_paired'])})")
    feature = traits(row["defining"][:1])
    where = "in every repeat, not significant" if trend else "across repeats"
    share = percent_fine(row["share"])
    return f"{name} ({share} of {units}; {feature}): {listing(parts, 3)} {where}"


def _label_line(result: PopulationsResult, label: str) -> str | None:
    if result.label is None or not len(result.label):
        return None
    best = result.label.sort_values("held", ascending=False).iloc[0]
    return (
        f"The cells labelled {label}: {percent(best['held'])} of them in {best['population']}, "
        f"which they make {percent(best['purity'])} of."
    )


_KIND_WORDS = {"main": "main (the rest)", "group": "group"}


def _found_words(pops: Populations) -> str:
    """``1 group of cells to validate``, in the report's words for its observations."""
    groups = len(pops.of_kind("group"))
    units = unit_plural(pops.settings.get("observation_unit") or "observation")
    return f"{count(groups, 'group')} of {units} to validate on the images"


def _rule_cell(row: Any) -> str:
    """``AR > 3.18 and MinFeret ≤ 17.7 (91%, 97%)`` for the table; ``–`` without a rule."""
    if not row.rule:
        return "–"
    if pd.isna(row.rule_recall):
        return row.rule
    return f"{row.rule} ({percent(row.rule_recall)}, {percent(row.rule_precision)})"


def rule_words(row: Any) -> str:
    """``picked by AR > 3.18 and MinFeret ≤ 17.7 (91% of its cells, 97% of those picked, on
    held-out cells)``, or empty without a rule."""
    if not row["rule"]:
        return ""
    if pd.isna(row["rule_recall"]):
        return f"picked by {row['rule']}"
    return (
        f"picked by {row['rule']} ({percent(row['rule_recall'])} of its cells, "
        f"{percent(row['rule_precision'])} of those picked, on held-out cells)"
    )


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
        f"{count(settings['observations'], setup.observation_unit or 'observation')} of every "
        f"condition pooled, clustered on the {count(len(settings['features']), 'declared feature')}"
        f" (HDBSCAN on a {settings['map_components']}-dimensional UMAP of their robust "
        f"standardisation; a cluster holds at least {settings['min_cells']} {units}), each "
        "cluster then judged on all the features"
    )
    if not settings.get("clustered", True):
        text = wrap(
            f"{method}: not clustered: too few {units} (or no feature that varies) for two "
            f"clusters of {settings['min_cells']}.",
            2,
            hang=4,
        )
        return Status.NOT_ASSESSED, f"not clustered: too few {units}", text
    if not pops.names:
        text = wrap(
            f"{method}: no group; the {units} form one continuum on these features.",
            2,
            hang=4,
        )
        return Status.OK, "no group: one continuum on these features", text
    lines = [wrap(f"{method}: {_found_words(pops)} ({_shares_line(pops, units)}).", 2, hang=4)]
    control = result.control
    heading = f"most different (vs {control})" if control is not None else "most different"
    tested = set(tested_populations(pops))
    rows = [
        [
            r.population,
            _KIND_WORDS.get(r.kind, r.kind),
            f"{r.observations} ({percent_fine(r.share)})",
            defined_by(r.defining[:2]) or "–",
            percent(r.purity) if pd.notna(r.purity) else "–",
            _rule_cell(r),
            r.fields,
            r.repeats,
            _enriched(result, r.population) if r.population in tested else "the rest (not tested)",
        ]
        for r in pops.table.itertuples()
    ]
    columns = [
        "population",
        "kind",
        units,
        "defined by",
        "neighbours inside",
        "rule (held-out: found, right)",
        "fields",
        "repeats",
        heading,
    ]
    lines.append(table(rows, columns, indent=4))
    significant, trend = _verdict(result)
    for name in pops.names:
        row = pops.table.set_index("population").loc[name]
        tests = _tests(result, name)
        if not len(tests):
            continue
        parts = []
        sided = one_sided(tests)
        for index, t in tests.iterrows():
            enriched = pd.notna(t["q_paired"]) and t["q_paired"] <= ALPHA
            consistent = bool(sided.loc[index])
            side = "higher" if t["d"] > 0 else "lower"
            versus = _versus(t)
            versus = f"{versus}, " if versus else ""
            if enriched:
                parts.append(
                    f"{t['condition']} {side}, significant across repeats ({versus}q "
                    f"{p_value(t['q_paired'])})"
                )
            elif consistent:
                q = f"q {p_value(t['q_paired'])}" if pd.notna(t["q_paired"]) else ""
                inside = ", ".join(x for x in (versus.rstrip(", "), q) if x)
                inside = f" ({inside})" if inside else ""
                parts.append(f"{t['condition']} {side} in every repeat, not significant{inside}")
        if parts:
            lines.append(
                wrap(
                    f"- {name} ({percent_fine(row['share'])} of {units}; "
                    f"{defined_by(row['defining'][:2])}; {rule_words(row)}): "
                    f"{'; '.join(parts)}.",
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
    if setup.rare_label is not None:
        line = _label_line(result, setup.rare_label)
        if line:
            lines.append(wrap(line, 4, hang=6))
    lines.append(
        wrap(
            f"Assumptions: The {units} of every condition are clustered together, then each "
            f"condition is compared with its own control on its share of {units} in each, "
            "per biological repeat (paired t-test on the log odds of the repeat shares, "
            "Benjamini-Hochberg within each sub-experiment over every population and "
            "condition); the main population, holding most of them, is the rest, listed and "
            "not tested. A cluster on the UMAP is kept as a group when "
            f"{percent(settings['purity'])} or more of its {units}' "
            f"{settings['neighbours']} nearest neighbours on all the features are in it too "
            "(neighbours inside). Whether a group is a separate state or the end of a "
            f"continuum (the most extreme {units} of a gradient) is not decided from the "
            "numbers: no test of a gap told the two apart reliably at these sizes, so open "
            "the images of its fields to validate it. Its rule is the simplest one on the "
            "features (at most two thresholds, a decision tree) that picks out its "
            f"{units}, with the share of them it finds and the share right of those it picks, "
            f"on {units} it was not fitted on. Groups that overlap strongly on every feature "
            "are not told apart. The clusters depend on the declared features and their "
            "standardisation; a group held by few fields may be an artefact of those images.",
            2,
            hang=4,
        )
    )
    found = _found_words(pops)
    if significant:
        headline = f"{found}; " + "; ".join(
            _finding(result, name, units) for name in _rarest(result, significant)[:3]
        )
        status = Status.OK
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
    "PopulationsResult",
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

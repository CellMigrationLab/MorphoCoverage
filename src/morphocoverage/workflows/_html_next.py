"""What to do next, after each answer of the HTML report (private; ``html_report``):
the analysis that follows from what was found, with the code to run it (owner, 2026-10-02).

- ``conditions_next``: for a feature significant across repeats, plan the imaging around it
  (``optimise`` with it as the readout); with fewer than 3 biological repeats, or a condition
  not yet significant, how many repeats to plan.
- ``populations_next``: for a population (the most enriched, else the largest but the main
  one), label it and declare it as the rare phenotype, so ``report`` says whether it was seen
  in enough fields and ``optimise`` plans for it.
- ``enough_next``: for a feature not yet shown in every repeat, plan its fields.
- ``fields_next``: run the report again without the fields found to be artefacts.

The code reads ``result`` (the ``Report``), ``objects`` and ``fovs``, as named in the
Python examples of ``docs/workflows.md``.
"""

from __future__ import annotations

import html
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from .._wording import count, percent_fine
from ..data_check import validate_input_tables
from ..metadata import MORPHOBRICKS_GROUP_COLUMNS
from ..sampling_unit.contrast_repeats import MIN_REPEATS
from ..sampling_unit.contrast_sampling import PRACTICAL
from ..sampling_unit.populations import traits
from ..sampling_unit.repeat_plan_text import field_words
from ._enough_targets import HELD_BACK
from ._populations import rule_words, tested_populations

# The reservoir optimise holds back to judge each stop, as in docs/workflows.md (one
# value with the fingerprint's answer, ``_enough_targets.HELD_BACK``).
RESERVOIR = HELD_BACK
# A population this labelled is the declared rare label's: not declared again.
LABELLED = 0.9
# The fields named to open for a group.
FIELDS_SHOWN = 3
_COUNTS = (2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96)


def fov_counts(fovs: pd.DataFrame) -> list[int]:
    """Up to 4 field counts optimise can replay: within the fields a well keeps once the
    reservoir is held back (its median well); empty when wells are too small to plan from."""
    groups = [c for c in MORPHOBRICKS_GROUP_COLUMNS if c in fovs]
    sizes = fovs.groupby(groups, dropna=False).size() if groups else pd.Series([len(fovs)])
    usable = int(np.floor(float(sizes.median()) * (1 - RESERVOIR)))
    counts = [c for c in _COUNTS if c <= usable]
    if len(counts) > 4:
        picks = np.linspace(0, len(counts) - 1, 4).round().astype(int)
        counts = [counts[i] for i in picks]
    return counts


def _block(steps: Sequence[tuple[str, str]]) -> str:
    """The "What to do next" box: each step's sentence and, when it has one, its code,
    folded (owner, 2026-10-02: the sentence says what to do, the code is there when wanted)."""
    parts = "".join(
        f"<p>{html.escape(line)}</p>"
        + (
            f"<details><summary>Show the code</summary><pre><code>{html.escape(code)}"
            "</code></pre></details>"
            if code
            else ""
        )
        for line, code in steps
    )
    return f'<div class="next"><h3>What to do next</h3>{parts}</div>'


# The code runs in a notebook as it is; a script needs the guard, as it uses several processes.
_GUARD = '# In a script, run this under: if __name__ == "__main__":\n'


def _optimise_code(feature: str, counts: list[int], readout: str | None = None) -> str:
    setup = (
        "setup = result.setup\n"
        if feature == readout
        else f"setup = replace(result.setup, readout={feature!r})\n"
    )
    return (
        _GUARD
        + ("" if feature == readout else "from dataclasses import replace\n")
        + "from morphocoverage.workflows import optimise\n\n"
        + setup
        + f"plan = optimise(objects, fovs, setup, fov_counts={counts}, "
        f"reservoir_fraction={RESERVOIR})\n"
        "print(plan.summary())"
    )


def _optimise_line(feature: str, counts: list[int], readout: str | None = None) -> str:
    if not counts:
        return (
            f"To plan the imaging for {feature}, image more fields per well: these wells are "
            "too small for optimise to replay."
        )
    which = "(the readout)" if feature == readout else "with it as the readout"
    return (
        f"To plan the fields per well for {feature} (how many hold its mean to the precision "
        f"asked, and which stopping rule to run live), run optimise on these data {which}:"
    )


def _repeat_counts(discovery: Any) -> dict[Any, int]:
    tests = discovery.tests
    return {c: int(rows["repeats"].max()) for c, rows in tests.groupby("condition", sort=False)}


def _repeats_line(discovery: Any) -> str:
    """With fewer than MIN_REPEATS biological repeats: how many to plan."""
    few = sorted({n for n in _repeat_counts(discovery).values() if n < MIN_REPEATS})
    if not few:
        return ""
    have = " or ".join(str(n) for n in few)
    return (
        f"With {have} biological repeat{'s' if few != [1] else ''}, no difference can be called "
        f"significant: the test across repeats needs at least {MIN_REPEATS}. Plan at least "
        f"{MIN_REPEATS} (more for small effects), each with its own control; the candidates "
        "above say which features to watch."
    )


def conditions_next(result: Any, fovs: pd.DataFrame) -> str:
    """What to do after "Do the conditions differ?"; empty without a feature discovery."""
    discovery = result.discovery
    if discovery is None or discovery.tests is None or not len(discovery.tests):
        return ""
    columns = set(result.setup.columns)
    tests = discovery.tests
    significant = tests[tests["confirmed"].fillna(False).astype(bool)]
    ranked = [f for f in discovery.ranking["feature"] if f in set(significant["feature"])]
    steps = []
    counts = fov_counts(fovs)
    feature = next((f for f in ranked if f in columns), None)
    if feature is not None:
        steps.append(
            (
                _optimise_line(feature, counts, result.setup.readout),
                _optimise_code(feature, counts, result.setup.readout) if counts else "",
            )
        )
    repeats = _repeats_line(discovery)
    if repeats:
        steps.append((repeats, ""))
    return _block(steps) if steps else ""


def _chosen_population(populations: Any) -> Any | None:
    """The group to follow: the most consistently enriched, else the largest (never the
    main population); its row of the populations table."""
    table = populations.populations.table
    tested = tested_populations(populations.populations)
    if not tested:
        return None
    name = tested[0]
    enrichment = populations.enrichment
    if enrichment is not None and len(enrichment) and "q_paired" in enrichment:
        rows = enrichment[enrichment["feature"].isin(tested)].sort_values("q_paired")
        if len(rows) and np.isfinite(float(rows["q_paired"].iloc[0])):
            name = str(rows["feature"].iloc[0])
    return table.set_index("population").loc[name].rename(name)


def _labelled(populations: Any, name: str, label: str | None) -> bool:
    """Whether ``name`` is the declared rare label's population (most of it labelled)."""
    overlap = getattr(populations, "label", None)
    if not label or overlap is None or not len(overlap):
        return False
    row = overlap[overlap["population"] == name]
    return bool(len(row)) and float(row["purity"].iloc[0]) >= LABELLED


def _group_fields(
    result: Any, populations: Any, objects: pd.DataFrame | None, fovs: pd.DataFrame, name: str
) -> list[str]:
    """The fields holding most of the group's cells (at most ``FIELDS_SHOWN``), named by
    their FOV key, to open."""
    features = populations.populations.settings.get("features") or []
    if objects is None or not features:
        return []
    data = validate_input_tables(
        objects, fovs=fovs, feature_cols=features[:1], **result.setup.common
    )
    key = [c for c in data.fov_key if c in data.objects]
    labels = populations.populations.labels.reindex(data.objects.index)
    mine = data.objects.loc[labels.eq(name).to_numpy(), key].astype(str)
    named = mine.agg(" · ".join, axis=1).value_counts()
    return [str(f) for f in named.index[:FIELDS_SHOWN]]


def populations_next(result: Any, fovs: pd.DataFrame, objects: pd.DataFrame | None = None) -> str:
    """What to do after "Are there populations or groups of cells?": validate the group on
    its images, then label it with its rule and study it as the rare phenotype; empty
    without a group."""
    populations = (getattr(result, "results", {}) or {}).get("populations")
    row = _chosen_population(populations) if populations is not None else None
    if row is None:
        return ""
    name, share = str(row.name), float(row["share"])
    label = result.setup.rare_label
    if _labelled(populations, name, label):
        line = (
            f"{name} is the declared rare phenotype, {label}: most of its cells carry the "
            "label. Its sampling is answered in “Did I image enough?”, with the label "
            "itself; nothing to declare again."
        )
        return _block([(line, "")])
    fields = _group_fields(result, populations, objects, fovs, name)
    where = f" (most of its cells are in {', '.join(fields)})" if fields else ""
    validate = (
        f"First validate {name} ({percent_fine(share)} of the cells; "
        f"{traits(row['defining'][:2])}): open the images of its fields{where} and check "
        "that its cells are one state, not the end of a gradient or an artefact."
    )
    rule = rule_words(row)
    study = (
        f"If it is, label its cells with its rule ({rule}) and declare it as the rare "
        "phenotype: the report then says in “Did I image enough?” whether it was seen in "
        "enough fields of every well, and optimise plans the fields to find it. The rule "
        "labels new acquisitions the same way; adjust its thresholds after looking at the "
        "images."
        if rule
        else "If it is, label its cells (the clustering's labels) and declare it as the rare "
        "phenotype: the report then says in “Did I image enough?” whether it was seen in "
        "enough fields of every well, and optimise plans the fields to find it."
    )
    counts = fov_counts(fovs)
    phenotype = f"{name}: {traits(row['defining'][:2])}"
    labelled = (
        f"objects[{name!r}] = {row['rule_code']}\n"
        if row["rule"]
        else 'found = result.results["populations"].populations.labels\n'
        f"objects[{name!r}] = found.reindex(objects.index).eq({name!r})\n"
    )
    code = (
        _GUARD + "from dataclasses import replace\n"
        "from morphocoverage import RareTarget\n"
        "from morphocoverage.workflows import optimise, report\n\n"
        + labelled
        + "setup = replace(\n"
        "    result.setup,\n"
        f"    rare=RareTarget(object_prevalence={round(share, 4)}, detection_probability=0.95,\n"
        f"                    min_fov_occurrences=3, phenotype={phenotype!r}),\n"
        f"    rare_label={name!r},\n"
        ")\n"
        "result = report(objects, fovs, setup)\n"
        + (
            f"plan = optimise(objects, fovs, setup, fov_counts={counts}, "
            f"reservoir_fraction={RESERVOIR})\n"
            if counts
            else ""
        )
    )
    return _block([(validate, ""), (study, code.rstrip("\n"))])


def enough_next(result: Any, fovs: pd.DataFrame, features: Any) -> str:
    """What to do after "Did I image enough?": the feature that needs the most cells to be
    shown in every repeat, its fields planned; the repeats when there are too few."""
    steps = []
    counts = fov_counts(fovs)
    table = getattr(features, "table", None)
    if table is not None and len(table) and "fields_low" in table:
        # In the table's order (significant across repeats first): the first not yet shown
        # in every comparison whose need can be projected.
        open_ = table[(table["shown"] < table["comparisons"]) & table["fields_low"].notna()]
        open_ = open_[open_["feature"].isin(set(result.setup.columns))]
        if len(open_):
            row = open_.iloc[0]
            low, high = int(row["fields_low"]), int(row["fields_high"])
            about = f"{low}" if low == high else f"{low}-{high}"
            unit = field_words(next(iter(features.contrasts.values())))[1]
            needing = int(row["needing"])
            # Only the comparisons more fields can show (audit, 2026-10-02).
            small = int(row["too_small"])
            beyond = (
                f"; {count(small, 'comparison')} would not show even with {PRACTICAL} times "
                f"the {unit}"
                if small
                else ""
            )
            line = (
                f"{row['feature']}: the {count(needing, 'comparison')} not yet shown would need "
                f"about {about} {unit} a side in its repeat{beyond}. "
                + _optimise_line(row["feature"], counts, result.setup.readout)
            )
            code = _optimise_code(row["feature"], counts, result.setup.readout) if counts else ""
            steps.append((line, code))
    if result.discovery is not None and len(result.discovery.tests):
        repeats = _repeats_line(result.discovery)
        if repeats:
            steps.append((repeats, ""))
    return _block(steps) if steps else ""


def fields_next(fields: Sequence[Sequence[Any]], key: Sequence[str]) -> str:
    """What to do after "Fields to check": once the images confirm a field is an artefact,
    the report again without it (``fields``: the flagged fields' values of ``key``, the
    field's identity, one copy of each pair the integrity checks name; the code lists them
    all, to delete the line of each one to keep)."""
    if not fields or not key:
        return ""
    shown = list(dict.fromkeys(tuple(str(v) for v in field) for field in fields))
    listed = "".join(f"    {f!r},\n" for f in shown)
    code = (
        _GUARD + "from morphocoverage.workflows import report\n\n"
        f"key = {list(key)!r}\n"
        "# Every field flagged above (one copy of each pair): delete the line of any to keep.\n"
        f"artefacts = [\n{listed}]\n"
        "keep = ~fovs[key].astype(str).apply(tuple, axis=1).isin(artefacts)\n"
        "inside = objects[key].astype(str).apply(tuple, axis=1).isin(artefacts)\n"
        "result = report(objects[~inside], fovs[keep], result.setup, integrity=True)"
    )
    line = (
        "Open the images of the fields above: keep those that show biology. If you confirm "
        "that some should be excluded (out of focus, debris, a field exported twice), run the "
        "report again without them."
    )
    return _block([(line, code)])


__all__ = ["conditions_next", "enough_next", "fields_next", "fov_counts", "populations_next"]

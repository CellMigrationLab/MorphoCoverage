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

from .._wording import percent_fine
from ..metadata import MORPHOBRICKS_GROUP_COLUMNS
from ..sampling_unit.contrast_repeats import MIN_REPEATS
from ..sampling_unit.repeat_plan_text import field_words
from ._populations import tested_populations

# The reservoir optimise holds back to judge each stop, as in docs/workflows.md.
RESERVOIR = 0.35
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
    """The "What to do next" box: each step's sentence and, when it has one, its code."""
    parts = "".join(
        f"<p>{html.escape(line)}</p>"
        + (f"<pre><code>{html.escape(code)}</code></pre>" if code else "")
        for line, code in steps
    )
    return f'<div class="next"><h3>What to do next</h3>{parts}</div>'


def _optimise_code(feature: str, counts: list[int]) -> str:
    return (
        "from dataclasses import replace\n"
        "from morphocoverage.workflows import optimise\n\n"
        f"setup = replace(result.setup, readout={feature!r})\n"
        f"plan = optimise(objects, fovs, setup, fov_counts={counts}, "
        f"reservoir_fraction={RESERVOIR})\n"
        "print(plan.summary())"
    )


def _optimise_line(feature: str, counts: list[int]) -> str:
    if not counts:
        return (
            f"To plan the imaging for {feature}, image more fields per well: these wells are "
            "too small for optimise to replay."
        )
    return (
        f"To plan the imaging for {feature} (how many fields each well needs, and which "
        "stopping rule to run live), make it the readout and run optimise on these data:"
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
            (_optimise_line(feature, counts), _optimise_code(feature, counts) if counts else "")
        )
    repeats = _repeats_line(discovery)
    if repeats:
        steps.append((repeats, ""))
    return _block(steps) if steps else ""


def _chosen_population(populations: Any) -> tuple[str, float, str] | None:
    """The population to follow: the most consistently enriched, else the largest of those
    tested (never the main one); its name, share and definition."""
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
    row = table.set_index("population").loc[name]
    return name, float(row["share"]), str(row["defined_by"])


def populations_next(result: Any, fovs: pd.DataFrame) -> str:
    """What to do after "Are there distinct populations?"; empty without one to follow."""
    populations = (getattr(result, "results", {}) or {}).get("populations")
    chosen = _chosen_population(populations) if populations is not None else None
    if chosen is None:
        return ""
    name, share, defined = chosen
    counts = fov_counts(fovs)
    phenotype = f"{name}: {defined.split(' (d ')[0]}" if defined else name
    lines = [
        (
            f"To study {name} ({percent_fine(share)} of the cells) further, label its cells "
            "and declare it as the rare phenotype: the report then says in “Did I image "
            "enough?” whether it was seen in enough fields of every well, and optimise plans "
            "the fields to find it."
        )
    ]
    code = (
        "from dataclasses import replace\n"
        "from morphocoverage import RareTarget\n"
        "from morphocoverage.workflows import optimise, report\n\n"
        'found = result.results["populations"].populations.labels\n'
        f"objects[{name!r}] = found.reindex(objects.index).eq({name!r})\n"
        "setup = replace(\n"
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
    return _block([(lines[0], code.rstrip("\n"))])


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
            line = (
                f"{row['feature']} would be shown in every repeat with about {about} {unit} a "
                "side. " + _optimise_line(row["feature"], counts)
            )
            steps.append((line, _optimise_code(row["feature"], counts) if counts else ""))
    if result.discovery is not None and len(result.discovery.tests):
        repeats = _repeats_line(result.discovery)
        if repeats:
            steps.append((repeats, ""))
    return _block(steps) if steps else ""


def fields_next(fields: Sequence[Sequence[Any]], key: Sequence[str]) -> str:
    """What to do after "Fields to check": the report again without the fields whose images
    show an artefact (``fields``: their values of ``key``, the field's identity)."""
    if not fields or not key:
        return ""
    shown = [tuple(str(v) for v in field) for field in fields[:3]]
    listed = ",\n    ".join(repr(f) for f in shown)
    more = "    # ... the others whose images show an artefact\n" if len(fields) > 3 else ""
    code = (
        "from morphocoverage.workflows import report\n\n"
        f"key = {list(key)!r}\n"
        f"artefacts = [\n    {listed},\n{more}]\n"
        "keep = ~fovs[key].astype(str).apply(tuple, axis=1).isin(artefacts)\n"
        "result = report(objects.merge(fovs.loc[keep, key], on=key), fovs[keep], result.setup)"
    )
    lines = [
        (
            "Open the images of the fields above. Keep those that show biology; run the "
            "report again without those that show an artefact (out of focus, debris, a field "
            "exported twice):"
        )
    ]
    return _block([(lines[0], code)])


__all__ = ["conditions_next", "enough_next", "fields_next", "fov_counts", "populations_next"]

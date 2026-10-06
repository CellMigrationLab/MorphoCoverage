"""What to do next, after the answers of the HTML report that call for a look at the images
(private; ``html_report``; owner, 2026-10-02): the step that follows from what was found, with
the code to run it. Planning the next experiment, with ``optimise`` and ``live``, is the page's
last section (``_html_plan``, which reuses ``fov_counts``, ``_optimise_code`` and
``_repeats_line``).

- ``populations_next``: for a group (the most enriched, else the largest but the main one),
  validate it on its images (for example a few of its fields, with their counts), then,
  when it is rare and not mostly one condition, label it with its rule and declare it as the
  rare phenotype, so ``report`` says whether it was seen in enough fields (owner, 2026-10-06,
  audit 10).
- ``fields_next``: run the report again without the fields found to be artefacts.

The code reads ``result`` (the ``Report``), ``objects`` and ``fovs``, as named in the
Python examples of ``docs/report/running_a_report.md``.
"""

from __future__ import annotations

import html
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from .._wording import count, listing, percent_fine
from ..data_check import validate_input_tables
from ..metadata import MORPHOBRICKS_GROUP_COLUMNS
from ..sampling_unit.contrast_repeats import MIN_REPEATS
from ..sampling_unit.populations import traits
from ..schema import observation_words
from ._enough_targets import HELD_BACK
from ._population_context import RARE_SHARE, by_evidence, mostly_words
from ._population_spaces import described, found_words
from ._populations import labelled_populations, rule_words, tested_populations

# The reservoir optimise holds back to judge each stop, as in docs/report/questions.md (one
# value with the fingerprint's answer, ``_enough_targets.HELD_BACK``).
RESERVOIR = HELD_BACK
# The fields named to open for a group.
FIELDS_SHOWN = 3
_COUNTS = (2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96)


def fov_counts(fovs: pd.DataFrame, group_cols: Sequence[str] | None = None) -> list[int]:
    """Up to 4 field counts optimise can replay: within the fields a well keeps once the
    reservoir is held back (its median well); empty when wells are too small to plan from.
    The wells are the analysis groups: ``group_cols`` (``Setup.group_cols``) when set, else
    the MorphoBricks group columns."""
    names = MORPHOBRICKS_GROUP_COLUMNS if group_cols is None else group_cols
    groups = [c for c in names if c in fovs]
    sizes = fovs.groupby(groups, dropna=False).size() if groups else pd.Series([len(fovs)])
    usable = int(np.floor(float(sizes.median()) * (1 - RESERVOIR)))
    counts = [c for c in _COUNTS if c <= usable]
    if len(counts) > 4:
        picks = np.linspace(0, len(counts) - 1, 4).round().astype(int)
        counts = [counts[i] for i in picks]
    return counts


def rule_labels(table: pd.DataFrame, terms: Sequence[Sequence[Any]]) -> pd.Series:
    """Which rows of ``table`` a rule's terms pick (a missing value is not picked)."""
    picked = pd.Series(True, index=table.index)
    for feature, op, threshold in terms:
        column = pd.to_numeric(table[feature], errors="coerce")
        picked &= column > threshold if op == ">" else column <= threshold
    return picked


def rule_code(terms: Sequence[Sequence[Any]], table: str) -> str:
    """A rule's terms as pandas code on ``table`` (as ``_population_rule.Rule.code``)."""
    return " & ".join(f'({table}["{f}"] {op} {t:.6g})' for f, op, t in terms)


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


def _mean_tolerances(setup: Any) -> tuple[tuple[float, ...], str]:
    """The tolerances a mean goal plans for, and the code that declares them: the setup's,
    or the default when it declares no mean target (``tolerances=()``, the whole
    distribution; owner, 2026-10-06)."""
    if setup.tolerances:
        return tuple(setup.tolerances), ""
    from dataclasses import fields

    default = next(f.default for f in fields(setup) if f.name == "tolerances")
    return default, f", tolerances={default!r}"


def _optimise_code(
    feature: str, counts: list[int], readout: str | None = None, extra: str = ""
) -> str:
    """The code that plans the fields for the mean of ``feature`` (``extra``: more fields
    of the setup to replace, ``, tolerances=(0.05, 0.025)``)."""
    replaced = ("" if feature == readout else f", readout={feature!r}") + extra
    setup = f"setup = replace(result.setup{replaced})\n" if replaced else "setup = result.setup\n"
    return (
        _GUARD
        + ("from dataclasses import replace\n" if replaced else "")
        + "from morphocoverage.workflows import optimise\n\n"
        + setup
        + f"plan = optimise(objects, fovs, setup, fov_counts={counts}, "
        f"reservoir_fraction={RESERVOIR})\n"
        "print(plan.summary())"
    )


def _repeat_counts(discovery: Any) -> dict[Any, int]:
    tests = discovery.tests
    return {c: int(rows["repeats"].max()) for c, rows in tests.groupby("condition", sort=False)}


def _repeats_line(discovery: Any) -> str:
    """With fewer than MIN_REPEATS biological repeats: how many to plan. When only some
    conditions have so few, those are named (the others are planned in repeats)."""
    counts = _repeat_counts(discovery)
    pilots = {c: n for c, n in counts.items() if n < MIN_REPEATS}
    if not pilots:
        return ""
    few = sorted(set(pilots.values()))
    have = " or ".join(str(n) for n in few)
    plural = "s" if few != [1] else ""
    named = "" if len(pilots) == len(counts) else f" for {listing([str(c) for c in pilots], 6)}"
    unit = observation_words(getattr(discovery, "settings", {}))[0]
    return (
        f"With {have} biological repeat{plural}{named}, no difference can be called "
        f"significant: the test across repeats needs at least {MIN_REPEATS}. Plan at least "
        f"{MIN_REPEATS} (more for small effects), each with its own control; the candidates "
        f"of the {unit}-level screen, in “Do the conditions differ?”, say which features to "
        "watch."
    )


def pilot_conditions(discovery: Any) -> set[str]:
    """The conditions with fewer than MIN_REPEATS biological repeats holding the control."""
    if discovery is None or discovery.tests is None or not len(discovery.tests):
        return set()
    return {str(c) for c, n in _repeat_counts(discovery).items() if n < MIN_REPEATS}


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
        # The smallest q, then the largest effect (audit 10).
        rows = by_evidence(enrichment[enrichment["feature"].isin(tested)])
        if len(rows):
            name = str(rows["feature"].iloc[0])
    return table.set_index("population").loc[name].rename(name)


def _labelled(populations: Any, name: str, label: str | None) -> bool:
    """Whether ``name`` is the declared rare label's population (most of it labelled)."""
    overlap = getattr(populations, "label", None)
    return bool(label) and name in labelled_populations(overlap)


def _group_fields(
    result: Any, populations: Any, objects: pd.DataFrame | None, fovs: pd.DataFrame, name: str
) -> list[tuple[str, int]]:
    """Fields holding the group's cells, the most first (at most ``FIELDS_SHOWN``), each
    with its count of them, named by their FOV key, to open."""
    features = populations.populations.settings.get("features") or []
    if objects is None or not features:
        return []
    data = validate_input_tables(
        objects, fovs=fovs, feature_cols=features[:1], **result.setup.common
    )
    key = [c for c in data.fov_key if c in data.objects]
    inside = populations.populations.membership(name).reindex(data.objects.index)
    mine = data.objects.loc[inside.eq(1).to_numpy(), key].astype(str)
    named = mine.agg(" · ".join, axis=1).value_counts()
    return [(str(field), int(held)) for field, held in named.iloc[:FIELDS_SHOWN].items()]


def not_rare(populations: Any, name: str, share: float, units: str) -> str:
    """Why a group is not suggested as the rare phenotype, or empty: mostly one condition (its
    shift), or more than ``RARE_SHARE`` of the cells (owner, 2026-10-06, audit 10, A13)."""
    mostly = (getattr(populations, "context", {}) or {}).get(name, {}).get("mostly")
    if mostly:
        return f"It is {mostly_words(mostly, units)}, so it is not a rare phenotype to declare."
    if share > RARE_SHARE:
        return (
            f"It holds {percent_fine(share)} of the {units}, more than "
            f"{percent_fine(RARE_SHARE)}: not a rare phenotype to declare."
        )
    return ""


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
    units = result.setup.unit_words
    if _labelled(populations, name, label):
        line = (
            f"Nothing to declare again for {name}: the label {label} already marks its "
            f"{units}, and “Did I image enough?” says whether they were seen in enough fields."
        )
        return _block([(line, "")])
    fields = _group_fields(result, populations, objects, fovs, name)
    cells = int(row["observations"])
    # A few of its fields, with their counts: not "most of its cells" (audit 10, A13).
    shown = sum(held for _, held in fields)
    where = (
        f", for example {listing([f for f, _ in fields], FIELDS_SHOWN)} ({shown} of its "
        f"{count(cells, result.setup.unit_word, units)})"
        if fields
        else ""
    )
    # A group only another feature table found has no rule in the explainable features.
    what = traits(row["defining"][:2]) if described(row) else found_words(row)
    validate = (
        f"First validate {name} ({percent_fine(share)} of the {units}; "
        f"{what}): open the images of its fields{where}; check "
        f"that its {units} are one state, not the end of a gradient or an artefact."
    )
    why_not = not_rare(populations, name, share, units)
    if why_not:  # a whole condition, or not rare: never suggested as the rare phenotype
        return _block([(f"{validate} {why_not}", "")])
    rule = rule_words(row, units)
    study = (
        f"If it is, label its {units} with its rule, {rule}, and declare it as the rare "
        "phenotype: the report then says in “Did I image enough?” whether it was seen in "
        "enough fields of every well. The rule labels new acquisitions the same way; adjust "
        "its thresholds after looking at the images."
        if rule
        else f"If it is, label its {units} (the clustering's labels) and declare it as the "
        "rare phenotype: the report then says in “Did I image enough?” whether it was seen in "
        "enough fields of every well."
    )
    phenotype = f"{name}: {what}" if described(row) else name
    # The share of cells the rule labels, which the code declares (the cluster's share
    # without a rule).
    if row["rule"] and objects is not None:
        share = float(rule_labels(objects, row["rule_terms"]).mean())
    labelled = (
        f"objects[{name!r}] = {row['rule_code']}\n"
        if row["rule"]
        else 'found = result.results["populations"].populations\n'
        f"objects[{name!r}] = found.membership({name!r}).reindex(objects.index).eq(1)\n"
    )
    code = (
        _GUARD + "from dataclasses import replace\n"
        "from morphocoverage import RareTarget\n"
        "from morphocoverage.workflows import report\n\n" + labelled + "setup = replace(\n"
        "    result.setup,\n"
        f"    rare=RareTarget(object_prevalence={round(share, 4)}, detection_probability=0.95,\n"
        f"                    min_fov_occurrences=3, phenotype={phenotype!r}),\n"
        f"    rare_label={name!r},\n"
        ")\n"
        "result = report(objects, fovs, setup)\n"
    )
    return _block([(validate, ""), (study, code.rstrip("\n"))])


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


__all__ = ["fields_next", "fov_counts", "populations_next", "rule_code", "rule_labels"]

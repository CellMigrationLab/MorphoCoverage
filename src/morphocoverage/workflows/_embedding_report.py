"""The report of an embedding-only Setup (private; ``retrospective.report``).

A Setup with no readout and no explainable features (owner, 2026-10-06) reads the other
feature tables alone: the questions that need explainable features say so once each, with the
➖ marker (``NOT_APPLICABLE``), and the rest is answered on the tables:

- "Do the conditions differ?": the whole-table test of each table (``_whole_table``), the
  only test, which the answer leads with; the per-feature discovery, the heatmap and the
  SuperPlots are not applicable;
- "Did I image enough?": the fingerprint of each table (on its first principal components,
  ``_fingerprint_space``) and the labelled phenotype, as the user chose; per feature it is
  not applicable, and the mean-precision rules (no readout) are said once, in the
  declaration;
- unusual observations on the tables' weighted combined space (``_feature_space``);
- the maps, the populations, the separation and the copy checks of each table, as in any
  report.

The Foundation, question 1 (what the features can see) and the fields to check (influence
on the readout) are not applicable.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import pandas as pd

from .._integrity_tables import integrity_by_table
from .._style import DASH
from .._wording import count
from ..data_check import check_tables
from ..retrospective import dataset_overview
from ._document import assemble, with_reasons
from ._enough_targets import fingerprint_precision
from ._feature_space import (
    FeatureTables,
    analysed_tables,
    centring_line,
    separation_in_report,
    table_columns,
    table_lines,
)
from ._populations import populations_in_report
from ._rare_share import unlabelled_note
from ._results import PLAN_TITLE, Plan, Report
from ._shared import control_checks, one_per_field, rare_clustering
from ._whole_table import whole_table_in_report

if TYPE_CHECKING:
    from .setup import Setup

NOT_APPLICABLE = "not applicable: no explainable features"
# The parts of a question that need explainable features, by question.
_PARTS = {
    "unit": "",
    "controls": "",
    "inspect": "",
    "enough": "➖ Per feature (the difference from the control, the mean's precision): ",
    "conditions": "➖ Per feature (the discovery, the heatmap, SuperPlots): ",
    "profile": "➖ Fields against random observations: ",
}


def embedding_reasons(results: dict[str, Any]) -> dict[str, str]:
    """By question, what needs explainable features, once each (see the module); a question
    without any answer on the tables is ``NOT_APPLICABLE`` as a whole."""
    answered = {
        "enough": results.get("fingerprint_precision") is not None
        or results.get("rare_clustering") is not None,
        "conditions": results.get("whole_table") is not None,
        "profile": results.get("clustering") is not None or "rare_clustering" in results,
    }
    reasons = {}
    for key, part in _PARTS.items():
        reasons[key] = f"{part}{NOT_APPLICABLE}" if part and answered.get(key) else NOT_APPLICABLE
    return reasons


def embedding_report(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    setup: Setup,
    data_check: Any,
    tables: FeatureTables | None,
    integrity: bool,
) -> Report:
    """The report of an embedding-only Setup (see the module); ``objects`` already holds the
    tables' columns (``_feature_space.attached``)."""
    if not setup.feature_tables:
        raise ValueError(
            "a Setup without a readout reads the other feature tables alone: pass them, "
            "feature_tables={name: table}, or declare a readout and the explainable features"
        )
    common = setup.common
    if data_check is None:  # the tables were attached after the inputs were checked
        data_check = check_tables(
            objects,
            fovs,
            feature_cols=table_columns(setup),
            control=setup.reference_control,
            **common,
        )
    per_field = not one_per_field(objects, fovs, setup)
    overview = dataset_overview(
        objects,
        fovs,
        feature_cols=table_columns(setup)[:1],  # validated by one; no explainable feature
        control=setup.single_control,
        data_check=data_check,
        feature_tables=table_lines(setup, tables, centring_line(objects, fovs, setup)),
        **common,
    )
    counts = {**overview.counts, "features": 0}
    results: dict[str, Any] = {
        "overview": replace(overview, features=(), counts=counts, value_checks=())
    }
    side: dict[str, str] = {}
    # With normalize, the tables centred per repeat (never for a test, owner 2026-10-06).
    analysed = analysed_tables(objects, fovs, setup)
    if setup.reference_control is not None:
        results.update(control_checks(analysed, fovs, setup, side, per_field))
    if integrity:  # on each table apart, the union said
        results["integrity"] = integrity_by_table(
            objects, fovs, tables=dict(setup.feature_tables), **common
        )
    if setup.rare_label is not None:
        results["rare_clustering"] = rare_clustering(objects, fovs, setup)
    # The only test of "Do the conditions differ?" (owner, 2026-10-06).
    results["whole_table"] = whole_table_in_report(objects, fovs, setup)
    if setup.fingerprint:
        results["fingerprint_precision"] = fingerprint_precision(objects, fovs, setup)
    results["populations"] = populations_in_report(analysed, fovs, setup)
    results["separation"] = separation_in_report(analysed, fovs, setup)
    reasons = embedding_reasons(results)
    reasons.update({k: v for k, v in side.items() if k not in reasons})
    document = with_reasons(assemble(results, setup), reasons)
    unlabelled = unlabelled_note(objects, setup)
    if unlabelled:
        document = replace(document, notes=(*document.notes, unlabelled))
    return Report(results, document, setup, reasons, {})


def embedding_plan(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    setup: Setup,
    results: dict[str, Any],
    reasons: dict[str, str],
    space: Any,
) -> Plan:
    """``optimise`` for an embedding-only Setup (owner, 2026-10-06): the declared targets (the
    fingerprint of each table, the labelled phenotype) judged on held-back fields, across
    samples when a hierarchy allows; no stopping rule to replay (no readout), and no effort
    projection (it reads the readout)."""
    no_rule = "not applicable: no stopping rule, as there is no readout"
    for key in ("bought", "order"):
        reasons.setdefault(key, no_rule)
    reasons.setdefault("effort", NOT_APPLICABLE)
    document = assemble(results, setup)
    unit = setup.unit_word
    title = (
        f"{PLAN_TITLE} {DASH} {count(len(objects), unit, setup.unit_words)} in "
        f"{count(len(fovs), 'field')}"
    )
    unread = [
        f"The fingerprint, {c}, is not assessable: {r}." for c, r in space.not_assessable.items()
    ]
    document = replace(
        with_reasons(document, reasons), title=title, notes=(*document.notes, *unread)
    )
    # No settings sentence: there is no replay, so no orders or reservoirs to name.
    return Plan(results, document, setup, reasons, {})


__all__ = ["NOT_APPLICABLE", "embedding_plan", "embedding_reasons", "embedding_report"]

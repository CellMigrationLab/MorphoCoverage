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

from .._control import NO_CONTROL
from .._integrity_tables import integrity_by_table
from .._style import DASH
from .._wording import count
from ..data_check import check_tables
from ..metadata import REPEAT_COLUMN
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
from ._one_per_field import one_per_field, one_per_field_reasons
from ._populations import populations_in_report
from ._rare_share import unlabelled_note
from ._results import PLAN_TITLE, Plan, Report
from ._shared import control_checks, rare_clustering
from ._whole_table import whole_table_in_report

if TYPE_CHECKING:
    from .setup import Setup

NOT_APPLICABLE = "not applicable: no explainable features"
CENTRED_VALUES = (
    "the other feature tables centred per biological repeat (Batch, below), never divided"
)
# The parts of a question that need explainable features, by question.
_PARTS = {
    "unit": "",
    "controls": "",
    "inspect": "",
    "enough": "➖ Per feature (the difference from the control, the mean's precision): ",
    "conditions": "➖ Per feature (the discovery, the heatmap, SuperPlots): ",
    "profile": "➖ Fields against random observations: ",
}


def embedding_reasons(
    results: dict[str, Any],
    setup: Setup | None = None,
    fovs: pd.DataFrame | None = None,
    per_field: bool = True,
) -> dict[str, str]:
    """By question, what needs explainable features, once each (see the module); a question
    without any answer on the tables is ``NOT_APPLICABLE`` as a whole, or says the real
    reason when another one keeps it from running (``_other_reasons``, audit 8B)."""
    answered = {
        "enough": results.get("fingerprint_precision") is not None
        or results.get("rare_clustering") is not None,
        "conditions": results.get("whole_table") is not None,
        "profile": any(results.get(k) is not None for k in ("unusual", "clustering"))
        or "rare_clustering" in results,
    }
    other = _other_reasons(setup, fovs, per_field) if setup is not None else {}
    reasons = {}
    for key, part in _PARTS.items():
        whole = other.get(key, NOT_APPLICABLE)
        reasons[key] = f"{part}{NOT_APPLICABLE}" if part and answered.get(key) else whole
    # One observation per field: what does not apply within a field, beside the rest.
    if not per_field and "profile" in other:
        reasons["profile"] = _one_reason([reasons["profile"], other["profile"]])
    return reasons


def _one_reason(parts: list[str]) -> str:
    """Reasons as one sentence: "not applicable" said once (audit 10, C13)."""
    head = "not applicable: "
    said = [p.replace(f"; {head}", "; ") for p in dict.fromkeys(parts)]
    return "; ".join([said[0], *(p.removeprefix(head) for p in said[1:])])


def _other_reasons(setup: Setup, fovs: pd.DataFrame | None, per_field: bool) -> dict[str, str]:
    """By question, why it has no answer on the tables when the lack of explainable features
    is not the reason: no control, no repeat column, no target declared, one observation per
    field."""
    units = setup.unit_words
    found: dict[str, str] = {}
    if not setup.fingerprint and setup.rare_label is None:
        found["enough"] = (
            "not applicable: no target declared; declare the fingerprint of a table "
            "(fingerprint) or a labelled phenotype (rare with rare_label)"
        )
    if setup.reference_control is None:
        found["conditions"] = f"not applicable: {NO_CONTROL}"  # plan v2 (owner, 2026-10-06)
        found["profile"] = (
            f"not applicable: unusual {units} are judged against the control, and none is declared"
        )
    elif fovs is not None and REPEAT_COLUMN not in fovs:
        found["conditions"] = (
            "not applicable: no biological repeat column, and the whole-table test compares "
            "the repeats"
        )
    for key, reason in ({} if per_field else one_per_field_reasons(setup)).items():
        found[key] = _one_reason([r for r in (found.get(key), reason) if r])  # within a field
    return found


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
    unusual = setup.reference_control is not None  # whether they run (plan v2: per field too)
    overview = dataset_overview(
        objects,
        fovs,
        feature_cols=table_columns(setup)[:1],  # validated by one; no explainable feature
        # normalize centres the tables, never divides them (owner, 2026-10-06).
        normalized=CENTRED_VALUES if setup.normalize is not None else None,
        control=setup.single_control,
        data_check=data_check,
        feature_tables=table_lines(
            setup, tables, centring_line(objects, fovs, setup, unusual), unusual
        ),
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
    results["populations"] = populations_in_report(analysed, fovs, setup, objects)
    results["separation"] = separation_in_report(analysed, fovs, setup, raw=objects)
    reasons = embedding_reasons(results, setup, fovs, per_field)
    reasons.update({k: v for k, v in side.items() if k not in reasons})
    document = with_reasons(assemble(results, setup, per_field), reasons)
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
    why = "there is no readout" if setup.readout is None else "no mean target (tolerances=())"
    no_rule = f"not applicable: no stopping rule, as {why}"
    for key in ("bought", "order"):
        reasons.setdefault(key, no_rule)
    reasons.setdefault("effort", NOT_APPLICABLE if setup.readout is None else no_rule)
    document = assemble(results, setup)
    unit = setup.unit_word
    title = (
        f"{PLAN_TITLE} {DASH} {count(len(objects), unit, setup.unit_words)} in "
        f"{count(len(fovs), 'field')}"
    )
    unread = space.unread()
    unread += space.small_groups(setup.unit_word, setup.unit_words)  # audit 8
    document = replace(
        with_reasons(document, reasons), title=title, notes=(*document.notes, *unread)
    )
    # No settings sentence: there is no replay, so no orders or reservoirs to name.
    return Plan(results, document, setup, reasons, {})


__all__ = ["NOT_APPLICABLE", "embedding_plan", "embedding_reasons", "embedding_report"]

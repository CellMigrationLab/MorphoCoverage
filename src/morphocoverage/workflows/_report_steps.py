"""Steps of ``report`` and ``optimise`` (private; split out of ``workflows.retrospective``):
the dataset overview, the positive control, the questions between the conditions, the
report's document, the targets ``optimise`` can judge, its sampling reliability and the
analyses of its replay.

Shared-private: used by ``workflows.retrospective``. The steps that call an analysis tests
replace there (``sufficiency_report``, ``protocol_reliability``, ``inspect_fovs``) stay in
that module.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import pandas as pd

from ..metadata import MORPHOBRICKS_GROUP_COLUMNS
from ..retrospective import (
    MicroscopistReport,
    RepresentationControl,
    adequacy_points,
    dataset_overview,
    effort_projection,
    order_efficiency,
    redundancy_distance,
    representation_control,
    sampling_reliability,
    stopping_calibration,
)
from ..retrospective.hierarchy import SamplingHierarchy
from ._acquisition_block import acquisition_in_report, gap_moved, with_acquisition
from ._acquisition_gap import with_gap_caution, with_gaps
from ._contrasts import experiments
from ._distribution import distribution_by_feature
from ._document import assemble, with_reasons
from ._enough_features import enough_by_feature
from ._enough_targets import NO_TARGET, fingerprint_precision
from ._feature_space import (
    analysed_tables,
    centring_line,
    separation_in_report,
    table_columns,
    table_lines,
)
from ._populations import populations_in_report
from ._rare_share import unlabelled_note
from ._shared import conditions
from ._whole_table import whole_table_in_report
from .setup import Setup


def overview(
    objects: pd.DataFrame, fovs: pd.DataFrame, setup: Setup, data_check: Any, tables: Any
) -> Any:
    """What the tables hold, with the data check, the feature tables and the controls."""
    return dataset_overview(
        objects,
        fovs,
        feature_cols=setup.columns,
        normalized=setup.normalized_sentence(fovs, objects),
        control=setup.single_control,
        data_check=data_check,
        feature_tables=table_lines(setup, tables, centring_line(objects, fovs, setup)),
        control_experiments=experiments(setup),
        **setup.common,
    )


def positive_control(
    novel: pd.DataFrame, fovs: pd.DataFrame, setup: Setup, space: tuple[Any, list[str]]
) -> Any:
    """R1: the positive control's fields against the reference, on the definition of new
    and the columns it reads (``space``)."""
    combined, novelty_cols = space
    positive = setup.positive
    assert positive is not None
    return representation_control(
        novel,
        fovs,
        feature_cols=novelty_cols,
        novelty=combined,
        control=RepresentationControl(
            positive=positive.selector,
            reference=setup.reference_control,
            planted_objects=positive.planted_objects,
        ),
        pair_cols=list(positive.resolved_pair_cols(fovs)),
        fovs_seen=positive.fovs_seen,
        draws=positive.draws,
        random_state=setup.random_state,
        fov_col=setup.fov_col,
        observation_unit=setup.observation_unit,
        n_jobs=setup.n_jobs,
        # Group columns beyond condition and repeat (a well, say) still name the FOV.
        identity_cols=[c for c in (setup.group_cols or ()) if c not in MORPHOBRICKS_GROUP_COLUMNS],
        # The leave-one-field-out rates within the inspection's analysis groups.
        lofo_group_cols=setup.common["group_cols"],
    )


def comparisons(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    setup: Setup,
    prepared: pd.DataFrame,
    reasons: dict[str, str],
    results: dict[str, Any],
) -> list[dict]:
    """The conditions compared, the acquisition, every feature's and table's answers, the
    fingerprint's precision and the populations, added to ``results`` (in place); returns
    the controls apart in intensity."""
    # The tests read the raw values and normalise to each condition's own control
    # themselves, so their raw columns are raw (owner, 2026-10-05); the populations read
    # the figures' values.
    compared = objects
    results.update(conditions(compared, fovs, setup, reasons))
    gaps = with_gaps(results, objects, fovs, setup)  # several controls apart in intensity
    gap_moved(results)  # said in the acquisition block, beside the fields far from their group
    results["acquisition"] = acquisition_in_report(objects, fovs, setup)
    if "contrast" in results:
        # Question 2 for every feature alike: the report is neutral (owner, 2026-10-01).
        results["enough_features"] = enough_by_feature(
            objects, fovs, compared, setup, results.get("discovery")
        )
        results["distribution"] = distribution_by_feature(objects, fovs, setup)
    # Every other table tested whole, beside the features (owner, 2026-10-06).
    results["whole_table"] = whole_table_in_report(objects, fovs, setup)
    if setup.fingerprint:
        results["fingerprint_precision"] = fingerprint_precision(objects, fovs, setup)
    # Question 10: populations on every condition pooled, then their enrichment.
    # The tables centred per repeat with normalize, as the features are divided.
    analysed = analysed_tables(prepared, fovs, setup)
    results["populations"] = populations_in_report(analysed, fovs, setup, objects)
    # Per feature table, how far the conditions separate (descriptive, owner 2026-10-05).
    results["separation"] = separation_in_report(analysed, fovs, setup, raw=objects)
    return gaps


def report_document(
    results: dict[str, Any],
    setup: Setup,
    per_field: bool,
    reasons: dict[str, str],
    gaps: list[dict],
    objects: pd.DataFrame,
) -> MicroscopistReport:
    """The report's document: the assembled answers, the reasons, the acquisition block and
    the note on unlabelled observations."""
    document = assemble(results, setup, per_field)
    document = with_gap_caution(with_reasons(document, reasons), gaps)
    document = with_acquisition(document, results.get("acquisition"), gaps)
    unlabelled = unlabelled_note(objects, setup)
    if unlabelled:
        # Said once, where the label is used: the notes every answer is read with.
        document = replace(document, notes=(*document.notes, unlabelled))
    return document


def check_criteria(space: Any) -> list[Any]:
    """The criteria ``optimise`` judges on held-back fields; none is refused, with why."""
    criteria = space.setup.criteria()
    if criteria:
        return criteria
    # An embedding-only Setup whose targets cannot be judged.
    unread = space.unread()
    if unread:  # declared, as the report says it (tolerances=() or no readout)
        raise ValueError(
            f"optimise judges the declared targets on held-back fields. {NO_TARGET} "
            + " ".join(unread)
        )
    raise ValueError(
        "optimise judges the declared targets on held-back fields, and this Setup has "
        "none it can judge: without a readout, declare the fingerprint of a table or a "
        "labelled phenotype (rare, rare_label)"
    )


def reliability(
    prepared: pd.DataFrame,
    fovs: pd.DataFrame,
    setup: Setup,
    criteria: list[Any],
    fov_counts: Sequence[int],
    reservoir_fraction: float,
    n_splits: int,
    draws: int,
) -> Any:
    """P1: how often ``k`` random fields preserve each criterion in held-back fields."""
    return sampling_reliability(
        prepared,
        fovs,
        feature_cols=setup.judged_columns or table_columns(setup),
        criteria=criteria,
        fov_counts=fov_counts,
        reservoir_fraction=reservoir_fraction,
        n_splits=n_splits,
        draws=draws,
        random_state=setup.random_state,
        n_jobs=setup.n_jobs,
        **setup.common,
    )


def replay_results(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    setup: Setup,
    replay: Any,
    redundancy_bins: Sequence[float] | None,
    hierarchy: SamplingHierarchy | None,
    strata: Sequence[str],
) -> dict[str, Any]:
    """The stopping rules' calibration and adequacy points from the replay, the order
    efficiency and redundancy distance with stage coordinates, and the effort projection
    with a hierarchy (P3 to P8, P10)."""
    results: dict[str, Any] = {}
    results["calibration"] = stopping_calibration(replay)
    results["adequacy"] = adequacy_points(replay)
    if setup.stage_cols is not None:
        results["efficiency"] = order_efficiency(replay, results["adequacy"])
        if redundancy_bins is not None:
            results["redundancy"] = redundancy_distance(
                objects,
                fovs,
                feature_cols=[setup.readout],
                stage_cols=setup.stage_cols,
                bin_edges=list(redundancy_bins),
                random_state=setup.random_state,
                **setup.common,
            )
    if hierarchy is not None:
        results["effort"] = effort_projection(
            objects,
            fovs,
            feature=setup.readout,
            hierarchy=hierarchy,
            strata=strata,
            random_state=setup.random_state,
            observation_unit=setup.observation_unit,
        )
    return results


__all__ = [
    "check_criteria",
    "comparisons",
    "overview",
    "positive_control",
    "reliability",
    "replay_results",
    "report_document",
]

"""``report``, ``optimise`` and ``benchmark``: the retrospective workflows on a new dataset.

Each takes the two linked tables (every acquired FOV, including empty ones,
and the observations) and a ``Setup``, runs the package's analyses in the
order of ``docs/report/questions.md`` and ``docs/optimise/optimise.md``, and returns every
result with a ``summary()`` in sentences:

- ``report`` (REPORT: the Foundation, then R1 to R8): fields against objects, the representation
  control, what each rule says, the sampling profile with rare-object
  clustering, the fields to inspect, each condition against the control, and the populations;
- ``optimise`` (OPTIMISE): how often ``k`` random fields preserve the
  readout in held-back fields, across samples, what the extra fields bought,
  how reliable each stopping rule was, more fields or more samples, and the
  order of acquisition;
- ``benchmark``: fields saved against a fixed field count at matched risk,
  for rule families swept over threshold grids (``retrospective.frontier``;
  defined in ``workflows.benchmarks`` and re-exported here).

Their scientific choices are the ``Setup``'s fields and the explicit
arguments; the few defaults they carry (``protocol_threshold``, ``alphas``,
the numbers of replayed orders, the benchmark's resamples and grids) are
printed in the summaries, and every random order, draw and resample is
seeded from ``Setup.random_state``. When the setup
asks for it, the observation table is normalised to the control condition's
mean, repeat by repeat (``Setup.prepared``), for the outcome criteria judged
on held-back fields, the condition contrast and the feature ranking; every
stopping rule and every other analysis reads the raw values, so a rule gives
the verdict here that it gives live. Each document opens with the setup's
declaration of what the acquisition was asked to preserve; every replay runs
under the objective made of the declared targets (``Setup.objective()``), and
``optimise`` and ``benchmark`` judge every declared criterion
(``Setup.criteria()``: the mean, the fingerprint, the labelled rare state) on
held-back fields. An analysis the declared analysis groups cannot support
(``group_cols`` that pool the repeats a comparison needs apart) says "not
assessed" and why. ``document`` merges a report and a plan of one setup
into the full microscopist's report, with the settings of both.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields, replace
from typing import Any

import pandas as pd

from .._integrity_tables import integrity_by_table
from .._style import DASH
from .._wording import count
from ..feature_tables import EXPLAINABLE
from ..metadata import MORPHOBRICKS_GROUP_COLUMNS
from ..retrospective import (
    MicroscopistReport,
    RepresentationControl,
    Summary,
    adequacy_points,
    dataset_overview,
    effort_projection,
    inspect_fovs,
    order_efficiency,
    protocol_reliability,
    redundancy_distance,
    representation_control,
    sampling_reliability,
    stopping_calibration,
)
from ..sampling_unit import (
    fov_variability,
)
from ..schema import resolve_observation_unit
from ..sufficiency import sufficiency_report
from ._acquisition_gap import with_gap_caution, with_gaps
from ._contrasts import experiments
from ._distribution import distribution_by_feature
from ._document import (
    assemble,
    no_hierarchy_reasons,
    plan_settings,
    protocol_reason,
    report_settings,
    with_reasons,
    with_settings,
)
from ._enough_features import enough_by_feature
from ._enough_targets import fingerprint_precision
from ._feature_space import (
    attached,
    novelty_space,
    separation_in_report,
    table_lines,
)
from ._fingerprint_space import fingerprint_space
from ._label_control import label_control
from ._populations import populations_in_report
from ._rare_share import unlabelled_note
from ._result_influence import influence_on_result, with_result_influence
from ._results import (
    CALIBRATION_SHORT,
    CALIBRATION_TITLE,
    PLAN_QUESTIONS,
    PLAN_TITLE,
    REPORT_ORDERS,
    REPORT_QUESTIONS,
    Plan,
    Report,
)
from ._shared import (
    check_workflow_inputs,
    conditions,
    control_checks,
    one_per_field,
    one_per_field_reasons,
    rare_clustering,
    replay_rules,
    report_only,
    reservoir_sizes,
    sufficiency_rules,
)
from ._shared import orders as _orders
from ._whole_table import whole_table_in_report
from .benchmarks import Benchmark, Family, benchmark, default_families
from .setup import Setup


def report(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    setup: Setup,
    *,
    integrity: bool = False,
    feature_tables: Mapping[str, pd.DataFrame] | None = None,
) -> Report:
    """REPORT on a finished acquisition: what can be trusted, and what cannot.

    Runs, as the setup allows: ``fov_variability`` and, with a control,
    ``object_rule_check`` (the Foundation); ``representation_control`` with a positive
    control and a definition of new (R1), its fields paired within each repeat
    unless ``PositiveControl.pair_cols`` says otherwise; ``sufficiency_report``
    for the setup's rules in the recorded order and ``REPORT_ORDERS`` random
    orders seeded from ``random_state``, with next-field novelty when
    ``novelty`` is set (R2, R3); ``unusual_objects`` and ``fov_clustering``
    with a control and features, and ``fov_clustering`` of the declared rare
    phenotype's ``rare_label`` (R4); ``inspect_fovs`` (R5);
    ``condition_contrast`` when the control is
    declared by condition and the FOV table holds other conditions, and
    with several declared features ``feature_ranking``, the gated ranking
    that chooses what ``plots`` draws by default (R7). The contrast and the
    ranking read the raw values and give their tests raw and normalised to the control of
    the same repeat (owner, 2026-10-05); every other analysis, the stopping rules included,
    reads the raw values. The analyses
    against the control need the control's columns (and, for the contrast, the
    condition and the repeat) kept apart by the analysis groups; with
    ``group_cols`` that pool them they are "not assessed", with the reason.

    With ``integrity=True`` (opt-in) the overview also reports the exact
    data-integrity checks (``integrity_checks``: copied, reordered or scaled
    fields, repeated blocks and field summaries) on every declared column.

    ``feature_tables`` are other feature tables for the same observations, by name (a DINO
    embedding, owner 2026-10-05): checked and attached (``attach_feature_tables``), never
    tested feature by feature, they join the maps, the populations (in each table and in the
    weighted combined space, ``_feature_space``), the copy checks (on each table apart) and
    their own fingerprints, never novelty or unusual observations (owner, 2026-10-06); the
    report's ``setup`` names them.
    """
    given = feature_tables is not None
    objects, fovs, data_check = check_workflow_inputs(
        objects, fovs, setup, warn=False, tables_given=given
    )
    objects, setup, tables = attached(objects, fovs, setup, feature_tables)
    common = setup.common
    readout = setup.readout
    # One observation per field (one cell per image): the questions about fields do not
    # apply, and are said so once, not answered with warnings.
    per_field = not one_per_field(objects, fovs, setup)
    reasons: dict[str, str] = {} if per_field else one_per_field_reasons(setup)
    novelty = setup.novelty if per_field or "enough" not in reasons else None
    results: dict[str, Any] = {
        "overview": dataset_overview(
            objects,
            fovs,
            feature_cols=setup.columns,
            normalized=setup.normalized_sentence(fovs, objects),
            control=setup.single_control,
            data_check=data_check,
            feature_tables=table_lines(setup, tables),
            control_experiments=experiments(setup),
            **common,
        ),
    }
    if per_field:
        results["variability"] = fov_variability(objects, fovs, feature=readout, **common)
    if setup.reference_control is not None:
        results.update(control_checks(objects, fovs, setup, reasons, per_field))
    if setup.positive is None and setup.rare_label is not None:
        # The labelled rare state is a positive control of its own (owner, 2026-10-02).
        results["label_control"] = label_control(objects, fovs, setup)
    # "New" is judged on the declared features alone, whatever the tables (owner, 2026-10-06).
    combined, novelty_cols = novelty_space(setup)
    if setup.positive is not None:  # Setup guarantees features and novelty with it
        positive = setup.positive
        results["controls"] = representation_control(
            objects,
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
            identity_cols=[
                c for c in (setup.group_cols or ()) if c not in MORPHOBRICKS_GROUP_COLUMNS
            ],
            # The leave-one-field-out rates within the inspection's analysis groups.
            lofo_group_cols=common["group_cols"],
        )
    if integrity:  # on each feature table apart, the union said (owner, 2026-10-06)
        tables_read = {EXPLAINABLE: setup.columns, **(setup.feature_tables or {})}
        results["integrity"] = integrity_by_table(objects, fovs, tables=tables_read, **common)
    if setup.rare_label is not None:
        results["rare_clustering"] = rare_clustering(objects, fovs, setup)
    prepared = setup.prepared(objects, fovs)
    # The tests read the raw values and normalise to each condition's own control
    # themselves, so their raw columns are raw (owner, 2026-10-05); the populations read
    # the figures' values.
    compared = objects
    results.update(conditions(compared, fovs, setup, reasons))
    gaps = with_gaps(results, objects, fovs, setup)  # several controls apart in intensity
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
    results["populations"] = populations_in_report(prepared, fovs, setup)
    # Per feature table, how far the conditions separate (descriptive, owner 2026-10-05).
    results["separation"] = separation_in_report(prepared, fovs, setup)
    rules, novelty_target = sufficiency_rules(setup)
    if novelty is None:
        novelty_target = None
    sufficiency = sufficiency_report(
        objects,
        fovs,
        # The rules read the same columns, and the same raw values, in report,
        # optimise, benchmark and live.
        feature_cols=list(dict.fromkeys([*setup.columns, *(novelty_cols if novelty else [])])),
        strategies=rules,
        objective=setup.objective(),
        # sufficiency_report seeds a permutation by its order's seed alone.
        orders=_orders(setup, permuted=REPORT_ORDERS, dispersed=0, seed_permuted=True),
        novelty=combined if novelty is not None else None,
        novelty_target=novelty_target,
        few_fovs=setup.resolved_few_fovs if novelty is not None else None,
        acquisition_order_col=setup.acquisition_order_col,
        # "New" is judged on the declared features only, as in the controls and the
        # inspection; the rules read the readout.
        novelty_cols=novelty_cols if novelty is not None else None,
        # The order check beside every stop: the readout against order and stage.
        readout=setup.readout,
        stage_cols=setup.stage_cols,
        n_jobs=setup.n_jobs,
        **common,
    )
    results["sufficiency"] = report_only(sufficiency, setup)
    # Every reason at once: new states on the declared features, influence, empty
    # or crowded fields and stage neighbours on the readout.
    novel = novelty is not None and bool(setup.features)
    # With a contrast, a field's influence is on the result (the repeat means the paired
    # test compares, on the values it compares, _result_influence); without one, on the
    # field-weighted mean.
    on_result = influence_on_result(results.get("contrast"))
    results["inspection"] = inspect_fovs(
        objects,
        fovs,
        feature_cols=[readout],
        summaries=[] if on_result else [Summary(readout, "mean")],
        novelty=combined if novel else None,
        novelty_cols=novelty_cols if novel else None,
        stage_cols=setup.stage_cols,
        label_col=setup.rare_label,
        **common,
    )
    if on_result:
        results["inspection"] = with_result_influence(
            results["inspection"], compared, fovs, results["contrast"], setup
        )
    settings = {
        "recorded_order": setup.acquisition_order_col is not None,
        "permuted_orders": REPORT_ORDERS,
        "random_state": setup.random_state,
    }
    document = with_gap_caution(with_reasons(assemble(results, setup), reasons), gaps)
    unlabelled = unlabelled_note(objects, setup)
    if unlabelled:
        # Said once, where the label is used: the notes every answer is read with.
        document = replace(document, notes=(*document.notes, unlabelled))
    return Report(results, document, setup, reasons, settings)


def optimise(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    setup: Setup,
    *,
    fov_counts: Sequence[int],
    reservoir_fraction: float,
    n_splits: int = 5,
    draws: int = 50,
    permuted_orders: int = 10,
    dispersed_orders: int = 5,
    protocol_threshold: float = 0.9,
    redundancy_bins: Sequence[float] | None = None,
) -> Plan:
    """OPTIMISE: what the next experiment should do, from what this one preserved.

    ``fov_counts`` are the numbers of fields to judge (for example
    ``[2, 4, 8, 12]``) and ``reservoir_fraction`` the share of each group's
    fields held back as the reference, a design choice with no default. Runs
    ``sampling_reliability`` with one curve per declared criterion (each
    tolerance of the readout's mean, each fingerprint criterion, the labelled
    rare state) (P1), ``protocol_reliability`` across samples when a hierarchy
    is declared or derived from a ``technical_repeat`` column and every sample
    and stratum is an analysis group (P2), a replay of the setup's rules
    against the same held-back fields for their calibration and adequacy points
    (P3 to P6), ``order_efficiency`` and, with ``redundancy_bins``,
    ``redundancy_distance`` when stage coordinates are set (P7, P8), and
    ``effort_projection`` with that hierarchy (P10: more fields, or more
    technical repeats?). The outcome criteria read the normalised values when
    ``normalize`` is set; the rules, the effort projection and the redundancy
    distance read the raw values. ``protocol_threshold`` (default 0.9, a
    provisional preset) is the share of random acquisitions of ``k`` fields
    that must preserve a criterion for a sample to count as adequately
    represented; the summary prints it. A count of fields is a recommendation
    only once it holds on samples it was not chosen on.
    """
    objects, fovs, _ = check_workflow_inputs(objects, fovs, setup)
    common = setup.common
    hierarchy = setup.resolved_hierarchy(objects, fovs)
    strata = setup.resolved_effort_strata(fovs)
    prepared = setup.compared_values(objects, fovs)
    # A wide table's fingerprint on its first components, as the report reads it.
    space = fingerprint_space(prepared, fovs, setup)
    criteria = space.setup.criteria()
    reasons: dict[str, str] = {}
    results: dict[str, Any] = {}
    results["reliability"] = sampling_reliability(
        prepared,
        fovs,
        feature_cols=setup.judged_columns,
        criteria=criteria,
        fov_counts=fov_counts,
        reservoir_fraction=reservoir_fraction,
        n_splits=n_splits,
        draws=draws,
        random_state=setup.random_state,
        n_jobs=setup.n_jobs,
        **common,
    )
    settings = {
        "protocol_threshold": protocol_threshold,
        "reservoir_fraction": reservoir_fraction,
        "n_splits": n_splits,
        "draws": draws,
        "random_state": setup.random_state,
    }
    if hierarchy is None:
        reasons.update(no_hierarchy_reasons(setup, fovs))
    else:
        groups = results["reliability"].group_cols
        reason = protocol_reason(hierarchy, strata, groups)
        if reason is not None:
            reasons["next_time"] = reason
        else:
            results["protocol"] = protocol_reliability(
                results["reliability"],
                hierarchy,
                threshold=protocol_threshold,
                strata=strata,
                random_state=setup.random_state,
            )
    replay = replay_rules(
        objects,
        prepared,
        fovs,
        setup,
        strategies=list(setup.stopping_rules()),
        criteria=criteria,
        reservoir_fraction=reservoir_fraction,
        orders=_orders(setup, permuted_orders, dispersed_orders),
        acquisition_order_col=setup.acquisition_order_col,
        stage_cols=setup.stage_cols,
        n_splits=n_splits,
        random_state=setup.random_state,
        incremental=True,
        n_jobs=setup.n_jobs,
    )
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
                **common,
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
    document = assemble({k: v for k, v in results.items() if k != "calibration"}, setup)
    settings = {
        **settings,
        "reservoir_fovs": reservoir_sizes(replay),
        "recorded_order": setup.acquisition_order_col is not None,
        "permuted_orders": permuted_orders,
        "dispersed_orders": dispersed_orders if setup.stage_cols is not None else 0,
    }
    unit = resolve_observation_unit(setup.observation_unit)[0]
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
    return Plan(results, document, setup, reasons, settings)


def _same_setup(first: Setup, other: Setup) -> bool:
    """Whether two setups declare the same analysis (``n_jobs`` changes no result)."""
    return all(
        getattr(first, f.name) == getattr(other, f.name)
        for f in fields(Setup)
        if f.name != "n_jobs"
    )


def document(*parts: Report | Plan) -> MicroscopistReport:
    """The full microscopist's report from a ``report`` and a ``plan`` of the same data.

    Every part must come from one setup (``n_jobs`` aside): the declaration
    printed first is the one every answer was computed under. The settings
    sentence of each part is kept among the closing notes.
    """
    if not parts:
        raise ValueError("document takes the results of report() and optimise()")
    merged: dict[str, Any] = {}
    reasons: dict[str, str] = {}
    sentences: list[str] = []
    for part in parts:
        if not isinstance(part, Report | Plan):
            raise TypeError("document takes the results of report() and optimise()")
        if not _same_setup(parts[0].setup, part.setup):
            raise ValueError(
                "document merges parts of one setup: these were computed under different "
                "Setups, so one declaration cannot describe them; run report and optimise "
                "with the same Setup"
            )
        merged.update(part.results)
        reasons.update(part.not_assessed)
        if part.settings:
            writer = report_settings if isinstance(part, Report) else plan_settings
            sentences.append(writer(part.settings))
    merged_document = with_reasons(assemble(merged, parts[0].setup), reasons)
    return with_settings(merged_document, sentences)


__all__ = [
    "CALIBRATION_SHORT",
    "CALIBRATION_TITLE",
    "PLAN_QUESTIONS",
    "REPORT_ORDERS",
    "REPORT_QUESTIONS",
    "Benchmark",
    "Family",
    "Plan",
    "Report",
    "benchmark",
    "default_families",
    "document",
    "optimise",
    "report",
]

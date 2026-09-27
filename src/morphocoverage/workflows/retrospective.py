"""``report``, ``optimise`` and ``benchmark``: the retrospective workflows on a new dataset.

Each takes the two linked tables (every acquired FOV, including empty ones,
and the observations) and a ``Setup``, runs the package's analyses in the
order of ``docs/questions.md``, and returns every result with a ``summary()``
in sentences:

- ``report`` (REPORT: the Foundation, then R1 to R7): fields against objects, the representation
  control, what each rule says, the sampling profile with rare-object
  clustering, the fields to inspect, and each condition against the control;
- ``optimise`` (OPTIMISE / PLAN): how often ``k`` random fields preserve the
  readout in held-back fields, across samples, what the extra fields bought,
  how reliable each stopping rule was, more fields or more samples, and the
  order of acquisition;
- ``benchmark``: fields saved against a fixed field count at matched risk,
  for rule families swept over threshold grids (``retrospective.frontier``).

Their scientific choices are the ``Setup``'s fields and the explicit
arguments; the few defaults they carry (``protocol_threshold``, ``alphas``,
the numbers of replayed orders) are printed in the summaries. When the setup
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
assessed" and why. ``document`` merges a report and a plan into the full
microscopist's report.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ..metadata import MORPHOBRICKS_GROUP_COLUMNS
from ..retrospective import (
    MicroscopistReport,
    RepresentationControl,
    Summary,
    adequacy_points,
    dataset_overview,
    effort_projection,
    inspect_fovs,
    microscopist_report,
    order_efficiency,
    protocol_reliability,
    redundancy_distance,
    representation_control,
    sampling_reliability,
    stopping_calibration,
)
from ..retrospective.frontier import FrontierResult, frontier, sweep
from ..sampling_unit import (
    FeatureRanking,
    fov_variability,
)
from ..strategies import DescriptorSpreadStrategy, FOVBootstrapStrategy, ScalarPrecisionStrategy
from ..strategies.base import SamplingStrategy
from ..sufficiency import sufficiency_report
from ._shared import check_tables as _check_tables
from ._shared import (
    conditions,
    control_checks,
    no_hierarchy_reasons,
    plan_settings,
    protocol_reason,
    replay_rules,
    report_only,
    with_reasons,
)
from ._shared import orders as _orders
from .setup import Setup

REPORT_QUESTIONS = ("unit", "controls", "enough", "profile", "inspect", "conditions")
PLAN_QUESTIONS = ("next_time", "bought", "effort", "order")


@dataclass(frozen=True)
class Report:
    """The REPORT answers: ``results`` by analysis, ``document`` the assembled report.

    ``not_assessed`` holds, by question, why an analysis was not run.
    """

    results: dict[str, Any]
    document: Any
    setup: Setup
    not_assessed: dict[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        """The report questions, in sentences."""
        return self.document.summary(REPORT_QUESTIONS)

    @property
    def ranking(self) -> FeatureRanking | None:
        """The gated feature ranking (question 9), when several features were declared."""
        return self.results.get("ranking")


def report(objects: pd.DataFrame, fovs: pd.DataFrame, setup: Setup) -> Report:
    """REPORT on a finished acquisition: what can be trusted, and what cannot.

    Runs, as the setup allows: ``fov_variability`` and, with a control,
    ``object_rule_check`` (the Foundation); ``representation_control`` with a positive
    control and a definition of new (R1); ``sufficiency_report`` for the
    setup's rules, with next-field novelty when ``novelty`` is set (R2, R3);
    ``unusual_objects`` and ``fov_clustering`` with a control and features
    (R4); ``inspect_fovs`` (R5); ``condition_contrast`` when the control is
    declared by condition and the FOV table holds other conditions, and
    with several declared features ``feature_ranking``, the gated ranking
    that chooses what ``plots`` draws by default (R7). The contrast and the
    ranking read the normalised values when ``normalize`` is set; every other
    analysis, the stopping rules included, reads the raw values. The analyses
    against the control need the control's columns (and, for the contrast, the
    condition and the repeat) kept apart by the analysis groups; with
    ``group_cols`` that pool them they are "not assessed", with the reason.
    """
    _check_tables(objects, fovs, setup)
    common = setup.common
    readout = setup.readout
    reasons: dict[str, str] = {}
    results: dict[str, Any] = {
        "overview": dataset_overview(
            objects,
            fovs,
            feature_cols=setup.columns,
            normalized=setup.normalized_sentence(fovs, objects),
            **common,
        ),
        "variability": fov_variability(objects, fovs, feature=readout, **common),
    }
    if setup.control is not None:
        results.update(control_checks(objects, fovs, setup, reasons))
    if setup.positive is not None:  # Setup guarantees features and novelty with it
        positive = setup.positive
        results["controls"] = representation_control(
            objects,
            fovs,
            feature_cols=list(setup.features),
            novelty=setup.novelty,
            control=RepresentationControl(
                positive=positive.selector,
                reference=setup.control,
                planted_objects=positive.planted_objects,
            ),
            pair_cols=list(positive.pair_cols),
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
        )
    results.update(conditions(setup.prepared(objects, fovs), fovs, setup, reasons))
    sufficiency = sufficiency_report(
        objects,
        fovs,
        # The rules read the same columns, and the same raw values, in report,
        # optimise, benchmark and live.
        feature_cols=setup.columns,
        strategies=list(setup.stopping_rules()),
        objective=setup.objective(),
        orders=_orders(setup, permuted=5, dispersed=0),
        novelty=setup.novelty,
        novelty_target=setup.novelty_target if setup.novelty is not None else None,
        few_fovs=setup.resolved_few_fovs,
        acquisition_order_col=setup.acquisition_order_col,
        # "New" is judged on the declared features only, as in the controls and the
        # inspection; the rules read the readout.
        novelty_cols=list(setup.features) if setup.novelty is not None else None,
        **common,
    )
    results["sufficiency"] = report_only(sufficiency, setup)
    # Every reason at once: new states on the declared features, influence, empty
    # or crowded fields and stage neighbours on the readout.
    novel = setup.novelty is not None and bool(setup.features)
    results["inspection"] = inspect_fovs(
        objects,
        fovs,
        feature_cols=[readout],
        summaries=[Summary(readout, "mean")],
        novelty=setup.novelty if novel else None,
        novelty_cols=list(setup.features) if novel else None,
        stage_cols=setup.stage_cols,
        **common,
    )
    document = microscopist_report(**results, declaration=setup.declaration())
    return Report(results, with_reasons(document, reasons), setup, reasons)


@dataclass(frozen=True)
class Plan:
    """The OPTIMISE answers: ``results`` by analysis, ``document`` the assembled report.

    ``not_assessed`` holds, by question, why an analysis was not run;
    ``settings`` the computational settings and presets of the run.
    """

    results: dict[str, Any]
    document: Any
    setup: Setup
    not_assessed: dict[str, str] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """The planning questions, with the calibration of each stopping rule, in sentences."""
        parts = [self.document.summary(PLAN_QUESTIONS)]
        calibration = self.results.get("calibration")
        if calibration is not None:
            parts.append(
                "Would a simple rule have been sufficient, and how reliable is each stopping "
                "rule? (P5, P6)\n" + calibration.summary()
            )
        if self.settings:
            parts.append(plan_settings(self.settings))
        return "\n\n".join(parts)


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
    _check_tables(objects, fovs, setup)
    common = setup.common
    criteria = setup.criteria()
    hierarchy = setup.resolved_hierarchy(objects, fovs)
    strata = setup.resolved_effort_strata(fovs)
    prepared = setup.prepared(objects, fovs)
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
    if hierarchy is None:
        reasons.update(no_hierarchy_reasons(setup, fovs))
    else:
        groups = results["reliability"].group_cols
        reason = protocol_reason(setup, hierarchy, strata, groups)
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
    document = microscopist_report(
        **{k: v for k, v in results.items() if k != "calibration"},
        declaration=setup.declaration(),
    )
    settings = {
        "protocol_threshold": protocol_threshold,
        "reservoir_fraction": reservoir_fraction,
        "n_splits": n_splits,
        "draws": draws,
        "permuted_orders": permuted_orders,
        "dispersed_orders": dispersed_orders if setup.stage_cols is not None else 0,
        "random_state": setup.random_state,
    }
    return Plan(results, with_reasons(document, reasons), setup, reasons, settings)


def document(*parts: Report | Plan) -> MicroscopistReport:
    """The full microscopist's report from a ``report`` and a ``plan`` of the same data."""
    merged: dict[str, Any] = {}
    reasons: dict[str, str] = {}
    for part in parts:
        if not isinstance(part, Report | Plan):
            raise TypeError("document takes the results of report() and optimise()")
        merged.update(part.results)
        reasons.update(part.not_assessed)
    declaration = parts[0].setup.declaration() if parts else ()
    return with_reasons(microscopist_report(**merged, declaration=declaration), reasons)


@dataclass(frozen=True)
class Family:
    """A rule family for the benchmark: one strategy, the quantity it thresholds, a grid.

    ``strategy`` is built at any threshold of the grid; the sweep re-decides
    every threshold from its stored evidence (``redecide``), so the replay runs
    once per family.
    """

    name: str
    strategy: SamplingStrategy
    quantity: str
    thresholds: tuple[float, ...]


def default_families(readout: str, n_resamples: int = 200) -> list[Family]:
    """The rule families and grids of the locked benchmark (protocol version 3).

    The JCB rule and the JCS descriptor spread, each per object and per FOV,
    and the FOV bootstrap of the mean, on ``readout``. The grids reach from a
    setting that never stops to one that stops at the first field on the
    committed data; widen them if your data need it.
    """
    jcb = (0.0025, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.4)
    jcs = (0.001, 0.002, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16)
    boot = (0.0005, 0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064)
    return [
        Family(
            "JCB per object",
            ScalarPrecisionStrategy(readout, jcb[0], "object", "upper_95"),
            "averaging_cv",
            jcb,
        ),
        Family(
            "JCB per field",
            ScalarPrecisionStrategy(readout, jcb[0], "fov", "upper_95"),
            "averaging_cv",
            jcb,
        ),
        Family(
            "JCS spread per object",
            DescriptorSpreadStrategy(readout, jcs[0], "object", n_resamples, 0),
            "max_descriptor_spread",
            jcs,
        ),
        Family(
            "JCS spread per field",
            DescriptorSpreadStrategy(readout, jcs[0], "fov", n_resamples, 0),
            "max_descriptor_spread",
            jcs,
        ),
        Family(
            "FOV bootstrap of the mean",
            # On the readout only, whatever other features the setup declares.
            FOVBootstrapStrategy("centroid", boot[0], None, n_resamples, 0, 2, features=(readout,)),
            "fov_bootstrap_centroid_max_se",
            boot,
        ),
    ]


@dataclass(frozen=True)
class Benchmark:
    """Fields saved at matched risk, one ``FrontierResult`` per criterion."""

    frontiers: dict[str, FrontierResult]
    families: tuple[str, ...]
    settings: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Every criterion's scores, in sentences, then the risk levels and settings."""
        blocks = [result.summary() for result in self.frontiers.values()]
        if self.settings:
            settings = self.settings
            levels = " and ".join(f"{alpha:.0%}" for alpha in settings["alphas"])
            units = ", ".join(settings["unit_cols"]) or "one pooled unit"
            blocks.append(
                f"Settings: risk levels {levels} of stopping too early (alphas, a preset); "
                f"{settings['reservoir_fraction']:.0%} of each group's fields held back, "
                f"{settings['n_splits']} hold-outs, {settings['permuted_orders']} random "
                f"orders; intervals over {units}."
            )
        return "\n\n".join(blocks)


def benchmark(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    setup: Setup,
    *,
    reservoir_fraction: float,
    families: Sequence[Family] | None = None,
    alphas: Sequence[float] = (0.05, 0.10),
    unit_cols: Sequence[str] | None = None,
    n_splits: int = 5,
    permuted_orders: int = 10,
    min_fovs: int = 1,
    n_boot: int = 2000,
) -> Benchmark:
    """BENCHMARK: which rule needs the fewest fields for the same risk of stopping too early.

    Each family (default: ``default_families(setup.readout)``) is replayed
    once against held-back fields and swept over its grid; at each risk level
    in ``alphas`` its cheapest setting from which the risk stays at or below
    the level is scored against a fixed field count, with intervals over the
    independent units ``unit_cols`` (default: the analysis groups). ``alphas``
    (default 5% and 10%, a preset) are printed in the summary. The rules run on
    the raw values, as live; the criteria judge the normalised values when
    ``normalize`` is set. Settings chosen here describe these data; score them
    on data they were not chosen on before using them
    (``retrospective.frontier.transfer``).
    """
    _check_tables(objects, fovs, setup)
    families = list(default_families(setup.readout) if families is None else families)
    if not families or any(not isinstance(f, Family) for f in families):
        raise TypeError("families must be a non-empty sequence of Family")
    criteria = setup.criteria()
    # The rules on the raw values, as live; the criteria on the normalised ones.
    replay = replay_rules(
        objects,
        setup.prepared(objects, fovs),
        fovs,
        setup,
        strategies=[f.strategy for f in families],
        criteria=criteria,
        reservoir_fraction=reservoir_fraction,
        orders=_orders(setup, permuted_orders, dispersed=0),
        acquisition_order_col=setup.acquisition_order_col,
        n_splits=n_splits,
        random_state=setup.random_state,
        min_fovs=min_fovs,
        keep_results=True,
        n_jobs=setup.n_jobs,
    )
    swept = [
        sweep(
            replay,
            config_id=f.strategy.config_id,
            quantity=f.quantity,
            thresholds=list(f.thresholds),
            name=f.name,
        )
        for f in families
    ]
    units = list(replay.group_cols) if unit_cols is None else list(unit_cols)
    labels = [criterion.label for criterion in criteria]
    # Frontiers are keyed by label; two criteria with one label (tolerances 5% and
    # 5.02% both print as 5.0%) keep both, the second named by its criterion.
    keys = [
        label if labels.count(label) == 1 or labels.index(label) == i else f"{label} ({c.name})"
        for i, (label, c) in enumerate(zip(labels, criteria, strict=True))
    ]
    frontiers = {
        key: frontier(
            replay,
            swept,
            criterion=criterion.name,
            alphas=list(alphas),
            unit_cols=units,
            n_boot=n_boot,
            random_state=setup.random_state,
            label=criterion.label,
        )
        for key, criterion in zip(keys, criteria, strict=True)
    }
    settings = {
        "reservoir_fraction": reservoir_fraction,
        "alphas": list(alphas),
        "unit_cols": units,
        "n_splits": n_splits,
        "permuted_orders": permuted_orders,
        "families": {
            f.name: {"quantity": f.quantity, "thresholds": list(f.thresholds)} for f in families
        },
    }
    return Benchmark(frontiers, tuple(f.name for f in families), settings)


__all__ = [
    "PLAN_QUESTIONS",
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

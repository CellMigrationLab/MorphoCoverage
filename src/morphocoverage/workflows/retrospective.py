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

They add no scientific choice of their own: every one is in the ``Setup`` or
an explicit argument. When the setup asks for it, the object table is first
normalised to the control condition's mean, repeat by repeat
(``Setup.prepared``), and every document says so in its overview. Each
document opens with the setup's declaration of what the acquisition was asked
to preserve; every replay runs under the objective
made of the declared targets (``Setup.objective()``), and ``optimise`` and
``benchmark`` judge every declared criterion (``Setup.criteria()``: the mean,
the fingerprint, the labelled rare state) on held-back fields. ``document`` merges a report and a
plan into the full microscopist's report.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from matplotlib.figure import Figure

from ..benchmark import AcquisitionOrder
from ..metadata import MORPHOBRICKS_GROUP_COLUMNS
from ..plots import plots_of_data
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
    replay_acquisitions,
    representation_control,
    sampling_reliability,
    stopping_calibration,
)
from ..retrospective.frontier import FrontierResult, frontier, sweep
from ..sampling_unit import (
    condition_contrast,
    fov_clustering,
    fov_variability,
    object_rule_check,
    unusual_objects,
)
from ..strategies import DescriptorSpreadStrategy, FOVBootstrapStrategy, ScalarPrecisionStrategy
from ..strategies.base import SamplingStrategy
from ..sufficiency import sufficiency_report
from .setup import Setup

CONDITION = MORPHOBRICKS_GROUP_COLUMNS[0]
REPORT_QUESTIONS = ("unit", "controls", "enough", "profile", "inspect", "conditions")
PLAN_QUESTIONS = ("next_time", "bought", "effort", "order")
_UNUSUAL = "morphocoverage_unusual"


def _orders(setup: Setup, permuted: int, dispersed: int) -> list[AcquisitionOrder]:
    orders = [AcquisitionOrder.recorded()] if setup.acquisition_order_col else []
    orders += [AcquisitionOrder.permuted(seed) for seed in range(permuted)]
    if setup.stage_cols is not None:
        orders += [AcquisitionOrder.dispersed(seed) for seed in range(dispersed)]
    return orders


def _check_tables(objects: Any, fovs: Any, setup: Any) -> None:
    if not isinstance(objects, pd.DataFrame) or not isinstance(fovs, pd.DataFrame):
        raise TypeError("objects and fovs must be pandas DataFrames (every acquired FOV in fovs)")
    if not isinstance(setup, Setup):
        raise TypeError("setup must be a morphocoverage.workflows.Setup")
    if setup.rare_label is not None and setup.rare_label not in objects.columns:
        raise ValueError(
            f"rare_label {setup.rare_label!r} is not a column of the object table: the rare "
            "phenotype cannot be judged on held-back fields without its labels"
        )


@dataclass(frozen=True)
class Report:
    """The REPORT answers: ``results`` by analysis, ``document`` the assembled report."""

    results: dict[str, Any]
    document: Any
    setup: Setup

    def summary(self) -> str:
        """The report questions, in sentences."""
        return self.document.summary(REPORT_QUESTIONS)


def report(objects: pd.DataFrame, fovs: pd.DataFrame, setup: Setup) -> Report:
    """REPORT on a finished acquisition: what can be trusted, and what cannot.

    Runs, as the setup allows: ``fov_variability`` and, with a control,
    ``object_rule_check`` (the Foundation); ``representation_control`` with a positive
    control and a definition of new (R1); ``sufficiency_report`` for the
    setup's rules, with next-field novelty when ``novelty`` is set (R2, R3);
    ``unusual_objects`` and ``fov_clustering`` with a control and features
    (R4); ``inspect_fovs`` (R5); ``condition_contrast`` when the control is
    declared by condition and the FOV table holds other conditions (R7).
    """
    _check_tables(objects, fovs, setup)
    common = setup.common
    readout = setup.readout
    objects = setup.prepared(objects, fovs)
    results: dict[str, Any] = {
        "overview": dataset_overview(
            objects,
            fovs,
            feature_cols=setup.columns,
            normalized=setup.normalized_sentence(fovs),
            **common,
        ),
        "variability": fov_variability(objects, fovs, feature=readout, **common),
    }
    if setup.control is not None:
        results["object_rules"] = object_rule_check(
            objects,
            fovs,
            feature=readout,
            targets=setup.tolerances,
            control=setup.control,
            random_state=setup.random_state,
            **common,
        )
        if setup.features:
            unusual = unusual_objects(
                objects,
                fovs,
                feature_cols=list(setup.features),
                reference=setup.control,
                random_state=setup.random_state,
                **common,
            )
            results["unusual"] = unusual
            results["clustering"] = fov_clustering(
                objects.assign(**{_UNUSUAL: unusual.labels}),
                fovs,
                label_col=_UNUSUAL,
                states=[True],
                random_state=setup.random_state,
                **common,
            )
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
    contrasted = setup.control is not None and CONDITION in setup.control
    if contrasted and CONDITION in fovs.columns and fovs[CONDITION].nunique() > 1:
        results["contrast"] = condition_contrast(
            objects,
            fovs,
            feature=readout,
            control=setup.control,
            random_state=setup.random_state,
            **common,
        )
    novelty_features = list(setup.features) if setup.novelty is not None else []
    results["sufficiency"] = sufficiency_report(
        objects,
        fovs,
        feature_cols=[readout, *[f for f in novelty_features if f != readout]],
        strategies=list(setup.stopping_rules()),
        objective=setup.objective(),
        orders=_orders(setup, permuted=5, dispersed=0),
        novelty=setup.novelty,
        novelty_target=setup.novelty_target if setup.novelty is not None else None,
        few_fovs=3 if setup.novelty is not None else None,
        acquisition_order_col=setup.acquisition_order_col,
        # "New" is judged on the declared features only, as in the controls and the
        # inspection; the rules read the readout.
        novelty_cols=list(setup.features) if setup.novelty is not None else None,
        **common,
    )
    if setup.novelty is not None and setup.features:
        results["inspection"] = inspect_fovs(
            objects,
            fovs,
            feature_cols=list(setup.features),
            novelty=setup.novelty,
            count_ratio=None,
            **common,
        )
    else:
        results["inspection"] = inspect_fovs(
            objects,
            fovs,
            feature_cols=[readout],
            summaries=[Summary(readout, "mean")],
            stage_cols=setup.stage_cols,
            **common,
        )
    return Report(results, microscopist_report(**results, declaration=setup.declaration()), setup)


@dataclass(frozen=True)
class Plan:
    """The OPTIMISE answers: ``results`` by analysis, ``document`` the assembled report."""

    results: dict[str, Any]
    document: Any
    setup: Setup

    def summary(self) -> str:
        """The planning questions, with the calibration of each stopping rule, in sentences."""
        parts = [self.document.summary(PLAN_QUESTIONS)]
        calibration = self.results.get("calibration")
        if calibration is not None:
            parts.append(
                "Would a simple rule have been sufficient, and how reliable is each stopping "
                "rule? (P5, P6)\n" + calibration.summary()
            )
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
    ``sampling_reliability`` for each tolerance (P1), ``protocol_reliability``
    across samples when ``setup.hierarchy`` is set (P2), a replay of the
    setup's rules against the same held-back fields for their calibration and
    adequacy points (P3 to P6), ``order_efficiency`` and, with
    ``redundancy_bins``, ``redundancy_distance`` when stage coordinates are
    set (P7, P8), and ``effort_projection`` with a hierarchy (P10). A count
    of fields is a recommendation only once it holds on samples it was not
    chosen on.
    """
    _check_tables(objects, fovs, setup)
    common = setup.common
    criteria = setup.criteria()
    objects = setup.prepared(objects, fovs)
    results: dict[str, Any] = {}
    results["reliability"] = sampling_reliability(
        objects,
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
    if setup.hierarchy is not None:
        results["protocol"] = protocol_reliability(
            results["reliability"],
            setup.hierarchy,
            threshold=protocol_threshold,
            strata=list(setup.effort_strata),
        )
    replay = replay_acquisitions(
        objects,
        fovs,
        feature_cols=setup.judged_columns,
        strategies=list(setup.stopping_rules()),
        criteria=criteria,
        objective=setup.objective(),
        reservoir_fraction=reservoir_fraction,
        orders=_orders(setup, permuted_orders, dispersed_orders),
        acquisition_order_col=setup.acquisition_order_col,
        stage_cols=setup.stage_cols,
        n_splits=n_splits,
        random_state=setup.random_state,
        incremental=True,
        n_jobs=setup.n_jobs,
        **common,
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
    if setup.hierarchy is not None:
        results["effort"] = effort_projection(
            objects,
            fovs,
            feature=setup.readout,
            hierarchy=setup.hierarchy,
            strata=list(setup.effort_strata),
            random_state=setup.random_state,
            observation_unit=setup.observation_unit,
        )
    document = microscopist_report(
        **{k: v for k, v in results.items() if k != "calibration"},
        declaration=setup.declaration(),
    )
    return Plan(results, document, setup)


def document(*parts: Report | Plan) -> MicroscopistReport:
    """The full microscopist's report from a ``report`` and a ``plan`` of the same data."""
    merged: dict[str, Any] = {}
    for part in parts:
        if not isinstance(part, Report | Plan):
            raise TypeError("document takes the results of report() and optimise()")
        merged.update(part.results)
    declaration = parts[0].setup.declaration() if parts else ()
    return microscopist_report(**merged, declaration=declaration)


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
    """The rule families and grids of the locked benchmark (protocol version 2).

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
            FOVBootstrapStrategy("centroid", boot[0], None, n_resamples, 0, 2),
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
        """Every criterion's scores, in sentences."""
        return "\n\n".join(result.summary() for result in self.frontiers.values())


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
    independent units ``unit_cols`` (default: the analysis groups). Settings
    chosen here describe these data; score them on data they were not chosen
    on before using them (``retrospective.frontier.transfer``).
    """
    _check_tables(objects, fovs, setup)
    families = list(default_families(setup.readout) if families is None else families)
    if not families or any(not isinstance(f, Family) for f in families):
        raise TypeError("families must be a non-empty sequence of Family")
    criteria = setup.criteria()
    objects = setup.prepared(objects, fovs)
    replay = replay_acquisitions(
        objects,
        fovs,
        feature_cols=setup.judged_columns,
        strategies=[f.strategy for f in families],
        criteria=criteria,
        objective=setup.objective(),
        reservoir_fraction=reservoir_fraction,
        orders=_orders(setup, permuted_orders, dispersed=0),
        acquisition_order_col=setup.acquisition_order_col,
        n_splits=n_splits,
        random_state=setup.random_state,
        min_fovs=min_fovs,
        keep_results=True,
        n_jobs=setup.n_jobs,
        **setup.common,
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
    frontiers = {
        criterion.label: frontier(
            replay,
            swept,
            criterion=criterion.name,
            alphas=list(alphas),
            unit_cols=units,
            n_boot=n_boot,
            random_state=setup.random_state,
            label=criterion.label,
        )
        for criterion in criteria
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


def plots(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    setup: Setup,
    *,
    features: Sequence[str] | None = None,
    folder: str | Path | None = None,
    **settings: Any,
) -> dict[str, list[Figure]]:
    """PlotsOfData-style figures of the readout (or ``features``), on the prepared table.

    Within each condition the biological repeats sit side by side; every
    object is a point coloured by technical repeat (a well), with each
    biological repeat's median and its
    interval from resampling whole fields (``plots.plots_of_data``, which
    also takes ``settings`` such as ``summary`` or ``max_conditions``).
    With ``folder`` each figure is also saved there as
    ``<feature>[_<part>].png``. Returns the figures by feature.
    """
    _check_tables(objects, fovs, setup)
    chosen = [setup.readout] if features is None else list(features)
    undeclared = sorted(set(chosen) - set(setup.columns))
    if undeclared:
        raise ValueError(f"features to plot must be declared in the setup: {undeclared}")
    figures = plots_of_data(
        setup.prepared(objects, fovs),
        fovs,
        features=chosen,
        random_state=settings.pop("random_state", setup.random_state),
        **settings,
        **setup.common,
    )
    if folder is not None:
        target = Path(folder)
        target.mkdir(parents=True, exist_ok=True)
        for feature, parts in figures.items():
            for index, figure in enumerate(parts, start=1):
                suffix = f"_{index}" if len(parts) > 1 else ""
                figure.savefig(target / f"{feature}{suffix}.png", dpi=150)
    return figures

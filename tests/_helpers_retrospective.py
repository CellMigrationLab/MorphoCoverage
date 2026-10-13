"""Builders shared by the retrospective tests: plates, acquisitions, replays and reports."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scenarios import FEATURE_COLS

from morphocoverage import FixedFOVStrategy, ObjectiveConfig, SamplingObjective
from morphocoverage.benchmark import AcquisitionOrder
from morphocoverage.evidence import Component, Direction, Evidence, Quantity
from morphocoverage.outcomes import RelativeMeanError
from morphocoverage.policies import ThresholdPolicy
from morphocoverage.retrospective import (
    OutcomeCriterion,
    ReliabilityResult,
    RepresentationControl,
    SamplingHierarchy,
    Summary,
    inspect_fovs,
    order_efficiency,
    replay_acquisitions,
    representation_control,
)
from morphocoverage.strategies import SupportNoveltyEstimator
from morphocoverage.strategies.base import Estimator, SamplingStrategy
from morphocoverage.sufficiency import sufficiency_report
from morphocoverage.transforms import Standardize

POPULATION = ObjectiveConfig(SamplingObjective.POPULATION)


def _weak_control_report():
    from scenarios import _fov_table, _objects_for

    from morphocoverage.retrospective import RepresentationControl, representation_control

    rng = np.random.default_rng(1)
    ref_f = _fov_table(20, condition="MOCK")
    pos_f = _fov_table(20, condition="KD")
    ref_o, pos_o = (
        _objects_for(ref_f, np.full(20, 30), rng),
        _objects_for(pos_f, np.full(20, 30), rng),
    )
    pos_o[list(FEATURE_COLS)] += 0.8
    objects, fovs = pd.concat([ref_o, pos_o], ignore_index=True), pd.concat([ref_f, pos_f])
    novelty = SupportNoveltyEstimator(np.sqrt(2), 1, Standardize(), 2, 3)
    controls = representation_control(
        objects,
        fovs,
        feature_cols=list(FEATURE_COLS),
        novelty=novelty,
        control=RepresentationControl(
            positive={"condition": ["KD"]}, reference={"condition": ["MOCK"]}
        ),
        pair_cols=[],
        fovs_seen=10,
        draws=5,
    )
    sufficiency = sufficiency_report(
        objects,
        fovs,
        feature_cols=list(FEATURE_COLS),
        strategies=[FixedFOVStrategy(10)],
        objective=POPULATION,
        orders=[AcquisitionOrder.permuted(0)],
        novelty=novelty,
    )
    return controls, sufficiency


FEATURES = ["feature_a", "feature_b"]


NOVELTY = SupportNoveltyEstimator(1.0, 1, None, 3, 3)


def _plate(seed=0, shift=6.0, plates=("P1", "P2"), fovs_per_well=12):
    """Per plate: a reference well and a positive well shifted by ``shift`` in feature_a."""
    rng = np.random.default_rng(seed)
    fov_rows, object_rows = [], []
    for plate in plates:
        for well, condition, offset in (("A1", "MOCK", 0.0), ("B1", "KD", shift)):
            for i in range(fovs_per_well):
                name = f"{plate}_{well}_{i}"
                fov_rows.append(
                    {
                        "condition": condition,
                        "biological_repeat": plate,
                        "well": well,
                        "image_file": name,
                    }
                )
                values = rng.normal(0.0, 1.0, size=(25, 2))
                values[:, 0] += offset
                object_rows.append(
                    pd.DataFrame(values, columns=FEATURES).assign(
                        condition=condition, biological_repeat=plate, well=well, image_file=name
                    )
                )
    return pd.concat(object_rows, ignore_index=True), pd.DataFrame(fov_rows)


def _control(objects, fovs, control, **kwargs):
    settings = {"pair_cols": ["biological_repeat"], "fovs_seen": 6, "draws": 10}
    settings.update(kwargs)
    return representation_control(
        objects, fovs, feature_cols=FEATURES, novelty=NOVELTY, control=control, **settings
    )


BOTH = RepresentationControl(
    positive={"condition": "KD"}, reference={"condition": "MOCK"}, planted_objects=5
)


def _simulate(rng, components, repeats, samples, fovs, cells, balanced=True):
    """Observation and FOV tables of a nested design with known variance components."""
    v_rep, v_sample, v_fov, v_obs = components
    object_rows, fov_rows = [], []
    for r in range(repeats):
        rep = rng.normal(0, np.sqrt(v_rep))
        n_samples = samples if balanced else rng.integers(max(2, samples - 2), samples + 3)
        for s in range(n_samples):
            sample = rng.normal(0, np.sqrt(v_sample))
            n_fovs = fovs if balanced else rng.integers(max(2, fovs - 3), fovs + 4)
            for f in range(n_fovs):
                field = rng.normal(0, np.sqrt(v_fov))
                name = f"r{r}_s{s}_f{f}"
                fov_rows.append({"biological_repeat": f"R{r}", "well": f"W{s}", "image_file": name})
                m = cells if balanced else rng.integers(max(1, cells - 10), cells + 11)
                values = rep + sample + field + rng.normal(0, np.sqrt(v_obs), m)
                object_rows.append(pd.DataFrame({"image_file": name, "x": values}))
    fovs_table = pd.DataFrame(fov_rows).assign(condition="ctrl")
    objects = pd.concat(object_rows, ignore_index=True)
    objects = objects.merge(fovs_table, on="image_file")
    return objects, fovs_table


TRUTH = (0.5, 1.0, 0.3, 4.0)


MEAN = OutcomeCriterion(RelativeMeanError("feature_a"), 0.05, "lower_is_better", "the mean")


def _mixed(seed=0, n_easy=4, n_hard=4, n_fovs=40, cells=20):
    """Samples whose FOVs barely differ (easy) and samples whose FOVs differ a lot (hard).

    The difficulty is known by construction and recorded on the FOV table, so
    a test-only oracle can read it; no packaged strategy sees it.
    """
    rng = np.random.default_rng(seed)
    fov_rows, object_rows = [], []
    for s in range(n_easy + n_hard):
        hard = s >= n_easy
        spread = 0.6 if hard else 0.02
        for f in range(n_fovs):
            name = f"s{s}_f{f}"
            fov_rows.append(
                {
                    "condition": "c",
                    "biological_repeat": f"R{s}",
                    "image_file": name,
                    "difficulty": "hard" if hard else "easy",
                }
            )
            level = 10 + rng.normal(0, spread)
            values = level + rng.normal(0, 0.2, size=(cells, 2))
            object_rows.append(
                pd.DataFrame(values, columns=list(FEATURE_COLS)).assign(
                    condition="c", biological_repeat=f"R{s}", image_file=name
                )
            )
    return pd.concat(object_rows, ignore_index=True), pd.DataFrame(fov_rows)


@dataclass(frozen=True)
class _Oracle(Estimator):
    """Test only: knows each sample's difficulty and asks for ``easy`` or ``hard`` fields."""

    easy: int
    hard: int
    name: str = "oracle"

    @property
    def parameters(self):
        return {"easy": self.easy, "hard": self.hard}

    def estimate(self, data, *, objective):
        need = self.hard if (data.fovs["difficulty"] == "hard").any() else self.easy
        short = max(0, need - len(data.fovs))
        meta = {
            "direction": Direction.LOWER_IS_BETTER,
            "units": "FOVs",
            "component": Component.POPULATION,
        }
        return Evidence({"fields_short": Quantity.available(short, **meta)})


def _oracle(easy, hard):
    return SamplingStrategy([_Oracle(easy, hard)], [ThresholdPolicy("fields_short", 0)])


def _replay(objects, fovs, strategies, *, orders=4, splits=3, seed=0, criteria=(MEAN,)):
    return replay_acquisitions(
        objects,
        fovs,
        feature_cols=list(FEATURE_COLS),
        strategies=strategies,
        criteria=list(criteria),
        objective=POPULATION,
        reservoir_fraction=0.3,
        orders=[AcquisitionOrder.permuted(s) for s in range(orders)],
        n_splits=splits,
        random_state=seed,
        keep_results=True,
        group_cols=["condition", "biological_repeat"],
    )


NEW, TAIL, DENSE = "f03", "f07", "f12"


def _acquisition(seed=5, duplicate_tail=1):
    """20 FOVs of 20 cells; one holds a new state, one a feature_b tail, one is dense.

    The tail cells are spread along feature_a, so they are not one new state.
    """
    rng = np.random.default_rng(seed)
    names = [f"f{i:02d}" for i in range(20)]
    fovs = pd.DataFrame(
        {
            "condition": "control",
            "biological_repeat": "R1",
            "image_file": names,
            "stage_x": np.arange(20) % 5 * 100.0,
            "stage_y": np.arange(20) // 5 * 100.0,
        }
    )
    rows = []
    for name in names:
        n = 100 if name == DENSE else 20
        values = rng.normal(0.0, 1.0, size=(n, 2))
        if name == TAIL:
            tail = np.column_stack([np.linspace(-15, 15, 19), np.linspace(3, 5, 19)])
            values = np.repeat(np.vstack([values[:1], tail]), duplicate_tail, axis=0)
        if name == NEW:
            state = np.column_stack([rng.normal(8, 0.1, 8), rng.normal(0, 0.1, 8)])
            values = np.vstack([values, state])
        # A positive baseline, as for areas or intensities: a relative change needs one.
        rows.append(pd.DataFrame(values + 10, columns=FEATURES).assign(image_file=name))
    objects = pd.concat(rows, ignore_index=True).assign(condition="control", biological_repeat="R1")
    return objects, fovs


INSPECTION_NOVELTY = SupportNoveltyEstimator(1.5, 1, None, 5, 5)


def _inspect(objects, fovs, **kwargs):
    settings = {"summaries": [Summary("feature_b", 0.97)], "novelty": INSPECTION_NOVELTY}
    settings.update(kwargs)
    return inspect_fovs(objects, fovs, feature_cols=FEATURES, **settings)


HIERARCHY = SamplingHierarchy("image_file", "well", "plate")


def _curve(preserved, stream=12, criterion="mean"):
    """A reliability result from {(plate, well): {k: preserved}}."""
    rows = []
    for (plate, well), by_k in preserved.items():
        for k, value in by_k.items():
            rows.append(
                {
                    "condition": "ctrl",
                    "plate": plate,
                    "well": well,
                    "criterion": criterion,
                    "fovs": k,
                    "preserved": value,
                    "stream_fovs": stream,
                }
            )
    curve = pd.DataFrame(rows)
    return ReliabilityResult(
        curve, pd.DataFrame(), ("condition", "plate", "well"), {"labels": {criterion: "the mean"}}
    )


STAGE = ("stage_x", "stage_y")


def _gradient(seed=0):
    """A 10 x 6 raster; the phenotype changes along y, the raster runs along x first."""
    rng = np.random.default_rng(seed)
    fov_rows, object_rows = [], []
    for i in range(60):
        x, y = i % 10, i // 10
        name = f"f{i:02d}"
        fov_rows.append(
            {
                "condition": "control",
                "biological_repeat": "R1",
                "image_file": name,
                "acquisition_order": i,
                "stage_x": x * 200.0,
                "stage_y": y * 200.0,
            }
        )
        values = rng.normal(10 + y, 0.5, size=(15, 2))
        object_rows.append(
            pd.DataFrame(values, columns=list(FEATURE_COLS)).assign(
                condition="control", biological_repeat="R1", image_file=name
            )
        )
    return pd.concat(object_rows, ignore_index=True), pd.DataFrame(fov_rows)


def _efficiency(objects, fovs, orders, tolerance=0.06):
    criterion = OutcomeCriterion(RelativeMeanError("feature_a"), tolerance, "lower_is_better")
    replay = replay_acquisitions(
        objects,
        fovs,
        feature_cols=list(FEATURE_COLS),
        strategies=[FixedFOVStrategy(5)],
        criteria=[criterion],
        objective=POPULATION,
        reservoir_fraction=0.3,
        orders=orders,
        acquisition_order_col="acquisition_order",
        stage_cols=STAGE,
        n_splits=3,
    )
    return order_efficiency(replay, per_fov_seconds=2.0, stage_speed=1000.0)

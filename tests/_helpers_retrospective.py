"""Builders shared by the retrospective tests: plates, acquisitions, replays and reports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scenarios import FEATURE_COLS, ORDER_COL, SCENARIOS

from morphocoverage import FixedFOVStrategy, ObjectiveConfig, SamplingObjective
from morphocoverage.benchmark import AcquisitionOrder
from morphocoverage.evidence import Component, Direction, Evidence, Quantity
from morphocoverage.outcomes import RareStateDetected, RelativeMeanError
from morphocoverage.policies import ThresholdPolicy
from morphocoverage.retrospective import (
    OutcomeCriterion,
    ReliabilityResult,
    RepresentationControl,
    SamplingHierarchy,
    Summary,
    adequacy_points,
    effort_projection,
    inspect_fovs,
    order_efficiency,
    protocol_reliability,
    replay_acquisitions,
    representation_control,
    sampling_reliability,
)
from morphocoverage.retrospective.controls import ControlResult
from morphocoverage.retrospective.reliability import _curve_row
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
                        "repeat_group": plate,
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
                fov_rows.append(
                    {
                        "biological_repeat": f"R{r}",
                        "repeat_group": f"R{r}",
                        "well": f"W{s}",
                        "image_file": name,
                    }
                )
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
                    "repeat_group": f"R{s}",
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
            "repeat_group": "R1",
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
                "repeat_group": "R1",
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


def _make(n_fovs=20, per_fov=8, empty=(), seed=0, groups=(("A", "r1"),)):
    """Objects and FOVs, one analysis group per (condition, repeat), FOVs on a 5-wide grid."""
    rng = np.random.default_rng(seed)
    objects, fovs = [], []
    for condition, repeat in groups:
        for i in range(n_fovs):
            position = f"p{i:02d}"
            fovs.append(
                {
                    "condition": condition,
                    "biological_repeat": repeat,
                    "repeat_group": repeat,
                    "position_id": position,
                    "x": float(i % 5) * 100.0,
                    "y": float(i // 5) * 100.0,
                }
            )
            if i in empty:
                continue
            shift = rng.normal(0, 1)
            objects += [
                {
                    "condition": condition,
                    "biological_repeat": repeat,
                    "repeat_group": repeat,
                    "position_id": position,
                    "f": 10 + shift + rng.normal(0, 1),
                }
                for _ in range(per_fov)
            ]
    return pd.DataFrame(objects), pd.DataFrame(fovs)


def _two_group_tables():
    rng = np.random.default_rng(0)
    rows = [
        {
            "condition": c,
            "biological_repeat": r,
            "repeat_group": r,
            "image_file": f"{c}{r}_{j}",
            "a": v,
        }
        for c in ("A", "B")
        for r in (1, 2)
        for j in range(12)
        for v in rng.normal(10, 1, 10)
    ]
    objects = pd.DataFrame(rows)
    fovs = objects[["condition", "biological_repeat", "image_file"]].drop_duplicates()
    return objects, fovs.reset_index(drop=True)


def _plate_curve(preserved, stream=12, unit="plate"):
    """A reliability result from {(plate, well): {k: preserved}}, one criterion."""
    rows = [
        {
            "condition": "ctrl",
            unit: plate,
            "well": well,
            "criterion": "mean",
            "fovs": k,
            "preserved": value,
            "stream_fovs": stream,
        }
        for (plate, well), by_k in preserved.items()
        for k, value in by_k.items()
    ]
    return ReliabilityResult(pd.DataFrame(rows), pd.DataFrame(), ("condition", unit, "well"))


V_SAMPLE, V_FIELD, V_OBJECT = 1.0, 0.2, 1.0


def _wells(fields, cells, seed=0, repeats=None):
    """Tables of wells with ``fields[s]`` fields of ``cells[s]`` objects (0: empty fields)."""
    rng = np.random.default_rng(seed)
    fov_rows, object_rows = [], []
    for s, (n_fields, m) in enumerate(zip(fields, cells, strict=True)):
        sample = rng.normal(0, np.sqrt(V_SAMPLE))
        for f in range(n_fields):
            name = f"s{s}_f{f}"
            row = {"well": f"W{s}", "image_file": name, "condition": "c"}
            if repeats is not None:
                row["biological_repeat"] = repeats[s]
            fov_rows.append(row)
            values = sample + rng.normal(0, np.sqrt(V_FIELD)) + rng.normal(0, np.sqrt(V_OBJECT), m)
            object_rows.append(pd.DataFrame({"image_file": name, "x": values}))
    fovs = pd.DataFrame(fov_rows)
    objects = pd.concat(object_rows, ignore_index=True).merge(fovs, on="image_file")
    return objects, fovs


OBJECTIVE = ObjectiveConfig()


MEAN_OF_F = OutcomeCriterion(RelativeMeanError("f"), 0.02, "lower_is_better", "the mean within 2%")


RARE = OutcomeCriterion(RareStateDetected("rare", 1), 1.0, "higher_is_better", "the rare state")


FIX3 = FixedFOVStrategy(target_fovs=3)


GROUPS = ("condition", "biological_repeat", "technical_repeat")


BY_WELL = SamplingHierarchy("image_file", "well", "biological_repeat")


def _tables(
    n_fovs=12,
    objs=5,
    seed=0,
    wells=("A1",),
    repeats=("R1",),
    rare_fovs=(),
    scale=1.0,
):
    """Objects and FOVs of one condition: repeats x wells x fields on a 4-wide grid."""
    rng = np.random.default_rng(seed)
    fov_rows, object_rows = [], []
    for repeat in repeats:
        for well in wells:
            for i in range(n_fovs):
                fid = f"ctrl_{repeat}_{well}_{i:02d}"
                base = {
                    "condition": "ctrl",
                    "biological_repeat": repeat,
                    "repeat_group": repeat,
                    "technical_repeat": well,
                }
                fov_rows.append(
                    {
                        **base,
                        "position_id": fid,
                        "order": i + 1,
                        "x": (i % 4) * 100.0,
                        "y": (i // 4) * 100.0,
                    }
                )
                object_rows += [
                    {
                        **base,
                        "position_id": fid,
                        "f": float(rng.normal(10.0, scale)),
                        "rare": bool(i in rare_fovs and j == 0),
                    }
                    for j in range(objs)
                ]
    return pd.DataFrame(object_rows), pd.DataFrame(fov_rows)


def _f_replay(objects, fovs, **kwargs):
    settings = {
        "feature_cols": ["f"],
        "strategies": [FIX3],
        "criteria": [MEAN_OF_F],
        "objective": OBJECTIVE,
        "reservoir_fraction": 0.3,
        "orders": [AcquisitionOrder.permuted(0)],
        "n_splits": 2,
        "fov_col": "position_id",
    }
    settings.update(kwargs)
    return replay_acquisitions(objects, fovs, **settings)


def _f_reliability(objects, fovs, **kwargs):
    settings = {
        "feature_cols": ["f"],
        "criteria": [MEAN_OF_F],
        "fov_counts": [2],
        "reservoir_fraction": 0.5,
        "n_splits": 3,
        "draws": 4,
        "fov_col": "position_id",
    }
    settings.update(kwargs)
    return sampling_reliability(objects, fovs, **settings)


def _repeat_curve(preserved, **extra):
    """A reliability result from {(repeat, well): {k: preserved}}, streams of 12 fields."""
    rows = [
        {
            "condition": "c",
            "biological_repeat": repeat,
            "repeat_group": repeat,
            "well": well,
            "criterion": "m",
            "fovs": k,
            "preserved": value,
            "stream_fovs": 12,
            **extra,
        }
        for (repeat, well), by_k in preserved.items()
        for k, value in by_k.items()
    ]
    return ReliabilityResult(
        pd.DataFrame(rows),
        pd.DataFrame(),
        ("condition", "biological_repeat", "well"),
        {"labels": {"m": "m"}},
    )


def _ninety_percent_result():
    """Two wells, each preserved in exactly 9 of 10 draws in each of 9 splits."""
    draws = pd.DataFrame(
        [
            {
                "condition": "c",
                "biological_repeat": "R1",
                "repeat_group": "R1",
                "technical_repeat": well,
                "split": split,
                "fovs": 4,
                "draw": draw,
                "criterion": "m",
                "value": 0.0,
                "status": "preserved" if draw < 9 else "not_preserved",
                "reservoir_fovs": 4,
                "stream_fovs": 8,
                "holdout": f"{well}-{split}",
            }
            for well in ("A1", "A2")
            for split in range(9)
            for draw in range(10)
        ]
    )
    curve = pd.DataFrame(
        [
            {**dict(zip(GROUPS, key, strict=True)), "criterion": "m", "fovs": 4, **_curve_row(runs)}
            for key, runs in draws.groupby(list(GROUPS))
        ]
    )
    return ReliabilityResult(curve, draws, GROUPS, {"labels": {"m": "the mean"}, "n_splits": 9})


SYNTHETIC = Path(__file__).resolve().parents[1] / "data" / "synthetic"


SYN_FEATURES = ["area", "intensity", "eccentricity"]


SYN_NOVELTY = SupportNoveltyEstimator(float(np.sqrt(3)), 1, Standardize(), 5, 3)


KEY = ["condition", "biological_repeat", "technical_repeat", "image_file"]


@pytest.fixture(scope="module")
def synthetic():
    objects = pd.read_csv(SYNTHETIC / "synthetic_cells.csv")
    fovs = pd.read_csv(SYNTHETIC / "synthetic_fovs.csv")
    # Each folder label is confirmed as its own repeat group, by which repeats pair (#617).
    fovs = fovs.assign(repeat_group=fovs["biological_repeat"])
    return objects, fovs


def _syn_inspection(objects, fovs, **kwargs):
    return inspect_fovs(
        objects,
        fovs,
        feature_cols=["area"],
        novelty=SYN_NOVELTY,
        novelty_cols=SYN_FEATURES,
        count_ratio=None,
        observation_unit="cell",
        **kwargs,
    )


def _pooled(**rates):
    """A ControlResult holding only pooled counts: kind -> (judged new, judged)."""
    pooled = pd.DataFrame(
        [
            {"candidate": k, "judged_new": n, "candidates": t, "not_judged": 0, "rate": n / t}
            for k, (n, t) in rates.items()
        ]
    )
    table = pooled.assign(stratum="all FOVs")
    return ControlResult(table, pooled, "assessed", ("biological_repeat",), {"fovs_seen": 10})


def _grid_well(rng, well, trend, n=6, pitch=500.0):
    fovs, objects = [], []
    for i in range(n):
        for j in range(n):
            key = {
                "condition": "c",
                "biological_repeat": "r",
                "repeat_group": "r",
                "technical_repeat": well,
                "position_id": f"{well}_{i}_{j}",
            }
            fovs.append({**key, "x_um": i * pitch, "y_um": j * pitch})
            level = 10 + trend * i + rng.normal(0, 0.3)
            objects += [{**key, "x": v} for v in level + rng.normal(0, 0.2, 10)]
    return fovs, objects


def _full(result):
    """The full summary with its line breaks as spaces (the text is wrapped)."""
    return " ".join(result.summary(detail="full").split())


def _curve_with_ceiling(preserved, ceiling, stream=18):
    rows, ceilings = [], []
    for well, by_k in preserved.items():
        for k, value in by_k.items():
            rows.append(
                {
                    "condition": "ctrl",
                    "plate": "P1",
                    "well": well,
                    "criterion": "var",
                    "fovs": k,
                    "preserved": value,
                    "stream_fovs": stream,
                }
            )
        ceilings.append(
            {
                "condition": "ctrl",
                "plate": "P1",
                "well": well,
                "criterion": "var",
                "splits": 10,
                "defined_splits": 10,
                "whole_stream_preserved": ceiling[well],
                "stream_fovs": stream,
            }
        )
    return ReliabilityResult(
        pd.DataFrame(rows),
        pd.DataFrame(),
        ("condition", "plate", "well"),
        {"labels": {"var": "the variance within 10%"}},
        pd.DataFrame(ceilings),
    )


@pytest.fixture(scope="module")
def gradient_replay():
    built = SCENARIOS["neighbour_correlation"]()
    orders = [AcquisitionOrder.recorded(), AcquisitionOrder.permuted(0)] + [
        AcquisitionOrder.dispersed(s) for s in range(2)
    ]
    return replay_acquisitions(
        built.objects,
        built.fovs,
        feature_cols=list(FEATURE_COLS),
        strategies=[FixedFOVStrategy(4)],
        criteria=[
            OutcomeCriterion(
                RelativeMeanError("feature_a"), 0.05, "lower_is_better", "the mean within 5%"
            )
        ],
        objective=POPULATION,
        reservoir_fraction=0.35,
        orders=orders,
        acquisition_order_col=ORDER_COL,
        stage_cols=("stage_x", "stage_y"),
        n_splits=3,
    )


def _two_plates(settings=None, **rates):
    """A ControlResult over two plates: kind -> ((new, judged) on P1, (new, judged) on P2),
    pooled as ``representation_control`` pools them."""
    table = pd.DataFrame(
        [
            {"plate": plate, "stratum": f"plate={plate}", "candidate": kind}
            | {"judged_new": n, "candidates": t, "not_judged": 0, "rate": n / t}
            for kind, per_plate in rates.items()
            for plate, (n, t) in zip(("P1", "P2"), per_plate, strict=True)
        ]
    )
    pooled = table.groupby("candidate", sort=False)[["judged_new", "candidates"]].sum()
    pooled = pooled.reset_index().assign(not_judged=0)
    pooled["rate"] = pooled["judged_new"] / pooled["candidates"]
    return ControlResult(
        table, pooled, "assessed", ("plate",), {"fovs_seen": 10, **(settings or {})}
    )


EMPTY = pd.DataFrame()


MEAN_OF_FEATURE_A = OutcomeCriterion(
    RelativeMeanError("feature_a"), 0.02, "lower_is_better", "the mean"
)


FIXED = [FixedFOVStrategy(3), FixedFOVStrategy(5)]


def _homogeneous_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    built = SCENARIOS["homogeneous"]()
    return built.objects, built.fovs


def _homogeneous_replay(**kwargs: object):
    objects, fovs = _homogeneous_tables()
    return replay_acquisitions(
        objects,
        fovs,
        feature_cols=list(FEATURE_COLS),
        strategies=FIXED,
        criteria=[MEAN_OF_FEATURE_A],
        objective=POPULATION,
        **kwargs,
    )


MEAN_A = OutcomeCriterion(RelativeMeanError("feature_a"), 0.05, "lower_is_better", "the mean")


MEAN_B = OutcomeCriterion(RelativeMeanError("feature_b"), 0.05, "lower_is_better", "mean b")


def _scenario_reliability(criteria=(MEAN_A,)):
    built = SCENARIOS["condition_repeat_groups"]()
    return sampling_reliability(
        built.objects,
        built.fovs,
        feature_cols=list(FEATURE_COLS),
        criteria=list(criteria),
        fov_counts=[2, 4],
        reservoir_fraction=0.3,
        n_splits=3,
        draws=5,
    )


def _mean_curve(preserved, stream=12, labels=None):
    """A reliability result from {(plate, well): {k: preserved}} (one criterion, "mean")."""
    rows = [
        {
            "condition": "ctrl",
            "plate": plate,
            "well": well,
            "criterion": "mean",
            "fovs": k,
            "preserved": value,
            "stream_fovs": stream,
        }
        for (plate, well), by_k in preserved.items()
        for k, value in by_k.items()
    ]
    settings = {"labels": labels or {"mean": "the mean"}}
    return ReliabilityResult(
        pd.DataFrame(rows), pd.DataFrame(), ("condition", "plate", "well"), settings
    )


def _plate_protocol(preserved, plates=("P1", "P2")):
    return protocol_reliability(
        _mean_curve(preserved), HIERARCHY, threshold=0.9, strata=["condition"]
    )


def _well_values(plates, value_5, value_10):
    return {(p, w): {5: value_5, 10: value_10} for p in plates for w in ("A1", "A2")}


def _adequacy():
    built = SCENARIOS["condition_repeat_groups"]()
    replay = replay_acquisitions(
        built.objects,
        built.fovs,
        feature_cols=list(FEATURE_COLS),
        strategies=[FixedFOVStrategy(3), FixedFOVStrategy(1000)],
        criteria=[MEAN_A, MEAN_B],
        objective=ObjectiveConfig(SamplingObjective.POPULATION),
        reservoir_fraction=0.3,
        n_splits=2,
    )
    return adequacy_points(replay)


def _effort(repeats=3):
    rng = np.random.default_rng(6)
    rows, fovs = [], []
    for r in range(repeats):
        for s in range(3):
            for f in range(5):
                name = f"r{r}_s{s}_f{f}"
                fovs.append(
                    {
                        "condition": "ctrl",
                        "biological_repeat": f"R{r}",
                        "repeat_group": f"R{r}",
                        "well": f"W{s}",
                        "image_file": name,
                    }
                )
                values = rng.normal(r * 0.5 + s * 0.3, 1.0, 15)
                rows.append(pd.DataFrame({"image_file": name, "x": values}))
    fov_table = pd.DataFrame(fovs)
    objects = pd.concat(rows, ignore_index=True).merge(fov_table, on="image_file")
    return effort_projection(
        objects,
        fov_table,
        feature="x",
        hierarchy=SamplingHierarchy("image_file", "well", "biological_repeat"),
        strata=["condition"],
        n_boot=50,
    )

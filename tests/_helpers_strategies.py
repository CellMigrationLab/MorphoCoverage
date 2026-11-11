"""Builders shared by the strategy tests: tables, runs, replays and reports of the rules."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scenarios import FEATURE_COLS, ORDER_COL, SCENARIOS

from morphocoverage import (
    ObjectiveConfig,
    SamplingObjective,
    evaluate_acquisition_prefixes,
    run_strategies,
    validate_input_tables,
)
from morphocoverage.benchmark import AcquisitionOrder
from morphocoverage.evaluator import objects_in_fovs, restrict_input
from morphocoverage.results import decision_view
from morphocoverage.sufficiency import sufficiency_report
from morphocoverage.workflows import LiveTables, live_session

POPULATION = ObjectiveConfig(SamplingObjective.POPULATION)


def _area_tables(values_per_fov, feature="area"):
    rows = [
        {"image_file": f"f{j}.tif", feature: value}
        for j, values in enumerate(values_per_fov)
        for value in values
    ]
    objects = pd.DataFrame(rows, columns=["image_file", feature])
    fovs = pd.DataFrame(
        {
            "image_file": [f"f{j}.tif" for j in range(len(values_per_fov))],
            "order": range(1, len(values_per_fov) + 1),
        }
    )
    return objects, fovs


def _run(objects, fovs, strategy, features=("area",)):
    [result] = run_strategies(
        objects, fovs=fovs, strategies=[strategy], feature_cols=list(features), objective=POPULATION
    )
    return result


def _area_replay(objects, fovs, strategy, incremental=False, features=("area",)):
    return evaluate_acquisition_prefixes(
        objects,
        fovs,
        strategies=[strategy],
        feature_cols=list(features),
        acquisition_order_col="order",
        objective=POPULATION,
        incremental=incremental,
    )


def _estimate_input(values_per_fov, features=("f0",)):
    """A ValidatedInput from per-FOV (n_i, d) arrays."""
    rows = [
        {"position_id": f"p{i:03d}", **dict(zip(features, row, strict=True))}
        for i, values in enumerate(values_per_fov)
        for row in np.asarray(values, dtype=float).reshape(len(values), -1)
    ]
    objects = pd.DataFrame(rows, columns=["position_id", *features])
    fovs = pd.DataFrame({"position_id": [f"p{i:03d}" for i in range(len(values_per_fov))]})
    return validate_input_tables(
        objects, fovs=fovs, feature_cols=list(features), fov_col="position_id", group_cols=[]
    )


DISTRIBUTION = ObjectiveConfig(SamplingObjective.DISTRIBUTION)


def _acquisition(values_per_fov, feature="area"):
    """Objects and an FOV table (with an acquisition order) from values per FOV."""
    rows = [
        {"position_id": f"p{j}", feature: value}
        for j, values in enumerate(values_per_fov)
        for value in values
    ]
    objects = pd.DataFrame(rows, columns=["position_id", feature])
    fovs = pd.DataFrame(
        {
            "position_id": [f"p{j}" for j in range(len(values_per_fov))],
            "acq": np.arange(1.0, len(values_per_fov) + 1),
        }
    )
    return objects, fovs


def _words(doc):
    """A docstring with its line breaks and indentation as single spaces."""
    return " ".join(doc.split())


def _area_run(objects, fovs, strategies, objective=POPULATION, feature="area", **kw):
    return run_strategies(
        objects,
        fovs=fovs,
        strategies=strategies,
        feature_cols=[feature],
        objective=objective,
        fov_col="position_id",
        **kw,
    )


def _summary(objects, fovs, strategies, objective=POPULATION):
    report = sufficiency_report(
        objects,
        fovs,
        feature_cols=["area"],
        strategies=strategies,
        objective=objective,
        orders=[AcquisitionOrder.recorded()],
        acquisition_order_col="acq",
        fov_col="position_id",
    )
    # The sentences per group and rule (the answer column), then the short summary, whose
    # wrapped lines are joined so that a caveat reads as one string.
    return "\n".join(report.enough["answer"]) + "\n" + " ".join(report.summary().split())


def _validated_values(values_per_fov, features=("a", "b")):
    """Validated data with one FOV (``f<j>.tif``) per array of ``values_per_fov``."""
    rows = []
    for j, values in enumerate(values_per_fov):
        for row in np.atleast_2d(np.asarray(values, dtype=float)):
            rows.append({"image_file": f"f{j}.tif", **dict(zip(features, row, strict=True))})
    names = [f"f{j}.tif" for j in range(len(values_per_fov))]
    return validate_input_tables(
        pd.DataFrame(rows), fovs=pd.DataFrame({"image_file": names}), feature_cols=list(features)
    )


def _fovs(data, rows):
    """The FOVs at ``rows`` of ``data.fovs`` with their observations."""
    fovs = data.fovs.iloc[list(rows)]
    return restrict_input(data, objects_in_fovs(data, fovs), fovs)


PERMUTED_NINE = [AcquisitionOrder.permuted(seed) for seed in range(9)]


BOTH = [AcquisitionOrder.recorded(), *PERMUTED_NINE]


def _report_both_orders(scenario, strategies, orders=BOTH, **kwargs):
    built = SCENARIOS[scenario]()
    return sufficiency_report(
        built.objects,
        built.fovs,
        feature_cols=list(FEATURE_COLS),
        strategies=strategies,
        objective=POPULATION,
        orders=orders,
        acquisition_order_col=ORDER_COL,
        **kwargs,
    )


def _ordered_acquisition(values_per_fov, feature="x"):
    """Objects and an FOV table (with an acquisition order) from values per FOV."""
    rows = [
        {"position_id": f"p{j}", feature: value}
        for j, values in enumerate(values_per_fov)
        for value in values
    ]
    objects = pd.DataFrame(rows, columns=["position_id", feature])
    fovs = pd.DataFrame(
        {
            "position_id": [f"p{j}" for j in range(len(values_per_fov))],
            "acq": np.arange(1.0, len(values_per_fov) + 1),
        }
    )
    return objects, fovs


def _x_replay(objects, fovs, strategies, incremental=False, objective=POPULATION):
    return evaluate_acquisition_prefixes(
        objects,
        fovs,
        strategies=strategies,
        feature_cols=["x"],
        acquisition_order_col="acq",
        objective=objective,
        fov_col="position_id",
        incremental=incremental,
    )


def _x_run(objects, fovs, strategies, objective=POPULATION, **kw):
    return run_strategies(
        objects,
        fovs=fovs,
        strategies=strategies,
        feature_cols=["x"],
        objective=objective,
        fov_col="position_id",
        **kw,
    )


def _joined_words(text):
    return " ".join(text.split())


def _quantity(result, name):
    return result.evidence.quantities[name]


def _well(seed, n_fovs=8, per_fov=12):
    rng = np.random.default_rng(seed)
    means = rng.normal(10, 1.5, n_fovs)
    return _ordered_acquisition([list(rng.normal(m, 2, per_fov)) for m in means])


def _mixture(rng, n):
    """90% of cells around 10, 10% around 40: a minority state."""
    return list(np.where(rng.random(n) < 0.1, rng.normal(40, 4, n), rng.normal(10, 2, n)))


READOUT = "intensity"


GRID = 6


def _serpentine(grid: int) -> list[int]:
    order = []
    for row in range(grid):
        columns = range(grid) if row % 2 == 0 else range(grid - 1, -1, -1)
        order += [row * grid + c for c in columns]
    return order


def _raster_well(*, drift: float, rare_rows: int = 0, seed: int = 3, objects_per_field: int = 30):
    """A 6 x 6 well in serpentine raster order; the readout drifts by ``drift`` per field.

    With ``rare_rows``, objects of the last rows carry a rare label (every
    fourth object), which the raster reaches last.
    """
    rng = np.random.default_rng(seed)
    object_rows, fov_rows = [], []
    for rank, position in enumerate(_serpentine(GRID)):
        row, column = divmod(position, GRID)
        name = f"f{position:02d}.tif"
        mean = 100.0 * (1 + drift * rank)
        values = mean * np.exp(0.1 * rng.standard_normal(objects_per_field))
        rare = np.zeros(objects_per_field, bool)
        if rare_rows and row >= GRID - rare_rows:
            rare[::4] = True
        object_rows.append(
            pd.DataFrame(
                {
                    "condition": "MOCK",
                    "biological_repeat": "R1",
                    "repeat_group": "R1",
                    "image_file": name,
                    READOUT: values,
                    "rare": rare,
                }
            )
        )
        fov_rows.append(
            {
                "condition": "MOCK",
                "biological_repeat": "R1",
                "repeat_group": "R1",
                "image_file": name,
                "order": rank,
                "stage_x": 500.0 * column,
                "stage_y": 500.0 * row,
            }
        )
    return pd.concat(object_rows, ignore_index=True), pd.DataFrame(fov_rows)


def _live(setup, objects, fovs):
    return LiveTables(live_session(setup)).update(objects, fovs)


def _overall_stop(update) -> bool:
    return "objective): stop" in decision_view(update.results)


def _area_values_tables(values_per_fov, **fov_columns):
    rows = [
        {"image_file": f"f{j}.tif", "area": v}
        for j, values in enumerate(values_per_fov)
        for v in values
    ]
    objects = pd.DataFrame(rows, columns=["image_file", "area"])
    fovs = pd.DataFrame(
        {"image_file": [f"f{j}.tif" for j in range(len(values_per_fov))], **fov_columns}
    )
    return objects, fovs


def _run_one(values_per_fov, strategy, **fov_columns):
    objects, fovs = _area_values_tables(values_per_fov, **fov_columns)
    [result] = run_strategies(
        objects, fovs=fovs, strategies=[strategy], feature_cols=["area"], objective=POPULATION
    )
    return result


ORDERS = [AcquisitionOrder.recorded(), *(AcquisitionOrder.permuted(s) for s in range(3))]


def _flat(text: str) -> str:
    return " ".join(text.split())


def _named_scenario_report(scenario: str, strategies, **kwargs):
    built = SCENARIOS[scenario]()
    return sufficiency_report(
        built.objects,
        built.fovs,
        feature_cols=list(FEATURE_COLS),
        strategies=strategies,
        objective=POPULATION,
        orders=ORDERS,
        acquisition_order_col="acquisition_order",
        **kwargs,
    )

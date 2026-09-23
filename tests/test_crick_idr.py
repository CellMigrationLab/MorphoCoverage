"""Real-data tests on the Crick Enhancing-Reproducibility subset (data/crick_idr).

These check data integrity and that the framework runs end to end on real
acquisitions. They make no claim about which strategy is better.
"""

from pathlib import Path

import pandas as pd
import pytest

from morphocoverage import (
    DescriptorPlateauStrategy,
    DescriptorSpreadStrategy,
    FixedFOVStrategy,
    FixedObjectStrategy,
    ObjectiveConfig,
    SamplingObjective,
    ScalarPrecisionStrategy,
    evaluate_acquisition_prefixes,
    validate_input_tables,
)
from morphocoverage.benchmark import (
    AcquisitionOrder,
    BenchmarkDataset,
    BenchmarkRegistry,
    run_benchmark,
)
from morphocoverage.outcomes import RandomCellRelativeError, RelativeMeanError

DATA = Path(__file__).resolve().parents[1] / "data" / "crick_idr"
POPULATION = ObjectiveConfig(SamplingObjective.POPULATION)


def _load(name):
    fovs = pd.read_csv(DATA / f"{name}_fovs.csv.gz", dtype={"repeat": str})
    objects = pd.read_csv(DATA / f"{name}_nuclei.csv.gz", dtype={"repeat": str})
    return objects, fovs


def _ratio(objects, protein):
    nuclear = objects[f"Nuclei_Intensity_MeanIntensity_{protein}"]
    return nuclear / (nuclear + objects[f"Cytoplasm_Intensity_MeanIntensity_{protein}"])


@pytest.fixture(scope="module")
def idr0139():
    objects, fovs = _load("idr0139")
    return objects.assign(Fascin_Ratio=_ratio(objects, "Fascin")), fovs


@pytest.fixture(scope="module")
def idr0028():
    objects, fovs = _load("idr0028_screenB")
    return objects.assign(YAPTAZ_Ratio=_ratio(objects, "YAPTAZ")), fovs


# Marcotti et al. 2026, Table 1: well, treatment, mean, median, 25th and 75th
# percentiles, SD and number of cells.
TABLE_1 = [
    ("J05", "Untreated", 0.465, 0.511, 0.418, 0.547, 0.131, 1362),
    ("E22", "SN0212398523", 0.488, 0.533, 0.448, 0.558, 0.140, 844),
    ("L08", "Leptomycin b", 0.579, 0.569, 0.543, 0.626, 0.108, 606),
    ("O02", "DMSO", 0.476, 0.524, 0.441, 0.554, 0.131, 1299),
]


@pytest.mark.parametrize("well, treatment, mean, median, q1, q3, sd, ncells", TABLE_1)
def test_idr0139_subset_reproduces_the_papers_table_1(
    idr0139, well, treatment, mean, median, q1, q3, sd, ncells
):
    objects, _ = idr0139
    cells = objects[objects["well"].eq(well)]
    ratio = cells["Fascin_Ratio"]
    assert set(cells["condition"]) == {treatment}
    assert len(ratio) == ncells
    assert round(ratio.mean(), 3) == mean
    assert round(ratio.median(), 3) == median
    assert round(ratio.std(), 3) == sd
    # Two printed quartiles differ by 1 in the third decimal (J05 Q3 0.5464 vs
    # 0.547, E22 Q1 0.4488 vs 0.448), presumably a different quantile method.
    assert ratio.quantile(0.25) == pytest.approx(q1, abs=0.0011)
    assert ratio.quantile(0.75) == pytest.approx(q3, abs=0.0011)


@pytest.mark.parametrize(
    "dataset, n_wells, fields_per_well", [("idr0139", 10, 4), ("idr0028_screenB", 12, 28)]
)
def test_tables_validate_with_morphobricks_defaults(dataset, n_wells, fields_per_well):
    objects, fovs = _load(dataset)
    features = [c for c in objects.columns if c.startswith(("Nuclei_", "Cytoplasm_"))]
    data = validate_input_tables(
        objects, fovs=fovs, feature_cols=features, group_cols=["condition", "repeat", "well"]
    )
    assert data.fov_col == "image_file"
    assert data.fov_key == ("condition", "repeat", "well", "image_file")
    assert len(data.fovs) == n_wells * fields_per_well
    assert fovs.groupby(["repeat", "well"]).size().eq(fields_per_well).all()
    # Location and identifiers are carried as metadata, not features.
    assert not {"nucleus_x_px", "nucleus_y_px", "label_id"} & set(data.feature_cols)


def test_well_names_reused_across_plates_stay_distinct(idr0028):
    _, fovs = idr0028
    j23 = fovs[fovs["well"].eq("J23")]
    assert set(j23["repeat"]) == {"1A", "2A"}
    assert len(j23) == 56


def test_idr0028_acquisition_order_is_defined_within_each_well(idr0028):
    _, fovs = idr0028
    assert not fovs.duplicated(["repeat", "well", "acquisition_order"]).any()
    assert (
        fovs.groupby(["repeat", "well"])["acquisition_order"]
        .agg(["min", "max"])
        .eq([1, 28])
        .all()
        .all()
    )


def test_all_baselines_replay_on_real_acquisitions(idr0028):
    objects, fovs = idr0028
    feature = "YAPTAZ_Ratio"
    strategies = [
        FixedFOVStrategy(10),
        FixedObjectStrategy(250),
        ScalarPrecisionStrategy(feature, 0.05, "object", "upper_95"),
        ScalarPrecisionStrategy(feature, 0.05, "fov", "upper_95"),
        DescriptorPlateauStrategy(feature, 0.01, 3, "acquisition_order"),
        DescriptorSpreadStrategy(feature, 0.02, "fov", 20, 0),
    ]
    results = evaluate_acquisition_prefixes(
        objects,
        fovs,
        strategies=strategies,
        feature_cols=[feature],
        acquisition_order_col="acquisition_order",
        objective=POPULATION,
        group_cols=["condition", "repeat", "well"],
    )
    assert len(results) == len(fovs) * len(strategies)
    counts = objects.groupby(["repeat", "well"]).size()
    for item in results:
        assert item.result.n_fovs == item.step
        if item.step == 28:
            group = item.result.group
            assert item.result.n_objects == counts[group["repeat"], group["well"]]


def test_benchmark_protocol_runs_on_a_real_acquisition(idr0028):
    # The Crick subset is test data, registered as development only to
    # exercise the protocol; it is not a MorphoCoverage benchmark dataset.
    objects, fovs = idr0028
    feature = "YAPTAZ_Ratio"
    registry = BenchmarkRegistry(
        [
            BenchmarkDataset(
                "idr0028_screenB",
                "development",
                lambda: (objects, fovs),
                feature_cols=[feature],
                acquisition_order_col="acquisition_order",
                group_cols=["condition", "repeat", "well"],
                area_col="area_um2",
            )
        ]
    )
    strategies = [
        FixedFOVStrategy(4),
        ScalarPrecisionStrategy(feature, 0.05, "fov", "upper_95"),
    ]
    records = run_benchmark(
        registry,
        "idr0028_screenB",
        strategies=strategies,
        metrics=[RelativeMeanError(feature), RandomCellRelativeError(feature)],
        objective=POPULATION,
        orders=[AcquisitionOrder.recorded(), AcquisitionOrder.permuted(0)],
    )
    assert len(records) == 12 * 2 * len(strategies)
    area = fovs["area_um2"].iloc[0]
    assert fovs["area_um2"].eq(area).all()
    for record in records:
        assert record.total_fovs == 28
        assert record.area == pytest.approx(record.n_fovs * area)
        assert all(value is not None and value >= 0 for value in record.outcomes.values())
        if record.strategy == "fixed_fov":
            assert (record.stop_step, record.n_fovs) == (4, 4)

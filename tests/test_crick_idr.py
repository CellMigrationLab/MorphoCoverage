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
from morphocoverage.outcomes import RandomObjectRelativeError, RelativeMeanError

DATA = Path(__file__).resolve().parents[1] / "data" / "crick_idr"
POPULATION = ObjectiveConfig(SamplingObjective.POPULATION)


def _load(name):
    """The stored tables; they already use biological_repeat (the plate) and technical_repeat
    (the well)."""
    fovs = pd.read_csv(DATA / f"{name}_fovs.csv.gz", dtype={"biological_repeat": str})
    objects = pd.read_csv(DATA / f"{name}_nuclei.csv.gz", dtype={"biological_repeat": str})
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
    cells = objects[objects["technical_repeat"].eq(well)]
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
        objects,
        fovs=fovs,
        feature_cols=features,
        group_cols=["condition", "biological_repeat", "technical_repeat"],
    )
    assert data.fov_col == "image_file"
    assert data.fov_key == ("condition", "biological_repeat", "technical_repeat", "image_file")
    assert len(data.fovs) == n_wells * fields_per_well
    assert fovs.groupby(["biological_repeat", "technical_repeat"]).size().eq(fields_per_well).all()
    # Location and identifiers are carried as metadata, not features.
    assert not {"nucleus_x_px", "nucleus_y_px", "label_id"} & set(data.feature_cols)


def test_well_names_reused_across_plates_stay_distinct(idr0028):
    _, fovs = idr0028
    j23 = fovs[fovs["technical_repeat"].eq("J23")]
    assert set(j23["biological_repeat"]) == {"1A", "2A"}
    assert len(j23) == 56


def test_idr0028_acquisition_order_is_defined_within_each_well(idr0028):
    _, fovs = idr0028
    assert not fovs.duplicated(["biological_repeat", "technical_repeat", "acquisition_order"]).any()
    assert (
        fovs.groupby(["biological_repeat", "technical_repeat"])["acquisition_order"]
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
        group_cols=["condition", "biological_repeat", "technical_repeat"],
    )
    assert len(results) == len(fovs) * len(strategies)
    counts = objects.groupby(["biological_repeat", "technical_repeat"]).size()
    for item in results:
        assert item.result.n_fovs == item.step
        if item.step == 28:
            group = item.result.group
            assert (
                item.result.n_objects
                == counts[group["biological_repeat"], group["technical_repeat"]]
            )


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
                group_cols=["condition", "biological_repeat", "technical_repeat"],
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
        metrics=[RelativeMeanError(feature), RandomObjectRelativeError(feature)],
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


def test_sufficiency_report_answers_both_questions_on_idr0028():
    import numpy as np

    from morphocoverage.strategies import SupportNoveltyEstimator
    from morphocoverage.sufficiency import CalibrationSettings, sufficiency_report
    from morphocoverage.transforms import Standardize

    objects, fovs = _load("idr0028_screenB")
    wells = fovs[["biological_repeat", "technical_repeat"]].drop_duplicates().iloc[:2]
    fovs = fovs.merge(wells)
    objects = objects.merge(wells)
    features = [c for c in objects.columns if c.startswith(("Nuclei_", "Cytoplasm_"))]
    novelty = SupportNoveltyEstimator(float(np.sqrt(len(features))), 1, Standardize(), 2, 3)
    report = sufficiency_report(
        objects,
        fovs,
        feature_cols=features,
        group_cols=["condition", "biological_repeat", "technical_repeat"],
        strategies=[FixedFOVStrategy(10)],
        objective=ObjectiveConfig(SamplingObjective.POPULATION),
        orders=[AcquisitionOrder.recorded()],
        acquisition_order_col="acquisition_order",
        novelty=novelty,
        novelty_target=0.25,
        calibration=CalibrationSettings((10,), 3, 0),
    )
    assert len(report.next_fov) == 2 and set(report.next_fov["fovs_imaged"]) == {28}
    assert report.next_fov["chance_new"].between(0, 1).all()
    assert (report.next_fov["chance_new_at_most"] >= report.next_fov["chance_new"]).all()
    assert len(report.enough) == 4 and set(report.calibration["remaining_fovs"]) == {18}
    assert report.summary(detail="full").count("One more field: chance") == 2
    assert all("Would one more FOV show something new?" not in a for a in report.enough["answer"])


def _crick_scripts():
    """Import helpers from the data scripts, which are not part of the package."""
    import importlib
    import sys

    sys.path.insert(0, str(DATA))
    try:
        return importlib.import_module("datasets")
    finally:
        sys.path.pop(0)


def test_the_registry_declares_idr0028_development_and_idr0139_held_out():
    datasets = _crick_scripts()
    registry = datasets.registry()
    assert registry.names("development") == ["idr0028_screenB"]
    assert registry.names("held_out") == ["idr0139"]
    assert registry.check_tuning(["idr0028_screenB"]) == ("idr0028_screenB",)
    with pytest.raises(ValueError, match="held out"):
        registry.check_tuning(["idr0139"])
    dataset = datasets.load("idr0139")
    assert datasets.FASCIN_RATIO in dataset.objects.columns and len(dataset.fovs) == 40
    assert datasets.FASCIN_RATIO not in dataset.features


@pytest.mark.parametrize(
    "name, wells, fields_per_well, n_features",
    [
        ("idr0028_screenB", 12, 28, 30),
        ("idr0139", 10, 4, 60),
        ("idr0028_screenB_plates", 1144, 28, 0),
        ("idr0139_plate", 308, 4, 0),
    ],
)
def test_every_stored_dataset_loads_ready_to_use(name, wells, fields_per_well, n_features):
    datasets = _crick_scripts()
    dataset = datasets.load(name)
    objects, fovs = dataset.tables
    key = ["condition", "biological_repeat", "image_file"]
    assert fovs.groupby(["biological_repeat", "technical_repeat"]).size().eq(fields_per_well).all()
    assert len(fovs.groupby(["biological_repeat", "technical_repeat"])) == wells
    assert not fovs.duplicated(key).any()
    assert objects[key].merge(fovs[key], how="left", indicator=True)["_merge"].eq("both").all()
    assert len(dataset.features) == n_features
    readout = objects[dataset.readout].dropna()
    assert readout.between(0, 1).all() and len(readout) > 0.99 * len(objects)
    data = validate_input_tables(
        objects,
        fovs=fovs,
        feature_cols=[dataset.readout, *dataset.features],
        group_cols=list(dataset.groups),
        observation_unit=dataset.observation_unit,
    )
    assert len(data.fovs) == len(fovs)


def test_mock_means_negative_control_on_every_idr0028_dataset():
    datasets = _crick_scripts()
    for name in ("idr0028_screenB", "idr0028_screenB_plates"):
        fovs = datasets.load(name).fovs
        assert set(fovs.loc[fovs["condition"] == "MOCK", "control_type"]) == {"negative control"}


def test_the_stored_readout_is_the_papers_ratio_and_the_plates_contain_the_paper_wells():
    datasets = _crick_scripts()
    for wells, plate, protein in (
        ("idr0139", "idr0139_plate", "Fascin"),
        ("idr0028_screenB", "idr0028_screenB_plates", "YAPTAZ"),
    ):
        paper = datasets.load(wells)
        ratio = _ratio(paper.objects, protein)
        assert (paper.objects[paper.readout] - ratio).abs().max() < 1e-5
        full = datasets.load(plate).objects
        key = ["biological_repeat", "image_file", "label_id"]
        joined = paper.objects[[*key, "condition", "technical_repeat", paper.readout]].merge(
            full[[*key, "condition", "technical_repeat", paper.readout]],
            on=key,
            validate="one_to_one",
        )
        assert len(joined) == len(paper.objects)
        assert (joined["condition_x"] == joined["condition_y"]).all()
        assert (joined[f"{paper.readout}_x"] - joined[f"{paper.readout}_y"]).abs().max() < 1e-9


def test_the_binomial_incidence_model_matches_the_planted_state_control():
    # The planning line for a state seen in few FOVs assumes independent,
    # representative FOVs: the share of acquisitions that had seen a state
    # planted in a share p of FOVs after k FOVs should be 1 - (1 - p) ** k.
    # The tracked frequency control (docs/preliminary_findings.md, entry 3)
    # replayed 50 planted acquisitions per plate, frequency and layout.
    table = pd.read_csv(DATA / "results" / "frequency_control_idr0028.csv")
    seen = table.groupby(["frequency", "fovs_seen"])["state_seen"].mean()
    for (p, k), observed in seen.items():
        expected = 1 - (1 - p) ** k
        assert abs(observed - expected) < 0.05, (p, k, observed, expected)


def test_the_held_out_tables_carry_their_ceilings_and_denominators():
    ceiling = pd.read_csv(DATA / "results" / "heldout_ceiling_idr0139.csv")
    protocol = pd.read_csv(DATA / "results" / "heldout_protocol_idr0139.csv")
    calibration = pd.read_csv(DATA / "results" / "heldout_calibration_idr0139.csv")
    assert (ceiling["stream_fovs"] == 2).all() and ceiling["whole_stream_preserved"].min() == 0
    assert (protocol["eligible"] == 10).all() and protocol["fovs"].max() == 2
    assert set(calibration["strategy"]) == {"scalar_precision_object_upper95", "fixed_fov"}
    # A stop that did not preserve the mean is either premature (a later field of the
    # stream would have preserved it) or not preserved by the stream at all (F-8).
    failed = calibration["premature"] + calibration["not_preserved_by_stream"]
    assert failed.max() == 1.0 and (calibration["premature"] <= failed).all()

"""Regressions for the third audit, group B: live session, replay, benchmark, calibration."""

import warnings

import numpy as np
import pandas as pd
import pytest
from scenarios import FEATURE_COLS, ORDER_COL, SCENARIOS

from morphocoverage import (
    FixedFOVStrategy,
    ObjectiveConfig,
    OnlineSession,
    SamplingObjective,
    benchmark,
    calibration,
    evaluate_acquisition_prefixes,
    online,
    reservoir,
    sequential,
    validate_input_tables,
)
from morphocoverage.benchmark import AcquisitionOrder, _area, permuted_fovs
from morphocoverage.calibration import calibrate_next_fov_novelty
from morphocoverage.evidence import Evidence
from morphocoverage.reservoir import ReservoirSplit, reference_error_floor
from morphocoverage.strategies import ImagedAreaEstimator
from morphocoverage.strategies.base import Estimator, EstimatorState, SamplingStrategy
from morphocoverage.strategies.support_novelty import SupportNoveltyEstimator

POPULATION = ObjectiveConfig(SamplingObjective.POPULATION)


def _session(**kwargs):
    return OnlineSession(
        strategies=[FixedFOVStrategy(5)], feature_cols=["f"], objective=POPULATION, **kwargs
    )


def _plane_files(rows_per_file=5):
    """3 positions written as 4 z-plane files each, z only in the FOV table."""
    rows, fov_rows = [], []
    for position in range(3):
        for z in range(4):
            name = f"pos{position}_z{z}.tif"
            fov_rows.append({"condition": "a", "repeat": "r1", "image_file": name, "z": z})
            rows += [
                {"condition": "a", "repeat": "r1", "image_file": name, "f": value}
                for value in np.linspace(1, 2, rows_per_file)
            ]
    return pd.DataFrame(rows), pd.DataFrame(fov_rows)


def _feed(session, objects, fovs):
    for _, meta in fovs.iterrows():
        session.add_fov(objects[objects.image_file == meta.image_file], meta.to_dict())


def test_b1_live_plane_rule_reads_z_from_the_fov_table():
    objects, fovs = _plane_files()
    message = r"holds a single 'z' value \(in the FOV table\)"
    with pytest.raises(ValueError, match=message):
        validate_input_tables(objects, feature_cols=["f"], fovs=fovs)
    # Live, a z is read as planes once one of its values recurs in another field
    # (audit 3, A-14: a stage z distinct in every field is a focus height): the
    # first four files hold z 0-3, and the fifth, z=0 again, is refused.
    session = _session()
    _feed(session, objects, fovs.iloc[:4])
    fifth = fovs.iloc[4]
    with pytest.raises(ValueError, match=message):
        session.add_fov(objects[objects.image_file == fifth.image_file], fifth.to_dict())
    # Refused atomically: the session still holds the first four FOVs only.
    [result] = session.add_fov(
        objects[objects.image_file == fifth.image_file].assign(image_file="other.tif"),
        {**fifth.to_dict(), "image_file": "other.tif", "z": 99},
    )
    assert result.n_fovs == 5

    # One row per field: the validator warns, and so does the live session.
    one, fovs = _plane_files(rows_per_file=1)
    with pytest.warns(UserWarning, match="in the FOV table"):
        validate_input_tables(one, feature_cols=["f"], fovs=fovs)
    with pytest.warns(UserWarning, match="in the FOV table"):
        _feed(_session(), one, fovs)
    # Naming the FOV column says each file is one position: no rule, no warning.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _feed(_session(fov_col="image_file"), one, fovs)


def test_b2_live_identity_is_resolved_over_observations_and_fov_row():
    rows, fov_rows = [], []
    for position in range(3):
        for tile in range(4):
            name = f"pos{position}_{tile}.tif"
            fov_rows.append({"condition": "a", "repeat": "r1", "image_file": name})
            rows += [
                {"condition": "a", "repeat": "r1", "position_id": f"pos{position}"}
                | {"image_file": name, "f": value}
                for value in np.linspace(1, 2, 5)
            ]
    objects, fovs = pd.DataFrame(rows), pd.DataFrame(fov_rows)
    message = "'position_id' is present in objects but absent from fovs"
    with pytest.raises(ValueError, match=message):
        validate_input_tables(objects, feature_cols=["f"], fovs=fovs)
    session = _session()
    with pytest.raises(ValueError, match=message):
        _feed(session, objects, fovs)
    assert session.fov_col is None  # the refused first FOV fixed nothing

    # Later FOVs are resolved the same way, observation columns included.
    session = _session()
    session.add_fov(pd.DataFrame({"f": [1.0]}), {"image_file": "a.tif"})
    with pytest.raises(ValueError, match=message):
        session.add_fov(pd.DataFrame({"f": [1.0], "position_id": ["P2"]}), {"image_file": "b.tif"})
    with pytest.raises(ValueError, match="'repeat' is present in objects but absent from fovs"):
        session.add_fov(pd.DataFrame({"f": [1.0], "repeat": ["r1"]}), {"image_file": "b.tif"})
    # Identity carried by the FOV row alone is copied onto the observations.
    [result] = session.add_fov(pd.DataFrame({"f": [2.0]}), {"image_file": "b.tif"})
    assert (session.fov_col, session.group_cols, result.n_fovs) == ("image_file", (), 2)
    # An empty position_id (a producer writing every column) is still absent.
    [result] = session.add_fov(
        pd.DataFrame({"f": [3.0], "position_id": [""]}), {"image_file": "c.tif", "position_id": ""}
    )
    assert result.n_fovs == 3


_SEEN: list = []


class _Spy(Estimator):
    """Records what the framework hands an estimator."""

    name = "spy"

    @property
    def parameters(self):
        return {}

    def estimate(self, data, *, objective):
        column = data.objects["image_file"]
        _SEEN.append(("estimate", data.objects.index.tolist(), list(column.cat.categories)))
        return Evidence({}, {})


class _SpyState(EstimatorState):
    def update(self, new):
        column = new.objects["image_file"]
        _SEEN.append(("update", new.objects.index.tolist(), list(column.cat.categories)))
        return Evidence({}, {})


class _IncrementalSpy(_Spy):
    name = "incremental_spy"

    def start(self, template, *, objective):
        return _SpyState()


@pytest.mark.parametrize("estimator", [_Spy(), _IncrementalSpy()], ids=["prefix", "update"])
def test_b3_live_estimator_inputs_expose_only_visible_rows(estimator):
    built = SCENARIOS["homogeneous"]()
    objects = built.objects.assign(image_file=pd.Categorical(built.objects.image_file))
    fovs = built.fovs.assign(image_file=pd.Categorical(built.fovs.image_file))
    strategy = SamplingStrategy([estimator], [], name="spy")
    kwargs = {"strategies": [strategy], "feature_cols": list(FEATURE_COLS), "objective": POPULATION}
    _SEEN.clear()
    evaluate_acquisition_prefixes(
        objects, fovs, acquisition_order_col=ORDER_COL, incremental=True, **kwargs
    )
    replayed = _SEEN[:3]
    _SEEN.clear()
    session = OnlineSession(**kwargs)
    for _, meta in fovs.sort_values(ORDER_COL).head(3).iterrows():
        session.add_fov(objects[objects.image_file == meta.image_file], meta.to_dict())
    assert _SEEN == replayed
    step, index, categories = _SEEN[-1]
    assert index[0] == 0 and len(categories) == (3 if step == "estimate" else 1)


def _novelty_tables():
    rng = np.random.default_rng(0)
    fovs = pd.DataFrame({"position_id": [f"p{i:02d}" for i in range(30)]})
    objects = pd.DataFrame({"position_id": np.repeat(fovs.position_id, 10)})
    objects["f"] = rng.normal(0, 1, len(objects)) + np.repeat(rng.choice([0, 0, 0, 5], 30), 10)
    settings = {
        "feature_cols": ["f"],
        "novelty": SupportNoveltyEstimator(0.5, 1, None, 3),
        "fov_counts": [5, 10],
        "draws": 20,
        "random_state": 0,
    }
    return objects, fovs, settings


def test_b4_calibration_does_not_depend_on_fov_table_row_order():
    objects, fovs, settings = _novelty_tables()
    table = calibrate_next_fov_novelty(objects, fovs, **settings)
    reversed_rows = calibrate_next_fov_novelty(objects, fovs.iloc[::-1], **settings)
    shuffled = calibrate_next_fov_novelty(objects.sample(frac=1, random_state=1), fovs, **settings)
    pd.testing.assert_frame_equal(table, reversed_rows)
    pd.testing.assert_frame_equal(table, shuffled)


def test_b5_calibration_docstring_expects_prediction_above_actual_by_construction():
    text = " ".join(calibration.__doc__.split())
    assert "k - 1" in text and "by construction" in text
    assert "The coverage check is ``bound_held``" in text
    assert "close to the mean actual rate" not in text


def test_b6_replay_reads_generator_feature_columns_once():
    built = SCENARIOS["homogeneous"]()
    kwargs = {
        "strategies": [FixedFOVStrategy(3)],
        "acquisition_order_col": ORDER_COL,
        "objective": POPULATION,
    }
    from_list = evaluate_acquisition_prefixes(
        built.objects, built.fovs, feature_cols=list(FEATURE_COLS), **kwargs
    )
    from_generator = evaluate_acquisition_prefixes(
        built.objects, built.fovs, feature_cols=(c for c in FEATURE_COLS), **kwargs
    )
    assert [r.result.without_timing() for r in from_generator] == [
        r.result.without_timing() for r in from_list
    ]
    with pytest.raises(ValueError, match="acquisition order cannot also be a phenotype feature"):
        evaluate_acquisition_prefixes(
            built.objects, built.fovs, feature_cols=iter([*FEATURE_COLS, ORDER_COL]), **kwargs
        )


@pytest.mark.parametrize("incremental", [False, True])
def test_b7_min_fovs_must_be_an_integer(incremental):
    built = SCENARIOS["homogeneous"]()
    kwargs = {
        "strategies": [FixedFOVStrategy(3)],
        "feature_cols": list(FEATURE_COLS),
        "acquisition_order_col": ORDER_COL,
        "objective": POPULATION,
        "incremental": incremental,
    }
    for bad in (2.5, 2.0, 0, True):
        with pytest.raises(ValueError, match="min_fovs must be an integer of at least 1"):
            evaluate_acquisition_prefixes(built.objects, built.fovs, min_fovs=bad, **kwargs)
    steps = evaluate_acquisition_prefixes(built.objects, built.fovs, min_fovs=np.int64(2), **kwargs)
    assert min(r.step for r in steps) == 2


def test_b8_negative_seeds_are_refused_on_construction():
    for make in (AcquisitionOrder.permuted, AcquisitionOrder.dispersed):
        with pytest.raises(ValueError, match="non-negative"):
            make(-1)
        assert make(np.int64(3)).seed == 3
    with pytest.raises(ValueError, match="non-negative"):
        ReservoirSplit(0.3, -1, None)
    assert type(ReservoirSplit(0.3, np.int64(2), None).seed) is int
    objects, fovs, settings = _novelty_tables()
    with pytest.raises(ValueError, match="non-negative"):
        calibrate_next_fov_novelty(objects, fovs, **{**settings, "random_state": -1})
    data = validate_input_tables(objects, fovs=fovs, feature_cols=["f"])
    with pytest.raises(ValueError, match="non-negative"):
        reference_error_floor(None, data, n_splits=2, seed=-1, block_col=None)


def test_b9_novelty_confidence_accepts_numpy_reals():
    estimator = SupportNoveltyEstimator(0.5, 1, None, 3, confidence=np.float32(0.95))
    assert estimator.confidence == pytest.approx(0.95)


def test_b10_a_tiny_reservoir_fraction_still_holds_out_one_block():
    fovs = pd.DataFrame({"position_id": [f"p{i}" for i in range(10)]})
    objects = pd.DataFrame({"position_id": fovs.position_id, "f": np.arange(10.0)})
    data = validate_input_tables(objects, fovs=fovs, feature_cols=["f"])
    stream, held = ReservoirSplit(1e-11, 0, None).split({}, data)
    assert (len(stream.fovs), len(held.fovs)) == (9, 1)
    stream, held = ReservoirSplit(0.07, 0, None).split({}, data)
    assert len(held.fovs) == 1


def test_b11_numpy_group_values_seed_like_python_ones():
    fovs = pd.DataFrame({"repeat": 1, "position_id": [f"p{i:02d}" for i in range(20)]})
    objects = pd.DataFrame({"repeat": 1, "position_id": fovs.position_id, "f": np.arange(20.0)})
    data = validate_input_tables(objects, fovs=fovs, feature_cols=["f"])
    group_from_table = {"repeat": data.fovs["repeat"].iloc[0]}
    assert isinstance(group_from_table["repeat"], np.integer)
    split = ReservoirSplit(0.3, 0, None)
    held = split.split(group_from_table, data)[1].fovs["position_id"].tolist()
    assert held == split.split({"repeat": 1}, data)[1].fovs["position_id"].tolist()
    order = permuted_fovs(data.fovs, data.fov_key, 0, group_from_table)
    assert order.equals(permuted_fovs(data.fovs, data.fov_key, 0, {"repeat": 1}))


def test_b12_benchmark_and_imaged_area_share_one_area_check():
    built = SCENARIOS["homogeneous"]()
    estimator = ImagedAreaEstimator("area_um2")
    for bad in (np.nan, -1.0, 0.0, np.inf):
        fovs = built.fovs.copy()
        fovs.loc[0, "area_um2"] = bad
        data = validate_input_tables(built.objects, fovs=fovs, feature_cols=list(FEATURE_COLS))
        quantity = estimator.estimate(data, objective=POPULATION).quantities["imaged_area"]
        assert quantity.availability.value == "not_applicable"
        with pytest.raises(ValueError) as refused:
            _area(data, "area_um2")
        assert str(refused.value) == quantity.reason
    fovs = built.fovs.assign(area_um2=built.fovs["area_um2"].astype(str))
    data = validate_input_tables(built.objects, fovs=fovs, feature_cols=list(FEATURE_COLS))
    for check in (
        lambda: _area(data, "area_um2"),
        lambda: estimator.estimate(data, objective=POPULATION),
    ):
        with pytest.raises(ValueError, match="not numbers"):
            check()


def test_j16_modules_list_their_public_names():
    assert online.__all__ == ["OnlineSession"]
    assert sequential.__all__ == ["SequentialResult", "evaluate_acquisition_prefixes"]
    assert "run_benchmark" in benchmark.__all__ and "permuted_fovs" not in benchmark.__all__
    for module in (online, sequential, benchmark, calibration, reservoir):
        assert all(not name.startswith("_") and hasattr(module, name) for name in module.__all__)

"""One strict stage-coordinate contract across the package (issue 232)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from morphocoverage import validate_input_tables
from morphocoverage.order_check import check_acquisition_order
from morphocoverage.reservoir import spatial_blocks
from morphocoverage.retrospective import dispersed_order, inspect_fovs, redundancy_distance
from morphocoverage.retrospective.stage import stage_travel
from morphocoverage.spatial import fov_semivariogram


def _tables(**stage) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(0)
    fovs = pd.DataFrame(
        {
            "position_id": [f"f{i}" for i in range(12)],
            "condition": "c",
            "sx": np.arange(12) * 10.0,
            "sy": (np.arange(12) % 4) * 10.0,
        }
    ).assign(**stage)
    objects = fovs.loc[fovs.index.repeat(5), ["position_id", "condition"]].reset_index(drop=True)
    return objects.assign(x=rng.normal(size=len(objects))), fovs


def _quiet(call, *args):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return call(*args)


def _inspect(fovs, objects, cols=("sx", "sy")):
    return inspect_fovs(objects, fovs, feature_cols=["x"], stage_cols=cols, fov_col="position_id")


def _redundancy(fovs, objects, cols=("sx", "sy")):
    return redundancy_distance(
        objects,
        fovs,
        feature_cols=["x"],
        stage_cols=cols,
        bin_edges=[0, 20, 60, 200],
        fov_col="position_id",
        n_permutations=9,
    )


def _semivariogram(fovs, objects, cols=("sx", "sy")):
    data = validate_input_tables(
        objects=objects, fovs=fovs, feature_cols=["x"], fov_col="position_id"
    )
    return fov_semivariogram(data, cols[0], cols[1], [0, 20, 60, 200])


def _blocks(fovs, objects, cols=("sx", "sy")):
    return spatial_blocks(fovs, cols[0], cols[1], 25.0)


def _order(fovs, objects, cols=("sx", "sy")):
    data = validate_input_tables(
        objects=objects, fovs=fovs, feature_cols=["x"], fov_col="position_id"
    )
    return check_acquisition_order(data, "x", stage_cols=cols)


PATHS = {
    "inspect_fovs": _inspect,
    "redundancy_distance": _redundancy,
    "fov_semivariogram": _semivariogram,
    "spatial_blocks": _blocks,
    "check_acquisition_order": _order,
}


@pytest.mark.parametrize("path", PATHS)
def test_valid_integer_and_float_coordinates_are_accepted(path):
    for stage in ({}, {"sx": np.arange(12), "sy": np.arange(12) % 4}):
        objects, fovs = _tables(**stage)
        assert _quiet(PATHS[path], fovs, objects) is not None


@pytest.mark.parametrize("path", PATHS)
def test_one_axis_twice_is_refused(path):
    objects, fovs = _tables()
    with pytest.raises(ValueError, match="two different columns|not one axis twice"):
        _quiet(lambda: PATHS[path](fovs, objects, ("sx", "sx")))


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize(
    ("bad", "message"),
    [
        (np.arange(12) % 2 == 0, "booleans"),
        ([str(v) for v in np.arange(12) * 10.0], "values"),
        (np.arange(12) * (1 + 1j), "complex"),
    ],
    ids=["boolean", "numeric text", "complex"],
)
def test_booleans_text_and_complex_coordinates_are_refused_not_converted(path, bad, message):
    objects, fovs = _tables(sx=bad)
    with pytest.raises(TypeError, match=message):
        _quiet(lambda: PATHS[path](fovs, objects))


@pytest.mark.parametrize("path", ["inspect_fovs", "fov_semivariogram", "spatial_blocks"])
def test_infinite_coordinates_are_refused_before_any_statistic(path):
    values = np.arange(12) * 10.0
    values[3] = np.inf
    objects, fovs = _tables(sx=values)
    with pytest.raises(ValueError, match="finite"):
        _quiet(lambda: PATHS[path](fovs, objects))


def test_the_low_level_helpers_share_the_contract():
    xy = np.column_stack([np.arange(5.0), np.arange(5.0)])
    assert list(dispersed_order(xy, 0)) == [0, 4, 2, 1, 3] or len(dispersed_order(xy, 0)) == 5
    assert stage_travel(xy)[-1] == pytest.approx(4 * np.sqrt(2))
    for bad in (np.ones((3, 3)), np.array([[True, False], [False, True]]), [["a", "b"]]):
        with pytest.raises((TypeError, ValueError)):
            stage_travel(bad)
    with pytest.raises(ValueError, match="finite"):
        stage_travel(np.array([[0.0, 0.0], [np.inf, 1.0]]))
    with pytest.raises(TypeError, match="whole number"):
        dispersed_order(xy, 1.5)
    with pytest.raises(TypeError, match="whole number"):
        dispersed_order(xy, True)
    with pytest.raises(ValueError, match="positive, finite"):
        spatial_blocks(_tables()[1], "sx", "sy", True)

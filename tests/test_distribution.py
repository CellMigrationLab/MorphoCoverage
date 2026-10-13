"""Beyond the mean (``sampling_unit.distribution``, ``workflows._distribution``)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
from _helpers_sampling_unit import _signed, _three

from morphocoverage.sampling_unit.distribution import distribution_contrast
from morphocoverage.statistics import BIMODAL, bimodality_coefficient
from morphocoverage.workflows._distribution import split_words


def _distribution_tables(factor: float, spread: float = 1.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Three repeats of 6 fields of 20 cells a side: ``a`` the control's lognormal cells
    times ``factor``, their log spread times ``spread``."""
    rng = np.random.default_rng(13)
    rows = []
    for repeat in ("r1", "r2", "r3"):
        for condition in ("ctrl", "a"):
            for field in range(6):
                logs = rng.normal(0, 0.5, 20)
                if condition == "a":
                    logs = logs * spread + np.log(factor)
                for label, value in enumerate(np.exp(logs)):
                    rows.append(
                        {
                            "condition": condition,
                            "biological_repeat": repeat,
                            "image_file": f"{repeat}_{condition}_{field}.tif",
                            "label_id": label,
                            "area": 100 * value,
                        }
                    )
    objects = pd.DataFrame(rows)
    fovs = objects[["condition", "biological_repeat", "image_file"]].drop_duplicates()
    return objects, fovs


def test_beyond_the_mean_takes_out_the_mean_shift_on_the_features_scale() -> None:
    # C2: a condition that only moved its mean read "differs beyond the mean", the quantile
    # shifts holding the mean's shift (a25), and a fold change moves every quantile of a raw
    # positive feature.
    objects, fovs = _distribution_tables(2.2)
    result = distribution_contrast(
        objects, fovs, features=["area"], control={"condition": "ctrl"}, observation_unit="cell"
    )
    assert result.settings["log_scale"] == {"area": True}
    shifts = result.tests[result.tests["statistic"].isin(["q10", "q50", "q90"])]
    assert (shifts["mean"].abs() < 0.2).all()
    assert result.differing().empty
    # On the difference scale, the quantile shifts beyond the mean are the skew's alone.
    raw = distribution_contrast(
        objects,
        fovs,
        features=["area"],
        control={"condition": "ctrl"},
        observation_unit="cell",
        scales={"area": "difference"},
    )
    assert raw.settings["log_scale"] == {"area": False}
    # A wider distribution, the same median on the log scale: its spread differs.
    objects, fovs = _distribution_tables(1.0, spread=2.0)
    wide = distribution_contrast(
        objects, fovs, features=["area"], control={"condition": "ctrl"}, observation_unit="cell"
    )
    assert set(wide.differing()["statistic"].iloc[0]) >= {"spread"}


def _fields_design(seed: int = 0):
    """3 repeats; the control one field of 60 cells, the condition three fields of 20 cells
    shifted -3, 0 and +3 cell SDs: no effect beyond the field design."""
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(3):
        for condition, shifts, cells in (("ctrl", [0.0], 60), ("a", [-3.0, 0.0, 3.0], 20)):
            for f, shift in enumerate(shifts):
                rows += [
                    {
                        "condition": condition,
                        "biological_repeat": f"R{r}",
                        "image_file": f"{condition}{r}_{f}.tif",
                        "area": 10 + shift + value,
                    }
                    for value in rng.normal(size=cells)
                ]
    objects = pd.DataFrame(rows)
    fovs = objects[["condition", "biological_repeat", "image_file"]].drop_duplicates()
    return objects, fovs.reset_index(drop=True)


def test_a5_beyond_the_mean_reads_field_balanced_subsamples():
    from morphocoverage.sampling_unit.distribution import distribution_contrast

    objects, fovs = _fields_design()
    kwargs = {"features": ["area"], "control": {"condition": "ctrl"}, "fov_col": "image_file"}
    tests = distribution_contrast(objects, fovs, random_state=3, **kwargs).tests
    spread = tests.set_index("statistic").loc["spread"]
    # Pooled, the condition's SD would be sqrt(7) times the control's (log 0.97).
    assert abs(spread["mean"]) < 0.25 and spread["high"] < 0.4
    again = distribution_contrast(objects, fovs, random_state=3, **kwargs).tests
    np.testing.assert_allclose(again["mean"], tests["mean"])


def test_a5_seed10_beyond_the_mean_reads_the_features_the_fingerprint_reads():
    from morphocoverage.multivariate_outcomes import CovarianceError
    from morphocoverage.retrospective.criteria import OutcomeCriterion
    from morphocoverage.workflows import Setup
    from morphocoverage.workflows._distribution import (
        NOT_RUN,
        distribution_by_feature,
        with_distribution,
    )

    objects, fovs = _signed(offset_b=3.0)
    objects = objects.assign(d0=objects["area"] * 2, d1=objects["positive"] / 3)

    def setup(columns):
        criterion = OutcomeCriterion(CovarianceError(columns, True), 0.2, "lower_is_better")
        return Setup(
            readout="area",
            features=["area", "positive"],
            control={"condition": "ctrl"},
            feature_tables={"DINO": ("d0", "d1")},
            fingerprint=(criterion,),
            fov_col="image_file",
        )

    one = distribution_by_feature(objects, fovs, setup(("positive", "d0")))
    assert one.settings["features"] == ["positive"]
    dino = setup(("d0", "d1"))
    assert distribution_by_feature(objects, fovs, dino) is None

    @dataclass(frozen=True)
    class Report:  # the two texts with_distribution writes into
        short: dict
        sections: dict

    empty = Report({"conditions": "Do they differ?"}, {"conditions": "Do they differ?"})
    said = with_distribution(empty, None, "cells", dino)
    assert " ".join(NOT_RUN.split()) in " ".join(said.short["conditions"].split())


def test_beyond_the_mean_trend_only_from_six_repeats():
    # The trend line from TREND_REPEATS repeats, as per feature (owner, 2026-10-06, audit 8).
    from morphocoverage.workflows._distribution import _consistent

    rows = pd.DataFrame(
        {
            "feature": ["a", "b"],
            "repeats": [3, 6],
            "lower": [3, 6],
            "higher": [0, 0],
            "q": [0.4, 0.4],
        }
    )
    assert list(_consistent(rows)["feature"]) == ["b"]


def test_a_condition_that_splits_its_cells_is_said_to():
    rng = np.random.default_rng(3)
    assert bimodality_coefficient(rng.normal(size=400)) < BIMODAL
    assert (
        bimodality_coefficient(np.r_[rng.normal(-2.5, 1, 200), rng.normal(2.5, 1, 200)]) > BIMODAL
    )
    objects, fovs = _three()
    split = objects["condition"] == "drug"
    halves = np.where(np.arange(split.sum()) % 2 == 0, -3.0, 3.0)
    objects.loc[split, "x"] = objects.loc[split, "x"] + halves
    result = distribution_contrast(
        objects, fovs, features=["x"], control={"condition": "ctrl"}, observation_unit="cell"
    )
    [row] = result.splits().to_dict("records")
    assert row["condition"] == "drug" and row["split_repeats"] == 3
    assert split_words(result, "drug").startswith("its cells may split into two groups on x")
    assert split_words(result, "ctrl") == ""


def test_the_distribution_families_are_counted_as_corrected():
    """#302: families of 4 and 8 tests read as "6 tests in each of 2 sub-experiments"."""
    from morphocoverage.sampling_unit.distribution import DistributionContrast, tests_words

    def result(sizes):
        experiment = [e for e, n in enumerate(sizes, start=1) for _ in range(n)]
        tests = pd.DataFrame({"p": 0.5, "experiment": experiment})
        return DistributionContrast(tests, {"families": len(sizes)})

    assert tests_words(result([4, 8])) == "4 to 8 tests in each of 2 sub-experiments"
    assert tests_words(result([4, 4])) == "4 tests in each of 2 sub-experiments"
    assert tests_words(result([8])) == "8 tests"


@pytest.mark.parametrize("alpha", [5, 0, 1, -0.1, float("nan"), True, "0.05"])
def test_the_distribution_gate_refuses_an_alpha_that_is_not_a_probability(alpha):
    """#310: alpha=5 (meant as 5%) passed every finite q without a word."""
    objects, fovs = _three()
    with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
        distribution_contrast(
            objects, fovs, features=["x"], control={"condition": "ctrl"}, alpha=alpha
        )


def test_a_skewed_condition_is_not_said_to_split_in_two():
    objects, fovs = _three()
    drug = objects["condition"] == "drug"
    rng = np.random.default_rng(8)
    objects.loc[drug, "x"] = 10 + rng.exponential(2.0, drug.sum())
    result = distribution_contrast(
        objects, fovs, features=["x"], control={"condition": "ctrl"}, observation_unit="cell"
    )
    assert result.splits().empty

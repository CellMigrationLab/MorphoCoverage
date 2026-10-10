"""Audit A2: question 2 gives no chance of a new state where the definition of new is not
informative at the group's size (the same null-rate guard as question 3)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from morphocoverage import ObjectiveConfig, SamplingObjective
from morphocoverage.benchmark import AcquisitionOrder
from morphocoverage.strategies import SupportNoveltyEstimator
from morphocoverage.sufficiency import sufficiency_report
from morphocoverage.transforms import RobustStandardize
from morphocoverage.workflows import default_novelty


def _homogeneous(n_features: int, n_fields: int, per_field: int = 1):
    rng = np.random.default_rng(0)
    names = [f"f{i}" for i in range(n_features)]
    rows = [
        {"condition": "a", "position_id": f"p{i}", **dict(zip(names, rng.normal(size=n_features)))}
        for i in range(n_fields)
        for _ in range(per_field)
    ]
    fovs = pd.DataFrame({"condition": "a", "position_id": [f"p{i}" for i in range(n_fields)]})
    return pd.DataFrame(rows), fovs, names


def _report(novelty, objects, fovs, names):
    return sufficiency_report(
        objects,
        fovs,
        feature_cols=names,
        strategies=[],
        objective=ObjectiveConfig(SamplingObjective.POPULATION),
        orders=[AcquisitionOrder.permuted(0)],
        novelty=novelty,
        fov_col="position_id",
        group_cols=["condition"],
    )


def test_a_definition_that_calls_a_homogeneous_group_new_gives_no_chance():
    objects, fovs, names = _homogeneous(17, 6)
    report = _report(default_novelty(names), objects, fovs, names)
    row = report.next_fov.iloc[0]
    assert (
        isinstance(row["uninformative"], str)
        and "not informative with 6 fields" in row["uninformative"]
    )
    short = " ".join(report.summary().split())
    full = " ".join(report.summary(detail="full").split())
    assert "not informative" in short and "no chance of a new state" in short
    assert "One more field: not informative" in full
    assert "Fields still needed" not in full
    assert "chance" not in short.split("One more field")[1].split(".")[0].replace(
        "no chance of a new state", ""
    )
    assert report.next_fov["answer"].iloc[0].startswith("Not informative")


def test_an_informative_definition_keeps_its_chance():
    objects, fovs, names = _homogeneous(2, 8)
    novelty = SupportNoveltyEstimator(3.0, 1, RobustStandardize(), 2, 3)
    report = _report(novelty, objects, fovs, names)
    assert report.next_fov["uninformative"].isna().all()
    assert "chance" in " ".join(report.summary(detail="full").split())

"""Tests of ``retrospective.criteria``: did a set of fields preserve what the experiment needs?"""

import re

import numpy as np
import pandas as pd
import pytest
from _helpers_benchmark import MEAN_F
from _helpers_retrospective import RARE, _adequacy, _f_reliability, _f_replay, _make, _tables
from _helpers_strategies import _flat
from _helpers_workflows import _sictrl_tables

from morphocoverage import FixedFOVStrategy
from morphocoverage.benchmark import AcquisitionOrder
from morphocoverage.data_check import validate_input_tables
from morphocoverage.multivariate_outcomes import CovarianceError, EnergyError, QuantileError
from morphocoverage.outcomes import RareStateDetected, RelativeMeanError
from morphocoverage.retrospective import (
    OutcomeCriterion,
    SamplingHierarchy,
    protocol_reliability,
    sampling_reliability,
    stopping_calibration,
)
from morphocoverage.retrospective.criteria import Preservation
from morphocoverage.workflows import Setup
from morphocoverage.workflows._enough_targets import fingerprint_precision
from morphocoverage.workflows._fingerprint_space import noun
from morphocoverage.workflows._guided_ask import ask_tolerance


def test_f1_an_empty_subset_is_undefined_when_the_reference_itself_is_undefined():
    objects, fovs = _make(n_fovs=4, empty=(0,))
    empty = validate_input_tables(
        objects.iloc[:0], fovs=fovs.iloc[:1], feature_cols=["f"], group_cols=[]
    )
    reference = validate_input_tables(
        objects[objects["position_id"] != "p00"], fovs=fovs.iloc[1:], feature_cols=["f"]
    )
    assert MEAN_F.assess(reference, empty) == (Preservation.NOT_PRESERVED, None)
    missing = objects[objects["position_id"] != "p00"].copy()
    missing.iloc[0, missing.columns.get_loc("f")] = np.nan
    undefined = validate_input_tables(missing, fovs=fovs.iloc[1:], feature_cols=["f"])
    assert MEAN_F.assess(undefined, empty) == (Preservation.UNDEFINED, None)
    # A rare state the reference does not show cannot be judged either.
    objects["rare"] = 0
    rare = OutcomeCriterion(RareStateDetected("rare", 1), 1, "higher_is_better")
    no_state = validate_input_tables(objects, fovs=fovs, feature_cols=["f"])
    assert rare.assess(no_state, empty)[0] is Preservation.UNDEFINED


def test_f8_a_low_ceiling_is_not_called_unjudgeable():
    objects, fovs = [], []
    for plate in range(2):
        for well in range(2):
            o, f = _make(n_fovs=8, seed=10 * plate + well)
            for table in (o, f):
                table["plate"], table["well"] = f"b{plate}", f"w{well}"
            objects.append(o)
            fovs.append(f)
    criterion = OutcomeCriterion(RelativeMeanError("f"), 0.01, "lower_is_better")
    reliability = sampling_reliability(
        pd.concat(objects, ignore_index=True),
        pd.concat(fovs, ignore_index=True),
        feature_cols=["f"],
        criteria=[criterion],
        fov_counts=[2],
        reservoir_fraction=0.3,
        n_splits=4,
        draws=5,
        group_cols=["condition", "biological_repeat", "plate", "well"],
    )
    result = protocol_reliability(
        reliability,
        SamplingHierarchy("position_id", "well", "plate"),
        threshold=0.9,
        strata=["condition", "biological_repeat"],
    )
    assert result.difficulty["ceiling"].notna().all()
    text = " ".join(result.summary(detail="full").split())
    short = " ".join(result.summary().split())
    assert "could not judge" not in text and "could not judge" not in short
    assert "even every available field together was enough in fewer than 90%" in text
    assert "every available field together was enough in fewer than 90%" in short
    # Why it is not counted as needing more fields: said once, under the table.
    assert short.count("cap what can be shown") == 1


def _calibration():
    objects, fovs = _tables(n_fovs=16, scale=3.0)
    # A tolerance no stream of these fields meets (0.0005 was met by one hold-out once
    # the reservoirs stopped depending on the group's labels, audit 4 P9-2).
    tight = OutcomeCriterion(RelativeMeanError("f"), 0.0002, "lower_is_better", "the mean")
    replay = _f_replay(
        objects,
        fovs,
        strategies=[FixedFOVStrategy(target_fovs=3), FixedFOVStrategy(target_fovs=50)],
        criteria=[tight],
        orders=[AcquisitionOrder.permuted(s) for s in range(4)],
        n_splits=5,
    )
    return stopping_calibration(replay)


def test_f8_a_stop_the_whole_stream_could_not_fix_is_not_too_early():
    result = _calibration()
    runs = result.runs[result.runs["strategy"] == "fixed_fov"]
    assert (runs["status_at_end"] == "not_preserved").all()
    stops = result.table[result.table["stopped"] == 1.0].iloc[0]
    assert stops["premature"] == 0 and stops["not_preserved_by_stream"] == 1.0
    total = stops[["correct", "premature", "not_preserved_by_stream", "undefined_at_stop"]]
    assert total.sum() == pytest.approx(stops["stopped"])
    text = " ".join(result.summary(detail="full").split())
    # Audit 4, P2-9: the total not preserved is the headline, the split its bounds.
    assert "100%, 0% to 100% too early" in text
    assert "not even every available field together preserved it" in text
    # The short form keeps the headline share and says that the split is in the full form.
    short = result.summary()
    assert "100%" in short and "not even every available field preserved it" in short


def test_f9_a_rule_that_never_stopped_says_whether_the_last_field_had_it():
    text = _calibration().summary(detail="full")
    never = text.split("fixed 50 fields:")[1].split("\n\n")[0]
    assert re.search(r"0% of \d+ .*100%, 0% had it by the last field", never)
    assert "fixed 50 fields" in _calibration().summary() and "never stopped" in never


def test_f11_too_few_objects_for_the_metric_is_not_preserved():
    objects = pd.DataFrame(
        {
            "condition": "c",
            "biological_repeat": "r",
            "position_id": ["p0", "p1", "p1", "p2"],
            "a": [1.0, 2.0, 4.0, 3.0],
            "b": [5.0, 4.0, 7.0, 1.0],
        }
    )
    fovs = pd.DataFrame(
        {"condition": "c", "biological_repeat": "r", "position_id": ["p0", "p1", "p2"]}
    )
    data = validate_input_tables(objects, fovs=fovs, feature_cols=["a", "b"], fov_col="position_id")
    one = validate_input_tables(
        objects.iloc[:1], fovs=fovs.iloc[:1], feature_cols=["a", "b"], fov_col="position_id"
    )
    metric = CovarianceError(("a", "b"), standardize=False)
    assert metric.min_observations == 2 and RelativeMeanError("a").min_observations == 1
    assert EnergyError(("a",), False, None, 0).min_observations == 2
    criterion = OutcomeCriterion(metric, 0.5, "lower_is_better")
    assert criterion.assess(data, one) == (Preservation.NOT_PRESERVED, None)
    # A reference that leaves the metric undefined still gives undefined.
    flat = validate_input_tables(
        objects.assign(a=1.0, b=2.0), fovs=fovs, feature_cols=["a", "b"], fov_col="position_id"
    )
    assert criterion.assess(flat, one)[0] == Preservation.UNDEFINED


def test_f16_reliability_says_what_it_rests_on_at_the_edges():
    objects, fovs = _tables()
    beyond = _f_reliability(objects, fovs, fov_counts=[20]).summary()
    assert "Judged on 0 or more" not in beyond
    assert "Judged on 6 or more held-back fields" in beyond
    nothing = _f_reliability(objects, fovs, criteria=[RARE]).summary()
    assert "Judged on" not in nothing
    assert "the criterion could not be judged against any of them" in nothing


def test_a_covariance_criterion_is_called_the_covariance_fingerprint():
    covariance = OutcomeCriterion(CovarianceError(("a", "b"), True), 0.1, "lower_is_better")
    quantiles = OutcomeCriterion(QuantileError(("a",), True, (0.1, 0.9)), 0.1, "lower_is_better")
    assert noun(covariance) == "covariance fingerprint"
    assert noun(quantiles) == "fingerprint"
    said = []
    ask_tolerance(lambda _: "", said.append, ["a", "b"])
    assert said[0].startswith("The fingerprint judged is a covariance fingerprint: the covariance")
    assert "not the whole distribution" in said[0]


def test_fingerprint_precision_without_an_error_to_lower_is_none():
    objects, fovs = _sictrl_tables()
    criterion = OutcomeCriterion(RelativeMeanError("area"), 0.9, "higher_is_better")
    assert (
        fingerprint_precision(objects, fovs, Setup(readout="area", fingerprint=(criterion,)))
        is None
    )


def test_adequacy_short_tables_the_rules_and_full_keeps_every_sentence():
    result = _adequacy()
    short, full = result.summary(), result.summary(detail="full")
    assert "fixed 3 fields" in short and "fixed 1000 fields" in short
    assert "never" in short  # a rule waiting for 1000 fields never says stop
    assert "Each target on its own" in short and "mean b" in short
    assert "condition=" not in short and "condition=" not in full
    assert _flat(short).count("'Reliably' means") == 1
    # The full form: every group's sentence with its medians and tail, and the config ids.
    groups = result.realized[list(result.group_cols)].drop_duplicates()
    assert full.count("when was it enough?") == len(groups)
    assert "in 90% of runs" in full and "config " in full

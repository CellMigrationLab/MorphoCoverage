"""Regressions of the 2026-09-26 function audit: planning, inspection and protocol (F1-F16).

Each test names the finding it pins (the first function audit, docs/audit/README.md)
and fails on the code before the fix.
"""

import numpy as np
import pandas as pd
import pytest
from test_retrospective_controls import BOTH, _control, _plate
from test_retrospective_effort import TRUTH, _simulate
from test_retrospective_frontier import MEAN, _mixed, _oracle, _replay
from test_retrospective_protocol import HIERARCHY, _curve

from morphocoverage.retrospective import (
    SamplingHierarchy,
    Summary,
    effort_projection,
    inspect_fovs,
    protocol_reliability,
)
from morphocoverage.retrospective._text import median_beyond as _median_beyond
from morphocoverage.retrospective.effort import _contributions
from morphocoverage.retrospective.effort_design import _se
from morphocoverage.retrospective.frontier import family, frontier
from morphocoverage.retrospective.frontier_text import _setting


def _full(result):
    """The full summary with its line breaks as spaces (the text is wrapped)."""
    return " ".join(result.summary(detail="full").split())


def test_f1_unequal_fields_weigh_the_field_term_by_their_cluster_size():
    # 10 fields of 2 cells and 2 of 100: N = 220, sum n^2 = 20040.
    sizes = np.array([2] * 10 + [100] * 2, dtype=float)
    n, cluster = sizes.sum(), float(np.sum(sizes**2) / sizes.sum())
    design = {
        "has_repeat": False,
        "repeats": 1,
        "samples": 1,
        "total_fovs": len(sizes),
        "yield": 1.0,
        "per_fov": n / len(sizes),
        "cluster_size": cluster,
        "sample_weight": 1.0,  # a single sample holds every observation
    }
    components = np.array([0.0, 1.0, 1.0])
    truth = np.sqrt(1.0 * np.sum(sizes**2) / n**2 + 1.0 / n)
    assert _se(components, design, 1, len(sizes)) == pytest.approx(truth)
    shares = _contributions(components, design)
    field_part, cell_part = np.sum(sizes**2) / n**2, 1 / n
    assert shares[1] == pytest.approx(field_part / (field_part + cell_part))


def test_f2_one_repeat_or_one_sample_is_explained_not_a_linear_algebra_error():
    rng = np.random.default_rng(1)
    objects, fovs = _simulate(rng, TRUTH, 1, 4, 5, 10)
    hierarchy = SamplingHierarchy("image_file", "well", "biological_repeat")
    result = effort_projection(
        objects, fovs, feature="x", hierarchy=hierarchy, strata=[], n_boot=20
    )
    assert "There is a single biological repeat here" in result.summary()
    objects, fovs = _simulate(rng, TRUTH, 1, 1, 5, 10)
    result = effort_projection(
        objects, fovs, feature="x", hierarchy=SamplingHierarchy("image_file", "well"), strata=[]
    )
    assert result.components.empty
    assert "a single 'well' unit holds observations" in result.skipped["reason"].iloc[0]
    # One stratum that cannot be estimated does not stop the others.
    first, first_fovs = _simulate(rng, TRUTH, 3, 3, 4, 10)
    second, second_fovs = _simulate(rng, TRUTH, 1, 1, 4, 10)
    second, second_fovs = second.assign(condition="B"), second_fovs.assign(condition="B")
    result = effort_projection(
        pd.concat([first, second]),
        pd.concat([first_fovs, second_fovs]),
        feature="x",
        hierarchy=hierarchy,
        strata=["condition"],
        n_boot=20,
    )
    assert set(result.components["condition"]) == {"ctrl"}
    assert list(result.skipped["condition"]) == ["B"]


def test_f3_no_alternative_at_the_same_total_is_not_called_an_overlap():
    rng = np.random.default_rng(2)
    objects, fovs = _simulate(rng, (0.0, 1.0, 0.3, 4.0), 1, 12, 6, 10)
    result = effort_projection(
        objects,
        fovs,
        feature="x",
        hierarchy=SamplingHierarchy("image_file", "well"),
        strata=[],
        allocations=[(6, 20)],
        n_boot=20,
    )
    assert "no allocation compared has the acquired total of fields" in _full(result)
    assert "no allocation compared has the acquired total of fields" in " ".join(
        result.summary().split()
    )
    assert "overlap" not in _full(result) and "overlap" not in result.summary()


def test_f4_checks_that_could_not_be_judged_are_counted_not_printed_as_nan():
    objects, fovs = _plate()
    rows = objects.index[(objects["biological_repeat"] == "P1") & (objects["condition"] == "KD")][
        :1
    ]
    objects.loc[rows, "feature_a"] = np.nan
    text = _control(objects, fovs, BOTH).summary()
    assert "nan" not in text.lower()
    assert "1 not judged" in text and "could not be judged" in text


def test_f5_f11_f13_inspection_says_what_it_could_not_look_for():
    from test_retrospective_controls import FEATURES, NOVELTY

    objects, fovs = _plate(shift=0.0)
    objects.loc[objects.index[0], "feature_b"] = np.nan
    result = inspect_fovs(
        objects,
        fovs,
        feature_cols=FEATURES,
        novelty=NOVELTY,
        count_ratio=None,
        group_cols=["condition", "biological_repeat"],
    )
    assert "New states were not looked for in" in result.summary()
    assert "missing values in feature_b (1 observation)" in result.summary()
    labels = [Summary("area", q).label for q in (0.01, 0.02, 0.03, 0.11, 0.22, 0.025)]
    assert labels[:5] == [f"{n} percentile of area" for n in ("1st", "2nd", "3rd", "11th", "22nd")]
    assert labels[5] == "2.5th percentile of area"


def test_f11_missing_values_never_make_a_field_weigh_less():
    objects = pd.DataFrame(
        {
            "image_file": ["a", "a", "a", "a", "b", "c"],
            "x": [1.0, np.nan, np.nan, np.nan, 3.0, 2.0],
        }
    )
    fovs = pd.DataFrame({"image_file": ["a", "b", "c"]})
    result = inspect_fovs(
        objects,
        fovs,
        feature_cols=["x"],
        summaries=[Summary("x", "mean")],
        count_ratio=None,
        influence_threshold=0.2,
        # The threshold alone: 3 fields are too few to judge whether one stands out.
        influence_spread=None,
    )
    # Every field weighs one: the mean is 2 and leaving out b gives 1.5 (a 25% change).
    detail = result.fovs.set_index("image_file")["detail"]
    assert detail["b"] == "removing it lowers the field-weighted mean of x by 25%"


def test_f6_order_medians_are_over_the_same_runs():
    assert _median_beyond(pd.Series([10.0, np.nan, np.nan])) is None
    assert _median_beyond(pd.Series([10.0, 20.0, np.nan])) == 20.0
    assert _median_beyond(pd.Series([None, None])) is None


def test_f7_f12_one_unit_and_half_field_medians_are_stated_plainly():
    one = {("P1", "A1"): {5: 0.95, 10: 0.99}, ("P1", "A2"): {5: 0.8, 10: 0.95}}
    text = protocol_reliability(_curve(one), HIERARCHY, threshold=0.9, strata=["condition"])
    short = " ".join(text.summary().split())
    text = _full(text)
    assert "With a single plate, no number can be chosen on other plates" in text
    assert "the others reached no number" not in text
    assert "median 7.5, range 5–10" in text
    assert "with a single plate, no number can be chosen on other plates" in short
    assert "7.5 (5-10)" in short


def test_f8_f14_frontier_refuses_foreign_replays_and_says_when_there_is_no_interval():
    mixed = _mixed()
    one = _replay(*mixed, [_oracle(2, 12)], seed=0)
    other = _replay(*mixed, [_oracle(2, 12)], seed=1)
    alien = family(other, "oracle", {_oracle(2, 12).config_id: 12})
    with pytest.raises(ValueError, match="other reservoirs or orders"):
        frontier(one, [alien], criterion=MEAN.name, alphas=[0.1], unit_cols=["biological_repeat"])
    own = family(one, "oracle", {_oracle(2, 12).config_id: 12})
    result = frontier(one, [own], criterion=MEAN.name, alphas=[0.1], unit_cols=[])
    assert "No interval: there is a single unit to resample" in result.summary()
    assert "resample the 1 units" not in result.summary()


def test_f15_family_settings_are_printed_as_given():
    assert _setting(0.05) == "0.05" and _setting(12) == "12" and _setting("strict") == "strict"


def test_f16_a_technical_repeat_is_named_as_such_in_the_protocol():
    from morphocoverage.retrospective import ReliabilityResult

    rows = [
        {
            "condition": "ctrl",
            "biological_repeat": repeat,
            "technical_repeat": well,
            "criterion": "mean",
            "fovs": 5,
            "preserved": 0.95,
            "stream_fovs": 12,
        }
        for repeat, well in (("R1", "A1"), ("R1", "A2"), ("R2", "A1"))
    ]
    curve = ReliabilityResult(
        pd.DataFrame(rows),
        pd.DataFrame(),
        ("condition", "biological_repeat", "technical_repeat"),
        {"labels": {"mean": "the mean"}},
    )
    hierarchy = SamplingHierarchy("image_file", "technical_repeat", "biological_repeat")
    result = protocol_reliability(curve, hierarchy, threshold=0.9, strata=["condition"])
    for text in (_full(result), " ".join(result.summary().split())):
        assert (
            "ields per technical repeat" in text
            and "3 technical repeats in 2 biological repeats" in text
        )
        assert "technical_repeat" not in text

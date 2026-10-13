"""Unusual observations split in two (owner, 2026-10-06, plan v2): an acquisition anomaly on
the raw values, as quality control in "What is in the dataset", and biological unusualness on
values relative to each own control in its repeat, exploratory."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from morphocoverage.sampling_unit import unusual_objects
from morphocoverage.sampling_unit.acquisition import AcquisitionAnomalies, acquisition_anomalies
from morphocoverage.workflows import Setup, report
from morphocoverage.workflows._shared import control_checks
from morphocoverage.workflows._unusual_values import centred_on_own_control, ratio_scale

FEATURES = ("area", "intensity_mean", "ecc")
# The share of the control's observations called unusual (``unusual_objects``' quantile):
# the null rate of biological unusualness.
NULL_RATE = 0.01
CTRL = {"condition": "ctrl"}


def _tables(
    conditions: tuple[str, ...],
    *,
    repeats: int = 4,
    fields: int = 6,
    cells: int = 15,
    bright: dict | None = None,
    planted: str | None = None,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every condition drawn alike (no biological difference): ``bright`` multiplies the
    intensity of the observations whose (column, value) it names, an acquisition gap; with
    ``planted``, a tenth of that condition's cells form a distinct state (``planted`` True)."""
    rng = np.random.default_rng(seed)
    rows, fov_rows = [], []
    for condition in conditions:
        for r in range(1, repeats + 1):
            repeat = f"R{r}"
            batch = rng.normal(0, 0.05)  # a small repeat effect, shared by its conditions
            for f in range(fields):
                key = {"condition": condition, "biological_repeat": repeat, "position_id": f}
                fov_rows.append(key)
                field = rng.normal(0, 0.05)
                for _ in range(cells):
                    rows.append(
                        {
                            **key,
                            "area": 100 * np.exp(batch + field + rng.normal(0, 0.2)),
                            "intensity_mean": 1000 * np.exp(batch + field + rng.normal(0, 0.2)),
                            "ecc": 0.5 + rng.normal(0, 0.05),
                            "planted": False,
                        }
                    )
    objects = pd.DataFrame(rows)
    for (column, value), fold in (bright or {}).items():
        objects.loc[objects[column].eq(value), "intensity_mean"] *= fold
    if planted is not None:
        mine = np.flatnonzero(objects["condition"].eq(planted).to_numpy())
        chosen = rng.choice(mine, size=len(mine) // 10, replace=False)
        objects.loc[chosen, "planted"] = True
        objects.loc[chosen, "area"] *= 2.0
        objects.loc[chosen, "ecc"] += 0.25
    return objects, pd.DataFrame(fov_rows)


def _flat(text: str) -> str:
    return " ".join(text.split())


def _share(unusual, objects: pd.DataFrame, mask: np.ndarray) -> float:
    labels = unusual.labels[mask].dropna().astype(bool)
    return float(labels.mean())


@pytest.fixture(scope="module")
def two_experiments():
    """Two experiments with their own controls, the second imaged 10 times brighter."""
    objects, fovs = _tables(("cA", "a1", "cB", "b1"), bright={("condition", "cB"): 10.0})
    objects.loc[objects["condition"].eq("b1"), "intensity_mean"] *= 10.0
    setup = Setup(
        readout="area",
        features=FEATURES,
        control={"condition": {"cA": ["a1"], "cB": ["b1"]}},
        observation_unit="cell",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return objects, report(objects, fovs, setup)


def test_a_ten_fold_gap_between_experiments_is_acquisition_not_biology(two_experiments):
    objects, result = two_experiments
    unusual = result.results["unusual"]
    treated = objects["condition"].isin(["a1", "b1"]).to_numpy()
    # No biological difference: no more than the null rate, with Monte Carlo slack.
    assert _share(unusual, objects, treated) <= 3 * NULL_RATE
    assert "each relative to its own control's mean in its repeat" in unusual.settings["space"]
    short = result.summary()
    overview = _flat(short[: short.index("What this acquisition was asked to preserve")])
    # The acquisition block names the gap, apart from the checks of the values.
    block = overview[overview.index("Acquisition anomalies (quality control)") :]
    assert "➖ Acquisition anomalies" in overview  # 6 fields a group: none judged
    assert "Controls apart in intensity: cA and cB differ 9.9-fold in median intensity_mean" in (
        block
    )
    glance = _flat(short[: short.index("What is in the dataset")])
    assert "acquisition: controls apart in intensity" in glance
    profile = _flat(short[short.index("3. What is limiting my sampling?") :])
    assert "exploratory" in profile.split("6. Which fields")[0]


def test_a_bright_repeat_neither_reads_as_biology_nor_hides_a_population():
    """One control, one repeat imaged 10 times brighter and shifted: on raw values the
    shifted repeat's control widens the threshold, and most of a planted population in every
    repeat is missed; relative to the control of their own repeat, the population is found
    and the bright repeat's controls are not unusual."""
    objects, fovs = _tables(("ctrl", "a"), planted="a", seed=3)
    bright = objects["biological_repeat"].eq("R4")
    objects.loc[bright, "intensity_mean"] *= 10.0
    objects.loc[bright, "ecc"] += 0.3
    setup = Setup(
        readout="area", features=FEATURES, control={"condition": "ctrl"}, observation_unit="cell"
    )
    planted = objects["planted"].to_numpy()
    raw = unusual_objects(objects, fovs, feature_cols=FEATURES, reference={"condition": "ctrl"})
    assert _share(raw, objects, planted) < 0.5
    unusual = control_checks(objects, fovs, setup, {})["unusual"]
    assert _share(unusual, objects, planted) >= 0.9
    controls = objects["condition"].eq("ctrl").to_numpy()
    assert _share(unusual, objects, controls & bright.to_numpy()) <= 3 * NULL_RATE


def test_a_planted_population_is_found_by_the_biological_block():
    objects, fovs = _tables(("ctrl", "a"), planted="a", seed=4)
    setup = Setup(
        readout="area", features=FEATURES, control={"condition": "ctrl"}, observation_unit="cell"
    )
    unusual = control_checks(objects, fovs, setup, {})["unusual"]
    planted = objects["planted"].to_numpy()
    assert _share(unusual, objects, planted) >= 0.9
    others = objects["condition"].eq("a").to_numpy() & ~planted
    assert _share(unusual, objects, others) <= 3 * NULL_RATE


def test_values_are_relative_to_the_control_of_the_same_repeat_on_the_test_scale():
    objects, fovs = _tables(("ctrl", "a"), repeats=2, fields=2, cells=3)
    objects["velocity"] = np.random.default_rng(5).normal(0, 1, len(objects))
    second = objects["biological_repeat"].eq("R2")
    objects.loc[second, "area"] *= 10.0
    objects.loc[second, "velocity"] += 4.0
    columns = [*FEATURES, "velocity"]
    setup = Setup(readout="area", features=tuple(columns), control={"condition": "ctrl"})
    relative = centred_on_own_control(objects, fovs, setup, columns)
    control = objects["condition"].eq("ctrl")
    for repeat in ("R1", "R2"):
        mine = control & objects["biological_repeat"].eq(repeat)
        # A positive feature: the log ratio to the control's mean; a signed one: the difference.
        assert np.exp(relative.loc[mine, "area"]).mean() == pytest.approx(1.0)
        assert relative.loc[mine, "velocity"].mean() == pytest.approx(0.0, abs=1e-12)
    assert relative.index.equals(objects.index)


def test_a_declared_scale_is_the_one_biological_unusualness_reads():
    """Plan v2: ``ratio_scale`` reads the feature's scale declared in the Setup, else the
    automatic rule; a declared ratio on a value that is not positive is refused."""
    objects, fovs = _tables(("ctrl", "a"), repeats=2, fields=2, cells=3)
    columns = list(FEATURES)
    declared = Setup(
        readout="area",
        features=FEATURES,
        control={"condition": "ctrl"},
        scales={"area": "difference"},
    )
    automatic = Setup(readout="area", features=FEATURES, control={"condition": "ctrl"})
    values = objects["area"].to_numpy()
    assert ratio_scale(automatic, "area", values)  # every value positive: a log ratio
    assert not ratio_scale(declared, "area", values)
    assert ratio_scale(declared, "ecc", objects["ecc"].to_numpy())  # undeclared: automatic
    relative = centred_on_own_control(objects, fovs, declared, columns)
    mine = objects["condition"].eq("ctrl") & objects["biological_repeat"].eq("R1")
    assert relative.loc[mine, "area"].mean() == pytest.approx(0.0, abs=1e-9)  # a difference
    shifted = objects.assign(ecc=objects["ecc"] - 1.0)  # signed values
    ratio = Setup(
        readout="area", features=FEATURES, control={"condition": "ctrl"}, scales={"ecc": "ratio"}
    )
    assert not ratio_scale(automatic, "ecc", shifted["ecc"].to_numpy())
    with pytest.raises(ValueError, match="ratio scale"):
        ratio_scale(ratio, "ecc", shifted["ecc"].to_numpy())


def test_a_field_far_from_its_condition_and_repeat_is_an_acquisition_anomaly():
    objects, fovs = _tables(("ctrl", "a"), repeats=2, fields=10, cells=5)
    far = objects["condition"].eq("a") & objects["biological_repeat"].eq("R1")
    objects.loc[far & objects["position_id"].eq(3), "intensity_mean"] *= 10.0
    found = acquisition_anomalies(objects, fovs, feature_cols=FEATURES)
    assert found.found == 1
    row = found.fields.iloc[0]
    assert (row["condition"], row["biological_repeat"], row["position_id"]) == ("a", "R1", 3)
    assert row["feature"] == "intensity_mean" and row["z"] > 5
    line = found.line()
    assert line.startswith("1 field 5 SDs or more from the other fields of its condition")
    assert "an engineering quality-control threshold, never a biological verdict" in line
    clean = acquisition_anomalies(*_tables(("ctrl",), fields=10), feature_cols=FEATURES)
    assert clean.found == 0 and clean.line().startswith("none: no field 5 SDs or more")


def _scenario(name: str):
    """A scenario of ``data/scenarios`` (``make_scenarios.load``), its Setup built and its
    tables attached as the report attaches them."""
    import sys
    from pathlib import Path

    from morphocoverage.workflows._feature_space import attached

    folder = Path(__file__).resolve().parents[1] / "data" / "scenarios"
    sys.path.insert(0, str(folder))
    try:
        import make_scenarios
    finally:
        sys.path.remove(str(folder))
    loaded = make_scenarios.load(name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        objects, setup, _ = attached(
            loaded.objects, loaded.fovs, Setup(**loaded.setup), loaded.feature_tables
        )
    return objects, loaded.fovs, setup


def test_acquisition_qc_reads_the_test_scale_and_one_scale_pooled_over_groups():
    """Audit 10 (A1, B13, C17; owner, 2026-10-06): positive features on the log scale, one
    robust scale per feature pooled over the groups' within-group differences, each flag's
    relative difference and the count expected by chance said."""
    from morphocoverage.workflows._acquisition_block import acquisition_in_report

    # a18: no field planted; the SD of 7 other fields read one at 14.9 and one at 6.5 SDs.
    null = acquisition_in_report(*_scenario("a18_tagged_none_image_tech_fp_expl"))
    assert null.found == 0 and null.judged == 48
    assert set(null.settings["scales"].values()) == {"log"}
    # One image per field: the check reads single images, and says so.
    assert "each field holds one image, so this reads single images" in null.line()
    # e16: one field's areas are 4 times its group's; it alone is flagged, as a fold.
    planted = acquisition_in_report(*_scenario("e16_acquisition_block"))
    assert planted.found == 1
    row = planted.fields.iloc[0]
    assert row["feature"] == "area" and row["field"].startswith("a1 · repeat 2026_01_05")
    assert 3.5 < row["fold"] < 4.5 and row["log_scale"]
    line = _flat(planted.line())
    assert "(area, 9.3 SDs above, ×3.9)" in line
    assert "about 0.00017 expected by chance in the 288 values of a field and feature" in line
    # A null log-normal, field medians of skewed cells: none on the log scale.
    rng = np.random.default_rng(3)
    rows = [
        {"condition": "c", "biological_repeat": r, "position_id": f, "x": 10 * np.exp(v)}
        for r in range(6)
        for f in range(10)
        for v in rng.normal(rng.normal(0, 0.15), 0.8, 20)
    ]
    objects = pd.DataFrame(rows)
    fovs = objects[["condition", "biological_repeat", "position_id"]].drop_duplicates()
    assert acquisition_anomalies(objects, fovs, feature_cols=["x"]).found == 0
    # A declared difference scale is read on the values, its flag a difference.
    objects.loc[objects["position_id"].eq(4) & objects["biological_repeat"].eq(2), "x"] += 400
    declared = acquisition_anomalies(objects, fovs, feature_cols=["x"], scales={"x": "difference"})
    assert declared.settings["scales"] == {"x": "values"}
    assert declared.found == 1 and declared.fields.iloc[0]["relative"] is None


def test_one_image_per_field_lists_a_qc_field_once_not_again_as_a_new_state():
    """Audit 10 (A1c): with one observation per field the acquisition anomalies read single
    observations, so a field they name is not listed again as a new state in question 6."""
    from morphocoverage.workflows import default_novelty

    # 30 fields a group: novelty on 3 features is informative there (null rate 5%).
    objects, fovs = _tables(("ctrl", "a"), repeats=3, fields=30, cells=1, seed=4)
    far = objects["condition"].eq("a") & objects["biological_repeat"].eq("R2")
    far &= objects["position_id"].eq(5)
    objects.loc[far, "area"] *= 1000.0
    setup = Setup(
        readout="area",
        features=FEATURES,
        control={"condition": "ctrl"},
        novelty=default_novelty(FEATURES),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = report(objects, fovs, setup)
    named = result.results["acquisition"].fields
    assert list(named["position_id"]) == [5]
    inspection = result.results["inspection"].fovs
    again = inspection["reason"].eq("new state") & inspection["position_id"].eq(5)
    again &= inspection["condition"].eq("a") & inspection["biological_repeat"].eq("R2")
    assert not again.any()
    full = _flat(result.summary(detail="full"))
    assert "1 field named among the acquisition anomalies is not listed again" in full


def test_unusual_shares_read_against_the_controls_and_say_when_they_restate_a_shift():
    """Audit 10 (A5, B8, C32, A15/B9; owner, 2026-10-06): the controls' own share flagged is
    always printed as the background; the shares are of each scored group; one feature is
    "not informative beyond the mean shift in question 9", with no table of shares; a group
    with no unusual observation reads "– (none unusual; refits flag k)", never 0% agreement."""
    objects, fovs = _tables(("ctrl", "a"), repeats=3, fields=6, cells=15, planted="a")
    found = unusual_objects(objects, fovs, feature_cols=list(FEATURES), reference=CTRL)
    short = _flat(found.summary())
    assert "Background: the controls' own cells, scored the same way against the other " in short
    controls = found.groups[found.groups["reference_group"]]
    flagged, scored = int(controls["unusual"].sum()), int(controls["scored"].sum())
    assert f"unusual, {flagged} of {scored}; a condition's share reads against it" in short
    assert "Background" in _flat(found.summary(detail="full"))
    # A group with no unusual observation: no agreement, the refits' count instead.
    none = found.groups[(found.groups["status"] == "ok") & (found.groups["unusual"] == 0)]
    assert len(none) and none["seed_agreement"].isna().all()
    assert "– (none unusual" in short and "agree on only 0%" not in short
    # One feature: the shares restate the shift; no table, no stability caution.
    one = unusual_objects(objects, fovs, feature_cols=["area"], reference=CTRL)
    text = _flat(one.summary())
    assert "not informative beyond the mean shift in question 9, so no share is given" in text
    assert "refits agree" not in text and "unstable" not in text and "Background" not in text


def test_the_glance_reads_each_scored_group_and_one_feature_says_nothing():
    from morphocoverage.retrospective.report_sampling import states_lines, unusual_alone

    objects, fovs = _tables(("ctrl", "a"), repeats=3, fields=6, cells=1, planted=None)
    found = unusual_objects(objects, fovs, feature_cols=list(FEATURES), reference=CTRL)
    glance = unusual_alone(found)
    assert "of each group that is not a control, against " in glance
    assert "of the controls' own (exploratory)" in glance
    one = unusual_objects(objects, fovs, feature_cols=["area"], reference=CTRL)
    assert unusual_alone(one) == ""
    assert "not informative beyond the mean shift" in states_lines(None, one, "cells")[0]


def test_the_fields_not_judged_for_quality_control_are_counted_beside_those_judged_a18():
    groups = pd.DataFrame(
        {"condition": ["a", "b", "c"], "fields": [9, 5, 6], "judged": [9, 0, 0], "anomalies": 0}
    )
    anomalies = AcquisitionAnomalies(pd.DataFrame(), groups, ("condition",), {})
    line = anomalies.line()
    assert line.endswith(
        "9 fields judged in 1 group; 11 fields in 2 groups of fewer than 8 fields counted, "
        "not judged"
    )

"""Unusual observations split in two (owner, 2026-10-06, plan v2): an acquisition anomaly on
the raw values, as quality control in "What is in the dataset", and biological unusualness on
values relative to each own control in its repeat, exploratory."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from morphocoverage.sampling_unit import unusual_objects
from morphocoverage.sampling_unit.acquisition import acquisition_anomalies
from morphocoverage.workflows import Setup, report
from morphocoverage.workflows._shared import control_checks
from morphocoverage.workflows._unusual_values import centred_on_own_control, ratio_scale

FEATURES = ("area", "intensity_mean", "ecc")
# The share of the control's observations called unusual (``unusual_objects``' quantile):
# the null rate of biological unusualness.
NULL_RATE = 0.01


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
    block = overview[overview.index("Acquisition anomalies (raw values, quality control)") :]
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
    assert line.startswith("1 field 5 SDs or more from the other fields of its condition")  # A22
    assert "an engineering quality-control threshold, never a biological verdict" in line
    clean = acquisition_anomalies(*_tables(("ctrl",), fields=10), feature_cols=FEATURES)
    assert clean.found == 0 and clean.line().startswith("none: no field 5 SDs or more")

"""Data-integrity checks (issue #49): each finds the copy it is for, and nothing on clean data."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from _helpers_integrity import FEATURES, _check, _field, _replace_field

from morphocoverage import integrity_checks

DATA = Path(__file__).resolve().parents[1] / "data" / "crick_idr"
sys.path.insert(0, str(DATA))
from datasets import load


@pytest.fixture(scope="module")
def idr0028():
    objects, fovs = load("idr0028_screenB").tables
    return objects, fovs


def test_clean_datasets_have_no_findings(idr0028):
    result = _check(*idr0028)
    assert result.findings.empty
    assert (result.checks["status"] == "checked").all()
    assert "✅ Data integrity — nothing found by the exact copy checks" in result.summary()


def test_a_field_exported_twice_under_another_name_is_a_copied_field(idr0028):
    objects, fovs = idr0028
    source = objects[_field(objects, fovs, 0)]
    copied = _replace_field(objects, fovs, 40, source.copy())
    result = _check(copied, fovs)
    assert result.findings["check"].tolist() == ["copied field"]
    assert result.findings["observations"].iloc[0] == len(source)
    assert "structure to inspect" in result.summary()


def test_the_same_observations_in_another_order_are_a_reordered_copy(idr0028):
    objects, fovs = idr0028
    source = objects[_field(objects, fovs, 0)]
    shuffled = source.sample(frac=1, random_state=0)
    result = _check(_replace_field(objects, fovs, 40, shuffled), fovs)
    assert result.findings["check"].tolist() == ["reordered copy"]


def test_a_change_of_units_is_a_scaled_copy(idr0028):
    objects, fovs = idr0028
    source = objects[_field(objects, fovs, 0)].copy()
    source["Cytoplasm_Intensity_MeanIntensity_YAPTAZ"] *= 0.65**2  # another intensity scale
    source["Nuclei_Intensity_MeanIntensity_YAPTAZ"] += 100.0  # an offset
    result = _check(_replace_field(objects, fovs, 40, source), fovs)
    assert result.findings["check"].tolist() == ["scaled copy"]


def test_a_block_pasted_into_another_field_is_a_repeated_block(idr0028):
    objects, fovs = idr0028
    source = objects[_field(objects, fovs, 0)]
    target = objects[_field(objects, fovs, 40)].copy()
    pasted = pd.concat([target.iloc[:5], source.iloc[3:15], target.iloc[5:]])
    result = _check(_replace_field(objects, fovs, 40, pasted), fovs)
    [finding] = result.findings.itertuples()
    # The whole run of 12 pasted observations, not the 10 of block_length (P1-11).
    assert finding.check == "repeated block" and finding.observations == 12
    assert "12 consecutive observations" in finding.detail
    assert "from observation 4 of the first field and 6 of the second" in finding.detail


def test_same_means_from_different_observations_are_a_repeated_summary(idr0028):
    objects, fovs = idr0028
    source = objects[_field(objects, fovs, 0)].copy()
    # Shift every nucleus's value by a zero-sum amount: every row differs, every mean is the
    # same. (Swapping two values kept the other rows, which are now shared observations.)
    column = "Cytoplasm_Intensity_MeanIntensity_YAPTAZ"
    shift = np.linspace(-1, 1, len(source)) * source[column].std() / 10
    source[column] = source[column].to_numpy() + (shift - shift.mean())
    result = _check(_replace_field(objects, fovs, 40, source), fovs)
    assert result.findings["check"].tolist() == ["repeated field summary"]


def test_a_small_field_inside_another_is_a_copied_small_field(idr0028):
    """Fields of 2-4 observations (tracks per position) are below min_objects; a copy is still
    found when every observation of the smaller field is in the other."""
    objects, fovs = idr0028
    rows = objects[_field(objects, fovs, 0)].iloc[:4].copy()
    rows[FEATURES] *= 1.5  # values no other field holds, so only the copies share them
    extra = objects[_field(objects, fovs, 41)].iloc[:1]
    copied = _replace_field(objects, fovs, 40, rows)
    copied = _replace_field(copied, fovs, 41, pd.concat([rows, extra]))
    result = _check(copied, fovs)
    [finding] = result.findings.itertuples()
    assert finding.check == "copied small field" and finding.observations == 4
    assert "all 4 observations of the first field are also in the second" in finding.detail


def _one_cell_fields(n_features, copies):
    """Twelve fields of one cell each, ``n_features`` continuous features; the first cell's
    row is written again in ``copies`` other fields."""
    rng = np.random.default_rng(1)
    names = [f"f{i}.tif" for i in range(12)]
    fovs = pd.DataFrame({"condition": "a", "biological_repeat": "r", "image_file": names})
    values = rng.normal(100, 10, (12, n_features)).round(3)
    values[1 : 1 + copies] = values[0]
    objects = pd.DataFrame(values, columns=[f"x{j}" for j in range(n_features)]).assign(
        condition="a", biological_repeat="r", image_file=names
    )
    features = [f"x{j}" for j in range(n_features)]
    return integrity_checks(objects, fovs, feature_cols=features, observation_unit="cell")


def test_a_cell_row_pasted_into_another_one_cell_field_is_found():
    result = _one_cell_fields(n_features=3, copies=1)
    [finding] = result.findings.itertuples()
    assert finding.check == "copied small field" and finding.observations == 1
    assert "the two fields hold the same single observation" in finding.detail
    # Pasted into two other fields, the row recurs in three: set aside, as documented.
    assert _one_cell_fields(n_features=3, copies=2).findings.empty


def test_one_shared_observation_needs_three_continuous_features():
    result = _one_cell_fields(n_features=2, copies=1)
    assert result.findings.empty
    status = result.checks.set_index("check")
    assert status.loc["copied small field", "status"] == "not applicable"
    assert (
        "one shared observation can be a coincidence" in status.loc["copied small field", "reason"]
    )


def test_quantized_features_make_the_copy_checks_not_applicable():
    rng = np.random.default_rng(0)
    fovs = pd.DataFrame(
        {"condition": "a", "biological_repeat": "r", "image_file": [f"f{i}.tif" for i in range(6)]}
    )
    objects = pd.DataFrame(
        {
            "condition": "a",
            "biological_repeat": "r",
            "image_file": np.repeat(fovs["image_file"], 20).to_numpy(),
            "count": rng.integers(0, 3, 120).astype(float),
        }
    )
    result = integrity_checks(objects, fovs, feature_cols=["count"])
    status = dict(zip(result.checks["check"], result.checks["status"], strict=True))
    assert status["copied field"] == "not applicable"
    assert status["repeated block"] == "not applicable"
    assert status["repeated field summary"] == "not applicable"
    assert result.findings.empty
    assert "cannot be told from a coincidence" in " ".join(result.summary().split())


def test_the_synthetic_acquisition_has_no_findings():
    objects = pd.read_csv(DATA.parent / "synthetic" / "synthetic_cells.csv")
    fovs = pd.read_csv(DATA.parent / "synthetic" / "synthetic_fovs.csv")
    result = integrity_checks(
        objects, fovs, feature_cols=["area", "intensity"], observation_unit="cell"
    )
    assert result.findings.empty, result.summary()


def test_arguments_are_checked():
    fovs = pd.DataFrame({"condition": "a", "biological_repeat": "r", "image_file": ["f.tif"]})
    objects = pd.DataFrame(
        {"condition": "a", "biological_repeat": "r", "image_file": "f.tif", "x": [1.0]}
    )
    with pytest.raises(ValueError, match="block_length"):
        integrity_checks(objects, fovs, feature_cols=["x"], block_length=1)
    with pytest.raises(ValueError, match="min_distinct_share"):
        integrity_checks(objects, fovs, feature_cols=["x"], min_distinct_share=2)


def test_the_report_carries_the_integrity_checks_only_when_asked(idr0028):
    from morphocoverage.workflows import Setup, report

    objects, fovs = idr0028
    wells = fovs["technical_repeat"].isin(fovs["technical_repeat"].unique()[:2])
    objects = objects[objects["technical_repeat"].isin(fovs.loc[wells, "technical_repeat"])]
    setup = Setup(readout=FEATURES[1], features=tuple(FEATURES), observation_unit="nucleus")
    plain = report(objects, fovs[wells], setup)
    checked = report(objects, fovs[wells], setup, integrity=True)
    assert "integrity" not in plain.results
    assert "Data integrity" not in plain.summary(detail="full")
    assert "➖ Copy checks not run (report(..., integrity=True) runs them)" in plain.summary()
    full = checked.summary(detail="full")
    assert "Data integrity" in full and checked.results["integrity"].findings.empty
    assert "✅ Integrity: nothing found." in checked.summary()

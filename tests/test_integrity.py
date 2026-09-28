"""Data-integrity checks (issue #49): each finds the copy it is for, and nothing on clean data."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from morphocoverage import integrity_checks

DATA = Path(__file__).resolve().parents[1] / "data" / "crick_idr"
sys.path.insert(0, str(DATA))
from datasets import load

FEATURES = [
    "Nuclei_Intensity_IntegratedIntensity_YAPTAZ",
    "Nuclei_Intensity_MeanIntensity_YAPTAZ",
    "Cytoplasm_Intensity_MeanIntensity_YAPTAZ",
]


@pytest.fixture(scope="module")
def idr0028():
    objects, fovs = load("idr0028_screenB").tables
    return objects, fovs


def _field(objects, fovs, i):
    row = fovs.iloc[i]
    mask = pd.Series(True, index=objects.index)
    for column in ("condition", "repeat", "technical_repeat", "image_file"):
        mask &= objects[column].eq(row[column])
    return mask


def _replace_field(objects, fovs, target, rows):
    """``objects`` with field ``target``'s observations replaced by ``rows``."""
    keep = objects[~_field(objects, fovs, target)]
    identity = fovs.iloc[target][["condition", "repeat", "technical_repeat", "image_file"]]
    return pd.concat([keep, rows.assign(**identity.to_dict())], ignore_index=True)


def _check(objects, fovs):
    return integrity_checks(objects, fovs, feature_cols=FEATURES, observation_unit="nucleus")


def test_clean_datasets_have_no_findings(idr0028):
    result = _check(*idr0028)
    assert result.findings.empty
    assert (result.checks["status"] == "checked").all()
    assert "Nothing was found" in result.summary()


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
    assert finding.check == "repeated block" and finding.observations == 10
    assert "from observation 4 of the first field and 6 of the second" in finding.detail


def test_same_means_from_different_observations_are_a_repeated_summary(idr0028):
    objects, fovs = idr0028
    source = objects[_field(objects, fovs, 0)].copy()
    # Swap two nuclei's areas: the rows differ, every mean is the same.
    values = source["Cytoplasm_Intensity_MeanIntensity_YAPTAZ"].to_numpy().copy()
    values[[0, 1]] = values[[1, 0]]
    source["Cytoplasm_Intensity_MeanIntensity_YAPTAZ"] = values
    source = source.iloc[::-1]  # and keep no run of 10 rows in the same order
    result = _check(_replace_field(objects, fovs, 40, source), fovs)
    assert result.findings["check"].tolist() == ["repeated field summary"]


def test_quantized_features_make_the_copy_checks_not_applicable():
    rng = np.random.default_rng(0)
    fovs = pd.DataFrame(
        {"condition": "a", "repeat": "r", "image_file": [f"f{i}.tif" for i in range(6)]}
    )
    objects = pd.DataFrame(
        {
            "condition": "a",
            "repeat": "r",
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
    assert "cannot be told from a coincidence" in result.summary()


def test_the_synthetic_acquisition_has_no_findings():
    objects = pd.read_csv(DATA.parent / "synthetic" / "synthetic_cells.csv")
    fovs = pd.read_csv(DATA.parent / "synthetic" / "synthetic_fovs.csv")
    result = integrity_checks(
        objects, fovs, feature_cols=["area", "intensity"], observation_unit="cell"
    )
    assert result.findings.empty, result.summary()


def test_arguments_are_checked():
    fovs = pd.DataFrame({"condition": "a", "repeat": "r", "image_file": ["f.tif"]})
    objects = pd.DataFrame({"condition": "a", "repeat": "r", "image_file": "f.tif", "x": [1.0]})
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
    assert "integrity" not in plain.results and "Data integrity" not in plain.summary()
    assert "Data integrity" in checked.summary() and checked.results["integrity"].findings.empty

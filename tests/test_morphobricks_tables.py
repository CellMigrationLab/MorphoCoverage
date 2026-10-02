"""A table as fiberfeatures writes it (data/morphobricks/README.md): read as it is, its patch
geometry never offered as features, a typing error of a shared column refused with the name
meant, and a missing value reported with its column."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from morphocoverage import check_tables, validate_input_tables
from morphocoverage.metadata import canonical_spelling
from morphocoverage.strategies import FOVBootstrapEstimator
from morphocoverage.transforms import Standardize
from morphocoverage.workflows import scan_design

TABLE = Path(__file__).resolve().parents[1] / "data" / "morphobricks" / "fiberfeatures_patches.csv"
FEATURES = [
    "nematic_order",
    "linearity_mean",
    "entropy_mean",
    "bundle_evidence_mean",
    "pixel_category_network_fraction",
]
FIELD = ["condition", "biological_repeat", "technical_repeat", "image_file"]


@pytest.fixture
def patches():
    table = pd.read_csv(TABLE)
    return table, table[FIELD].drop_duplicates()


def test_the_table_is_read_as_fiberfeatures_writes_it(patches):
    check = check_tables(*patches, feature_cols=FEATURES, observation_unit="patch")
    assert check.ok and not check.warnings
    assert "A field is identified by 'image_file'" in check.summary()


def test_the_scan_offers_the_features_and_not_where_the_patches_were_cut(patches):
    scan = scan_design(*patches, control={"condition": "condition_2"}, observation_unit="patch")
    offered = scan.settings["feature_cols"]
    assert sorted(offered) == sorted(FEATURES)
    geometry = [c for c in patches[0] if c.startswith("patch_")]
    assert len(geometry) > 10 and not set(geometry) & set(offered)


@pytest.mark.parametrize(
    ("typed", "meant"),
    [
        ("biological_repat", "biological_repeat"),
        ("biologcial_repeat", "biological_repeat"),
        ("techincal_repeat", "technical_repeat"),
        ("conditon", "condition"),
        ("image_flie", "image_file"),
    ],
)
def test_a_typing_error_of_a_shared_column_is_refused_with_the_name_meant(patches, typed, meant):
    objects, fovs = (table.rename(columns={meant: typed}) for table in patches)
    check = check_tables(objects, fovs, feature_cols=FEATURES, observation_unit="patch")
    assert not check.ok
    assert f"Column {typed!r} looks like the MorphoBricks column {meant!r}" in check.summary()
    with pytest.raises(ValueError, match=f"Did you mean {meant!r}"):
        validate_input_tables(objects, fovs=fovs, feature_cols=FEATURES)


def test_other_columns_are_not_taken_for_typing_errors(patches):
    """Every column fiberfeatures and CellPatcher write, and longer names of a repeat."""
    names = [*patches[0].columns, "technical_repeat_id", "mask_file", "patch_file", "image_id"]
    assert [n for n in names if canonical_spelling(n)] == []


def test_a_missing_value_is_reported_with_its_column(patches):
    objects, fovs = patches
    objects = objects.copy()
    objects.loc[objects.index[:2], "entropy_mean"] = np.nan
    data = validate_input_tables(objects, fovs=fovs, feature_cols=FEATURES)
    fitted = Standardize().fit(data)
    assert fitted.reason.startswith("missing values in entropy_mean (2 observations)")
    evidence = FOVBootstrapEstimator("centroid", None, 50, 0, 2).estimate(data, objective=None)
    assert (
        evidence.diagnostics["undefined_reason"]
        == "missing values in entropy_mean (2 observations)"
    )

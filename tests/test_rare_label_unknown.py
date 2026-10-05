"""A missing value in the rare label is unknown everywhere (#344; owner, 2026-10-05): never a
negative, left out of every share's numerator and denominator, and counted once among the
report's notes."""

import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from test_rare_fingerprint import FEATURES, _screen, _setup

from morphocoverage.workflows import report
from morphocoverage.workflows._html_plan import _incidence
from morphocoverage.workflows._label_control import label_control
from morphocoverage.workflows._populations import populations_in_report
from morphocoverage.workflows._rare_share import share_by_field, unlabelled_note, with_share
from morphocoverage.workflows._shared import rare_clustering


def _partly_labelled():
    """The screen with the label missing for every cell of 4 control fields and for 10% of
    the other cells."""
    objects, fovs = _screen()
    rng = np.random.default_rng(5)
    images = objects.loc[objects["condition"] == "siCTRL", "image_file"].unique()[:4]
    missing = objects["image_file"].isin(images) | (rng.random(len(objects)) < 0.1)
    label = objects["mitotic"].astype(object).where(~missing, np.nan)
    return objects.assign(mitotic=label), fovs, missing.to_numpy(), images


@pytest.fixture(scope="module")
def tables():
    return _partly_labelled()


def test_the_label_check_leaves_the_unlabelled_cells_out(tables):
    objects, fovs, missing, _ = tables
    check = label_control(objects, fovs, _setup())
    known = objects.loc[~missing, "mitotic"].astype(bool)
    assert check.observations == len(known) and check.labelled == int(known.sum())
    assert check.chance == pytest.approx(known.mean())
    assert check.recognized >= 0.9


def test_the_share_and_its_field_shares_leave_the_unlabelled_cells_out(tables):
    objects, fovs, missing, images = tables
    setup = _setup()
    shared = with_share(objects, setup)
    assert shared["mitotic share"].isna().sum() == missing.sum()
    fields = share_by_field(shared, fovs, setup).set_index("image_file")["mitotic share"]
    partly = objects.loc[~objects["image_file"].isin(images), "image_file"].iloc[0]
    inside = objects[(objects["image_file"] == partly) & ~missing]
    assert fields[partly] == pytest.approx(inside["mitotic"].astype(float).mean())
    # A field whose cells all lack a label has no share, not a share of 0.
    assert fields[images].isna().all()


def test_the_plan_leaves_fields_without_any_label_out_of_the_incidence(tables):
    objects, fovs, missing, images = tables
    result = SimpleNamespace(setup=_setup())
    incidence, where = _incidence(result, objects, fovs, objects["mitotic"])
    known = objects[~missing]
    held = known[known["mitotic"].astype(bool)].groupby("condition")["image_file"].nunique()
    looked = fovs[~fovs["image_file"].isin(images)].groupby("condition")["image_file"].size()
    expected = (held / looked).dropna()
    assert where == expected.idxmin() and incidence == pytest.approx(expected.min())
    # Read as negatives, the 4 control fields would lower the control's incidence.
    filled = objects["mitotic"].where(~missing, False)
    assert where == "siCTRL" and _incidence(result, objects, fovs, filled)[0] < incidence


def test_the_clustering_and_the_populations_leave_the_unlabelled_cells_out(tables):
    objects, fovs, _, images = tables
    setup = _setup()
    clustering = rare_clustering(objects, fovs, setup)
    unlabelled = clustering.table.groupby("condition")["fovs_unlabelled"].sum()
    assert unlabelled["siCTRL"] == len(images) and unlabelled.drop("siCTRL").eq(0).all()
    found = populations_in_report(objects, fovs, setup)
    labels = found.populations.labels.reindex(objects.index)
    known = objects["mitotic"].notna() & objects[list(FEATURES)].notna().all(axis=1)
    positive = known & objects["mitotic"].eq(True)
    for row in found.label.itertuples():
        inside = labels == row.population
        assert row.held == pytest.approx((positive & inside).sum() / positive.sum())
        assert row.purity == pytest.approx((positive & inside).sum() / (inside & known).sum())


def test_the_report_counts_the_unlabelled_cells_once(tables):
    objects, fovs, missing, _ = tables
    setup = _setup(fingerprint=())
    note = unlabelled_note(objects, setup)
    assert note.startswith(f"{int(missing.sum())} cells of {len(objects)} carry no mitotic label")
    assert unlabelled_note(objects.assign(mitotic=False), setup) is None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = report(objects, fovs, setup)
    assert result.document.notes.count(note) == 1
    flat = " ".join(result.summary(detail="full").split())
    assert flat.count("carry no mitotic label") == 1
    assert pd.notna(result.results["label_control"].recognized)

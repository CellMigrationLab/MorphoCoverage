"""The figures place the canonical ``repeat_group`` as their biological-repeat level.

MorphoFeature's notebooks group by ``("condition", "repeat_group")``: biological repeats are
kept apart and paired by their confirmed repeat group, never by the folder label in
``biological_repeat``. These tests guard against a figure (``plots_of_data``, the pairwise
matrix under it, the HTML report's interactive copy) that refuses those groups, falls back to
the folder label, or labels its repeats by folder; and they check that
``group_cols`` naming both columns are refused, and that the ``biological_repeat`` path is
unchanged.
"""

from __future__ import annotations

import json
import re

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from morphocoverage.plots import plots_of_data
from morphocoverage.plots_pairwise import pairwise_contrasts
from morphocoverage.workflows import Setup, default_novelty, html_report, report

# The same folder labels in both conditions name different repeat groups (fixture v30's
# trap): pairing or labelling by folder would put drug's G2 beside ctrl's G1.
CROSSED = {
    ("ctrl", "r1"): "G1",
    ("ctrl", "r2"): "G2",
    ("ctrl", "r3"): "G3",
    ("drug", "r1"): "G2",
    ("drug", "r2"): "G3",
    ("drug", "r3"): "G1",
}
# Different folder labels in each condition for the same repeat groups (fixture v03's trap).
RENAMED = {
    ("ctrl", "rep1"): "G1",
    ("ctrl", "rep2"): "G2",
    ("ctrl", "rep3"): "G3",
    ("drug", "experiment_21"): "G1",
    ("drug", "experiment_22"): "G2",
    ("drug", "experiment_23"): "G3",
}
FEATURES = ("area", "speed", "aspect")
BOTH = "group_cols hold both 'repeat_group' and 'biological_repeat'"


def _tables(groups: dict = CROSSED, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Five fields of six cells per folder; each repeat group shifts every value its own way,
    so a cluster drawn from the wrong repeat shows."""
    rng = np.random.default_rng(seed)
    rows = []
    for (condition, folder), group in groups.items():
        shift = {"G1": 0.0, "G2": 3.0, "G3": 6.0}[group] + (2.0 if condition == "drug" else 0.0)
        for field in range(5):
            for label in range(6):
                rows.append(
                    {
                        "condition": condition,
                        "biological_repeat": folder,
                        "repeat_group": group,
                        "image_file": f"{condition}_{folder}_{field}.tif",
                        "label_id": label + 1,
                        "area": rng.normal(20 + shift, 1),
                        "speed": rng.normal(5, 1),
                        "aspect": rng.normal(2, 0.2),
                    }
                )
    objects = pd.DataFrame(rows)
    fovs = objects[
        ["condition", "biological_repeat", "repeat_group", "image_file"]
    ].drop_duplicates()
    return objects, fovs


def _summaries(group_cols, groups: dict = CROSSED, **options) -> pd.DataFrame:
    objects, fovs = _tables(groups)
    figure = plots_of_data(
        objects, fovs, features=["area"], group_cols=group_cols, n_boot=50, **options
    )["area"][0]
    return figure.cluster_summaries, figure.caption


def test_the_repeat_level_is_the_repeat_group_when_the_groups_name_it():
    from morphocoverage._repeat_groups import repeat_level

    assert repeat_level(None) == "biological_repeat"
    assert repeat_level(("condition", "biological_repeat")) == "biological_repeat"
    assert repeat_level(("condition", "repeat_group")) == "repeat_group"
    with pytest.raises(ValueError, match=BOTH):
        repeat_level(("condition", "biological_repeat", "repeat_group"))


def test_the_figure_places_the_repeat_groups_labelled_by_their_values():
    summaries, caption = _summaries(("condition", "repeat_group"))
    assert sorted(summaries["biological_repeat"].unique()) == ["G1", "G2", "G3"]
    assert len(summaries) == 6
    # Each cluster holds its repeat group's cells, not its folder's: drug's G1 is folder r3.
    objects, _ = _tables()
    drug_g1 = summaries.query("condition == 'drug' and biological_repeat == 'G1'")
    cells = objects.query("condition == 'drug' and repeat_group == 'G1'")["area"]
    assert drug_g1["summary"].item() == pytest.approx(np.median(cells))
    # The repeat group is a biological repeat, named as one without its column.
    assert "biological repeat" in caption and "(repeat_group)" not in caption


def test_repeat_col_none_pools_the_repeat_groups():
    summaries, _ = _summaries(("condition", "repeat_group"), repeat_col=None)
    assert len(summaries) == 2 and summaries["biological_repeat"].isna().all()


def test_the_biological_repeat_path_is_unchanged():
    # Called directly on the tables as given, the folder labels are what the caller asked
    # to keep apart; with no group_cols the default biological_repeat is placed, as before.
    for group_cols in (("condition", "biological_repeat"), None):
        summaries, caption = _summaries(group_cols)
        assert sorted(summaries["biological_repeat"].unique()) == ["r1", "r2", "r3"]
        assert "(biological_repeat)" not in caption


def test_group_cols_naming_both_repeat_columns_are_refused():
    with pytest.raises(ValueError, match=BOTH):
        _summaries(("condition", "biological_repeat", "repeat_group"))
    objects, fovs = _tables()
    with pytest.raises(ValueError, match=BOTH):
        pairwise_contrasts(
            objects,
            fovs,
            feature="area",
            group_cols=("condition", "biological_repeat", "repeat_group"),
        )


def test_the_pairwise_matrix_pairs_the_repeat_groups():
    objects, fovs = _tables(RENAMED)
    table = pairwise_contrasts(
        objects, fovs, feature="area", group_cols=("condition", "repeat_group")
    )
    # The three repeat groups are shared although no folder label is.
    assert table["shared_repeats"].tolist() == [3]


def _feature_specs(page: str) -> dict:
    specs = json.loads(re.search(r'id="feature-data">(.*?)</script>', page).group(1))
    return next(s for s in specs if s["feature"] == "area")["views"]["raw"]


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_the_page_is_written_and_shows_the_repeat_groups(tmp_path):
    objects, fovs = _tables(RENAMED)
    drawn = {}
    for group_cols in (("condition", "repeat_group"), ("condition", "biological_repeat")):
        setup = Setup(
            readout="area",
            features=FEATURES,
            control={"condition": "ctrl"},
            observation_unit="cell",
            group_cols=group_cols,
        )
        result = report(objects, fovs, setup)
        folder = tmp_path / group_cols[1]
        drawn[group_cols[1]] = _feature_specs(
            html_report(result, objects, fovs, folder).read_text()
        )
    clusters = drawn["repeat_group"]["clusters"]
    assert sorted({c["biological_repeat"] for c in clusters}) == ["G1", "G2", "G3"]
    # Every point's hover names its repeat group first, then its field.
    labels = drawn["repeat_group"]["points"]["label"]
    assert {label.split(" · ")[0] for label in labels} == {"G1", "G2", "G3"}
    # The workflows read the tables paired by repeat group (#617): the folder-label path
    # draws the same repeats, the same numbers.
    assert clusters == drawn["biological_repeat"]["clusters"]
    assert drawn["repeat_group"]["pairwise"] == drawn["biological_repeat"]["pairwise"]


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_the_page_refuses_group_cols_naming_both_repeat_columns(tmp_path):
    objects, fovs = _tables()
    setup = Setup(
        readout="area",
        features=FEATURES,
        control={"condition": "ctrl"},
        observation_unit="cell",
        group_cols=("condition", "biological_repeat", "repeat_group"),
    )
    result = report(objects, fovs, setup)
    with pytest.raises(ValueError, match=BOTH):
        html_report(result, objects, fovs, tmp_path)


def _default_groups(fovs: pd.DataFrame) -> tuple[str, ...]:
    """MorphoFeature's ``notebook_results.default_groups``: the condition and the canonical
    ``repeat_group``, refused (never the folder label) when the FOV table has none."""
    if "repeat_group" not in fovs.columns:
        raise ValueError("the coverage FOV table has no 'repeat_group' column")
    return ("condition", "repeat_group")


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_the_morphofeature_notebook_writes_its_page(tmp_path):
    """Section 4 of MorphoFeature's ``notebooks/MorphoBricks_end_to_end.ipynb`` (at
    70d4753): the widgets' defaults (the first control, no positive control, novelty with the
    default definition, tolerances 0.05 and 0.025, ``default_groups``), then ``report`` and
    ``html_report`` into ``<output>/report``."""
    objects, fovs = _tables(CROSSED)
    features = tuple(c for c in objects.columns if c.startswith(("area", "speed", "aspect")))
    setup = Setup(
        readout="area",
        features=features,
        control={"condition": "ctrl"},
        positive=None,
        novelty=default_novelty(features),
        tolerances=(0.05, 0.025),
        acquisition_order_col=None,
        stage_cols=None,
        fov_col=None,
        group_cols=_default_groups(fovs),
        observation_unit="cell",
        n_jobs=1,
    )
    report_result = report(objects, fovs, setup)
    page = html_report(report_result, objects, fovs, tmp_path / "report")
    assert page.exists()
    clusters = _feature_specs(page.read_text())["clusters"]
    assert sorted({c["biological_repeat"] for c in clusters}) == ["G1", "G2", "G3"]

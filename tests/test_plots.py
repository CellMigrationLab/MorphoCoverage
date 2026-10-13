"""PlotsOfData-style figures: objects as points, intervals from fields, legible limits."""

from __future__ import annotations

from types import SimpleNamespace

import matplotlib
from _helpers_sampling_unit import (
    _two_repeats,
    _well_tables,
    _well_tables_with_empty_fields,
    _without_technical,
)
from _helpers_tables import _four_cell_tables, _plate_tables
from _helpers_workflows import READOUT, _acquisition, _setup
from scenarios import FEATURE_COLS

import morphocoverage._plots_clusters as intervals_module
import morphocoverage.plots as plots_module
import morphocoverage.workflows.figures as figures_module
from morphocoverage import validate_input_tables
from morphocoverage._palette import COLOURS
from morphocoverage._plots_text import title
from morphocoverage._plots_text import title as _title
from morphocoverage.plots import _Layout, _layout
from morphocoverage.retrospective import dataset_overview
from morphocoverage.sampling_unit import FeatureRanking

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

from morphocoverage.plots import plots_of_data
from morphocoverage.workflows import Setup, plots

# The well is named as technical_col: only technical_repeat is found by itself (audit 3, H-5).
GROUPS = ["condition", "biological_repeat", "well"]


def _tables(n_conditions: int = 2, n_repeats: int = 2, n_fields: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    rows, fovs = [], []
    for c in range(n_conditions):
        for r in range(n_repeats):
            for j in range(n_fields):
                name = f"c{c}_r{r}_f{j}"
                fovs.append(
                    {
                        "condition": f"c{c}",
                        "biological_repeat": f"r{r}",
                        "well": "w",
                        "image_file": name,
                    }
                )
                offset = rng.normal(0, 0.5)
                for _ in range(15):
                    rows.append(
                        {
                            "condition": f"c{c}",
                            "biological_repeat": f"r{r}",
                            "well": "w",
                            "image_file": name,
                            "x": c + offset + rng.normal(),
                            "y": rng.normal(),
                        }
                    )
    return pd.DataFrame(rows), pd.DataFrame(fovs)


def test_one_figure_per_feature_with_points_summary_and_interval():
    objects, fovs = _tables()
    figures = plots_of_data(
        objects, fovs, features=["x", "y"], n_boot=50, group_cols=GROUPS, technical_col="well"
    )
    assert set(figures) == {"x", "y"} and all(len(v) == 1 for v in figures.values())
    figure = figures["x"][0]
    assert isinstance(figure, Figure)
    ax = figure.axes[0]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["c0", "c1"]
    assert len(ax.collections) >= 4  # points per condition and repeat, plus legend handles
    # Per condition and repeat cluster: the interval bar, its caps and the median line.
    assert sum(isinstance(c, LineCollection) for c in ax.collections) == 12
    title = ax.figure.caption
    assert "resampling its fields (one well per repeat)" in title
    assert "each biological repeat side by side" in title
    # Repeats are labelled above their clusters; points are shaped by the well, and one
    # well per repeat needs no legend.
    assert sorted(t.get_text() for t in ax.texts) == ["r0", "r0", "r1", "r1"]
    assert ax.get_legend() is None
    assert "its shape the well" in title


def test_each_part_of_a_split_figure_describes_its_own_wells():
    """#179: both parts read the wells of all 14 conditions, not those they show."""
    objects, fovs = _tables(n_conditions=14, n_repeats=1, n_fields=4)
    first = sorted(fovs["condition"].unique())[:7]  # the conditions of the first part
    two = fovs["condition"].isin(first) & fovs["image_file"].str.endswith(("f2", "f3"))
    fovs.loc[two, "well"] = "w2"
    objects["well"] = objects["image_file"].map(dict(zip(fovs["image_file"], fovs["well"])))
    parts = plots_of_data(
        objects, fovs, features=["x"], n_boot=20, group_cols=GROUPS, technical_col="well"
    )["x"]
    assert "whole wells with their fields (2 per repeat)\n" in parts[0].caption
    assert "one well" not in parts[0].caption and parts[0].axes[0].get_legend() is not None
    assert "resampling its fields (one well per repeat)" in parts[1].caption
    assert "2 to 3 wells" not in parts[1].caption and parts[1].axes[0].get_legend() is None


def test_many_conditions_split_and_many_repeats_are_pooled():
    objects, fovs = _tables(n_conditions=14, n_repeats=1, n_fields=2)
    figures = plots_of_data(
        objects, fovs, features=["x"], n_boot=20, group_cols=GROUPS, technical_col="well"
    )
    assert len(figures["x"]) == 2
    # Split evenly over the two figures, never 12 and 2.
    assert "conditions 1 to 7 of 14" in figures["x"][0].axes[0].figure.caption
    assert "conditions 8 to 14 of 14" in figures["x"][1].axes[0].figure.caption
    objects, fovs = _tables(n_conditions=1, n_repeats=9, n_fields=2)
    figures = plots_of_data(
        objects, fovs, features=["x"], n_boot=20, group_cols=GROUPS, technical_col="well"
    )
    ax = figures["x"][0].axes[0]
    assert "each condition's biological repeats pooled (more than the 8" in ax.figure.caption
    assert sum(isinstance(c, LineCollection) for c in ax.collections) == 3 and not ax.texts
    # Too many technical replicates of one biological repeat to colour: a note instead of a
    # legend (well names are counted per repeat, so 9 fields a repeat, each its own well).
    objects, fovs = _tables(n_conditions=1, n_repeats=9, n_fields=9)
    wells = {name: f"w{i % 9}" for i, name in enumerate(fovs["image_file"])}
    objects["well"], fovs["well"] = objects["image_file"].map(wells), fovs["image_file"].map(wells)
    figures = plots_of_data(
        objects, fovs, features=["x"], n_boot=20, group_cols=GROUPS, technical_col="well"
    )
    ax = figures["x"][0].axes[0]
    assert ax.get_legend() is None
    assert any("9 wells per biological repeat" in t.get_text() for t in ax.texts)


def test_limits_and_declarations_are_enforced():
    objects, fovs = _tables()
    with pytest.raises(ValueError, match="max_features"):
        plots_of_data(
            objects, fovs, features=["x"] * 25, n_boot=5, group_cols=GROUPS, technical_col="well"
        )
    with pytest.raises(ValueError, match="summary"):
        plots_of_data(
            objects,
            fovs,
            features=["x"],
            summary="mode",
            n_boot=5,
            group_cols=GROUPS,
            technical_col="well",
        )
    setup = Setup(readout="x", group_cols=tuple(GROUPS))
    with pytest.raises(ValueError, match="declared in the setup"):
        plots(objects, fovs, setup, features=["y"], n_boot=5)


def test_workflow_plots_save_one_png_per_figure(tmp_path):
    objects, fovs = _tables()
    # Normalising needs positive control means: values around zero are shifted up.
    objects[["x", "y"]] += 10.0
    setup = Setup(
        readout="x",
        features=("x", "y"),
        control={"condition": "c0"},
        normalize="control_mean",
        group_cols=tuple(GROUPS),
    )
    figures = plots(
        objects, fovs, setup, features=["x", "y"], folder=tmp_path, n_boot=20, technical_col="well"
    )
    assert set(figures) == {"x", "y"}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["x.png", "y.png"]
    # Without a condition or repeat column the plot still draws: one group, one colour.
    bare_objects = objects.drop(columns=["condition", "biological_repeat"])
    bare_fovs = fovs.drop(columns=["condition", "biological_repeat"])
    figures = plots_of_data(
        bare_objects,
        bare_fovs,
        features=["x"],
        n_boot=10,
        group_cols=["well"],
        technical_col="well",
    )
    assert [t.get_text() for t in figures["x"][0].axes[0].get_xticklabels()] == ["all fields"]


def test_the_morphobricks_technical_repeat_is_the_default_colour_and_bootstrap_unit():
    objects, fovs = _tables()
    wells = {name: f"w{i % 2}" for i, name in enumerate(fovs["image_file"])}
    objects["technical_repeat"] = objects["image_file"].map(wells)
    fovs["technical_repeat"] = fovs["image_file"].map(wells)
    # No group_cols: the MorphoBricks defaults find condition, repeat and technical_repeat.
    bare = objects.drop(columns="well"), fovs.drop(columns="well")
    figures = plots_of_data(*bare, features=["x"], n_boot=30)
    ax = figures["x"][0].axes[0]
    title = ax.figure.caption
    assert "its shape the well\n" in title
    legend = ax.get_legend()
    assert legend.get_title().get_text().startswith("well,\nin order")
    labels = [t.get_text() for t in legend.get_texts()]
    assert labels == ["1st", "2nd"]


def test_a_field_without_observations_must_have_its_condition_in_the_fov_table():
    """#200: a layout column only the observations carry cannot place an empty field, which
    is refused rather than left out of the resampling."""
    rng = np.random.default_rng(0)
    objects = pd.DataFrame(
        {
            "position_id": np.repeat(["a", "b", "c", "d"], 5),
            "treatment": np.repeat(["X", "X", "Y", "Y"], 5),
            "f": rng.normal(size=20),
        }
    )
    fovs = pd.DataFrame({"position_id": ["a", "b", "c", "d", "e"]})
    arguments = {"features": ["f"], "condition_col": "treatment", "repeat_col": None}
    with pytest.raises(ValueError, match="add 'treatment' to the FOV table"):
        plots_of_data(objects, fovs, **arguments)
    fovs["treatment"] = ["X", "X", "Y", "Y", "Y"]
    assert len(plots_of_data(objects, fovs, **arguments)["f"]) == 1
    # Every field holds observations: the observations' values place them, as before.
    assert plots_of_data(objects, fovs.iloc[:4].drop(columns="treatment"), **arguments)["f"]


def test_e5_plot_interval_says_how_many_technical_repeats_and_warns_with_two_or_three():
    rng = np.random.default_rng(0)
    rows, fovs = [], []
    for condition in ("c0", "c1"):
        for field_index in range(6):
            name = f"{condition}_{field_index}"
            well = f"w{field_index % 2}"
            fovs.append({"condition": condition, "technical_repeat": well, "image_file": name})
            rows += [
                {"condition": condition, "technical_repeat": well, "image_file": name, "x": v}
                for v in rng.normal(size=10)
            ]
    figures = plots_of_data(pd.DataFrame(rows), pd.DataFrame(fovs), features=["x"], n_boot=20)
    title = figures["x"][0].axes[0].figure.caption
    assert "whole wells with their fields (2 per condition" in title
    assert "with 2 to 3 wells the interval is only a guide" in title


def test_h9_plots_raise_on_a_misspelt_column():
    objects, fovs = _two_repeats()
    common = {"features": ["x"], "n_boot": 10}
    for argument in ("condition_col", "repeat_col", "technical_col"):
        with pytest.raises(ValueError, match=f"{argument} 'Wrong' is not a column"):
            plots_of_data(objects, fovs, **{argument: "Wrong"}, **common)
    # The default condition and repeat columns may be absent.
    figures = plots_of_data(
        objects.drop(columns="biological_repeat"),
        fovs.drop(columns="biological_repeat"),
        group_cols=["condition", "technical_repeat"],
        **common,
    )
    assert len(figures["x"]) == 1


def test_h12_the_legend_counts_wells_within_each_repeat():
    spec = [("ctrl", f"r{r}", f"r{r}_w{i}", 3, 0, 0.1) for r in range(9) for i in range(2)]
    objects, fovs = _well_tables(spec, per_field=5)
    ax = plots_of_data(objects, fovs, features=["x"], n_boot=10)["x"][0].axes[0]
    assert not ax.texts
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == ["1st", "2nd"]


def _caps(figure):
    segments = [
        s
        for c in figure.axes[0].collections
        if isinstance(c, LineCollection)
        for s in c.get_segments()
    ]
    return sorted({round(float(s[0][1]), 6) for s in segments if abs(s[0][1] - s[1][1]) < 1e-12})


def test_h17_every_cluster_draws_from_its_own_generator():
    objects, fovs = _well_tables([("ctrl", "r1", f"w{i}", 6, i * 0.5, 0.5) for i in range(3)])
    objects["y"] = objects["x"] * 2
    alone = _caps(plots_of_data(objects, fovs, features=["y"])["y"][0])
    after = _caps(plots_of_data(objects, fovs, features=["x", "y"])["y"][0])
    assert alone == after
    more, more_fovs = _well_tables(
        [("ctrl", "r1", f"w{i}", 6, i * 0.5, 0.5) for i in range(3)]
        + [("drug", "r1", f"v{i}", 6, 1.0, 0.5) for i in range(3)]
    )
    more["y"] = more["x"] * 2
    ctrl = _caps(
        plots_of_data(more[more["condition"] == "ctrl"], more_fovs.iloc[:18], features=["y"])["y"][
            0
        ]
    )
    both = plots_of_data(more, more_fovs, features=["y"])["y"][0]
    assert set(ctrl) <= set(_caps(both))


def test_j8_plot_intervals_include_empty_fields(monkeypatch):
    rng = np.random.default_rng(0)
    fov_rows, rows = [], []
    for j in range(12):
        fov_rows.append({"condition": "A", "position_id": f"f{j}"})
        if j < 4:  # 4 fields with objects, 8 acquired empty fields
            rows += [
                {"condition": "A", "position_id": f"f{j}", "x": v}
                for v in rng.normal(10 + 2 * j, 1, size=20)
            ]
    captured = []
    original = intervals_module.bootstrap_nested

    def spy(units, *args, **kwargs):
        captured.append([len(unit) for unit in units])
        return original(units, *args, **kwargs)

    monkeypatch.setattr(intervals_module, "bootstrap_nested", spy)
    plots_of_data(
        pd.DataFrame(rows), pd.DataFrame(fov_rows), features=["x"], group_cols=[], n_boot=20
    )
    assert captured == [[12]]


def _spy(monkeypatch):
    """The units every plotted interval resampled: per call, the fields of each unit."""
    calls = []
    original = intervals_module.bootstrap_nested

    def spy(units, *args, **kwargs):
        calls.append([len(unit) for unit in units])
        return original(units, *args, **kwargs)

    monkeypatch.setattr(intervals_module, "bootstrap_nested", spy)
    return calls


def test_h2_pooled_repeats_resample_the_repeats_alone(monkeypatch):
    calls = _spy(monkeypatch)
    spec = [("ctrl", f"r{r}", f"w{i}", 3, float(r), 0.2) for r in range(10) for i in range(2)]
    objects, fovs = _well_tables_with_empty_fields(spec, per_field=5)
    title = plots_of_data(objects, fovs, features=["x"], n_boot=20)["x"][0].axes[0]
    # One call for the one pooled cluster: ten repeats, each with its 6 fields (2 wells).
    assert calls == [[6] * 10]
    text = title.figure.caption
    assert "resampling whole biological repeats with all their fields" in text
    assert not hasattr(plots_module, "_bootstrap_repeats")


def test_h3_plot_intervals_are_the_widened_nested_ones_with_the_small_unit_caveat(
    monkeypatch,
):
    calls = _spy(monkeypatch)
    spec = [("ctrl", "r1", f"w{i}", 4, 0.3 * i, 0.2) for i in range(3)]
    objects, fovs = _well_tables_with_empty_fields(spec, per_field=10)
    ax = plots_of_data(objects, fovs, features=["x"], n_boot=50)["x"][0].axes[0]
    assert calls == [[4, 4, 4]]  # the wells, each with its fields, into bootstrap_nested
    assert "with 2 to 3 wells the interval is only a guide" in (ax.figure.caption)


def test_h4_repeats_left_out_of_the_layout_are_pooled_not_merged(monkeypatch):
    calls = _spy(monkeypatch)
    spec = [("ctrl", f"r{r}", w, 4, 2.0 * r, 0.2) for r in range(3) for w in ("A", "B")]
    objects, fovs = _well_tables_with_empty_fields(spec, per_field=15)
    ax = plots_of_data(objects, fovs, features=["x"], repeat_col=None, n_boot=20)["x"][0].axes[0]
    # Three repeats, each with its two wells of four fields: never well A of all repeats.
    assert calls == [[8, 8, 8]]
    title = ax.figure.caption
    assert "repeat not shown apart, pooled within each cluster" in title
    assert "whole biological repeats with all their fields" in title
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == ["1st", "2nd"]


def test_h5_only_the_technical_repeat_column_is_taken_by_itself():
    spec = [(c, "r1", "w", 6, m, 0.2) for c, m in (("ctrl", 0.0), ("drug", 3.0))]
    objects, fovs = _without_technical(*_well_tables_with_empty_fields(spec, per_field=15))
    renamed = objects.rename(columns={"condition": "treatment"})
    renamed_fovs = fovs.rename(columns={"condition": "treatment"})
    with pytest.raises(ValueError, match=r"hold \['treatment'\], which the figure does not"):
        plots_of_data(
            renamed, renamed_fovs, features=["x"], group_cols=["treatment", "biological_repeat"]
        )
    figures = plots_of_data(
        renamed,
        renamed_fovs,
        features=["x"],
        condition_col="treatment",
        group_cols=["treatment", "biological_repeat"],
        n_boot=10,
    )
    ax = figures["x"][0].axes[0]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["ctrl", "drug"]
    assert "coloured by technical repeat" not in ax.figure.caption


def test_h6_a_missing_technical_repeat_value_raises():
    spec = [("ctrl", "r1", w, 4, 0.0, 0.2) for w in ("A", "B", "C")]
    objects, fovs = _without_technical(*_well_tables_with_empty_fields(spec, per_field=15))
    fovs["well"] = [p.split("_")[2] for p in fovs["position_id"]]
    fovs.loc[fovs.index[:2], "well"] = None
    with pytest.raises(ValueError, match="2 rows of the FOV table have no 'well' value"):
        plots_of_data(objects, fovs, features=["x"], technical_col="well")


def test_h16_a_technical_column_of_the_fov_table_only_is_used():
    spec = [("ctrl", "r1", w, 4, 0.0, 0.2) for w in ("A", "B", "C")]
    objects, fovs = _without_technical(*_well_tables_with_empty_fields(spec, per_field=10))
    fovs["well"] = [p.split("_")[2] for p in fovs["position_id"]]
    ax = plots_of_data(objects, fovs, features=["x"], technical_col="well", n_boot=10)["x"][0]
    title = ax.axes[0].figure.caption
    assert "its shape the well" in title
    assert "whole wells with their fields (3 per repeat" in title


def test_p9_10_control_first_then_sorted_whatever_the_row_order():
    cells, fovs = _plate_tables()
    cells = cells.assign(condition=cells["condition"].map({"control": "mock", "drug": "a_drug"}))
    fovs = fovs.assign(condition=fovs["condition"].map({"control": "mock", "drug": "a_drug"}))
    shuffled = fovs.sample(frac=1, random_state=7)
    shuffled_cells = cells.sample(frac=1, random_state=8)
    overview = dataset_overview(
        shuffled_cells, shuffled, feature_cols=["area"], control={"condition": "mock"}
    )
    assert overview.layout["condition"].tolist()[:2] == ["mock", "mock"]
    assert "(condition: mock, a_drug)" in overview.summary(detail="full")
    assert "(biological_repeat: 2026_01_01, 2026_01_02)" in overview.summary(detail="full")
    short = " ".join(overview.summary().split())
    assert "2 conditions (mock, a_drug), 2 biological repeats (2026_01_01, 2026_01_02)" in short
    data = validate_input_tables(shuffled_cells, fovs=shuffled, feature_cols=["area"])
    layout = _layout(data, "condition", "biological_repeat", None, 6, "mock")
    assert layout.conditions == ["mock", "a_drug"]
    assert layout.repeats == ["2026_01_01", "2026_01_02"]
    assert _layout(data, "condition", "biological_repeat", None, 6).conditions == ["a_drug", "mock"]


def test_p3_5_a_single_technical_repeat_interval_is_captioned():
    layout = _Layout(
        "condition", "biological_repeat", "technical_repeat", ["a"], ["r1"], False, 4, ()
    )
    title = _title("x", "", layout, "cell", "median", 0.95, clusters=[(1, [1]), (1, [3])])
    assert "with one well the interval shows field-to-field variation only" in title


def _pooled_wells():
    """9 biological repeats (more than are drawn side by side), 2 wells each named alike."""
    rng = np.random.default_rng(0)
    rows, fields = [], []
    for repeat in range(9):
        for well in range(2):
            shift = rng.normal(0, 3)
            for field in range(3):
                key = {
                    "condition": "c",
                    "biological_repeat": f"r{repeat}",
                    "technical_repeat": f"w{well}",
                    "image_file": f"f{field}.tif",
                }
                fields.append(key)
                rows += [{**key, "x": shift + rng.normal()} for _ in range(10)]
    return pd.DataFrame(rows), pd.DataFrame(fields)


def test_h1_pooled_repeats_keep_each_well_within_its_repeat(monkeypatch):
    captured = []
    original = intervals_module.bootstrap_nested

    def spy(units, *args, **kwargs):
        captured.append([len(unit) for unit in units])
        return original(units, *args, **kwargs)

    monkeypatch.setattr(intervals_module, "bootstrap_nested", spy)
    plots_of_data(*_pooled_wells(), features=["x"], n_boot=50)
    # 9 repeats, each with the 6 fields of its 2 wells (the repeats alone resampled: audit
    # 3, H-2), never 2 wells of 27 fields
    assert captured == [[6] * 9]


def test_h5_the_palette_note_and_limit_say_what_is_drawn():
    assert not any(colour in ("#8C8C8C", "#999999") for colour in COLOURS)
    with pytest.raises(ValueError, match="max_technical can be at most 8"):
        plots_of_data(*_pooled_wells(), features=["x"], n_boot=10, max_technical=9)
    figure = plots_of_data(*_pooled_wells(), features=["x"], n_boot=10, max_technical=1)["x"][0]
    texts = [t.get_text() for t in figure.axes[0].texts]
    assert any("the first 1 shaped in order, the others small dots" in t for t in texts)


def test_h10_df2_plots_give_the_top_places_to_the_best_ranked(monkeypatch):
    """The readout is plotted only when it ranks (owner, 2026-10-02: the report is neutral)."""
    seen = {}

    def spy(objects, fovs, *, features, value_note=None, **kwargs):
        seen["features"], seen["note"] = list(features), value_note
        return {}

    monkeypatch.setattr(figures_module, "plots_of_data", spy)
    ranking = FeatureRanking(
        pd.DataFrame(),
        pd.DataFrame({"feature": [READOUT, FEATURE_COLS[1]], "score": [3.0, 2.0]}),
        {},
    )
    objects, fovs = _acquisition(n_fovs=8, n_objects=6)
    plots(objects, fovs, _setup(), ranking=ranking, top=1)
    assert seen["features"] == [READOUT] and seen["note"] is None
    other_first = FeatureRanking(
        pd.DataFrame(),
        pd.DataFrame({"feature": [FEATURE_COLS[1], READOUT], "score": [3.0, 2.0]}),
        {},
    )
    plots(objects, fovs, _setup(), ranking=other_first, top=1)
    assert seen["features"] == [FEATURE_COLS[1]]
    plots(objects, fovs, _setup(normalize="control_mean"), ranking=ranking, top=1)
    assert seen["note"] == "relative to control"


def test_title_with_technical_repeats_and_no_cluster():
    layout = SimpleNamespace(
        repeat_col=None, technical_col="well", pooled_repeats=False, pooled_cols=(), max_repeats=8
    )
    assert title("area", "", layout, "cell", "mean", 0.95) == (
        "area\nevery cell a point in its condition's colour, its shape the well\nmean with its "
        "95% interval from resampling whole wells with their fields, or the fields where a "
        "condition has one well"
    )


def test_plots_of_data_refuses_no_feature():
    objects, fovs = _four_cell_tables()
    with pytest.raises(ValueError) as error:
        plots_of_data(objects, fovs, features=[])
    assert str(error.value) == "features must name at least one column"

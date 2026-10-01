"""PlotsOfData-style figures: objects as points, intervals from fields, legible limits."""

import matplotlib

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

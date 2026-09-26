"""PlotsOfData-style figures: objects as points, intervals from fields, legible limits."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from morphocoverage.plots import plots_of_data
from morphocoverage.workflows import Setup, plots

GROUPS = ["condition", "repeat", "well"]


def _tables(n_conditions: int = 2, n_repeats: int = 2, n_fields: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    rows, fovs = [], []
    for c in range(n_conditions):
        for r in range(n_repeats):
            for j in range(n_fields):
                name = f"c{c}_r{r}_f{j}"
                fovs.append(
                    {"condition": f"c{c}", "repeat": f"r{r}", "well": "w", "image_file": name}
                )
                offset = rng.normal(0, 0.5)
                for _ in range(15):
                    rows.append(
                        {
                            "condition": f"c{c}",
                            "repeat": f"r{r}",
                            "well": "w",
                            "image_file": name,
                            "x": c + offset + rng.normal(),
                            "y": rng.normal(),
                        }
                    )
    return pd.DataFrame(rows), pd.DataFrame(fovs)


def test_one_figure_per_feature_with_points_summary_and_interval():
    objects, fovs = _tables()
    figures = plots_of_data(objects, fovs, features=["x", "y"], n_boot=50, group_cols=GROUPS)
    assert set(figures) == {"x", "y"} and all(len(v) == 1 for v in figures.values())
    figure = figures["x"][0]
    assert isinstance(figure, Figure)
    ax = figure.axes[0]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["c0", "c1"]
    assert len(ax.collections) >= 4  # points per condition and repeat, plus legend handles
    assert len(ax.patches) == 2  # one interval box per condition
    assert "resampling whole fields" in ax.get_title()


def test_many_conditions_split_and_many_repeats_lose_their_colours():
    objects, fovs = _tables(n_conditions=14, n_repeats=1, n_fields=2)
    figures = plots_of_data(objects, fovs, features=["x"], n_boot=20, group_cols=GROUPS)
    assert len(figures["x"]) == 2
    assert "conditions 1 to 12 of 14" in figures["x"][0].axes[0].get_title()
    objects, fovs = _tables(n_conditions=1, n_repeats=9, n_fields=2)
    figures = plots_of_data(objects, fovs, features=["x"], n_boot=20, group_cols=GROUPS)
    texts = [t.get_text() for t in figures["x"][0].axes[0].texts]
    assert any("9 repeats (more than 8)" in t for t in texts)


def test_limits_and_declarations_are_enforced():
    objects, fovs = _tables()
    with pytest.raises(ValueError, match="max_features"):
        plots_of_data(objects, fovs, features=["x"] * 25, n_boot=5, group_cols=GROUPS)
    with pytest.raises(ValueError, match="summary"):
        plots_of_data(objects, fovs, features=["x"], summary="mode", n_boot=5, group_cols=GROUPS)
    setup = Setup(readout="x", group_cols=tuple(GROUPS))
    with pytest.raises(ValueError, match="declared in the setup"):
        plots(objects, fovs, setup, features=["y"], n_boot=5)


def test_workflow_plots_save_one_png_per_figure(tmp_path):
    objects, fovs = _tables()
    setup = Setup(
        readout="x",
        features=("x", "y"),
        control={"condition": "c0"},
        normalize="control_mean",
        group_cols=tuple(GROUPS),
    )
    figures = plots(objects, fovs, setup, features=["x", "y"], folder=tmp_path, n_boot=20)
    assert set(figures) == {"x", "y"}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["x.png", "y.png"]
    # Without a condition or repeat column the plot still draws: one group, one colour.
    bare_objects = objects.drop(columns=["condition", "repeat"])
    bare_fovs = fovs.drop(columns=["condition", "repeat"])
    figures = plots_of_data(bare_objects, bare_fovs, features=["x"], n_boot=10, group_cols=["well"])
    assert [t.get_text() for t in figures["x"][0].axes[0].get_xticklabels()] == ["all fields"]

"""Every pair of conditions for one feature (plots_pairwise), and the panel under a figure."""

import numpy as np
import pandas as pd
import pytest

from morphocoverage.plots_pairwise import add_pairwise, format_p, pairwise_contrasts
from morphocoverage.statistics import benjamini_hochberg
from morphocoverage.workflows import Setup, plots


def _tables(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    # ctrl and drug share repeats r1-r3; other was imaged in r4-r6 only, beside ctrl.
    design = {"ctrl": ["r1", "r2", "r3", "r4", "r5", "r6"], "drug": ["r1", "r2", "r3"]}
    design["other"] = ["r4", "r5", "r6"]
    shift = {"ctrl": 0.0, "drug": 3.0, "other": -1.0}
    for condition, repeats in design.items():
        for repeat in repeats:
            batch = 0.5 * int(repeat[1:])
            for field in range(4):
                for label in range(5):
                    rows.append(
                        {
                            "condition": condition,
                            "biological_repeat": repeat,
                            "image_file": f"{condition}_{repeat}_{field}.tif",
                            "label_id": label + 1,
                            "area": rng.normal(20 + shift[condition] + batch, 1),
                        }
                    )
    objects = pd.DataFrame(rows)
    fovs = objects[["condition", "biological_repeat", "image_file"]].drop_duplicates()
    return objects, fovs


def test_every_pair_gets_a_signed_d_and_its_randomization_test():
    objects, fovs = _tables()
    table = pairwise_contrasts(objects, fovs, feature="area", control="ctrl", n_permutations=499)
    assert list(zip(table["first"], table["second"], strict=True)) == [
        ("ctrl", "drug"),
        ("ctrl", "other"),
        ("drug", "other"),
    ]
    rows = table.set_index(["first", "second"])
    # Signed: the first condition minus the second.
    assert rows.loc[("ctrl", "drug"), "d"] < 0 < rows.loc[("ctrl", "other"), "d"]
    # Every pair is tested on the observations, a shift of 3 SD found at once.
    assert rows.loc[("ctrl", "drug"), "p"] == pytest.approx(1 / 500)
    # Benjamini-Hochberg over the three pairs.
    assert table["q"].tolist() == pytest.approx(
        benjamini_hochberg(table["p"].to_numpy()), nan_ok=True
    )
    # The paired test on the repeats' means is kept beside it, where 3 repeats imaged both.
    assert rows.loc[("ctrl", "drug"), "shared_repeats"] == 3
    assert rows.loc[("ctrl", "drug"), "paired_values"] == "log ratios"
    assert np.isfinite(rows.loc[("ctrl", "drug"), "p_paired"])
    # Imaged in different repeats (two sub-experiments): d describes, nothing is tested,
    # since the batch between their repeats would be in it (audit, 2026-10-02).
    apart = rows.loc[("drug", "other")]
    assert apart["shared_repeats"] == 0 and np.isnan(apart["p_paired"])
    assert np.isnan(apart["p"]) and np.isnan(apart["q"]) and np.isfinite(apart["d"])
    assert apart["status"] == "imaged in different repeats"
    # Seeded: the same table again.
    again = pairwise_contrasts(objects, fovs, feature="area", control="ctrl", n_permutations=499)
    assert again["p"].equals(table["p"])


def test_the_panel_is_drawn_under_each_figure_with_its_table():
    objects, fovs = _tables()
    setup = Setup(readout="area", control={"condition": "ctrl"}, observation_unit="cell")
    figure = plots(objects, fovs, setup)["area"][0]
    # A plot and, under it, panels B (d, randomization) and C (the paired t-test).
    assert len(figure.axes) == 3
    plot, panel, paired = figure.axes
    # The plot moved up, the panel sits under it.
    assert panel.get_position().y1 < plot.get_position().y0
    texts = [t.get_text() for t in panel.texts]
    # Panel C: the paired test's p and q, and – where fewer than 3 repeats imaged both.
    assert "–" in [t.get_text() for t in paired.texts]
    # The legend at the bottom, a paragraph a panel, lettered.
    legend = figure.legend_artist.get_text()
    assert legend.startswith("Legend:\n(A) ") and "\n\n(B) " in legend and "\n\n(C) " in legend
    assert any(t.endswith("†") for t in texts)  # drug and other: different repeats
    assert sum(t.endswith("†") for t in texts) == 1  # its d; it is not tested
    assert len(figure.pairwise) == 3
    summaries = figure.cluster_summaries
    assert set(summaries["condition"]) == {"ctrl", "drug", "other"}
    # Each condition only as wide as its own repeats.
    widths = summaries.groupby("condition")["x"].agg(lambda x: x.max() - x.min())
    assert widths.to_dict() == pytest.approx({"ctrl": 5, "drug": 2, "other": 2})
    bare = plots(objects, fovs, setup, pairwise=False)["area"][0]
    assert len(bare.axes) == 1


def test_one_condition_draws_no_panel_and_p_values_read_in_a_cell():
    objects, fovs = _tables()
    one = objects[objects["condition"] == "ctrl"]
    table = pairwise_contrasts(one, fovs[fovs["condition"] == "ctrl"], feature="area")
    assert table.empty
    from matplotlib.figure import Figure

    figure = Figure()
    figure.add_subplot(111)
    add_pairwise(figure, table, ["ctrl"])
    assert len(figure.axes) == 1
    assert [format_p(p) for p in (0.0004, 0.004, 0.04, 0.4)] == ["<0.001", "0.004", "0.040", "0.40"]

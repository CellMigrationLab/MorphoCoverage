"""State clustering runs only where it can reach 0.05 (owner, 2026-10-06, audit 10, A4): a
group whose smallest possible p cannot reach 0.05 after Bonferroni over the tests kept is
left out, as every test the design cannot make informative, and a state not found clustered
reads "no clustering shown", never "spread like random"."""

from __future__ import annotations

import numpy as np
import pandas as pd

from morphocoverage.retrospective.report_sampling import profile_answer
from morphocoverage.sampling_unit import fov_clustering


def _labelled(planted: bool, seed: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """18 groups of 10 fields of 20 cells: 14 groups hold one or two labelled cells, which no
    shuffle can show clustered; 4 hold 12, spread at random, or, with ``planted``, the first
    holds them in one field."""
    rng = np.random.default_rng(seed)
    rows, fov_rows = [], []
    for g in range(18):
        labelled = 1 + g % 2 if g < 14 else 12
        cells = np.zeros(200, dtype=bool)
        if planted and g == 14:
            cells[:12] = True  # field 0 holds all of them
        else:
            cells[rng.choice(200, size=labelled, replace=False)] = True
        for f in range(10):
            fov_rows.append({"condition": f"c{g}", "biological_repeat": "R1", "position_id": f})
        for i, state in enumerate(cells):
            rows.append(
                {"condition": f"c{g}", "biological_repeat": "R1", "position_id": i // 20}
                | {"rare": bool(state)}
            )
    return pd.DataFrame(rows), pd.DataFrame(fov_rows)


def test_groups_that_cannot_reach_005_are_left_out_of_the_family():
    objects, fovs = _labelled(planted=True)
    result = fov_clustering(objects, fovs, label_col="rare", states=[True])
    table = result.table
    small = table["state_objects"] < 3
    # One or two cells: no shuffle reaches fewer fields than chance, so never tested.
    assert not table.loc[small, "tested"].any()
    assert table.loc[small, "p_clustered"].isna().all()
    assert table.loc[~small, "tested"].all()
    planted = table[table["condition"] == "c14"].iloc[0]
    # Corrected over the 4 tests that can reach 0.05, not 18: the planted state is clustered.
    assert planted["q_clustered"] <= 0.05
    text = " ".join(result.summary().split())
    assert "in 1 of 4 groups where tested (c14 · repeat R1)" in text
    assert "Not tested in c0 · repeat R1, c1 · repeat R1, c2 · repeat R1 and 11 more" in text


def test_a_state_not_found_clustered_shows_no_clustering_rather_than_random():
    objects, fovs = _labelled(planted=False)
    result = fov_clustering(objects, fovs, label_col="rare", states=[True])
    parts = profile_answer(pd.DataFrame(), [result], None)
    assert any(
        part.startswith("no clustering shown for rare cells in the 4 groups") for part in parts
    )
    assert not any("spread like random" in part for part in parts)

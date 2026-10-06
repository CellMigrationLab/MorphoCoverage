"""Regression tests of the owner's audit-8 decisions on the tests and their words (F9b,
2026-10-06)."""

import numpy as np
import pandas as pd

from morphocoverage.sampling_unit.whole_table import ONE_EACH, whole_table_tests


def _table(shift_b: float = 0.0, batch: float = 0.0, images: int = 5, seed: int = 0):
    """One image per field, conditions ctrl, a (+0.5 on d0) and b (``shift_b`` on d0), 4
    repeats, repeat R0 shifted by ``batch`` on every column."""
    rng = np.random.default_rng(seed)
    fovs = pd.DataFrame(
        [
            {"condition": c, "biological_repeat": f"R{r}", "image_file": f"{c}{r}_{i}.tif"}
            for c in ("ctrl", "a", "b")
            for r in range(4)
            for i in range(images)
        ]
    )
    values = rng.normal(size=(len(fovs), 6))
    values[fovs["condition"].eq("a").to_numpy(), 0] += 0.5
    values[fovs["condition"].eq("b").to_numpy(), 0] += shift_b
    values[fovs["biological_repeat"].eq("R0").to_numpy()] += batch
    names = [f"d{j}" for j in range(6)]
    objects = pd.concat([fovs, pd.DataFrame(values, columns=names)], axis=1)
    tests = whole_table_tests(
        objects,
        fovs,
        tables={"DINO": names},
        pairs=[("a", "ctrl"), ("b", "ctrl")],
        fov_col="image_file",
        observation_unit="image",
        permutations=99,
    ).tests
    return tests.set_index("condition")


def test_a4_another_condition_or_a_batch_never_changes_a_comparison():
    # A4: each comparison is scaled on the within-group residuals of its own two sides.
    plain = _table()
    for other in (_table(shift_b=4.0), _table(batch=3.0)):
        for column in ("t", "p_repeats", "effect", "distance", "p_shuffle"):
            assert np.isclose(other.loc["a", column], plain.loc["a", column])


def test_a4_one_observation_a_side_and_repeat_is_not_tested():
    tests = _table(images=1)
    assert set(tests["status"]) == {ONE_EACH} and tests["p_repeats"].isna().all()

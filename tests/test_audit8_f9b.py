"""Regression tests of the owner's audit-8 decisions on the tests and their words (F9b,
2026-10-06)."""

import numpy as np
import pandas as pd
from scipy import stats

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


def _signed(offset_b: float, seed: int = 0):
    """ctrl, a and b, 4 repeats, 3 fields of 5 cells; ``area`` near 2 for ctrl and a, near
    ``offset_b`` for b; ``positive`` the same plus 10."""
    rng = np.random.default_rng(seed)
    rows = []
    for condition, centre in (("ctrl", 2.0), ("a", 2.5), ("b", offset_b)):
        for r in range(4):
            for f in range(3):
                for _ in range(5):
                    rows.append(
                        {
                            "condition": condition,
                            "biological_repeat": f"R{r}",
                            "image_file": f"{condition}{r}_{f}.tif",
                            "area": centre + rng.normal(scale=0.3),
                        }
                    )
    objects = pd.DataFrame(rows).assign(positive=lambda t: t["area"] + 10)
    fovs = objects[["condition", "biological_repeat", "image_file"]].drop_duplicates()
    return objects, fovs.reset_index(drop=True)


def test_a7_the_main_test_reads_one_scale_per_feature():
    from morphocoverage.sampling_unit.discovery import feature_discovery

    # A7: b's values below 0 put area on differences for every condition, a included.
    objects, fovs = _signed(offset_b=-1.0)
    tests = feature_discovery(
        objects,
        fovs,
        features=["area", "positive"],
        control={"condition": "ctrl"},
        n_permutations=9,
        fov_col="image_file",
    ).tests.set_index(["feature", "condition"])
    assert set(tests.loc["area", "paired_values"]) == {"raw"}
    assert set(tests.loc["positive", "paired_values"]) == {"normalised"}
    means = objects.groupby(["condition", "biological_repeat"])["area"].mean()
    controls = means["ctrl"].to_numpy()
    expected = stats.ttest_1samp(means["a"].to_numpy() - controls, 0.0).pvalue
    assert np.isclose(tests.loc[("area", "a"), "p_paired"], expected)
    # The contrast of the readout, its plan and its words read the same scale.
    from morphocoverage.sampling_unit.contrast import condition_contrast

    contrast = condition_contrast(
        objects,
        fovs,
        feature="area",
        control={"condition": "ctrl"},
        n_permutations=9,
        n_draws=5,
        fov_col="image_file",
    )
    assert contrast.settings["log_scale"] is False
    assert contrast.repeats["p_paired_t_normalised"].isna().all()
    assert contrast.repeats["fold_change"].isna().all()
    assert set(contrast.plan["planned_on"]) <= {"difference", "d"}


def test_seed11_the_trend_line_needs_six_repeats():
    from types import SimpleNamespace

    from morphocoverage.sampling_unit.reading_text import (
        CONSISTENT,
        NOT_SIGNIFICANT,
        condition_readings,
    )
    from morphocoverage.workflows._whole_table import whole_headline

    def reading(n):
        row = {
            "condition": "A",
            "control_condition": "CTRL",
            "repeats": n,
            "lower": 0,
            "higher": n,
            "d_mean": 1.0,
            "q_paired_t": 0.4,
            "q_paired_t_normalised": 0.3,
        }
        contrast = SimpleNamespace(
            table=pd.DataFrame(),
            settings={"repeat_col": "biological_repeat", "observation_unit": "cell"},
            repeats=pd.DataFrame([row]),
            cells=pd.DataFrame(),
            targets=pd.DataFrame(),
        )
        return condition_readings(contrast).iloc[0]

    assert reading(5)["verdict"] == NOT_SIGNIFICANT
    assert "higher in 5 of 5 biological repeats" in reading(5)["reading"]
    assert reading(6)["verdict"] == CONSISTENT

    def whole(n):
        tests = pd.DataFrame(
            [{"table": "DINO", "condition": "a", "status": "tested", "repeats": n}]
        ).assign(same_way=n, q_repeats=0.4)
        return whole_headline(SimpleNamespace(tests=tests))[1]

    assert whole(5) == "no whole-table shift significant across repeats"
    assert whole(6).endswith("a shifted the same way in every repeat")

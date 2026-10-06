"""The whole-table test (owner, 2026-10-06): a table's centre shifted the same way in every
biological repeat, the repeat-level verdict calibrated, the exploratory shuffle in the CSV only, a
Benjamini-Hochberg family per table and sub-experiment, and the words of question 9."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from morphocoverage.sampling_unit.whole_table import (
    projections,
    repeat_test,
    shuffle_test,
    whole_table_tests,
)
from morphocoverage.workflows import Setup
from morphocoverage.workflows._whole_table import (
    WHOLE_CSV,
    table_pairs,
    whole_headline,
    whole_table_html,
    whole_table_lines,
)

KEY = ["condition", "biological_repeat", "image_file"]


def _images(
    repeats: int = 4,
    images: int = 5,
    columns: int = 20,
    shift: float = 0.0,
    conditions: tuple[str, ...] = ("ctrl", "a", "b"),
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """One image per field; condition ``a`` shifted by ``shift`` on the first column."""
    rng = np.random.default_rng(seed)
    rows = [
        {"condition": c, "biological_repeat": f"R{r}", "image_file": f"{c}_{r}_{i}.tif"}
        for c in conditions
        for r in range(repeats)
        for i in range(images)
    ]
    fovs = pd.DataFrame(rows)
    values = rng.normal(size=(len(fovs), columns))
    values[fovs["condition"].eq("a").to_numpy(), 0] += shift
    names = [f"d{j}" for j in range(columns)]
    objects = pd.concat([fovs, pd.DataFrame(values, columns=names)], axis=1)
    return objects, fovs, names


def _tests(objects, fovs, names, pairs=(("a", "ctrl"), ("b", "ctrl")), **kwargs):
    return whole_table_tests(
        objects,
        fovs,
        tables={"DINO": names},
        pairs=list(pairs),
        fov_col="image_file",
        observation_unit="image",
        **kwargs,
    )


def test_projections_read_each_repeat_on_the_others_direction():
    deltas = np.array([[1.0, 0.0], [2.0, 0.0], [0.0, 3.0]])
    s = projections(deltas)
    # Repeat 1 against (2, 3) normalised: 1 * 2 / sqrt(13).
    assert np.isclose(s[0], 2 / np.sqrt(13))
    assert np.isclose(s[2], 0.0)


def test_the_t_is_corrected_for_the_repeats_the_projections_share():
    rng = np.random.default_rng(1)
    deltas = rng.normal(size=(5, 30)) + 0.4
    same, _mean, t, p = repeat_test(deltas)
    s = projections(deltas)
    plain = s.mean() / (s.std(ddof=1) / np.sqrt(5))
    assert np.isclose(t, plain * np.sqrt(3 / 8))
    assert np.isclose(p, stats.t.sf(t, 4)) and same == int((s > 0).sum())


def test_fewer_than_three_repeats_are_not_tested():
    objects, fovs, names = _images(repeats=2)
    tests = _tests(objects, fovs, names).tests
    assert set(tests["status"]) == {"not tested: 2 repeats"}
    assert tests["p_repeats"].isna().all() and tests["p_shuffle"].notna().all()
    _status, answer = whole_headline(_tests(objects, fovs, names))
    assert answer.startswith("no whole-table test: at most 2 repeats hold a condition")


def test_a_shift_in_every_repeat_is_found_and_said():
    objects, fovs, names = _images(repeats=5, shift=2.0)
    result = _tests(objects, fovs, names)
    tests = result.tests.set_index("condition")
    assert tests.loc["a", "same_way"] == 5 and tests.loc["a", "q_repeats"] <= 0.05
    assert tests.loc["a", "p_shuffle"] <= 0.05 and tests.loc["a", "effect"] > 0.3
    assert tests.loc["b", "q_repeats"] > 0.05
    lines = " ".join(" ".join(whole_table_lines(result)).split())
    assert "DINO (20 columns):" in lines
    assert "- a vs ctrl: shifted the same way in 5 of 5 repeats, p " in lines
    assert "across repeats; the centres " in lines and "p_shuffle" not in lines
    assert "exploratory p with the labels of the fields shuffled" in lines
    assert "never part of the verdict" in lines
    _status, answer = whole_headline(result)
    assert answer.startswith("the whole DINO table shifted across repeats in a (q ")


def test_the_shuffle_p_permutes_labels_within_repeats_only():
    rng = np.random.default_rng(0)
    means = rng.normal(size=(12, 4))
    side = np.tile([1, 1, 0], 4)
    repeat = np.repeat(np.arange(4), 3)
    first = shuffle_test(means, side, repeat, 199, np.random.default_rng(5))
    again = shuffle_test(means, side, repeat, 199, np.random.default_rng(5))
    assert first == again and 1 / 200 <= first[1] <= 1


def test_each_table_is_its_own_family_and_never_joins_the_features():
    objects, fovs, names = _images(repeats=4, shift=1.0)
    objects = objects.assign(**{f"v{j}": objects[f"d{j}"] * 2 for j in range(5)})
    result = whole_table_tests(
        objects,
        fovs,
        tables={"DINO": names, "Other": [f"v{j}" for j in range(5)]},
        pairs=[("a", "ctrl"), ("b", "ctrl")],
        fov_col="image_file",
        observation_unit="image",
    )
    tests = result.tests
    for _, rows in tests.groupby("table"):
        expected = stats.false_discovery_control(rows["p_repeats"].to_numpy())
        np.testing.assert_allclose(rows["q_repeats"], expected)


def test_missing_values_leave_their_observations_out_counted():
    objects, fovs, names = _images(repeats=3)
    objects.loc[0, "d3"] = np.nan
    result = _tests(objects, fovs, names)
    assert result.settings["left_out"] == {"DINO": 1}
    assert "1 observation with a missing value left out" in " ".join(whole_table_lines(result))


def test_the_page_carries_the_tests_as_csv(tmp_path):
    objects, fovs, names = _images(repeats=3)
    result = _tests(objects, fovs, names)
    markup = whole_table_html(result, tmp_path, lambda text: f"<pre>{text}</pre>", lambda f: "")
    assert "<h3>The whole tables</h3>" in markup
    saved = pd.read_csv(tmp_path / WHOLE_CSV)
    assert list(saved["condition"]) == ["a", "b"] and "q_repeats" in saved


def test_the_comparisons_are_each_condition_against_its_own_control():
    _objects, fovs, _names = _images(repeats=3, conditions=("ctrl", "a", "ctrl2", "b"))
    one = Setup(readout=None, control={"condition": "ctrl"})
    assert table_pairs(one, fovs) == [("a", "ctrl"), ("ctrl2", "ctrl"), ("b", "ctrl")]
    several = Setup(readout=None, control={"condition": {"ctrl": ["a"], "ctrl2": ["b"]}})
    assert sorted(table_pairs(several, fovs)) == [("a", "ctrl"), ("b", "ctrl2")]


@pytest.mark.parametrize("repeats", [3, 5])
def test_the_repeat_level_test_keeps_its_false_positives_near_five_percent(repeats):
    # A short version of data/simulations/whole_table.py: no effect, a batch per well.
    rng = np.random.default_rng(repeats)
    rejected, plain = 0, 0
    for _ in range(300):
        deltas = rng.normal(size=(repeats, 50)) * (1 / np.arange(1, 51)) ** 0.35
        rejected += repeat_test(deltas)[3] <= 0.05
        s = projections(deltas)
        plain += stats.ttest_1samp(s, 0, alternative="greater").pvalue <= 0.05
    assert rejected / 300 <= 0.08
    assert plain / 300 > rejected / 300  # the plain paired t is anti-conservative here


def _wells(wells: int, images: int = 4, batch: float = 0.0, seed: int = 0):
    """``wells`` technical repeats a side and repeat, each shifted by ``batch`` SDs."""
    rng = np.random.default_rng(seed)
    rows = [
        {
            "condition": c,
            "biological_repeat": f"R{r}",
            "technical_repeat": f"W{w}",
            "image_file": f"{c}_{r}_{w}_{i}.tif",
        }
        for c in ("ctrl", "a")
        for r in range(3)
        for w in range(wells)
        for i in range(images)
    ]
    fovs = pd.DataFrame(rows)
    names = [f"d{j}" for j in range(6)]
    well = fovs.groupby(["condition", "biological_repeat", "technical_repeat"]).ngroup()
    shifts = rng.normal(size=(well.max() + 1, 6)) * batch
    values = rng.normal(size=(len(fovs), 6)) + shifts[well.to_numpy()]
    objects = pd.concat([fovs, pd.DataFrame(values, columns=names)], axis=1)
    return objects, fovs, names


def test_named_wells_are_shuffled_not_fields():
    objects, fovs, names = _wells(wells=3)
    tests = _tests(objects, fovs, names, pairs=(("a", "ctrl"),)).tests
    assert tests.loc[0, "shuffled"] == "technical_repeat" and tests.loc[0, "units"] == 18
    assert tests.loc[0, "fields"] == 72 and np.isfinite(tests.loc[0, "p_shuffle"])


def test_one_well_a_side_gives_no_shuffle_p():
    objects, fovs, names = _wells(wells=1)
    tests = _tests(objects, fovs, names, pairs=(("a", "ctrl"),)).tests
    # One well a side in 3 repeats: 2 ** 3 = 8 assignments, too few for a p.
    assert tests.loc[0, "units"] == 6 and np.isnan(tests.loc[0, "p_shuffle"])


def test_shuffling_named_wells_holds_its_false_positives_under_a_well_batch():
    rejected = {"wells": 0, "fields": 0}
    for seed in range(40):
        objects, fovs, names = _wells(wells=3, images=5, batch=1.0, seed=seed)
        kwargs = {"pairs": (("a", "ctrl"),), "permutations": 199}
        p = _tests(objects, fovs, names, **kwargs).tests.loc[0, "p_shuffle"]
        rejected["wells"] += p <= 0.05
        unnamed = fovs.drop(columns="technical_repeat")
        objects = objects.drop(columns="technical_repeat")
        p = _tests(objects, unnamed, names, **kwargs).tests.loc[0, "p_shuffle"]
        rejected["fields"] += p <= 0.05
    assert rejected["wells"] <= 6 and rejected["fields"] >= 3 * max(rejected["wells"], 2)

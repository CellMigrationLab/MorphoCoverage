"""Regressions of the 2026-09-26 function audit: the report and the sampling-unit checks (G1-G14).

Each test names the finding it pins (the first function audit, docs/audit/README.md)
and fails on the code before the fix.
"""

import numpy as np
import pandas as pd
from test_sampling_unit import CONTRAST_GROUPS, _technical, _two_conditions

from morphocoverage.retrospective import microscopist_report
from morphocoverage.sampling_unit import (
    condition_contrast,
    descriptor_spread_check,
    feature_ranking,
    fov_clustering,
    fov_variability,
    unusual_objects,
)


def test_g1_fields_without_within_field_variation_are_explained_not_a_crash():
    objects = pd.DataFrame(
        {"position_id": ["a", "a", "b", "b", "c", "c"], "x": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]}
    )
    result = fov_variability(objects, pd.DataFrame({"position_id": ["a", "b", "c"]}), feature="x")
    assert "do not vary within any field" in result.summary()
    report = microscopist_report(variability=result)
    assert "design effect" in report.sections["unit"] or "design effect" in report.summary()


def test_g2_about_five_percent_of_random_fields_are_unusual_even_with_few_fields():
    rng = np.random.default_rng(1)
    shares = []
    for _ in range(300):
        objects = pd.DataFrame(
            {"image_file": np.repeat([f"f{j}" for j in range(3)], 40), "x": rng.normal(size=120)}
        )
        fovs = pd.DataFrame({"image_file": [f"f{j}" for j in range(3)]})
        shares.append(fov_variability(objects, fovs, feature="x").groups["unusual_fov_share"][0])
    # 1/sqrt(n_j) alone gave about 1.6% with three fields.
    assert 0.035 < np.mean(shares) < 0.065


COMMON = {
    "feature": "x",
    "control": {"condition": "control"},
    "group_cols": CONTRAST_GROUPS,
    "n_permutations": 100,
    "n_draws": 10,
    "object_counts": (50,),
}


def test_g3_bonferroni_counts_the_contrasts_tested():
    # Five wells a side, so the test within a repeat can reach p <= 0.05 and is run.
    objects, fovs = _two_conditions(shift=1.0, field_sd=0.3, n_wells=5)
    kept = fovs[~((fovs["biological_repeat"] == "R2") & (fovs["condition"] == "control"))]
    result = condition_contrast(
        objects[objects["image_file"].isin(kept["image_file"])], kept, **COMMON
    )
    assert "Bonferroni over 1 comparison:" in " ".join(result.summary(detail="full").split())
    assert "Bonferroni-corrected over 1 comparison." in " ".join(result.summary().split())


def test_g4_the_jcs_line_is_kept_without_a_repeat_column():
    objects, fovs = _two_conditions(shift=1.0, field_sd=0.3)
    one = objects[objects["biological_repeat"] == "R1"].drop(columns="biological_repeat")
    one_fovs = fovs[fovs["biological_repeat"] == "R1"].drop(columns="biological_repeat")
    result = condition_contrast(one, one_fovs, **{**COMMON, "group_cols": ["condition", "well"]})
    assert "JCS effect size" in result.summary()


def _constant_field_means():
    rows, fields = [], []
    for condition, mean in (("control", 10.0), ("treated", 12.0)):
        for j in range(6):
            name = f"{condition}_{j}"
            fields.append({"condition": condition, "image_file": name})
            rows += [
                {"condition": condition, "image_file": name, "x": v} for v in (mean - 1, mean + 1)
            ]
    return pd.DataFrame(rows), pd.DataFrame(fields)


def test_g5_an_undefined_field_d_gets_no_p_value():
    objects, fovs = _constant_field_means()
    # Every field holds 9 and 11, so every field mean is 10 on both sides (a perfect
    # separation, treated means 11 against 10, is an infinite d instead: H-6).
    above = objects["x"] > objects.groupby("condition")["x"].transform("mean")
    objects["x"] = np.where(above, 11.0, 9.0)
    result = condition_contrast(
        objects,
        fovs,
        feature="x",
        control={"condition": "control"},
        group_cols=["condition"],
        n_permutations=99,
        n_draws=5,
        object_counts=(4,),
        fields_independent=True,
    )
    row = result.table.iloc[0]
    assert np.isnan(row["p_value"]) and "do not vary" in row["status"]


def test_g6_a_perfect_separation_ranks_first_at_the_cap():
    objects, fovs = _constant_field_means()
    result = feature_ranking(
        objects,
        fovs,
        features=["x"],
        control={"condition": "control"},
        group_cols=["condition"],
        n_permutations=199,
        fields_independent=True,
    )
    assert result.tests["status"].iloc[0] == "ok"
    assert result.ranking["score"].iloc[0] == 20.0 and result.top() == ["x"]


def test_g7_the_gate_sentence_follows_the_floor():
    objects, fovs = _technical(
        *_two_conditions(shift=2.0, field_sd=0.3, seed=7, n_wells=4, n_fields=6)
    )
    text = feature_ranking(
        objects, fovs, features=["y", "x"], control={"condition": "control"}, n_permutations=400
    ).summary()
    # 70 assignments, 4 wells a side: the split and its mirror give a floor of 2/70,
    # under the gate; corrected over the 4 tests a lone contrast there gets q = 0.11
    # (audit 4 D4), which the sentence now says.
    assert "cannot fall below 0.029;" in text and "cannot pass the gate" not in text
    assert "a lone contrast at the floor cannot pass" in text


def test_g8_a_group_without_labels_is_not_said_to_lack_the_state():
    rng = np.random.default_rng(0)
    rows, fields = [], []
    for condition in ("MOCK", "KD"):
        for j in range(8):
            name = f"{condition}_{j}"
            fields.append({"condition": condition, "image_file": name})
            rows += [
                {"condition": condition, "image_file": name, "a": rng.normal(), "b": rng.normal()}
                for _ in range(30)
            ]
    objects, fovs = pd.DataFrame(rows), pd.DataFrame(fields)
    unusual = unusual_objects(
        objects,
        fovs,
        feature_cols=["a", "b"],
        reference={"condition": "MOCK"},
        group_cols=["condition"],
    )
    text = fov_clustering(
        objects.assign(unusual=unusual.labels),
        fovs,
        label_col="unusual",
        states=[True],
        group_cols=["condition"],
    ).summary()
    text = " ".join(text.split())
    assert "Not looked for in MOCK (none of the cells of its 8 fields is labelled)" in text
    assert "Not seen in KD (240 labelled cells in 8 fields)" in text


def test_g11_g12_spread_ratios_and_small_groups_are_stated():
    rng = np.random.default_rng(0)
    objects = pd.DataFrame(
        {
            "image_file": np.repeat([f"f{j}" for j in range(20)], 30),
            "x": rng.integers(0, 3, 600).astype(float),
            "condition": np.repeat(["A"] * 19 + ["B"], 30),
        }
    )
    fovs = objects[["image_file", "condition"]].drop_duplicates()
    text = descriptor_spread_check(
        objects, fovs, feature="x", threshold=200, group_cols=["condition"]
    ).summary()
    text = " ".join(text.split())
    assert "nanx" not in text and "no ratio" in text
    assert "the random-object samples never varied" in text
    assert "JCS check in B: not checked: 30 cells, fewer than the threshold of 200." in text


def test_g9_g10_g14_the_report_marks_questions_by_what_answers_them():
    redundancy = pd.DataFrame(
        {"redundancy_distance": [520.0], "smallest_lag": [336.0], "unit": "µm"}
    )
    report = microscopist_report(redundancy=redundancy)
    assert report.inputs["order"] == "not assessed"
    assert "8. Could I" in report.limits()
    empty = microscopist_report()
    assert "any of the analyses below" not in empty.summary()
    assert "adequacy_points" not in empty.sections["enough"]

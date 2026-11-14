"""Tests of ``sampling_unit.variability``: how variable the fields are; what object counts miss."""

from __future__ import annotations

import numpy as np
import pandas as pd
from _helpers_sampling_unit import _well_tables
from _helpers_tables import _one_condition_tables

from morphocoverage._acquisition import check_one_acquisition
from morphocoverage._style import width
from morphocoverage._wording import percent
from morphocoverage.retrospective.report import microscopist_report
from morphocoverage.sampling_unit import fov_variability, variability


def test_results_of_one_acquisition_with_other_settings_compose():
    objects, fovs = _one_condition_tables()
    first = fov_variability(objects, fovs, feature="x")
    again = fov_variability(objects.assign(x=objects["x"] + 5), fovs, feature="x")
    check_one_acquisition({"a": first, "b": again})
    assert microscopist_report(variability=first) is not None


def test_h16_the_variability_summary_states_missing_values():
    objects, fovs = _well_tables([("ctrl", "r1", "w1", 10, 0, 0.5)], per_field=20)
    objects.loc[np.random.default_rng(0).random(len(objects)) < 0.4, "x"] = np.nan
    missing = int(objects["x"].isna().sum())
    result = fov_variability(objects, fovs, feature="x")
    assert f"Cells with no value were left out: {missing}." in result.summary()
    full = result.summary(detail="full")
    assert "no value, left out" in full and f" {missing} " in full
    complete = fov_variability(objects.dropna(), fovs, feature="x")
    assert "left out" not in complete.summary()
    assert "left out" not in complete.summary(detail="full")


def test_h13_a_sparse_group_does_not_read_zero_per_field():
    rng = np.random.default_rng(0)
    fovs = pd.DataFrame({"position_id": [f"f{j}" for j in range(50)]})
    rows = [{"position_id": f"f{j}", "x": v} for j in range(10) for v in rng.normal(j, 1, 2)]
    text = fov_variability(pd.DataFrame(rows), fovs, feature="x").summary(detail="full")
    row = next(line for line in text.splitlines() if line.startswith("  all fields (pooled)"))
    assert row.split()[5] == "0.4"  # fields, objects, then per field


def test_h18_the_variance_components_are_cited_where_they_live():
    assert "statistics.nested_variance_components" in variability.__doc__
    assert "retrospective.effort.nested_variance_components" not in variability.__doc__


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


def test_the_field_variability_short_is_one_answer_and_one_table(foundation_checks):
    text = foundation_checks["variability"].summary()
    assert text.startswith("How variable are the fields (x)? A mean from whole fields has")
    assert "design effect" in text and "KD · repeat r1" in text
    assert all(width(line) <= 100 for line in text.splitlines())


def test_the_field_variability_full_keeps_every_number(foundation_checks):
    result = foundation_checks["variability"]
    text = result.summary(detail="full")
    for row in result.groups.to_dict("records"):
        for value in (
            str(int(row["fovs"])),
            str(int(row["observations"])),
            percent(row["icc"]),
            f"{row['design_effect']:.1f}",
            f"{row['error_ratio']:.1f}",
            percent(row["unusual_fov_share"]),
        ):
            assert value in text

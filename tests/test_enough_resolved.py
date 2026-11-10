"""Question 2 with a control: do the fields acquired resolve the difference from the control?"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from _helpers_sampling_unit import _technical, _two_conditions

from morphocoverage._style import CAUTION, Status
from morphocoverage.retrospective.report_enough import enough_short
from morphocoverage.retrospective.report_resolved import resolved_full
from morphocoverage.retrospective.report_status import enough_status
from morphocoverage.sampling_unit.contrast import condition_contrast
from morphocoverage.sampling_unit.contrast_sampling import (
    PRACTICAL,
    projected_fields,
    sampling_row,
)
from morphocoverage.sampling_unit.repeat_plan import repeat_plan
from morphocoverage.sampling_unit.repeat_plan_text import plan_parts
from morphocoverage.statistics import cohen_d
from morphocoverage.workflows._html_sections import _resolution

LAB = Path(__file__).resolve().parents[1] / "data" / "lab"
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import catalog


def _fields(rng, n_fields, mean, per_field=5):
    return [[rng.normal(mean, 1.0, per_field) for _ in range(n_fields)]]


def test_a_large_difference_is_resolved_and_none_is_too_small():
    rng = np.random.default_rng(0)
    t, c = _fields(rng, 20, 2.0), _fields(rng, 20, 0.0)
    d = float(np.mean(np.concatenate(t[0])) - np.mean(np.concatenate(c[0])))
    big = sampling_row(t, c, d, np.random.default_rng(1))
    assert big["resolved"] and big["d_low"] > 0 and np.isnan(big["fields_to_resolve"])
    # No difference: the interval spans 0, and the projected fields are far beyond practical.
    same = sampling_row(t, _fields(rng, 20, 2.0), 0.01, np.random.default_rng(1))
    assert not same["resolved"] and same["too_small"]
    assert same["fields_to_resolve"] > PRACTICAL * 20


def test_a_moderate_difference_gets_the_fields_that_would_resolve_it():
    rng = np.random.default_rng(3)
    t, c = _fields(rng, 10, 0.8, per_field=1), _fields(rng, 10, 0.0, per_field=1)
    d = cohen_d(np.concatenate(t[0]), np.concatenate(c[0]))  # 0.65 in this draw
    row = sampling_row(t, c, d, np.random.default_rng(2))
    assert not row["resolved"] and not row["too_small"]
    assert 10 < row["fields_to_resolve"] <= PRACTICAL * 10


def test_the_fields_without_observations_are_drawn_and_counted():
    """The bootstrap drew only the fields with observations and the projection counted only
    them, so a side where half the fields acquired were empty was told half the fields it
    would acquire (#289). Empty fields are now drawn too (owner, 2026-10-05): the interval
    carries the field-to-field variation in cell counts, and the projection counts every
    field acquired. Two cells a field: with one, the interval is the exact noncentral t one
    (audit 10, owner, 2026-10-06), which reads the cells, not the fields."""
    rng = np.random.default_rng(3)
    t, c = _fields(rng, 10, 0.6, per_field=2), _fields(rng, 10, 0.0, per_field=2)
    d = cohen_d(np.concatenate(t[0]), np.concatenate(c[0]))
    row = sampling_row(t, c, d, np.random.default_rng(2))
    empty = [np.empty(0)] * 10
    halved = sampling_row([t[0] + empty], [c[0] + empty], d, np.random.default_rng(2))
    # Half the draws' fields hold no cell: fewer cells a draw, a wider interval.
    assert halved["d_high"] - halved["d_low"] > row["d_high"] - row["d_low"]
    assert halved["fields_to_resolve"] > 2 * row["fields_to_resolve"]
    # Without empty fields, the draws are those of before: the interval is unchanged.
    again = sampling_row(t, c, d, np.random.default_rng(2))
    assert (again["d_low"], again["d_high"]) == (row["d_low"], row["d_high"])


def _wells(rng, n_wells, n_fields, mean, well_sd, field_sd, cells=10):
    shifts = rng.normal(0, well_sd, n_wells)
    return [
        [rng.normal(mean + s + rng.normal(0, field_sd), 1.0, cells) for _ in range(n_fields)]
        for s in shifts
    ], shifts


def _d(t, c):
    return cohen_d(
        np.concatenate([f for u in t for f in u]), np.concatenate([f for u in c for f in u])
    )


def _half(row):
    return (row["d_high"] - row["d_low"]) / 2


@pytest.mark.parametrize(
    ("well_sd", "field_sd", "between"), [(0.0, 0.5, (0, 0.3)), (0.5, 0.0, (0.6, 1.0))]
)
def test_the_projection_shrinks_the_part_within_wells_only(well_sd, field_sd, between):
    """#96: the interval from wells and their fields resampled holds a part between wells,
    which more fields of the same wells do not reduce. The projection takes it from a nested
    bootstrap (owner, 2026-10-05), and its half-width at 4 times the fields matches the
    interval of the same wells imaged with 4 times the fields; the whole interval shrinking
    as 1/sqrt(fields), as before, was far too narrow when the wells differ."""
    predicted, actual, shares, old = [], [], [], []
    for seed in range(4):
        rng = np.random.default_rng(seed)
        (t, t_shift), (c, c_shift) = (
            _wells(rng, 3, 6, 0.3, well_sd, field_sd),
            _wells(rng, 3, 6, 0.0, well_sd, field_sd),
        )
        row = sampling_row(t, c, _d(t, c), np.random.default_rng(1))
        share = row["between_share"]
        shares.append(share)
        predicted.append(_half(row) * np.sqrt(share + (1 - share) / 4))
        old.append(_half(row) / 2)
        # The same wells (their shifts), 4 times the fields.
        more_t = [
            [rng.normal(0.3 + s + rng.normal(0, field_sd), 1, 10) for _ in range(24)]
            for s in t_shift
        ]
        more_c = [
            [rng.normal(s + rng.normal(0, field_sd), 1, 10) for _ in range(24)] for s in c_shift
        ]
        actual.append(
            _half(sampling_row(more_t, more_c, _d(more_t, more_c), np.random.default_rng(1)))
        )
    assert between[0] <= np.mean(shares) <= between[1]
    assert np.mean(predicted) == pytest.approx(np.mean(actual), rel=0.2)
    if well_sd:
        assert np.mean(old) < 0.7 * np.mean(actual)


def test_a_difference_the_wells_alone_blur_needs_more_wells_not_fields():
    """When the part of the interval between wells alone is as wide as the difference, more
    fields will not resolve it: it is said plainly, with no field count (#96)."""
    rng = np.random.default_rng(0)
    # Wells 1 apart on each side, the sides 0.2 apart: 8 fields a well, no field effect.
    t = [[rng.normal(m, 1.0, 10) for _ in range(8)] for m in (-0.8, 0.2, 1.2)]
    c = [[rng.normal(m, 1.0, 10) for _ in range(8)] for m in (-1.0, 0.0, 1.0)]
    row = sampling_row(t, c, _d(t, c), np.random.default_rng(1))
    assert not row["resolved"] and row["between_wells"] and not row["too_small"]
    assert np.isnan(row["fields_to_resolve"]) and row["between_share"] > 0.5
    needed, between = projected_fields(24, _d(t, c), row["d_low"], row["d_high"], 0.0)
    assert not between and np.isfinite(needed)  # the old projection gave a count


def test_one_well_a_side_projects_from_its_fields_as_before():
    """With one well a side the interval holds no variation between wells: share 0, and the
    projection is that of the whole interval."""
    rng = np.random.default_rng(3)
    t, c = _fields(rng, 10, 0.8, per_field=1), _fields(rng, 10, 0.0, per_field=1)
    d = cohen_d(np.concatenate(t[0]), np.concatenate(c[0]))
    row = sampling_row(t, c, d, np.random.default_rng(2))
    assert row["between_share"] == 0 and not row["between_wells"]
    old = np.ceil(10 * (_half(row) / abs(d)) ** 2)
    assert row["fields_to_resolve"] == old


def test_too_few_fields_give_no_answer():
    rng = np.random.default_rng(0)
    row = sampling_row(_fields(rng, 1, 2.0), _fields(rng, 5, 0.0), 2.0, rng)
    assert np.isnan(row["d_low"]) and not row["resolved"]


def test_question_2_leads_with_the_resolution_when_a_control_is_declared():
    data = catalog.load("cct8_cdm_morphology")
    contrast = condition_contrast(
        *data.tables, feature="Area", control=data.control, observation_unit="cell"
    )
    status, headline = enough_status(None, None, contrast)
    assert status is Status.CAUTION
    assert headline.startswith(
        "to compare with the control, within single repeats: shown in 2 of 6 comparisons"
    )
    # One cell per field: the advice counts cells.
    assert "would need about" in headline and "cells a side" in headline
    short = " ".join(enough_short(None, None, None, contrast=contrast).split())
    assert "Within each repeat, does the difference from the control show?" in short
    # The exact noncentral t interval of one cell per field (audit 10, owner, 2026-10-06)
    # replaced the bootstrap's: about 30 cells a side, 42 before.
    assert "about 30 cells a side" in short


def test_every_difference_resolved_is_ok_and_the_precision_cautions_stay_in_the_full_form():
    objects, fovs = _technical(*_two_conditions(shift=2.0, field_sd=0.3, n_wells=1))
    contrast = condition_contrast(
        objects, fovs, feature="x", control={"condition": "control"}, n_draws=5
    )
    assert contrast.table["resolved"].all()
    status, headline = enough_status(None, None, contrast)
    assert status is Status.OK
    assert headline == (
        "to compare with the control, within single repeats: shown in 2 of 2 comparisons"
    )
    assert CAUTION not in enough_short(None, None, None, contrast=contrast)


def test_a_comparison_without_an_interval_stays_in_the_count():
    """#252: one repeat's treated well imaged in one field has no interval; it is still a
    comparison, said not assessed, never dropped from the denominator."""
    objects, fovs = _technical(*_two_conditions(shift=2.0, field_sd=0.3, n_wells=1))
    one = (fovs["condition"] == "treated") & (fovs["biological_repeat"] == "R2")
    fovs = fovs[~one | fovs["image_file"].str.endswith("_f0")]
    objects = objects[objects["image_file"].isin(fovs["image_file"])]
    contrast = condition_contrast(
        objects, fovs, feature="x", control={"condition": "control"}, n_draws=5
    )
    status, headline = enough_status(None, None, contrast)
    assert status is Status.CAUTION
    assert "shown in 1 of 2 comparisons" in headline
    # Fewer than 5 fields a side: not assessable (audit 10, owner, 2026-10-06).
    assert "1 not assessable (fewer than 5 fields a side)" in headline
    full = " ".join(resolved_full(contrast).split())
    assert "repeat R2 1 vs 12 +1.70 not assessable: fewer than 5 fields a side" in full
    page = " ".join(_resolution(contrast))
    assert "Partly" in page and "in 1 of 2 comparisons" in page and "not assessable" in page


@pytest.mark.parametrize("dataset_id", ["cct8_filopodia_screen"])
def test_the_screens_hits_are_resolved_in_every_repeat(dataset_id):
    data = catalog.load(dataset_id)
    contrast = condition_contrast(
        *data.tables, feature="Filopodia number", control=data.control, observation_unit="cell"
    )
    table = contrast.table
    hits = table["condition"].str.startswith(("siCCT8", "siCSNK1A1", "siSCRIB"))
    assert table.loc[hits, "resolved"].all()


def _blurred_by_wells():
    """Two repeats, three wells a condition 1 apart, the treated wells 0.2 higher."""
    rng = np.random.default_rng(5)
    objects, fovs = [], []
    for repeat in ("R1", "R2"):
        for condition, shift in (("control", 0.0), ("treated", 0.2)):
            for w, offset in enumerate((-1.0, 0.0, 1.0)):
                for f in range(8):
                    name = f"{repeat}_{condition}_w{w}_f{f}"
                    meta = {
                        "condition": condition,
                        "biological_repeat": repeat,
                        "repeat_group": repeat,
                        "technical_repeat": f"w{w}",
                        "image_file": name,
                    }
                    fovs.append(meta)
                    objects += [{**meta, "x": v} for v in rng.normal(shift + offset, 1, 10)]
    return pd.DataFrame(objects), pd.DataFrame(fovs)


def test_the_report_says_plainly_when_more_wells_not_fields_would_resolve_it():
    contrast = condition_contrast(
        *_blurred_by_wells(), feature="x", control={"condition": "control"}, n_draws=5
    )
    table = contrast.table
    assert table["between_wells"].all() and table["fields_to_resolve"].isna().all()
    _, headline = enough_status(None, None, contrast)
    assert "2 that more fields will not resolve: more wells or biological repeats" in headline
    assert "would need about" not in headline
    full = " ".join(resolved_full(contrast).split())
    assert "no, more fields will not resolve it: more wells or biological repeats" in full
    page = " ".join(_resolution(contrast))
    assert "In 2 comparisons more fields will not resolve it: more wells or biological" in page
    # The plan's fields a side per repeat take the same projection.
    plan = " ".join(" ".join(plan_parts(contrast, full=True)[0]).split())
    assert plan.endswith("more wells or biological repeats")


def test_twice_the_fields_do_not_reduce_the_repeats_needed_when_the_wells_blur_it():
    """#96: the plan halved the whole sampling error of each repeat with twice the fields,
    so a wells-limited difference read "more wells or biological repeats" for the fields
    and 29 repeats with twice the fields. Only the part within wells halves now: the two
    repeat counts agree."""
    contrast = condition_contrast(
        *_blurred_by_wells(), feature="x", control={"condition": "control"}, n_draws=5
    )
    [row] = contrast.plan.to_dict("records")
    assert np.isinf(row["fields_per_repeat"])
    assert row["repeats_needed_twice_fields"] == row["repeats_needed"]
    # Without a part between wells, twice the fields halve the whole error, as before.
    table = contrast.table.assign(between_share=0.0)
    [old] = repeat_plan(table).to_dict("records")
    assert old["repeats_needed_twice_fields"] < old["repeats_needed"]
    # All between wells: twice the fields change nothing, whatever the counts.
    [same] = repeat_plan(contrast.table.assign(between_share=1.0)).to_dict("records")
    assert same["repeats_needed_twice_fields"] == same["repeats_needed"]

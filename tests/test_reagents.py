"""Independent reagents for one target: do they agree across the repeats? (question 9)."""

import sys
from pathlib import Path

import pandas as pd
import pytest

from morphocoverage.retrospective.report_status import conditions_status
from morphocoverage.sampling_unit.contrast import condition_contrast
from morphocoverage.sampling_unit.contrast_repeats import reagent_agreement

LAB = Path(__file__).resolve().parents[1] / "data" / "lab"
sys.path.insert(0, str(LAB))
import catalog


def _table(signs):
    """Per-repeat means of two reagents of KD, against a control at 10."""
    rows = []
    for repeat, pair in enumerate(signs):
        for reagent, sign in zip(("siKD#1", "siKD#2"), pair, strict=True):
            rows.append(
                {
                    "biological_repeat": f"r{repeat}",
                    "condition": reagent,
                    "mean": 10 + sign,
                    "control_mean": 10.0,
                }
            )
    return pd.DataFrame(rows)


TARGETS = {"siKD#1": "KD", "siKD#2": "KD"}


def test_reagents_that_go_the_same_way_in_every_repeat_agree():
    [row] = reagent_agreement(_table([(-1, -2), (-1, -1), (-3, -1)]), TARGETS).to_dict("records")
    assert (row["repeats"], row["agree"], row["direction"]) == (3, 3, "lower")
    assert row["reagents"] == ["siKD#1", "siKD#2"]


def test_disagreement_and_flipping_sides_are_counted():
    [split] = reagent_agreement(_table([(-1, 1), (-1, -1), (1, 1)]), TARGETS).to_dict("records")
    assert (split["agree"], split["direction"]) == (2, "mixed")
    [none] = reagent_agreement(_table([(-1, 1), (1, -1)]), TARGETS).to_dict("records")
    assert (none["agree"], none["direction"]) == (0, "none")


def test_a_target_with_one_reagent_or_no_column_gives_nothing():
    assert reagent_agreement(_table([(-1, -1)]), {"siKD#1": "KD"}).empty
    assert reagent_agreement(_table([(-1, -1)]), {}).empty


@pytest.fixture(scope="module")
def screen():
    data = catalog.load("cct8_filopodia_screen")
    return data, condition_contrast(
        *data.tables, feature="Filopodia number", control=data.control, observation_unit="cell"
    )


def test_the_screens_reagents_agree_for_every_target(screen):
    _, contrast = screen
    targets = contrast.targets.set_index("target")
    assert set(targets.index) == {"CCT8", "CSNK1A1", "MAP4K4", "MINK1", "NF2", "SCRIB"}
    assert (targets["agree"] == 3).all()
    assert set(targets.loc[["MAP4K4", "NF2"], "direction"]) == {"higher"}
    short = contrast.summary()
    assert "Independent reagents for one target (target)" in short
    assert "all higher in 3 of 3" in short and "⚠️" not in short
    _, headline = conditions_status(contrast, None)
    assert headline.endswith(
        "; the reagents of a target go the same way in every repeat for 6 of 6 targets "
        "(described, not tested)"
    )


def test_without_the_column_the_contrast_is_unchanged(screen):
    data, contrast = screen
    objects, fovs = data.tables
    plain = condition_contrast(
        objects.drop(columns="target"),
        fovs.drop(columns="target"),
        feature="Filopodia number",
        control=data.control,
        observation_unit="cell",
    )
    assert plain.targets.empty and "Independent reagents" not in plain.summary()
    pd.testing.assert_frame_equal(plain.table, contrast.table)


def test_a_condition_with_two_targets_is_refused(screen):
    data, _ = screen
    objects, fovs = data.tables
    fovs = fovs.copy()
    fovs.loc[fovs.index[0], "target"] = "OTHER"
    with pytest.raises(ValueError, match="more than one target for condition"):
        condition_contrast(
            objects, fovs, feature="Filopodia number", control=data.control, observation_unit="cell"
        )


def test_reagents_that_share_no_repeat_are_not_compared():
    """#307: two reagents of KD in different repeats read "on one side in none of 0" and
    counted against agreement in the headline."""
    from types import SimpleNamespace

    from morphocoverage.sampling_unit.reading_text import condition_readings
    from morphocoverage.sampling_unit.reagents_text import reagents_headline, reagents_parts

    apart = _table([(-1, -1), (-1, -1)])
    apart = apart[~((apart["condition"] == "siKD#2") & (apart["biological_repeat"] == "r0"))]
    apart = apart[~((apart["condition"] == "siKD#1") & (apart["biological_repeat"] == "r1"))]
    agreeing = _table([(-1, -1), (-1, -2)]).replace({"siKD#1": "siB#1", "siKD#2": "siB#2"})
    targets = {**TARGETS, "siB#1": "B", "siB#2": "B"}
    rows = reagent_agreement(pd.concat([apart, agreeing]), targets)
    assert rows.set_index("target").loc["KD", "repeats"] == 0
    contrast = SimpleNamespace(
        targets=rows,
        settings={"repeat_col": "biological_repeat", "observation_unit": "cell"},
        repeats=pd.DataFrame(
            [{"condition": "siKD#1", "control_condition": "CTRL", "repeats": 1, "lower": 1}]
            + [{"condition": "siB#1", "control_condition": "CTRL", "repeats": 2, "lower": 2}],
        ).assign(
            higher=0, d_mean=-1.0, q_paired_t=float("nan"), q_paired_t_normalised=float("nan")
        ),
        table=pd.DataFrame(),
        cells=pd.DataFrame(),
    )
    lines, _ = reagents_parts(contrast)
    assert "not compared: no repeat holds them all" in "\n".join(lines)
    assert reagents_headline(contrast) == (
        "the reagents of a target go the same way in every repeat for 1 of 1 target "
        "(described, not tested; 1 more with no repeat holding all its reagents)"
    )
    readings = condition_readings(contrast).set_index("condition")["reading"]
    assert "siKD#2 (KD) not imaged in the same biological repeat" in readings["siKD#1"]
    assert "siB#2 (B) agrees in 2 of 2 biological repeats" in readings["siB#1"]

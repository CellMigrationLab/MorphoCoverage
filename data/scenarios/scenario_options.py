"""The scenarios of options no other dataset of the repository exercises, which
``make_scenarios.py`` writes beside the edges: declared scales, a trend over six repeats, and
the acquisition block (a field far from its condition and repeat, controls apart in
intensity), with the mistake a declared scale allows (``make_scenarios`` lists the truth's
keys).
"""

from __future__ import annotations

from make_scenarios import Design, Scenario
from scenario_axes import SEVERAL_CONTROL
from scenario_edges import CTRL, READOUT, _scenario

SIGNED = {"readout": "area", "features": ["area", "length", "velocity"], "observation_unit": "cell"}


def declared_scales() -> Scenario:
    """Scales declared against the automatic rule: the positive area on differences, the
    length on ratios; the signed velocity left to the rule (differences)."""
    return _scenario(
        "e14_declared_scales",
        "edge",
        "scales declared (area on differences, length on ratios), velocity on the automatic "
        "rule: each feature tested on its scale, the declaration said",
        Design(("ctrl", "a"), shifts={"a": 0.8}, seed=2015),
        {**SIGNED, **CTRL, "scales": {"area": "difference", "length": "ratio"}},
        {
            "results": ["contrast", "discovery"],
            "planted": {"area": ["a"], "length": ["a"]},
            "full": ["declared in the Setup: area, the difference; length, the ratio"],
        },
        columns=("area", "length", "velocity"),
    )


def trend_six_repeats() -> Scenario:
    """Six repeats, one condition shifted the same way in every repeat but by very unequal
    amounts: same direction in every repeat, not significant, the only design that says a
    trend (from 6 repeats); a strong condition beside it is significant."""
    design = Design(
        ("ctrl", "trend", "strong"),
        repeats=6,
        shifts={"strong": 0.8},
        repeat_shifts={"trend": (0.25, 0.25, 0.25, 0.25, 0.25, 2.0)},
        seed=2016,
    )
    return _scenario(
        "e15_trend_six_repeats",
        "edge",
        "a condition higher in each of 6 repeats by very unequal amounts: a trend, not "
        "significant; a strong condition beside it significant",
        design,
        {**READOUT, **CTRL},
        {
            "results": ["contrast", "discovery", "whole_table"],
            "planted": {"area": ["strong"], "length": ["strong"]},
            "trend": {"area": ["trend"], "length": ["trend"]},
            "null_features": ["ecc"],
        },
        tables={"DINO1": ("dino", None)},
    )


def acquisition_block() -> Scenario:
    """Two experiments whose controls differ more than 2-fold in intensity, 8 fields per
    condition and repeat, and one field whose areas are 4 times its neighbours': the
    acquisition block names both, as quality control."""
    design = Design(
        ("cA", "a1", "cB", "b1"),
        repeats=3,
        # The second experiment brighter: its intensity about exp(0.5 x 1.6), 2.2 times.
        shifts={"a1": 0.8, "cB": 1.6, "b1": 1.6},
        fields=8,
        cells=3,
        seed=2017,
    )
    scenario = _scenario(
        "e16_acquisition_block",
        "edge",
        "a field whose areas are 4 times those of the other fields of its condition and "
        "repeat, and two controls more than 2-fold apart in intensity: both in the acquisition "
        "block",
        design,
        {
            "readout": "area",
            "features": ["area", "length", "intensity_mean"],
            "observation_unit": "cell",
            "control": SEVERAL_CONTROL,
            "master_control": "cA",
        },
        {
            "results": ["acquisition", "contrast"],
            # Named among the fields 5 SDs or more away; the 5-SD threshold may name others.
            "full": [
                "a1 · repeat 2026_01_05 · field_03.tif (area,",
                "Controls apart in intensity: cA and cB differ",
            ],
        },
        columns=("area", "length", "intensity_mean"),
    )
    objects = scenario.objects
    field = (
        objects["condition"].eq("a1")
        & objects["biological_repeat"].eq(objects["biological_repeat"].iloc[0])
        & objects["image_file"].eq("field_03.tif")
    )
    objects.loc[field, "area"] = (objects.loc[field, "area"] * 4).round(4)
    return scenario


def ratio_on_signed() -> Scenario:
    """A ratio declared for a signed feature: refused where the values are read."""
    return _scenario(
        "m08_ratio_declared_on_signed_feature",
        "mistake",
        "a ratio declared for a signed feature (a velocity): refused, naming the value that is "
        "not positive",
        Design(("ctrl", "a"), shifts={"a": 0.8}, seed=3008),
        {**SIGNED, **CTRL, "scales": {"velocity": "ratio"}},
        {
            "outcome": "refused",
            "at": "report",
            "error": "ValueError",
            "match": "declared on the ratio scale",
        },
        columns=("area", "length", "velocity"),
    )


SCENARIOS = [declared_scales, trend_six_repeats, acquisition_block, ratio_on_signed]

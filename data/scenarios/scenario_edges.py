"""The edge and mistake scenarios ``make_scenarios.py`` writes: designs the axes do not
reach, and what users get wrong, each with what the package does now (``make_scenarios``
lists the truth's keys).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from make_scenarios import (
    Design,
    Scenario,
    acquisition,
    covariance,
    dino,
    observation_columns,
    table,
)
from scenario_axes import EXPLAINABLE, measured

# --- edge designs ----------------------------------------------------------------------

READOUT = {"readout": "area", "features": list(EXPLAINABLE), "observation_unit": "cell"}
CTRL = {"control": {"condition": "ctrl"}}


def _scenario(
    name: str,
    kind: str,
    checks: str,
    design: Design,
    setup: dict[str, Any],
    truth: dict[str, Any],
    *,
    columns: tuple[str, ...] = EXPLAINABLE,
    tables: dict[str, Any] | None = None,
) -> Scenario:
    """A scenario from its design: the observation columns, the other tables built by
    ``tables`` ({name: (columns or "dino", tag)}), and the Setup and truth as given."""
    keys, fovs, effect = acquisition(design)
    built = {}
    for offset, (table_name, (table_columns, tag)) in enumerate((tables or {}).items()):
        seed = design.seed + 10 + offset
        if table_columns == "dino":
            built[table_name] = (dino(keys, effect, seed), tag)
        elif tag == "explainable":
            built[table_name] = (measured(keys, effect, tuple(table_columns), seed), tag)
        else:
            built[table_name] = (table(keys, effect, tuple(table_columns), seed), tag)
    objects = observation_columns(keys, effect, columns, design.seed)
    return Scenario(name, kind, checks, objects, fovs, setup, {"outcome": "runs", **truth}, built)


def two_repeats() -> Scenario:
    """Two repeats: no paired test, the direction and d only; the whole table not tested."""
    design = Design(("ctrl", "a"), repeats=2, shifts={"a": 0.8}, seed=2001)
    return _scenario(
        "e01_two_repeats",
        "edge",
        "two repeats: the paired tests and the whole-table test need 3, said, nothing tested",
        design,
        {**READOUT, **CTRL},
        {
            "results": ["contrast", "discovery", "whole_table"],
            "untested_across": True,
            "untested_whole": "2 repeats",
        },
        tables={"DINO1": ("dino", None)},
    )


def control_elsewhere() -> Scenario:
    """The control imaged in other repeats than the condition: nothing is paired."""
    design = Design(
        ("ctrl", "a"),
        shifts={"a": 0.8},
        absent=frozenset({("ctrl", 2), ("ctrl", 3), ("a", 0), ("a", 1)}),
        seed=2002,
    )
    return _scenario(
        "e02_control_in_other_repeats",
        "edge",
        "a control imaged in other repeats than its condition: not compared across repeats",
        design,
        {**READOUT, **CTRL},
        {
            "results": ["contrast"],
            "absent": ["enough_features"],
            "untested_across": True,
            "untested_whole": "no repeat holds both",
            "overview": ["in every repeat of its sub-experiment: ctrl in"],
        },
        tables={"DINO1": ("dino", None)},
    )


def one_condition() -> Scenario:
    """One condition, declared the control: nothing to compare, nothing to separate."""
    design = Design(("ctrl",), seed=2003)
    return _scenario(
        "e03_one_condition",
        "edge",
        "one condition: no comparison, no separation; the rest runs",
        design,
        {**READOUT, **CTRL},
        {
            "results": ["variability", "sufficiency", "inspection"],
            "absent": ["contrast", "discovery", "separation", "whole_table"],
        },
        tables={"DINO1": ("dino", None)},
    )


def screen() -> Scenario:
    """A screen: 30 conditions and one control, 3 repeats, 3 fields each summarised in one row
    (one image per field), as screens often are."""
    conditions = ("ctrl", *(f"c{i:02d}" for i in range(1, 31)))
    design = Design(
        conditions, repeats=3, shifts={c: 0.8 for c in conditions[1:4]}, cells=1, seed=2004
    )
    return _scenario(
        "e04_screen_30_conditions",
        "edge",
        "a screen of 30 conditions against one control, the readout alone: its tests "
        "corrected over the conditions, every condition compared",
        design,
        {"readout": "area", "observation_unit": "image", **CTRL},
        {"results": ["contrast"], "absent": ["populations", "separation"], "compared": 30},
        columns=("area",),
    )


def no_shared_condition() -> Scenario:
    """No condition, the control included, is imaged in every repeat."""
    design = Design(
        ("ctrl", "a", "b"),
        shifts={"a": 0.8},
        absent=frozenset({("ctrl", 3), ("a", 0), ("b", 1)}),
        seed=2005,
    )
    return _scenario(
        "e05_no_condition_in_every_repeat",
        "edge",
        "no condition in every repeat: compared where both are; the other tables not centred",
        design,
        {**READOUT, **CTRL},
        {
            "results": ["contrast", "whole_table"],
            "untested_across": True,
            "overview": ["Batch: the other tables are not centred, as no condition is in every"],
        },
        tables={"DINO1": ("dino", None)},
    )


def parent_chain() -> Scenario:
    """Three controls in a chain of parents, each with one condition."""
    design = Design(
        ("cA", "a1", "cB", "b1", "cC", "c1"),
        repeats=3,
        shifts={"a1": 0.8, "cB": 0.2, "cC": 0.4},
        cells=4,
        seed=2006,
    )
    setup = {
        **READOUT,
        "control": {"condition": {"cA": ["a1"], "cB": ["b1"], "cC": ["c1"]}},
        "control_parents": {"cB": "cA", "cC": "cB"},
        "master_control": "cA",
    }
    return _scenario(
        "e06_chain_of_parent_controls",
        "edge",
        "controls in a chain of parents: each condition against its own control, each control "
        "against its parent",
        design,
        setup,
        {"results": ["contrast", "discovery"], "contrast_kinds": ["own_control", "vs_master_raw"]},
    )


def narrow_tables() -> Scenario:
    """An explainable table of 1 column and another table of 2."""
    design = Design(("ctrl", "a"), shifts={"a": 0.8}, fields=4, seed=2007)
    setup = {
        "readout": "area",
        "features": ["area", "length", "width"],
        "observation_unit": "cell",
        **CTRL,
        "fingerprint": [covariance(["e0", "e1"])],
        "feature_tables": {"pair": ["e0", "e1"]},
    }
    return _scenario(
        "e07_tables_of_one_and_two_columns",
        "edge",
        "a table of 1 column and one of 2: tested, the 2 columns fingerprinted; no separation "
        "line of their own",
        design,
        setup,
        {
            "results": ["contrast", "whole_table", "fingerprint_precision"],
            "separation_spaces": ["Combined", "Explainable"],
            "full": ["each of their 2 tables weighted by MFA"],
            "order_free": True,
        },
        columns=("area", "length"),
        tables={"single": (("width",), "explainable"), "pair": (("e0", "e1"), None)},
    )


def count_features() -> Scenario:
    """A count with many zeros beside the readout."""
    design = Design(("ctrl", "a"), shifts={"a": 0.8}, seed=2008)
    return _scenario(
        "e08_counts_with_many_zeros",
        "edge",
        "a count feature, mostly zeros, beside the readout: compared like any feature; two "
        "columns, so no separation score",
        design,
        {"readout": "area", "features": ["area", "count"], "observation_unit": "cell", **CTRL},
        {"results": ["contrast", "discovery"], "absent": ["separation"]},
        columns=("area", "count"),
    )


def signed_normalised() -> Scenario:
    """A signed feature (a velocity, negative on average) with normalisation."""
    design = Design(("ctrl", "a"), shifts={"a": 0.8}, seed=2009)
    return _scenario(
        "e09_signed_feature_normalised",
        "edge",
        "a signed feature divided by its control mean: refused, the mean is negative",
        design,
        {
            "readout": "area",
            "features": ["area", "velocity"],
            "observation_unit": "cell",
            **CTRL,
            "normalize": "control_mean",
        },
        {"outcome": "refused", "at": "report", "error": "ValueError", "match": "negative"},
        columns=("area", "velocity"),
    )


def z_planes() -> Scenario:
    """Each cell measured in 3 planes, split into one feature per plane first."""
    design = Design(("ctrl", "a"), shifts={"a": 0.8}, seed=2010)
    scenario = _scenario(
        "e10_features_per_z_plane",
        "edge",
        "one row per cell and plane, split per plane (planes_as_features): one feature per plane",
        design,
        {
            "readout": "area_z0",
            "features": ["area_z0", "area_z1", "area_z2"],
            "observation_unit": "cell",
            **CTRL,
        },
        {
            "results": ["contrast", "discovery"],
            "tested_features": ["area_z0", "area_z1", "area_z2"],
        },
        columns=("area",),
    )
    scenario.objects = _per_plane(scenario.objects, design.seed)
    scenario.planes = {"features": ["area"], "axes": ["z"]}
    return scenario


def _per_plane(objects: pd.DataFrame, seed: int) -> pd.DataFrame:
    """``objects`` with each row measured in planes 0, 1 and 2: the area shrinks away from
    the middle plane, with noise."""
    rng = np.random.default_rng(seed)
    planes = []
    for z, scale in enumerate((0.8, 1.0, 0.7)):
        plane = objects.assign(z=z)
        plane["area"] = (objects["area"] * scale * np.exp(rng.normal(0, 0.05, len(objects)))).round(
            4
        )
        planes.append(plane)
    return pd.concat(planes, ignore_index=True)


def unequal_fields() -> Scenario:
    """Very unequal fields per repeat: 12, 2, 6 and 3."""
    design = Design(("ctrl", "a"), shifts={"a": 0.8}, fields=(12, 2, 6, 3), seed=2011)
    return _scenario(
        "e11_unequal_fields_per_repeat",
        "edge",
        "very unequal fields per repeat: the cell-level screen also given balanced",
        design,
        {**READOUT, **CTRL},
        {"results": ["contrast", "discovery"], "balanced": ["a"]},
    )


def missing_in_one_table() -> Scenario:
    """Missing values in the embedding only."""
    scenario = _scenario(
        "e12_missing_values_in_one_table",
        "edge",
        "missing values in one table only: counted and kept, each analysis says how many it "
        "leaves out",
        Design(("ctrl", "a"), shifts={"a": 0.8}, seed=2012),
        {**READOUT, **CTRL},
        {"results": ["contrast", "whole_table"], "overview": ["missing value"], "left_out": True},
        tables={"DINO1": ("dino", None)},
    )
    frame, tag = scenario.tables["DINO1"]
    frame = frame.copy()
    frame.loc[frame.index[::9], ["d0", "d3"]] = float("nan")
    scenario.tables["DINO1"] = (frame, tag)
    return scenario


def known_effects() -> Scenario:
    """Six repeats, one strong condition and one null, a feature and an embedding."""
    # "null" would be read as a missing value by pandas: the null condition is "vehicle".
    design = Design(("ctrl", "strong", "vehicle"), repeats=6, shifts={"strong": 0.8}, seed=2014)
    return _scenario(
        "e13_known_effects_six_repeats",
        "edge",
        "a strong effect found on the features and the embedding, a null condition not",
        design,
        {**READOUT, **CTRL},
        {
            "results": ["contrast", "discovery", "whole_table"],
            "planted": {"area": ["strong"], "length": ["strong"]},
            "null_conditions": ["vehicle"],
            "null_features": ["ecc"],
            "whole_table_planted": {"DINO1": ["strong"]},
            "overview": ["medians of ctrl, strong and vehicle, the conditions in every repeat"],
        },
        tables={"DINO1": ("dino", None)},
    )


# --- user mistakes ---------------------------------------------------------------------


def dino_tagged_explainable() -> Scenario:
    """An embedding tagged explainable, no features declared."""
    return _scenario(
        "m01_dino_tagged_explainable",
        "mistake",
        "an embedding tagged explainable: run, its columns tested one by one, as declared",
        Design(("ctrl", "a"), repeats=3, shifts={"a": 0.8}, seed=3001),
        {"readout": "area", "observation_unit": "cell", **CTRL},
        {
            "results": ["contrast", "discovery"],
            "absent": ["whole_table"],
            "tested_features": [f"d{j}" for j in range(8)],
        },
        columns=("area",),
        tables={"DINO1": ("dino", "explainable")},
    )


def wrong_identity() -> Scenario:
    """A table whose labels start at 0 where the observation table's start at 1."""
    scenario = _scenario(
        "m02_table_keyed_by_wrong_identity",
        "mistake",
        "a table keyed by the wrong identity (labels from 0): refused, unmatched observations",
        Design(("ctrl", "a"), shifts={"a": 0.8}, seed=3002),
        {**READOUT, **CTRL},
        {"outcome": "refused", "at": "report", "error": "ValueError", "match": "cannot be used"},
        tables={"DINO1": ("dino", None)},
    )
    frame, tag = scenario.tables["DINO1"]
    scenario.tables["DINO1"] = (frame.assign(label_id=frame["label_id"] - 1), tag)
    return scenario


def unknown_tag() -> Scenario:
    """A table tagged with a misspelt tag."""
    return _scenario(
        "m03_unknown_tag",
        "mistake",
        "a misspelt tag: refused, naming the two tags",
        Design(("ctrl", "a"), shifts={"a": 0.8}, seed=3003),
        {**READOUT, **CTRL},
        {"outcome": "refused", "at": "report", "error": "ValueError", "match": "explainable"},
        tables={"fiber": (("fib_len", "fib_w"), "explanable")},
    )


def controls_without_members() -> Scenario:
    """Several controls declared with no condition under any."""
    return _scenario(
        "m04_several_controls_no_members",
        "mistake",
        "several controls with no condition under them: refused at the Setup",
        Design(("cA", "a1", "cB", "b1"), shifts={"a1": 0.8}, seed=3004),
        {**READOUT, "control": {"condition": {"cA": {}, "cB": {}}}, "master_control": "cA"},
        {
            "outcome": "refused",
            "at": "setup",
            "error": "ValueError",
            "match": "have no condition under them and no parent",
        },
    )


def normalise_without_control() -> Scenario:
    """Normalisation declared with no control."""
    return _scenario(
        "m05_normalize_without_control",
        "mistake",
        "normalisation without a control: refused at the Setup",
        Design(("ctrl", "a"), shifts={"a": 0.8}, seed=3005),
        {**READOUT, "normalize": "control_mean"},
        {"outcome": "refused", "at": "setup", "error": "ValueError", "match": "normalize needs"},
    )


def planes_as_rows() -> Scenario:
    """Rows per plane given as they are, without splitting them per plane."""
    scenario = _scenario(
        "m06_planes_given_as_rows",
        "mistake",
        "one row per cell and plane given as is: refused, naming planes_as_features",
        Design(("ctrl", "a"), shifts={"a": 0.8}, seed=3006),
        {"readout": "area", "features": ["area"], "observation_unit": "cell", **CTRL},
        {
            "outcome": "refused",
            "at": "report",
            "error": "ValueError",
            "match": "planes_as_features",
        },
        columns=("area",),
    )
    scenario.objects = _per_plane(scenario.objects, 3006)
    return scenario


def positive_seen_too_few() -> Scenario:
    """A positive control judged after fewer reference fields than the definition of new
    needs before any estimate (5): never judged."""
    setup = {
        **READOUT,
        **CTRL,
        "positive": {"selector": {"condition": "a"}, "fovs_seen": 2, "draws": 5},
        "novelty": "default",
    }
    return _scenario(
        "m07_positive_control_seen_too_few_fields",
        "mistake",
        "a positive control after 2 reference fields, where the definition of new needs 5: "
        "runs, the control never judged",
        Design(("ctrl", "a"), repeats=3, shifts={"a": 0.8}, fields=6, seed=3007),
        setup,
        {"results": ["controls"], "controls_status": "not assessed"},
    )


SCENARIOS = [
    two_repeats,
    control_elsewhere,
    one_condition,
    screen,
    no_shared_condition,
    parent_chain,
    narrow_tables,
    count_features,
    signed_normalised,
    z_planes,
    unequal_fields,
    missing_in_one_table,
    known_effects,
    dino_tagged_explainable,
    wrong_identity,
    unknown_tag,
    controls_without_members,
    normalise_without_control,
    planes_as_rows,
    positive_seen_too_few,
]

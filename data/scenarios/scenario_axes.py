"""The axes scenarios ``make_scenarios.py`` writes: the tables x the controls x normalisation x
the unit x the wells x the target, every pair of values in at least one scenario, and the
truth ``docs/common/what_happens.md`` gives for each (``make_scenarios`` lists the truth's keys).
"""

from __future__ import annotations

from typing import Any

from make_scenarios import (
    RARE_TARGET,
    Design,
    Scenario,
    acquisition,
    covariance,
    dino,
    final_names,
    observation_columns,
    table,
)

ONE_CONTROL = ("ctrl", "a")
SEVERAL = ("cA", "a1", "cB", "b1")
# One treatment key under two controls, so the rows across experiments exist.
SEVERAL_CONTROL = {"condition": {"cA": {"k1": "a1"}, "cB": {"k1": "b1"}}}
# a and a1 strongly shifted, b1 null; cB a little away from cA.
SHIFTS = {"a": 0.8, "a1": 0.8, "cB": 0.2}
EXPLAINABLE = ("area", "length", "ecc")

# A pairwise covering design (audit 8B's, with the fingerprint on explainable features
# moved to an embedding where there are none, and two cases added for the pairs it then
# missed): every pair of values of two axes is in at least one case.
AXES = (
    ("tagged", "several", False, "image", "none", "rare"),
    ("expl", "several", True, "cells", "tech", "positive"),
    ("mix", "one", True, "image", "none", "fp_dino"),
    ("embed1", "none", False, "cells", "tech", "fp_dino"),
    ("embed2", "one", False, "cells", "tech", "fp_dino"),
    ("expl", "none", False, "image", "none", "tol"),
    ("tagged", "one", True, "cells", "tech", "tol"),
    ("embed1", "several", True, "image", "tech", "fp_dino"),
    ("mix", "one", False, "cells", "tech", "rare"),
    ("embed2", "several", True, "cells", "none", "tol"),
    ("mix", "several", False, "image", "none", "positive"),
    ("embed2", "none", False, "image", "tech", "rare"),
    ("mix", "none", False, "cells", "none", "fp_expl"),
    ("embed1", "one", True, "image", "none", "rare"),
    ("expl", "one", False, "cells", "tech", "rare"),
    ("tagged", "several", True, "cells", "none", "fp_dino"),
    ("tagged", "one", True, "image", "tech", "positive"),
    ("tagged", "none", False, "image", "tech", "fp_expl"),
    ("expl", "none", False, "cells", "tech", "fp_expl"),
    ("expl", "one", False, "image", "tech", "fp_dino"),
    ("embed1", "one", True, "cells", "none", "tol"),
    ("mix", "none", False, "cells", "tech", "tol"),
    ("embed2", "several", False, "image", "tech", "fp_dino"),
    ("expl", "one", True, "image", "none", "fp_expl"),
    ("mix", "several", True, "cells", "tech", "fp_expl"),
)
AXIS_NAMES = ("tables", "controls", "normalize", "unit", "wells", "target")
# The page is built for these: every table kind, several controls, normalize, rare_label.
HTML = {1, 6, 8, 9, 13}


def _design(controls: str, unit: str, wells: str, target: str, seed: int) -> Design:
    """The layout of an axes case: 3 repeats, few fields, few observations; a fingerprint is
    judged in groups of 4 fields or more, on k principal components, k a fifth of the median
    group: its cases have 4 fields per group, of 3 cells (k = 2, judged) or one image each
    (not assessable)."""
    several = controls == "several"
    fields = (2 if wells == "tech" else 3) if unit == "cells" else 4
    return Design(
        conditions=SEVERAL if several else ONE_CONTROL,
        repeats=3,
        shifts=SHIFTS,
        # One repeat without the second control: its conditions have no own control there.
        absent=frozenset({("cB", 2)}) if several else frozenset(),
        fields=4 if target.startswith("fp") else fields,
        cells=(3 if target.startswith("fp") else 4) if unit == "cells" else 1,
        wells=2 if wells == "tech" else 0,
        seed=seed,
    )


def _control(controls: str) -> dict[str, Any]:
    """The Setup's control arguments for an axes case."""
    if controls == "one":
        return {"control": {"condition": "ctrl"}}
    if controls == "several":
        return {
            "control": SEVERAL_CONTROL,
            "control_parents": {"cB": "cA"},
            "master_control": "cA",
        }
    return {}


def _tables(
    kind: str, keys: Any, effect: Any, seed: int
) -> tuple[list[str], dict[str, Any], dict[str, Any], tuple[str, ...]]:
    """An axes case's observation columns, other tables, Setup arguments and features."""
    if kind == "expl":
        return [*EXPLAINABLE, "intensity_mean"], {}, {"readout": "area"}, EXPLAINABLE
    if kind == "tagged":
        tables = {
            "fiber": (measured(keys, effect, ("fib_len", "fib_w"), seed + 1), "explainable"),
            "profiler": (measured(keys, effect, ("prof_0", "prof_1"), seed + 2), "explainable"),
        }
        features = ("area", "fib_len", "fib_w", "prof_0")
        return ["area"], tables, {"readout": "area"}, features
    if kind in ("embed1", "embed2"):
        # Two models write the same column names: the second table's are prefixed.
        tables = {
            f"DINO{i + 1}": (dino(keys, effect, seed + 3 + i), None) for i in range(int(kind[-1]))
        }
        return [], tables, {"readout": None}, ()
    tables = {
        "fiber": (measured(keys, effect, ("fib_len", "fib_w"), seed + 1), "explainable"),
        "DINO1": (dino(keys, effect, seed + 3), "non-explainable"),
    }
    return (
        ["area", "length", "intensity_mean"],
        tables,
        {"readout": "area"},
        ("area", "length", "fib_len"),
    )


def measured(keys: Any, effect: Any, columns: tuple[str, ...], seed: int) -> Any:
    """An explainable table: positive measurements, the first following the effect."""
    return table(keys, effect, columns, seed, affected=1, baseline=10.0)


def _target(
    target: str,
    conditions: tuple[str, ...],
    features: tuple[str, ...],
    tables: dict[str, Any],
    observed: list[str],
) -> dict[str, Any]:
    """The Setup's target arguments for an axes case: a fingerprint on an embedding names
    every non-explainable table's columns as the report names them."""
    if target == "fp_expl":
        return {"fingerprint": [covariance(features)]}
    if target == "fp_dino":
        named = {
            name: final_names(tables, name, observed)
            for name, (_, tag) in tables.items()
            if tag != "explainable"
        }
        return {"fingerprint": [covariance(named["DINO1"])], "feature_tables": named}
    if target == "rare":
        return {"rare": RARE_TARGET, "rare_label": "rare"}
    if target == "positive":
        selector = {"condition": "a1" if "a1" in conditions else "a"}
        return {
            "positive": {"selector": selector, "fovs_seen": 5, "draws": 5},
            "novelty": "default",
        }
    return {}


def axes_scenario(index: int) -> Scenario:
    """Axes case ``index`` (``AXES``), with its truth."""
    kinds, controls, normalize, unit, wells, target = AXES[index]
    seed = 1000 + index
    design = _design(controls, unit, wells, target, seed)
    keys, fovs, effect = acquisition(design)
    columns, tables, setup, features = _tables(kinds, keys, effect, seed)
    if target == "fp_dino" and "DINO1" not in tables:
        tables["DINO1"] = (dino(keys, effect, seed + 3), None)
    if target == "rare":
        columns = [*columns, "rare"]
    objects = observation_columns(keys, effect, columns, seed, rare_conditions=("a", "b1"))
    if features:
        setup["features"] = list(features)
    setup["observation_unit"] = "cell" if unit == "cells" else "image"
    setup.update(_control(controls))
    if normalize:
        setup["normalize"] = "control_mean"
    setup.update(_target(target, design.conditions, features, tables, list(objects.columns)))
    axes = dict(zip(AXIS_NAMES, AXES[index], strict=True))
    name = (
        f"a{index + 1:02d}_{kinds}_{controls}{'_norm' if normalize else ''}_{unit}_{wells}_{target}"
    )
    truth = axes_truth(axes, tables)
    truth["html"] = index in HTML
    return Scenario(
        name=name,
        kind="axes",
        checks=axes_checks(axes, truth),
        objects=objects,
        fovs=fovs,
        setup=setup,
        truth=truth,
        tables=tables,
        axes=axes,
    )


TARGET_WORDS = {
    "tol": "the mean's tolerances",
    "fp_expl": "a fingerprint on the explainable features",
    "fp_dino": "a fingerprint on DINO1",
    "rare": "the rare label's share and clustering",
    "positive": "a positive control",
}


def axes_checks(axes: dict[str, Any], truth: dict[str, Any]) -> str:
    """What an axes scenario checks, in a few words for the README's table."""
    parts = ["embedding-only report" if str(axes["tables"]).startswith("embed") else "features"]
    parts.append(
        {
            "none": "no condition test",
            "one": "each condition against the control",
            "several": "own controls and the rows across experiments",
        }[axes["controls"]]
    )
    if axes["normalize"]:
        parts.append("normalised and centred views")
    if axes["unit"] == "image":
        parts.append("the field questions on the images")
    parts.append(TARGET_WORDS[axes["target"]])
    if truth.get("html"):
        parts.append("the page" + (", its CSVs naming tables" if truth["table_column"] else ""))
    return "; ".join(parts)


def axes_truth(axes: dict[str, Any], tables: dict[str, Any]) -> dict[str, Any]:
    """What ``docs/common/what_happens.md`` says runs for these inputs, and why the rest does
    not (``not_applicable``: a part of the reason in ``Report.not_assessed`` or the
    question's headline)."""
    embedding = str(axes["tables"]).startswith("embed")
    control = axes["controls"] != "none"
    per_field = axes["unit"] == "cells"
    target = axes["target"]
    other_tables = any(tag != "explainable" for _, tag in tables.values())
    ran = ["overview", "populations", "separation"]
    absent: list[str] = []
    reasons: dict[str, str] = {}
    full: list[str] = []
    controls_status = None
    (ran if target.startswith("fp") else absent).append("fingerprint_precision")
    (ran if target == "rare" else absent).append("rare_clustering")
    (ran if other_tables and control else absent).append("whole_table")
    if not per_field:  # the field questions on the images; only ideas within a field not
        absent += ["object_rules", "clustering"]
        if control or target == "rare":
            reasons["profile"] = "cluster in a few fields is not applicable"
    if embedding:
        absent += ["contrast", "discovery", "sufficiency", "inspection", "label_control"]
        reasons["controls"] = reasons["inspect"] = "no explainable features"
        reasons["unit"] = "no explainable features"
        (ran if control else absent).append("unusual")
        reasons["profile"] = "Fields against random" if control else "against the control"
        reasons["conditions"] = "Per feature" if control else "no control is declared"
        reasons["enough"] = "Per feature" if target != "tol" else "no target declared"
    else:
        ran += ["sufficiency", "inspection"]
        (ran if target == "rare" else absent).append("label_control")
        ran.append("variability")
        if control and not per_field:  # one observation per field: the same draws
            reasons["unit"] = "random images and whole fields are the same draws"
        if control:
            ran += ["contrast", "discovery", "enough_features", "ranking"]
            # Beyond the mean reads the explainable features the fingerprint reads only.
            (ran if target == "fp_expl" else absent).append("distribution")
            if target == "fp_dino":
                full.append("Beyond the mean: not compared, as the fingerprint reads no")
            ran += ["unusual"] + (["object_rules", "clustering"] if per_field else [])
        else:
            absent += ["contrast", "discovery", "unusual", "object_rules", "distribution"]
            # Plan v2 (owner, 2026-10-06): requested contrasts, not yet supported.
            reasons["conditions"] = "no control is declared"
        if target == "positive":  # judged by the field version of new on images
            ran.append("controls")
            if per_field:  # 5 reference fields seen, as the definition of new needs
                controls_status = "assessed"
            if axes["controls"] == "several":  # the own controls, pooled
                full.append("judged against the controls cA and cB pooled")
    truth: dict[str, Any] = {
        "outcome": "runs",
        "results": sorted(set(ran)),
        "absent": sorted(set(absent) - set(ran)),
        "not_applicable": dict(sorted(reasons.items())),
        "table_column": any(tag == "explainable" for _, tag in tables.values()),
    }
    if target.startswith("fp"):
        truth["headlines"] = {"enough": _fingerprint_headline(target, other_tables, per_field)}
    truth["separation_spaces"] = _spaces(axes, tables)
    if full:
        truth["full"] = full
    if controls_status is not None:
        truth["controls_status"] = controls_status
    if axes["controls"] == "several" and not embedding:
        truth["contrast_kinds"] = ["across_normalised", "own_control"]
    return truth


def _fingerprint_headline(target: str, other_tables: bool, per_field: bool) -> str:
    """The fingerprint's part of "Did I image enough?": judged in groups of 12 cells (k = 2),
    not assessable in groups of 4 images (k would be 0)."""
    # A covariance criterion: the covariance fingerprint (plan v2, owner, 2026-10-06).
    named = "the covariance fingerprint"
    if target == "fp_dino":
        named = "the covariance fingerprint of DINO1"
    elif other_tables:
        named = "the covariance fingerprint of the explainable features"
    return f"{named} within its tolerance" if per_field else f"{named} not assessable"


def _spaces(axes: dict[str, Any], tables: dict[str, Any]) -> list[str]:
    """The feature spaces the separation scores: the explainable features ("Features" when
    the observation table is the only table), each other table, and the combined space of
    two or more."""
    spaces = [name for name, (_, tag) in tables.items() if tag != "explainable"]
    if not str(axes["tables"]).startswith("embed"):
        spaces.append("Features" if not tables else "Explainable")
    return sorted(spaces + (["Combined"] if len(spaces) > 1 else []))


SCENARIOS = [lambda i=i: axes_scenario(i) for i in range(len(AXES))]

"""P2, incomplete and complex contrasts: the contrast layer's three regimes
(docs/common/validation.md, module P2).

Aim: choose how a requested contrast is computed when its conditions do not share every
repeat: directly paired, model-estimated by the fixed-block engine on per-repeat condition
summaries, or said not estimable. Data: ``scalar`` (lognormal, multiplicative effects) on
designs with several conditions; the summaries are the log cell-weighted repeat means. Each
point names one target contrast (``target``: condition weights) and the contrasts whose
estimability is checked (``checked``).

Methods: ``direct`` (today: the paired t over the repeats holding every condition of the
contrast, from 3 repeats; a contrast it cannot answer is not rejected, ``answered`` 0),
``fixed_block`` (the model wherever estimable), ``regime`` (direct, else model, else not
estimable: the candidate layer) and ``mixed`` (comparator: a random repeat intercept,
statsmodels). Hard gates: the algebraic estimability agrees with the graph on every checked
contrast (``estimability_agrees``), and a target that is not estimable is said so
(``detected``).
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pandas as pd

from .candidates import fixed_block, repeat_means
from .design import connected_incomplete, design_from, disconnected
from .runner import Method, Module, Point, Rule, main
from .scalar import ScalarData, generate, scalar_from

FEATURE = "x0"
BASE: dict[str, Any] = {"distribution": "lognormal", "cv": 0.3, "scale": "ratio"}


def simulate(params: dict[str, Any], rng: np.random.Generator) -> ScalarData:
    """One dataset of a point."""
    return generate(design_from(params), scalar_from(params), rng)


def _summaries(data: ScalarData) -> pd.DataFrame:
    table = repeat_means.summaries(data.objects, FEATURE, "cell")
    table = table[table["value"] > 0].copy()
    table["value"] = np.log(table["value"])
    return table[["repeat", "condition", "value"]]


def _weights(context: dict[str, Any]) -> dict[str, float]:
    return {str(k): float(v) for k, v in context["params"]["target"].items()}


def _truth(data: ScalarData, weights: dict[str, float]) -> float:
    """The contrast of the true log means (the first condition's truth 0)."""
    total = 0.0
    for condition, weight in weights.items():
        if (FEATURE, condition, "cell", "ratio") in data.truth:
            total += weight * data.truth[FEATURE, condition, "cell", "ratio"]
    return total


def _presence(table: pd.DataFrame) -> dict[str, list[str]]:
    return table.groupby("repeat", sort=False)["condition"].agg(list).to_dict()


def _checks(table: pd.DataFrame, context: dict[str, Any]) -> dict[str, float]:
    """Whether the algebra agrees with the graph on every checked contrast, and whether a
    target that is not estimable is said so."""
    presence = _presence(table)
    layout = fixed_block.Layout.from_presence(presence)
    agrees = True
    for weights in context["params"].get("checked", []):
        try:
            algebra = fixed_block.estimable(layout.matrix(), layout.contrast(weights))
        except KeyError:
            algebra = False
        agrees &= algebra == fixed_block.graph_estimable(presence, weights)
    out = {"estimability_agrees": float(agrees)}
    target = _weights(context)
    if not fixed_block.graph_estimable(presence, target):
        out["detected"] = float(fixed_block.regime(presence, target) == "not estimable")
    return out


def _result(found: dict[str, Any], data: ScalarData, weights: dict[str, float]) -> dict[str, float]:
    """The outputs of one answer; a contrast not answered is not rejected."""
    answered = found.get("regime") != "not estimable" and np.isfinite(found.get("p", np.nan))
    out = {k: found.get(k, np.nan) for k in ("estimate", "lo", "hi")}
    out.update(p=found["p"] if answered else 1.0, answered=float(answered))
    out["truth"] = _truth(data, weights)
    out["model"] = float(found.get("regime") == "model")
    return out


def direct(data: ScalarData, context: dict[str, Any]) -> dict[str, float]:
    """Today: the paired t where at least 3 repeats hold every condition of the contrast."""
    table, weights = _summaries(data), _weights(context)
    if fixed_block.regime(_presence(table), weights) == "direct":
        found = fixed_block.contrast(table, weights, context["alpha"])
    else:
        found = {"regime": "not estimable", "p": np.nan}
    return _result(found, data, weights)


def model(data: ScalarData, context: dict[str, Any]) -> dict[str, float]:
    """The fixed-block model wherever the contrast is estimable."""
    table, weights = _summaries(data), _weights(context)
    found = fixed_block.fit_model(table, weights, context["alpha"])
    return {**_result(found, data, weights), **_checks(table, context)}


def regime(data: ScalarData, context: dict[str, Any]) -> dict[str, float]:
    """The candidate layer: direct, else model, else not estimable."""
    table, weights = _summaries(data), _weights(context)
    found = fixed_block.contrast(table, weights, context["alpha"])
    return {**_result(found, data, weights), **_checks(table, context)}


def mixed(data: ScalarData, context: dict[str, Any]) -> dict[str, float]:
    """A random repeat intercept on the per-repeat summaries (statsmodels comparator)."""
    import warnings

    import statsmodels.formula.api as smf
    from scipy import stats

    table, weights = _summaries(data), _weights(context)
    if not fixed_block.graph_estimable(_presence(table), weights):
        return _result({"regime": "not estimable", "p": np.nan}, data, weights)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = smf.mixedlm("value ~ 0 + C(condition)", table, groups=table["repeat"]).fit(reml=True)
    names = {c: f"C(condition)[{c}]" for c in weights}
    vector = np.array([weights[c] if names[c] in fit.params.index else 0.0 for c in weights])
    index = [names[c] for c in weights]
    estimate = float(vector @ fit.params[index].to_numpy())
    se = float(np.sqrt(vector @ fit.cov_params().loc[index, index].to_numpy() @ vector))
    p = float(2 * stats.norm.sf(abs(estimate / se)))
    found = {
        "regime": "model",
        "estimate": estimate,
        "p": p,
        "lo": estimate - 1.96 * se,
        "hi": estimate + 1.96 * se,
    }
    return _result(found, data, weights)


METHODS = (
    Method("direct", direct, role="ours", simplicity=1),
    Method("fixed_block", model, simplicity=2),
    Method("regime", regime, simplicity=3),
    Method(
        "mixed",
        mixed,
        role="comparator",
        question="comparators",
        simplicity=4,
        max_replicates=1_000,
    ),
)

THREE = ("control", "A", "B")
FOUR = ("control", "A", "B", "C")
ARMS = ("C_A", "T_A", "C_B", "T_B")
INTERACTION = {"T_A": 1.0, "C_A": -1.0, "T_B": -1.0, "C_B": 1.0}
PAIRS = [{"A": 1.0, "control": -1.0}, {"A": 1.0, "B": -1.0}, {"B": 1.0, "control": -1.0}]


def _designs(n: int) -> dict[str, dict[str, Any]]:
    """The designs of P2 at ``n`` repeats, each with its target and checked contrasts."""
    half = n // 2
    chain = tuple(("P", "C") if r < half else ("C", "T") for r in range(n))
    return {
        "complete": {"conditions": THREE, "target": {"A": 1.0, "control": -1.0}, "checked": PAIRS},
        "incomplete": {
            "conditions": FOUR,
            "presence": connected_incomplete(FOUR, n),
            "target": {"A": 1.0, "B": -1.0},
            "checked": PAIRS,
        },
        "missing_control": {
            "conditions": THREE,
            "presence": tuple(("A", "B") if r == 0 else THREE for r in range(n)),
            "target": {"A": 1.0, "control": -1.0},
            "checked": PAIRS,
        },
        "chain": {
            "conditions": ("P", "C", "T"),
            "presence": chain,
            "target": {"T": 1.0, "P": -1.0},
            "checked": [{"T": 1.0, "P": -1.0}],
        },
        "four_arm_matched": {"conditions": ARMS, "target": INTERACTION, "checked": [INTERACTION]},
        "four_arm_separate": {
            "conditions": ARMS,
            "presence": disconnected(ARMS, n),
            "target": INTERACTION,
            "checked": [INTERACTION],
        },
    }


EFFECTS = {
    "complete": {"A": 1.0},
    "incomplete": {"A": 1.0},
    "missing_control": {"A": 1.0},
    "chain": {"C": 0.5, "T": 1.5},
    "four_arm_matched": {"T_A": 1.0, "C_B": 0.5, "T_B": 0.5},
    "four_arm_separate": {"T_A": 1.0, "C_B": 0.5, "T_B": 0.5},
}
CLASSES = {
    "complete": "complete",
    "incomplete": "incomplete",
    "missing_control": "incomplete",
    "chain": "chain",
    "four_arm_matched": "interaction",
    "four_arm_separate": "interaction",
}
STRESS = {
    "heteroscedastic": {"heteroscedastic": 2.0},
    "heterogeneous": {"effect_sd": 0.5},
    "extreme_repeat": {"extreme_repeat": 2.0},
    "qc_exclusion": {"missing": "condition", "missing_rate": 0.3},
}


def points() -> tuple[Point, ...]:
    """The design points of P2."""
    out = []
    for n in (4, 6):
        for name, design in _designs(n).items():
            base = {**BASE, "repeats": n, **design}
            out.append(Point(f"core_null_{name}_n{n}", "core", base))
            effects = tuple(EFFECTS[name].items())
            out.append(
                Point(
                    f"core_{name}_n{n}", "core", {**base, "effects": effects}, False, CLASSES[name]
                )
            )
    not_estimable = {
        **BASE,
        "repeats": 6,
        "conditions": FOUR,
        "presence": disconnected(FOUR, 6),
        "target": {"A": 1.0, "B": -1.0},
        "checked": PAIRS,
    }
    out.append(Point("core_disconnected_n6", "core", not_estimable, False, None))
    weak = tuple(
        ("control", "A") if r < 3 else (("A", "B") if r == 3 else ("B",)) for r in range(6)
    )
    out.append(
        Point(
            "stress_weak_bridge",
            "stress",
            {
                **BASE,
                "repeats": 6,
                "conditions": THREE,
                "presence": weak,
                "target": {"B": 1.0, "control": -1.0},
                "checked": PAIRS,
            },
        )
    )
    for name, change in STRESS.items():
        base = {**BASE, "repeats": 6, **_designs(6)["incomplete"], **change}
        out.append(Point(f"stress_{name}", "stress", base))
    return tuple(out)


MODULE = Module(
    name="p2",
    title="incomplete and complex contrasts",
    simulate=simulate,
    points=points(),
    methods=METHODS,
    rules=(Rule("main", hard=("estimability_agrees", "detected")),),
)

if __name__ == "__main__":
    main("decisions.p2_contrasts", sys.argv[1:])

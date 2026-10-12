"""The readable names of the known stopping-rule families (private), for _style.rule_label.

``_named`` reads a rule's name, thresholds and parameters (``_style._rule_parts``) and
names its family in words: ``"fixed 8 fields"``, ``"mean-precision rule, per nucleus, 5%"``;
one function per family, chosen by the rule's exact name or by its prefix.

Shared-private: ``_named``, used by ``_style``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from ._wording import count, plural, rate

_BOOTSTRAP = {
    "centroid": "mean",
    "quantiles": "quantiles",
    "covariance": "covariance",
    "eigenspectrum": "eigenspectrum",
    "subspace": "subspace",
}
# The support-novelty quantities (strategies.support_novelty.QUANTITIES), by name here so
# that this module imports no strategy; tests/test_style_foundation.py checks they agree.
_NOVELTY = {
    "lofo_novel_fov_upper": "nothing new ({})",
    "lofo_novel_fov_fraction": "new fields ({}, point estimate)",
    "lofo_novel_fraction": "new {units} ({})",
    "lofo_mean_fov_novel_fraction": "new {units} per field ({})",
}
# The incidence-coverage quantities (strategies.incidence_coverage.QUANTITIES): the
# strategy's name is its quantity.
_INCIDENCE = {
    "incidence_coverage": "state coverage ({})",
    "chao2_completeness": "Chao2 completeness ({})",
    "expected_new_states_next_fov": "expected new states ({})",
    "prob_new_state_next_fov": "new state in the next field ({})",
    "good_turing_new_state_estimate": "Good-Turing new state ({})",
    "lofo_new_state_fov_fraction": "fields with a new state ({})",
    "new_state_fov_upper": "no new labelled state ({})",
}
_PROBABILITY_KINDS = {*_NOVELTY, *(k for k in _INCIDENCE if k != "expected_new_states_next_fov")}


def _threshold(
    thresholds: Mapping[str, tuple[str, float]], parameters: Mapping[str, Any], *keys: str
) -> float | None:
    """The first of ``keys`` among the thresholds, then among the parameters."""
    for key in keys:
        if key in thresholds:
            return thresholds[key][1]
    for key in keys:
        value = parameters.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
    return None


def _bound(value: float | None, probability: bool, below: bool = True) -> str:
    """``≤5%`` (a probability), ``≥95%``, ``≤0.3`` (a number in the readout's units)."""
    if value is None:
        return "its threshold"
    sign = "≤" if below else "≥"
    return f"{sign}{rate(value) if probability else f'{value:g}'}"


def _per(unit_name: str, words: str) -> str:
    """``per nucleus`` for the object unit, ``per field`` for the FOV unit."""
    return "per field" if unit_name == "fov" else f"per {words}"


def _fixed_fov(name: str, thresholds: Mapping, parameters: Mapping, unit: str) -> str | None:
    """fixed 8 fields."""
    target = _threshold(thresholds, parameters, "n_fovs", "target_fovs")
    return None if target is None else f"fixed {count(target, 'field')}"


def _fixed_objects(name: str, thresholds: Mapping, parameters: Mapping, unit: str) -> str | None:
    """fixed 200 nuclei."""
    keys = ("n_objects", "target_objects", "n_observations", "target_observations")
    target = _threshold(thresholds, parameters, *keys)
    return None if target is None else f"fixed {count(target, unit, plural(unit))}"


def _fixed_area(name: str, thresholds: Mapping, parameters: Mapping, unit: str) -> str | None:
    """fixed imaged area 2.5."""
    target = _threshold(thresholds, parameters, "imaged_area", "target_area")
    return None if target is None else f"fixed imaged area {target:g}"


def _precision_plan(name: str, thresholds: Mapping, parameters: Mapping, unit: str) -> str | None:
    """mean-precision plan, 120 nuclei, or the planned count without its number."""
    plan = parameters.get("plan") or {}
    needed = plan.get("required_objects") if isinstance(plan, Mapping) else None
    return (
        f"mean-precision plan, {count(needed, unit, plural(unit))}"
        if needed
        else "mean-precision planned count"
    )


def _precision_rule(name: str, thresholds: Mapping, parameters: Mapping, unit: str) -> str | None:
    """mean-precision rule, per nucleus, 5%, point SD, pilot of 3 fields."""
    parts = name.split("_")[2:]
    if not parts or parts[0] not in ("object", "fov"):
        return None
    target = _threshold(
        thresholds, parameters, "averaging_cv", "pilot_projected_averaging_cv", "target_cv"
    )
    shown = "?" if target is None else rate(target)
    label = f"mean-precision rule, {_per(parts[0], unit)}, {shown}"
    if "upper95" not in parts:
        label += ", point SD"
    if "pilot" in parts:
        pilot = parameters.get("pilot_fovs")
        label += f", pilot of {count(pilot, 'field')}" if pilot else ", from a pilot"
    return label


def _descriptor_spread(
    name: str, thresholds: Mapping, parameters: Mapping, unit: str
) -> str | None:
    """JCS spread, per field, ≤0.3."""
    resampling = name.removeprefix("descriptor_spread_")
    if resampling not in ("object", "fov"):
        return None
    tolerance = _threshold(
        thresholds, parameters, "max_location_spread", "max_scale_spread", "tolerance"
    )
    return f"JCS spread, {_per(resampling, unit)}, {_bound(tolerance, False)}"


def _descriptor_plateau(
    name: str, thresholds: Mapping, parameters: Mapping, unit: str
) -> str | None:
    """JCS plateau, ≤0.3 over 5 fields."""
    tolerance = _threshold(
        thresholds, parameters, "max_location_range", "max_scale_range", "tolerance"
    )
    window = parameters.get("window")
    over = f" over {count(window, 'field')}" if window else ""
    return f"JCS plateau, {_bound(tolerance, False)}{over}"


def _fov_bootstrap(name: str, thresholds: Mapping, parameters: Mapping, unit: str) -> str | None:
    """field bootstrap of the mean, SE ≤0.1."""
    summary = name.removeprefix("fov_bootstrap_")
    if summary not in _BOOTSTRAP:
        return None
    keys = [q for q in thresholds if q.startswith("fov_bootstrap_")]
    tolerance = _threshold(thresholds, parameters, *keys, "tolerance")
    measure = "angle" if summary == "subspace" else "SE"
    return f"field bootstrap of the {_BOOTSTRAP[summary]}, {measure} {_bound(tolerance, False)}"


def _support_novelty(name: str, thresholds: Mapping, parameters: Mapping, unit: str) -> str | None:
    """new nuclei (≤5%) and the other support-novelty quantities."""
    quantity = parameters.get("quantity") or next((q for q in thresholds if q in _NOVELTY), None)
    if quantity not in _NOVELTY:
        return None
    bound = _bound(_threshold(thresholds, parameters, quantity, "tolerance"), True)
    return _NOVELTY[quantity].format(bound, units=plural(unit))


def _rare_incidence(name: str, thresholds: Mapping, parameters: Mapping, unit: str) -> str | None:
    """rare-state risk ≤5%."""
    risk = _threshold(thresholds, parameters, "rare_incidence_risk", "max_risk")
    return f"rare-state risk {_bound(risk, True)}"


def _incidence_plan(name: str, thresholds: Mapping, parameters: Mapping, unit: str) -> str | None:
    """rare-state field plan."""
    return "rare-state field plan"


def _incidence(name: str, thresholds: Mapping, parameters: Mapping, unit: str) -> str | None:
    """state coverage (≥95%) and the other incidence-coverage quantities."""
    higher = name in ("incidence_coverage", "chao2_completeness")
    target = _threshold(thresholds, parameters, name, "target")
    return _INCIDENCE[name].format(_bound(target, name in _PROBABILITY_KINDS, not higher))


_Family = Callable[[str, Mapping, Mapping, str], "str | None"]
# The families named by the rule's exact name, then those named by its prefix, in order (the
# planned mean-precision rule before the other mean-precision rules).
_EXACT: dict[str, _Family] = {
    "fixed_fov": _fixed_fov,
    "fixed_objects": _fixed_objects,
    "fixed_observations": _fixed_objects,
    "fixed_area": _fixed_area,
    "descriptor_plateau": _descriptor_plateau,
    "support_novelty": _support_novelty,
    "rare_incidence": _rare_incidence,
    "fov_incidence_plan": _incidence_plan,
    **dict.fromkeys(_INCIDENCE, _incidence),
}
_PREFIXES: tuple[tuple[str, _Family], ...] = (
    ("scalar_precision_planned", _precision_plan),
    ("scalar_precision_", _precision_rule),
    ("descriptor_spread_", _descriptor_spread),
    ("fov_bootstrap_", _fov_bootstrap),
)


def _named(name: str, thresholds: Mapping, parameters: Mapping, unit: str) -> str | None:
    """The readable name of a known rule family, or None."""
    family = _EXACT.get(name)
    if family is None:
        family = next((f for prefix, f in _PREFIXES if name.startswith(prefix)), None)
    return None if family is None else family(name, thresholds, parameters, unit)

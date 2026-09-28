"""The declaration of a ``Setup`` in sentences (private): what the acquisition must preserve.

``Setup.declaration()`` returns ``declaration(setup)``: one sentence per
target, then the representation, the novelty rule, the positive control, the
stopping rules (an averaging CV named as the standard error it is), the order
check beside every stop (or the random-sample assumption it rests on without
one), the values the rules read, and the live decision. Every preset
a workflow applies without being asked (``minimum_detection``, the default
stopping rules, the pairing of the positive control) is named here, so the
report never rests on a setting it does not print.

Shared-private: ``declaration`` is used by ``workflows.setup``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .._wording import count, listing, percent, rate
from ..evidence import Component
from ..order_check import RANDOM_SAMPLE_ASSUMPTION
from ..results import COMPONENT_WORDS
from ..retrospective.replay import strategy_label
from ..schema import resolve_observation_unit
from ..statistics import fovs_to_detect
from ..strategies import ScalarPrecisionStrategy

if TYPE_CHECKING:
    from .setup import Setup


def declaration(setup: Setup) -> tuple[str, ...]:
    """What ``setup`` asks the acquisition to preserve, in sentences (see the module)."""
    tolerances = " and ".join(rate(t) for t in setup.tolerances)
    units = setup.unit_words
    lines = [
        (
            f"Population summary: the mean of {setup.readout} over all {units}, within "
            f"{tolerances} of its value on held-back fields."
        )
    ]
    if setup.fingerprint:
        named = "; ".join(c.description or c.name for c in setup.fingerprint)
        lines.append(f"Phenotypic fingerprint: {named}; judged on held-back fields.")
    else:
        lines.append(
            "Phenotypic fingerprint: not declared (no distribution criterion in "
            "fingerprint), so the answers below speak of the mean, not of the "
            "distribution of phenotypes."
        )
    if setup.rare is not None:
        lines.append(_rare(setup))
    else:
        lines.append(
            "Rare phenotype: not declared (no RareTarget in rare), so no answer below "
            "says whether a rare state was sampled enough."
        )
    lines.append(f"Representation: readout {setup.readout}; {_representation(setup)}.")
    if setup.novelty is not None:
        lines.append(_novelty(setup))
    if setup.positive is not None:
        lines.append(_positive(setup))
    lines.append(_rules(setup))
    lines.append(_order(setup))
    if setup.normalize is not None:
        lines.append(
            "Values: every declared feature divided by the mean of the control "
            f"{dict(setup.control or {})} {units} of the same repeat (of the whole table "
            "without a repeat column) for the outcomes judged on held-back fields, the "
            "condition contrast, the feature ranking and the plots; every stopping rule "
            "runs on the raw values, in the report, optimise and benchmark as in the live "
            "decision, and so do the other analyses."
        )
    objective = setup.objective()
    targets = " and ".join(
        COMPONENT_WORDS[c].lower() for c in Component if c in objective.required_components
    )
    lines.append(
        f"Live decision: the {objective.objective.value} objective ({targets}); a rule "
        "decides only the targets it has evidence about, and a target no rule decides "
        "stays unsupported rather than assumed adequate."
    )
    return tuple(lines)


def _representation(setup: Setup) -> str:
    """What the declared features are used for; unusual observations need the control."""
    units = setup.unit_words
    if not setup.features:
        # Unusual observations and novelty read the declared features only (see report).
        return (
            f"no feature columns declared, so unusual {units} and new phenotypes are not assessed"
        )
    columns = count(len(setup.features), "declared feature column")
    if setup.novelty is None:
        new = "new phenotypes are not assessed (no definition of new: novelty)"
        if setup.control is None:
            return f"{columns}; {new}, and unusual {units} are not assessed without a control"
        return (
            f"{columns} for unusual {units} (against the control {units}, their repeats "
            f"pooled); {new}"
        )
    if setup.control is None:
        return (
            f"{columns} for new phenotypes (definition of new: {setup.novelty.name}); "
            f"unusual {units} are not assessed without a control"
        )
    return (
        f"{columns} for unusual {units} (against the control {units}, their repeats pooled) "
        f"and new phenotypes; definition of new: {setup.novelty.name}"
    )


def _novelty(setup: Setup) -> str:
    """The novelty-bound rule the report judges, and the "seen in few fields" count."""
    few = (
        f"A state that fewer than {setup.resolved_few_fovs} other fields show is counted as "
        "seen in few fields (few_fovs)."
    )
    if setup.novelty_target is None:
        return (
            "Novelty: 'Would one more field show something new?' is answered without a "
            "target (novelty_target=None), and no novelty rule is added to 'Did I image "
            f"enough?'. {few}"
        )
    target = rate(setup.novelty_target)
    if setup.novelty_rule_declared:
        judged = (
            "judged once in the report"
            if set(setup.features) == set(setup.columns)
            else "judged in the report on the readout and the features, as live, so the "
            "next-field answer states no target"
        )
        return (
            "Novelty rule: the upper bound of the chance that one more field shows something "
            f"new at most {target} (novelty_target) is one of the stopping rules (rules): run "
            f"live, calibrated by optimise and benchmark, and {judged}. {few}"
        )
    return (
        "Report-only rule: with this definition of new, 'Did I image enough?' also "
        "judges whether the chance that one more field shows something new is at most "
        f"{target} (its upper bound; novelty_target). It is not one of "
        "the stopping rules (rules): the live decision does not run it, and optimise "
        f"and benchmark do not calibrate it. {few}"
    )


def _positive(setup: Setup) -> str:
    """The positive control: what is checked, how fields are paired, the preset it must meet."""
    positive = setup.positive
    assert positive is not None
    if positive.pair_cols is None:
        paired = (
            "paired within each repeat when the FOV table has a repeat column (pair_cols "
            "default), across the whole table otherwise"
        )
    elif positive.pair_cols:
        paired = f"paired within each {listing(positive.pair_cols)} (pair_cols)"
    else:
        paired = (
            "paired across the whole table, every repeat pooled (pair_cols=()), so a shift "
            "between repeats can read as a detected phenotype"
        )
    return (
        f"Positive control: fields of {dict(positive.selector)} judged against the control "
        f"{dict(setup.control or {})} after {count(positive.fovs_seen, 'reference field')}, "
        f"{paired}. If fewer than {percent(setup.minimum_detection)} of its checks are "
        "recognized (minimum_detection, a provisional preset), every adequacy answer carries "
        "a warning."
    )


def _rules(setup: Setup) -> str:
    """The stopping rules the workflows assess, calibrate and run live; the default says so."""
    named = listing([strategy_label(rule) for rule in setup.stopping_rules()], limit=10)
    if setup.rules is None:
        unit = resolve_observation_unit(setup.observation_unit)[0]
        named += (
            f" (the default, rules=None: the JCB rule per {unit} and per field at "
            f"{rate(setup.tolerances[0])})"
        )
    else:
        named += " (rules)"
    line = f"Stopping rules: {named}; assessed in the report, calibrated by optimise, run live."
    targets = sorted(
        {r.target_cv for r in setup.stopping_rules() if isinstance(r, ScalarPrecisionStrategy)}
    )
    if targets:
        # The JCB criterion is the paper's: a CV of the mean, not a tolerance on it.
        cvs = " and ".join(f"{rate(t)} is one standard error of {rate(t)}" for t in targets)
        line += (
            f" The JCB rules stop on the averaging CV: a CV of {cvs} of the mean, not a "
            f"bound of ±{rate(targets[0])} on its error; whether the mean is within "
            f"{' and '.join(rate(t) for t in setup.tolerances)} is judged on held-back "
            "fields (report, optimise), never by the rules."
        )
    return line


def _order(setup: Setup) -> str:
    """The acquisition-order check printed beside every stop, or the assumption without it."""
    order, stage = setup.acquisition_order_col, setup.stage_cols
    advice = "Image fields in a random or dispersed order."
    if order is None and stage is None:
        return (
            "Acquisition order: no acquisition_order_col or stage_cols, so nothing is "
            f"checked and every stop {RANDOM_SAMPLE_ASSUMPTION}. {advice}"
        )
    tests = []
    if order is not None:
        tests.append(f"a trend of the field means of {setup.readout} with {order!r} (Spearman)")
    if stage is not None:
        tests.append(
            f"alike means in neighbouring fields by {listing(stage)} (Moran's I, permutation "
            "p-value), with the stage extent covered beside a rare-phenotype stop"
        )
    return (
        f"Acquisition order: every stop, live and in the report, is printed with a check of "
        f"the fields seen so far for {' and '.join(tests)}. A flagged trend is a caution, "
        "never a veto (it does not turn stop into continue), and finding none does not show "
        f"the fields to be a random sample of the well's positions. {advice}"
    )


def _rare(setup: Setup) -> str:
    target = setup.rare
    assert target is not None
    parts = []
    if target.object_prevalence is not None:
        parts.append(f"{rate(target.object_prevalence)} of {setup.unit_words}")
    if target.fov_incidence is not None:
        parts.append(f"{rate(target.fov_incidence)} of fields")
    fields = target.min_fov_occurrences
    line = (
        f"Rare phenotype: {target.phenotype}, present in {' and '.join(parts)}, to be seen "
        f"in at least {count(fields, 'field')} with "
        f"{rate(target.detection_probability)} probability."
    )
    if target.fov_incidence is not None:
        needed = fovs_to_detect(target.fov_incidence, target.detection_probability, fields)
        line += (
            f" At that field incidence the binomial plan asks for {needed} independent "
            "representative fields; fields that cluster the state need more."
        )
    else:
        line += (
            " No field incidence was declared, so no field count is planned: object "
            "prevalence is never converted to field incidence."
        )
    if setup.rare_label is not None:
        line += (
            f" Judged on held-back fields with the label column {setup.rare_label!r}, and its "
            "clustering in few fields reported with the sampling profile."
        )
    else:
        line += " No label column: stated and planned for, not judged on held-back fields."
    return line


__all__ = ["declaration"]

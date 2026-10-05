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

from .._caveats import assumption, caveat_lines
from .._control import control_text
from .._style import rule_label, wrap
from .._wording import count, listing, percent, rate
from ..evidence import Component
from ..order_check import RANDOM_SAMPLE_ASSUMPTION
from ..results import COMPONENT_WORDS
from ..retrospective.replay import strategy_label
from ..schema import resolve_observation_unit
from ..statistics import fovs_to_detect
from ..strategies import ScalarPrecisionStrategy
from ._contrasts import controls_sentence

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
    controls = controls_sentence(setup)
    if controls:  # several controls (owner, 2026-10-05)
        lines.append(f"{controls}.")
    if setup.novelty is not None:
        lines.append(_novelty(setup))
    if setup.positive is not None:
        lines.append(_positive(setup))
    lines.append(_rules(setup))
    lines.append(_order(setup))
    if setup.normalize is not None and controls:
        lines.append(
            f"Values: every declared feature divided by the mean of each condition's own "
            f"control {units} of the same repeat (a control by its parent's), for the plots and "
            "the populations; the raw values are where levels across experiments are read, "
            "and the tests read them, each normalising to its own control."
        )
    elif setup.normalize is not None:
        lines.append(
            "Values: every declared feature divided by the mean of the control "
            f"{control_text(setup.reference_control)} {units} of the same repeat (of the whole "
            "table without a repeat column) for the outcomes judged on held-back fields, the "
            "condition contrast, the feature ranking and the plots; every stopping rule "
            "runs on the raw values, in the report, optimise and benchmark as in the live "
            "decision, and so do the other analyses."
        )
    objective = setup.objective()
    targets = " and ".join(
        COMPONENT_WORDS[c].lower() for c in Component if c in objective.required_components
    )
    lines.append(
        f"Live decision: a stop needs every declared target ({targets}); a rule decides "
        "only the targets it has evidence about, and a target no rule decides stays "
        "unsupported rather than assumed adequate."
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
    if setup.feature_tables:  # owner, 2026-10-05: the other tables join them, weighted
        names = ", ".join(setup.feature_tables)
        tables = "tables" if len(setup.feature_tables) > 1 else "table"
        columns += f" and the other feature {tables} {names}, weighted to count as much"
    against = "each condition's own control" if setup.contrast_table is not None else "the control"
    if setup.novelty is None:
        new = "new phenotypes are not assessed (no definition of new: novelty)"
        if setup.control is None:
            return f"{columns}; {new}, and unusual {units} are not assessed without a control"
        return (
            f"{columns} for unusual {units} (against {against} {units}, their repeats "
            f"pooled); {new}"
        )
    if setup.control is None:
        return (
            f"{columns} for new phenotypes (new: {_new_words(setup)}); "
            f"unusual {units} are not assessed without a control"
        )
    return (
        f"{columns} for unusual {units} (against {against} {units}, their repeats pooled) "
        f"and new phenotypes (new: {_new_words(setup)})"
    )


def _new_words(setup: Setup) -> str:
    """What counts as new, plainly: ``at least 3 similar cells that no other field shows``."""
    least = getattr(setup.novelty, "min_novel_observations", None)
    similar = f"at least {least} similar {setup.unit_words}" if least else "similar observations"
    others = getattr(setup.novelty, "min_other_fovs", 1)
    if others == 1:
        return f"{similar} that no other field shows"
    return f"{similar} that fewer than {count(others, 'other field')} show"


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
        f"{control_text(setup.reference_control)} after "
        f"{count(positive.fovs_seen, 'reference field')}, "
        f"{paired}. If fewer than {percent(setup.minimum_detection)} of its checks are "
        "recognized (minimum_detection, a provisional preset), every adequacy answer carries "
        "a warning."
    )


def _rules(setup: Setup) -> str:
    """The stopping rules the workflows assess, calibrate and run live; the default says so."""
    unit = resolve_observation_unit(setup.observation_unit)[0]
    named = listing(
        [f"{rule_label(rule, unit)} [{strategy_label(rule)}]" for rule in setup.stopping_rules()],
        limit=10,
    )
    if setup.rules is None:
        # Per field only for units other than objects (Setup.stopping_rules).
        sides = " and ".join(
            f"per {unit if rule.resampling_unit == 'object' else 'field'}"
            for rule in setup.stopping_rules()
        )
        named += (
            f" (the default, rules=None: the mean-precision rule {sides} at "
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
        cvs = " and ".join(rate(t) for t in targets)
        line += (
            f" The mean-precision rules stop when the standard error of the mean is {cvs} of "
            f"the mean (its averaging CV), not on a bound of ±{rate(targets[0])} on its error; "
            f"whether the mean is within {' and '.join(rate(t) for t in setup.tolerances)} is "
            "judged on held-back fields by optimise, never by the rules or this report."
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
    elif setup.rare_label is not None:
        line += (
            " The fields to see it are counted from the share of fields holding it in the "
            "data (“Did I image enough?”); object prevalence is never converted to "
            "field incidence."
        )
    else:
        line += (
            " Without a field incidence or a label, no field count is planned: object "
            "prevalence is never converted to field incidence."
        )
    if setup.rare_label is not None:
        line += (
            f" Judged with the label column {setup.rare_label!r}: in the report from the "
            "fields acquired (“Did I image enough?”), on held-back fields by optimise "
            "and benchmark; the report also shows its clustering in few fields."
        )
    else:
        line += " No label column: stated and planned for, not judged on held-back fields."
    return line


def _rare_short(setup: Setup) -> tuple[str | None, list[str]]:
    """The rare phenotype in one line, and its caveats."""
    target = setup.rare
    if target is None:
        # An option left at its default is a standing note, not a caution about the data.
        return None, [
            assumption(
                "no rare phenotype declared (rare), so no answer says whether a rare state was "
                "sampled enough"
            )
        ]
    parts = []
    if target.object_prevalence is not None:
        parts.append(f"{rate(target.object_prevalence)} of {setup.unit_words}")
    if target.fov_incidence is not None:
        parts.append(f"{rate(target.fov_incidence)} of fields")
    line = (
        f"Rare phenotype: {target.phenotype} in {' and '.join(parts)}, to be seen in at least "
        f"{count(target.min_fov_occurrences, 'field')} with "
        f"{rate(target.detection_probability)} probability"
    )
    cautions = []
    if target.fov_incidence is not None:
        needed = fovs_to_detect(
            target.fov_incidence, target.detection_probability, target.min_fov_occurrences
        )
        line += f"; the binomial plan asks for {needed} representative fields"
    elif setup.rare_label is not None:
        line += "; the fields to see it are counted from the data"
    if setup.rare_label is not None:
        line += f"; judged on held-back fields with the label {setup.rare_label!r}"
    else:
        cautions.append(
            "no rare_label column: the rare phenotype is not judged on held-back fields"
        )
    return f"{line}.", cautions


def _novelty_short(setup: Setup) -> str:
    """The novelty rule in one line."""
    few = f"; seen in few fields: fewer than {setup.resolved_few_fovs} other fields"
    if setup.novelty_target is None:
        return f"Novelty: answered without a target (novelty_target=None){few}."
    target = rate(setup.novelty_target)
    if setup.novelty_rule_declared:
        return f"Novelty rule: nothing new ≤{target} is one of the stopping rules{few}."
    return (
        f"Report-only rule: nothing new ≤{target}, judged by the report only (not live, not "
        f"calibrated by optimise and benchmark){few}."
    )


def _order_short(setup: Setup) -> tuple[str | None, list[str]]:
    """The order check in one line, and its caveat."""
    order, stage = setup.acquisition_order_col, setup.stage_cols
    advice = "image fields in a random or dispersed order"
    if order is None and stage is None:
        return None, [
            assumption(
                f"No acquisition order or stage coordinates, so the order is not checked: every "
                f"stop {RANDOM_SAMPLE_ASSUMPTION}; {advice}"
            )
        ]
    tests = []
    if order is not None:
        tests.append(f"a trend with {order!r}")
    if stage is not None:
        tests.append(f"alike neighbours by {listing(stage)} (Moran's I)")
    line = f"Order check beside every stop: {' and '.join(tests)}."
    return line, [
        assumption(
            "a flagged order trend is a caution, never a veto, and finding none does not show "
            f"the fields to be a random sample of the well's positions; {advice}"
        )
    ]


def declaration_short(setup: Setup, indent: int = 2) -> str:
    """The declaration in a line per target and setting, each caveat once (a missing
    declaration as a ⚠️ line, a standing assumption under "Assumptions:").

    The report's short form of "What this acquisition was asked to preserve";
    ``declaration(setup)`` keeps every sentence.
    """
    tolerances = " and ".join(rate(t) for t in setup.tolerances)
    lines = [
        (
            f"Population summary: the mean of {setup.readout} over all {setup.unit_words}, within "
            f"{tolerances} (judged on held-back fields by optimise)."
        )
    ]
    cautions = []
    if setup.fingerprint:
        named = "; ".join(c.description or c.name for c in setup.fingerprint)
        lines.append(f"Phenotypic fingerprint: {named} (judged on held-back fields by optimise).")
    else:
        cautions.append(
            assumption(
                "no phenotypic fingerprint declared (fingerprint), so the answers speak of the "
                "mean, not of the distribution of phenotypes"
            )
        )
    rare, rare_cautions = _rare_short(setup)
    lines += [rare] if rare else []
    cautions += rare_cautions
    lines.append(f"Representation: readout {setup.readout}; {_representation(setup)}.")
    controls = controls_sentence(setup)
    lines += [f"{controls}."] if controls else []
    if setup.novelty is not None:
        lines.append(_novelty_short(setup))
    if setup.positive is not None:
        positive = setup.positive
        pooled = positive.pair_cols is not None and not positive.pair_cols
        lines.append(
            f"Positive control: fields of {dict(positive.selector)} against the control "
            f"{control_text(setup.reference_control)}; below "
            f"{percent(setup.minimum_detection)} recognized "
            "(minimum_detection, a provisional preset) every adequacy answer is warned."
        )
        if pooled:
            cautions.append(
                "the positive control's fields are paired across the whole table "
                "(pair_cols=()), so a shift between repeats can read as a detected phenotype"
            )
    unit = resolve_observation_unit(setup.observation_unit)[0]
    rules = listing([rule_label(rule, unit) for rule in setup.stopping_rules()], limit=10)
    default = " (the default)" if setup.rules is None else ""
    lines.append(f"Stopping rules: {rules}{default}; run live, calibrated by optimise.")
    targets = sorted(
        {r.target_cv for r in setup.stopping_rules() if isinstance(r, ScalarPrecisionStrategy)}
    )
    if targets:
        cvs = " and ".join(rate(t) for t in targets)
        cautions.append(
            assumption(
                f"the mean-precision rules stop when the standard error of the mean is {cvs} of "
                f"the mean (its averaging CV), not on a ±{rate(targets[0])} bound on its error; "
                f"whether the mean is within {tolerances} is judged on held-back fields by "
                "optimise"
            )
        )
    order, order_cautions = _order_short(setup)
    lines += [order] if order else []
    cautions += order_cautions
    if setup.normalize is not None and controls:
        lines.append(
            "Values: each condition divided by its own control's mean in the same repeat for "
            "the plots and the populations; levels across experiments are read on the raw "
            "values, which the tests and the stopping rules read."
        )
    elif setup.normalize is not None:
        lines.append(
            "Values: features divided by the control mean of the same repeat for the held-back "
            "outcomes, the contrast, the ranking and the plots; the stopping rules read the "
            "raw values."
        )
    objective = setup.objective()
    targets = listing(
        [COMPONENT_WORDS[c].lower() for c in Component if c in objective.required_components]
    )
    lines.append(
        f"Live decision: a stop needs every declared target ({targets}); a target no rule "
        "decides stays unsupported, never assumed adequate."
    )
    text = [wrap(line, indent, hang=indent + 2) for line in lines]
    return "\n".join([*text, *caveat_lines(cautions, indent=indent)])


__all__ = ["declaration", "declaration_short"]

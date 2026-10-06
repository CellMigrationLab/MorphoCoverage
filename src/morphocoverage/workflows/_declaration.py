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
from .._wording import count, listing, percent, plural, rate
from ..evidence import Component
from ..order_check import RANDOM_SAMPLE_ASSUMPTION
from ..results import COMPONENT_WORDS
from ..retrospective.replay import strategy_label
from ..schema import resolve_observation_unit
from ..statistics import fovs_to_detect
from ..strategies import ScalarPrecisionStrategy
from ._contrasts import controls_sentence, normalised_readers
from ._one_per_field import field_unit

if TYPE_CHECKING:
    from .setup import Setup


def declaration(setup: Setup, per_field: bool = True) -> tuple[str, ...]:
    """What ``setup`` asks the acquisition to preserve, in sentences (see the module);
    ``per_field=False`` (one observation per field) leaves out what then does not run."""
    tolerances = " and ".join(rate(t) for t in setup.tolerances)
    units = setup.unit_words
    lines = [
        (
            f"Population summary: the mean of {setup.readout} over all {units}, within "
            f"{tolerances} of its value on held-back fields."
            if setup.mean_target
            else _no_mean(setup)
        )
    ]
    if setup.fingerprint:
        named = "; ".join(c.description or c.name for c in setup.fingerprint)
        lines.append(f"Phenotypic fingerprint: {named}; judged on held-back fields.")
    elif setup.readout is None:
        lines.append(
            "Phenotypic fingerprint: not declared (no distribution criterion in fingerprint), "
            "so no answer below says whether the distribution was sampled enough."
        )
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
    lines.append(_representation_line(setup, per_field))
    controls = controls_sentence(setup)
    if controls:  # several controls (owner, 2026-10-05)
        lines.append(f"{controls}.")
    if novelty_runs(setup, per_field):  # the positive control is judged by it
        lines.append(_novelty(setup))
        lines += [_positive(setup)] if setup.positive is not None else []
    if setup.stopping_rules():  # an embedding-only Setup has none (NO_READOUT says why)
        lines.append(_rules(setup))
    lines.append(_order(setup))
    if setup.normalize is not None and setup.readout is None:
        lines.append(_tables_centred(per_field))
    elif setup.normalize is not None and controls:
        lines.append(
            f"Values: every declared feature divided by the mean of each condition's own "
            f"control {units} of the same repeat (a control by its parent's); "
            f"{normalised_readers(several=True)}."
        )
    elif setup.normalize is not None:
        lines.append(
            "Values: every declared feature divided by the mean of the control "
            f"{control_text(setup.reference_control)} {units} of the same repeat (of the whole "
            f"table without a repeat column); {normalised_readers()}; every stopping rule "
            "runs on the raw values in the report, optimise and benchmark as in the live "
            "decision."
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


# An embedding-only Setup's normalisation (owner, 2026-10-06).
TABLES_CENTRED = (
    "Values: the other feature tables centred per biological repeat, on the median of the "
    "per-condition medians of the conditions in every repeat, for the maps, the "
    "populations, unusual observations and the separation of the conditions; never divided, "
    "and never for a test, which reads the raw values."
)


def _tables_centred(per_field: bool) -> str:
    """``TABLES_CENTRED`` (unusual observations run with one per field too, plan v2)."""
    return TABLES_CENTRED


def _no_mean(setup: Setup) -> str:
    """The population-summary line without a mean target: ``NO_READOUT``, or ``NO_MEAN``."""
    return NO_READOUT if setup.readout is None else NO_MEAN.format(readout=setup.readout)


# A readout with tolerances=(), the guided "whole distribution" (owner, 2026-10-06).
NO_MEAN = (
    "Population summary: not declared (tolerances=()), the whole distribution being the aim, "
    "so the mean-precision rules are not declared; the tests still compare the mean of "
    "{readout} between the conditions."
)
# An embedding-only Setup's population summary, the one place it is said (owner, 2026-10-06).
NO_READOUT = (
    "Population summary: none, as there is no readout (no explainable features, an "
    "embedding-only report), so the mean-precision rules are not applicable; the targets "
    "are the fingerprint of each feature table and the labelled phenotype, when declared."
)


def novelty_runs(setup: Setup, per_field: bool = True) -> bool:
    """Whether new phenotypes are judged: a definition of new (with one observation per field,
    its field version, ``_one_per_field.field_novelty``, plan v2)."""
    return setup.novelty is not None


def _unusual_missing(setup: Setup, per_field: bool) -> str:
    """Why unusual observations are not assessed (empty when they are): they need the
    control."""
    return "not assessed without a control" if setup.reference_control is None else ""


def _unusual_words(setup: Setup, per_field: bool, label: bool = True) -> str:
    """``unusual cells (against the control cells, each relative to its own control in its
    repeat; exploratory)``: biological unusualness (plan v2); with one observation per field,
    unusual images. ``label=False`` leaves "exploratory" to the caller's sentence."""
    units = setup.unit_words if per_field else plural(field_unit(setup) or setup.unit_word)
    against = (
        "each condition's own control, every value relative to that control in its repeat"
        if setup.contrast_table is not None
        else "the control, every value relative to the control of its repeat"
    )
    return f"unusual {units} (against {against}{'; exploratory' if label else ''})"


def _representation_line(setup: Setup, per_field: bool = True) -> str:
    """``Representation: readout area; ...``, or what an embedding-only Setup reads."""
    if setup.readout is None:
        tables = setup.feature_tables or {}
        space = (
            "their weighted combined space"
            if len(tables) > 1
            else f"the {count(sum(len(c) for c in tables.values()), 'column')} of "
            f"{next(iter(tables), 'the table')}"
        )
        missing = _unusual_missing(setup, per_field)
        relative = "relative to the control of each repeat; exploratory"
        unusual = (
            f"{setup.unit_words} {missing}"
            if missing
            else f"{setup.unit_words} on {space} ({relative})"
        )
        then = "; no" if missing else ", no"
        tables_only = "Representation: the other feature tables only"
        return f"{tables_only}; unusual {unusual}{then} definition of new."
    return f"Representation: readout {setup.readout}; {_representation(setup, per_field)}."


def _representation(setup: Setup, per_field: bool = True) -> str:
    """What the declared features are used for; unusual observations need the control."""
    units = setup.unit_words
    if not setup.features:
        # Unusual observations and novelty read the declared features only (see report);
        # beside other tables, unusual observations read the readout alone (audit 8, seed 2).
        if setup.feature_tables and not _unusual_missing(setup, per_field):
            return (
                f"no feature columns declared, so new phenotypes are not assessed; unusual "
                f"{units} on the readout alone, the other tables never entering them"
            )
        return (
            f"no feature columns declared, so unusual {units} and new phenotypes are not assessed"
        )
    # The other feature tables never enter unusual observations or novelty (owner,
    # 2026-10-06): the declared features alone.
    columns = count(len(setup.features), "declared feature column")
    unusual = _unusual_words(setup, per_field)
    if setup.novelty is None:
        new = "new phenotypes are not assessed (no definition of new: novelty)"
        # A contrast table alone gives own controls: unusual observations read them.
        if setup.reference_control is None:
            return f"{columns}; {new}, and unusual {units} are not assessed without a control"
        return f"{columns} for {unusual}; {new}"
    # One observation per field: a state seen once, exploratory (module P10, plan v2); said
    # once with the unusual images, both exploratory.
    new = f"new phenotypes (new: {_new_words(setup)})"
    if setup.reference_control is None:
        seen = new if per_field else new.replace(" (", ", exploratory (", 1)
        return f"{columns} for {seen}; unusual {units} are not assessed without a control"
    if not per_field:
        return (
            f"{columns} for {_unusual_words(setup, per_field, False)} and {new}, both exploratory"
        )
    return f"{columns} for {unusual} and {new}"


def _new_words(setup: Setup) -> str:
    """What counts as new, plainly: ``at least 3 similar cells that no other field shows``, or
    on fields of one observation ``a cell that no other field shows``."""
    least = getattr(setup.novelty, "min_novel_observations", None)
    similar = f"at least {least} similar {setup.unit_words}" if least else "similar observations"
    one = f"{'an' if setup.unit_word[:1] in 'aeiou' else 'a'} {setup.unit_word}"
    similar = one if least == 1 else similar
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
    # Several own controls are one reference, pooled: named, never as a dict (F10).
    reference = setup.reference_control or {}
    pooled = [str(v) for spec in reference.values() if isinstance(spec, tuple) for v in spec]
    against = f"the control {control_text(reference)}"
    if len(pooled) > 1:
        against = f"the controls {listing(pooled)} pooled"
    return (
        f"Positive control: fields of {dict(positive.selector)} judged against {against} after "
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
            + (
                f"whether the mean is within {' and '.join(rate(t) for t in setup.tolerances)} "
                "is judged on held-back fields by optimise, never by the rules or this report."
                if setup.tolerances
                else "no mean target is declared (tolerances=())."
            )
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


def declaration_short(setup: Setup, indent: int = 2, per_field: bool = True) -> str:
    """The declaration in a line per target and setting, each caveat once (a missing
    declaration as a ⚠️ line, a standing assumption under "Assumptions:"); ``per_field`` as
    for ``declaration``.

    The report's short form of "What this acquisition was asked to preserve";
    ``declaration(setup)`` keeps every sentence.
    """
    tolerances = " and ".join(rate(t) for t in setup.tolerances)
    lines = [
        (
            f"Population summary: the mean of {setup.readout} over all {setup.unit_words}, within "
            f"{tolerances} (judged on held-back fields by optimise)."
            if setup.mean_target
            else _no_mean(setup)
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
                + (
                    "mean, not of the distribution of phenotypes"
                    if setup.readout is not None
                    else "tests between conditions only, not of a distribution sampled enough"
                )
            )
        )
    rare, rare_cautions = _rare_short(setup)
    lines += [rare] if rare else []
    cautions += rare_cautions
    lines.append(_representation_line(setup, per_field))
    controls = controls_sentence(setup)
    lines += [f"{controls}."] if controls else []
    if novelty_runs(setup, per_field):
        lines.append(_novelty_short(setup))
    if setup.positive is not None and novelty_runs(setup, per_field):
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
    if setup.stopping_rules():  # an embedding-only Setup has none (NO_READOUT says why)
        lines.append(f"Stopping rules: {rules}{default}; run live, calibrated by optimise.")
    targets = sorted(
        {r.target_cv for r in setup.stopping_rules() if isinstance(r, ScalarPrecisionStrategy)}
    )
    if targets:
        cvs = " and ".join(rate(t) for t in targets)
        cautions.append(
            assumption(
                f"the mean-precision rules stop when the standard error of the mean is {cvs} of "
                f"the mean (its averaging CV), not on a ±{rate(targets[0])} bound on its error"
                + (
                    f"; whether the mean is within {tolerances} is judged on held-back fields "
                    "by optimise"
                    if setup.tolerances
                    else ""
                )
            )
        )
    order, order_cautions = _order_short(setup)
    lines += [order] if order else []
    cautions += order_cautions
    if setup.normalize is not None and setup.readout is None:
        lines.append(_tables_centred(per_field))
    elif setup.normalize is not None and controls:
        lines.append(
            "Values: each condition divided by its own control's mean in the same repeat; "
            f"{normalised_readers(several=True)}."
        )
    elif setup.normalize is not None:
        lines.append(
            f"Values: features divided by the control mean of the same repeat; "
            f"{normalised_readers()}."
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


__all__ = ["declaration", "declaration_short", "novelty_runs"]

"""The questions of the guided session (private; ``guided.guide``): each prompt, its
default, and how an answer is read. ``ask`` is any ``prompt -> text`` function
(``input`` at a terminal or in a notebook, a scripted list in the tests)."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any

from .._wording import listing

Ask = Callable[[str], str]
Say = Callable[[str], None]
AIMS = {
    "population": "averages",
    "fingerprint": "fingerprint",
    "both": "fingerprint",
    # Without explainable features, the targets that need no readout (owner, 2026-10-06).
    "label": "rare",
    "fingerprint_label": "fingerprint",
}
AIM_TEXT = {
    "population": "compare the population averages of the conditions (the readout's mean)",
    "fingerprint": "compare their covariance fingerprints (how the features vary together)",
    "both": "the population averages and the covariance fingerprints",
    "label": "see the labelled phenotype in enough fields",
    "fingerprint_label": "the covariance fingerprint of each table and the labelled phenotype",
}
# The aims of an embedding-only session: no average is offered (owner, 2026-10-06).
TABLE_AIMS = ("fingerprint", "label", "fingerprint_label")
_YES, _NO = {"y", "yes", ""}, {"n", "no"}
_UNITS = ("cell", "nucleus", "track", "patch", "object")


def _numbered(
    options: Sequence[Any], default: Any | None, labels: Sequence[str] | None = None
) -> str:
    lines = []
    for number, option in enumerate(options, start=1):
        mark = "  (default)" if default is not None and option == default else ""
        label = labels[number - 1] if labels else option
        lines.append(f"  {number}. {label}{mark}")
    return "\n".join(lines)


def _again(say: Say, text: str) -> None:
    say(f"  {text} Please answer again.")


def yes_no(ask: Ask, say: Say, prompt: str, default: bool = True) -> bool:
    """A yes or no question; Enter takes the default."""
    hint = "[Y/n]" if default else "[y/N]"
    while True:
        answer = ask(f"{prompt} {hint} ").strip().lower()
        if answer == "":
            return default
        if answer in _YES:
            return True
        if answer in _NO:
            return False
        _again(say, f"{answer!r} is not yes or no.")


def choose(
    ask: Ask,
    say: Say,
    prompt: str,
    options: Sequence[Any],
    default: Any | None = None,
    labels: Sequence[str] | None = None,
) -> Any:
    """One of ``options`` (shown as ``labels`` when given), by number or by name; Enter takes
    the default when there is one."""
    say(f"{prompt}\n{_numbered(options, default, labels)}")
    names = {str(o).strip().lower(): o for o in options}
    while True:
        answer = ask("> ").strip()
        if answer == "" and default is not None:
            return default
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return options[int(answer) - 1]
        if answer.lower() in names:
            return names[answer.lower()]
        _again(say, f"{answer!r} is not one of the {len(options)} options.")


def _selection(answer: str, options: Sequence[Any]) -> list[Any] | str:
    """The options an answer names (``all``, ``1,3-5``, names separated by commas), in the
    options' order, or the reason it cannot be read."""
    if answer.strip().lower() in ("all", "*"):
        return list(options)
    names = {str(o).strip().lower(): i for i, o in enumerate(options)}
    chosen: set[int] = set()
    for part in (p.strip() for p in answer.split(",")):
        if not part:
            continue
        span = re.fullmatch(r"(\d+)\s*-\s*(\d+)", part)
        if span:
            low, high = int(span[1]), int(span[2])
            if not 1 <= low <= high <= len(options):
                return f"{part!r} is outside 1-{len(options)}."
            chosen.update(range(low - 1, high))
        elif part.isdigit() and 1 <= int(part) <= len(options):
            chosen.add(int(part) - 1)
        elif part.lower() in names:
            chosen.add(names[part.lower()])
        else:
            return f"{part!r} is not one of the options."
    if not chosen:
        return "nothing was chosen."
    return [options[i] for i in sorted(chosen)]


def choose_many(
    ask: Ask,
    say: Say,
    prompt: str,
    options: Sequence[Any],
    default: Sequence[Any] | None = None,
    *,
    enter_all: bool = True,
) -> list[Any]:
    """Several of ``options``: numbers, ranges or names; Enter takes ``default`` (all of them
    when ``None``, or, with ``enter_all=False``, nothing: the question is asked again)."""
    say(f"{prompt}\n{_numbered(options, None)}")
    if default is None:
        hint = "'all' (the default)" if enter_all else "'all'"
    else:
        hint = f"'all'; Enter takes {listing([str(d) for d in default], 6)}"
    say(f"  Enter {hint}, numbers and ranges (1,3-5), or names separated by commas.")
    while True:
        answer = ask("> ").strip()
        if not answer and default is not None:
            return list(default)
        if not answer and not enter_all:
            _again(say, "Nothing was chosen.")
            continue
        selected = _selection(answer or "all", options)
        if isinstance(selected, list):
            return selected
        _again(say, selected)


def number(ask: Ask, say: Say, prompt: str, default: float, low: float, high: float) -> float:
    """A number in ``(low, high]``; a percentage (``10%``) is read as a share."""
    while True:
        answer = ask(f"{prompt} [{default:g}] ").strip()
        if answer == "":
            return default
        try:
            value = float(answer.rstrip("%")) / (100 if answer.endswith("%") else 1)
        except ValueError:
            _again(say, f"{answer!r} is not a number.")
            continue
        if low < value <= high:
            return value
        _again(say, f"{value:g} is not between {low:g} and {high:g}.")


def text(ask: Ask, prompt: str, default: str | None = None) -> str:
    """Free text; Enter takes the default."""
    hint = f" [{default}]" if default else ""
    answer = ask(f"{prompt}{hint} ").strip()
    return answer or (default or "")


def ask_control(ask: Ask, say: Say, scan: Any) -> dict[str, Any] | None:
    """The negative control, by condition (or none), the scan's suggestion as the default."""
    conditions = list(scan.design["conditions"])
    if not conditions:
        say("No 'condition' column: there is no control to declare.")
        return None
    none = "no control"
    guess = scan.design.get("control_guess")
    choice = choose(
        ask,
        say,
        "Which condition is the control?",
        [*conditions, none],
        guess if guess is not None else none,
    )
    return None if choice == none else {"condition": choice}


def ask_positive(ask: Ask, say: Say, scan: Any, control: dict[str, Any] | None) -> Any:
    """An optional positive control: a condition known to change the phenotype, so the report
    can check that the features see a change at all (question 1). ``None`` for none."""
    if control is None:
        return None
    named = control.get("condition")
    controls = set(named) if isinstance(named, dict) else {named}  # several, or one
    others = [c for c in scan.design["conditions"] if c not in controls]
    if not others:
        return None
    none = "none"
    say(
        "A positive control is a condition known to change the phenotype (for example a "
        "treatment or siRNA with a known effect); the report then checks that the features "
        "can see a change. It is optional, and can be added later."
    )
    choice = choose(ask, say, "Is one of the conditions a positive control?", [*others, none], none)
    return None if choice == none else {"condition": choice}


# The aim in plain words (owner, 2026-10-05); the stored values stay the keys of AIMS.
AIM_QUESTION = (
    "Do you want to know whether the average of each feature differs, whether the whole "
    "distribution differs (shape, spread, tails), or both?"
)
AIM_LABELS = {
    "population": "the average",
    "fingerprint": "the whole distribution (shape, spread, tails)",
    "both": "both",
}


def ask_aim(ask: Ask, say: Say) -> str:
    """``population`` (the average), ``fingerprint`` (the whole distribution) or ``both``."""
    aims = ["population", "fingerprint", "both"]
    return choose(ask, say, AIM_QUESTION, aims, aims[0], [AIM_LABELS[a] for a in aims])


def ask_table_aim(ask: Ask, say: Say, label: str | None) -> str:
    """Without explainable features (owner, 2026-10-06): the whole distribution of each table
    (``fingerprint``), the labelled phenotype (``label``) or both (``fingerprint_label``);
    without a label the whole distribution, said, not asked. No average is offered: there is
    no readout."""
    if label is None:
        say(
            "With no explainable features and no labelled phenotype, the aim is the whole "
            "distribution of each table (its fingerprint)."
        )
        return "fingerprint"
    labels = [
        "the whole distribution of each table (its fingerprint)",
        f"the labelled phenotype ({label}) seen in enough fields",
        "both",
    ]
    prompt = "With no explainable features, what should the imaging get right?"
    return choose(ask, say, prompt, list(TABLE_AIMS), TABLE_AIMS[0], labels)


def label_columns(objects: Any, taken: Sequence[str]) -> list[str]:
    """The columns of ``objects`` (not in ``taken``) holding true/false or 0/1 for every row
    that has a value, both values present: candidate labels of a phenotype."""
    found = []
    for column in objects.columns:
        if column in taken:
            continue
        values = objects[column].dropna()
        if values.empty:
            continue
        boolean = str(values.dtype) == "bool" or set(values.unique()) <= {0, 1}
        if boolean and values.astype(bool).nunique() == 2:
            found.append(column)
    return found


def ask_rare_label(ask: Ask, say: Say, candidates: Sequence[str]) -> str | None:
    """An optional column marking a phenotype of interest (``Setup.rare_label``); ``None``
    without one, or when no column holds true/false or 0/1."""
    if not candidates:
        return None
    none = "none"
    prompt = (
        "Is there a column that marks a phenotype you care about (true/false per cell, e.g. "
        "mitotic)?"
    )
    choice = choose(ask, say, prompt, [*candidates, none], none)
    return None if choice == none else choice


# Feature names that say a quantity measured from zero, read as ratios to the control when
# confirmed (plan v2, owner, 2026-10-06).
RATIO_WORDS = ("intensity", "area")


def ask_scales(ask: Ask, say: Say, objects: Any, features: Sequence[str]) -> dict[str, str]:
    """The scale of each feature named like an intensity or an area (``RATIO_WORDS``) whose
    values are all positive: ``"ratio"`` proposed and confirmed, else ``"difference"``; the
    other features keep the automatic rule (``Setup.scales``). Empty when none is named so."""
    named = [
        f
        for f in features
        if any(w in f.lower() for w in RATIO_WORDS) and (objects[f].dropna() > 0).all()
    ]
    if not named:
        return {}
    prompt = (
        f"Test the changes of {listing(named, 6)}, named like an intensity or an area, as "
        "ratios to the control (fold changes)? No tests them as differences"
    )
    scale = "ratio" if yes_no(ask, say, prompt, True) else "difference"
    return dict.fromkeys(named, scale)


def scale_lines(scales: dict[str, str]) -> list[str]:
    """``Tested as ratios to the control: area.``: the saved scales in sentences."""
    lines = []
    for scale in ("ratio", "difference"):
        named = [f for f, s in scales.items() if s == scale]
        lines += [f"Tested as {scale}s to the control: {', '.join(named)}."] if named else []
    return lines


def ask_run(ask: Ask, say: Say) -> bool:
    """Whether to run the report now (the default) or save the answers and stop."""
    options = ["run now", "save and stop"]
    prompt = "Run the report now, or save the answers and stop?"
    return choose(ask, say, prompt, options, options[0]) == options[0]


def ask_features(ask: Ask, say: Say, candidates: Sequence[str]) -> list[str]:
    """The features to assess, among the scan's numeric columns."""
    return choose_many(ask, say, "Which features do you want to assess?", list(candidates))


def ask_readout(ask: Ask, say: Say, features: Sequence[str]) -> str:
    """The readout, among the chosen features (the only one when there is one)."""
    if len(features) == 1:
        return features[0]
    return choose(ask, say, "Which of them is the readout you report?", list(features), features[0])


def ask_unit(ask: Ask, say: Say, suggested: str | None) -> str:
    """What one row is."""
    options = list(_UNITS)
    return choose(ask, say, "What is one row?", options, suggested or "cell")


def ask_tolerance(ask: Ask, say: Say, features: Sequence[str]) -> float:
    """The fingerprint's tolerance: the covariance of the features within this share."""
    what = (
        f"the standardised features ({listing(list(features), 4)})"
        if features
        else "each table's first principal components"
    )
    say(
        f"The fingerprint judged is a covariance fingerprint: the covariance of {what}, not the "
        "whole distribution. No tolerance has a published meaning: state yours."
    )
    return number(ask, say, "Within what share (0.10 = 10%)?", 0.10, 0.0, 1.0)


__all__ = [
    "AIMS",
    "AIM_LABELS",
    "AIM_QUESTION",
    "AIM_TEXT",
    "RATIO_WORDS",
    "TABLE_AIMS",
    "Ask",
    "Say",
    "ask_aim",
    "ask_control",
    "ask_features",
    "ask_positive",
    "ask_rare_label",
    "ask_readout",
    "ask_run",
    "ask_scales",
    "ask_table_aim",
    "ask_tolerance",
    "ask_unit",
    "choose",
    "choose_many",
    "label_columns",
    "number",
    "scale_lines",
    "text",
    "yes_no",
]

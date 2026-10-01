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
AIMS = {"population": "averages", "fingerprint": "fingerprint", "both": "fingerprint"}
AIM_TEXT = {
    "population": "compare the population averages of the conditions (the readout's mean)",
    "fingerprint": "compare their phenotypic fingerprints (the distribution over the features)",
    "both": "the population averages and the fingerprints",
}
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


def choose_many(ask: Ask, say: Say, prompt: str, options: Sequence[Any]) -> list[Any]:
    """Several of ``options``: ``all`` (the default), numbers, ranges or names."""
    say(f"{prompt}\n{_numbered(options, None)}")
    say("  Enter 'all' (the default), numbers and ranges (1,3-5), or names separated by commas.")
    while True:
        answer = ask("> ").strip()
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
        "Do you have a control you want to log? (the conditions are compared with it)",
        [*conditions, none],
        guess if guess is not None else none,
    )
    return None if choice == none else {"condition": choice}


def ask_aim(ask: Ask, say: Say) -> str:
    """``population``, ``fingerprint`` or ``both``."""
    aims = list(AIMS)
    labels = [f"{key}: {AIM_TEXT[key]}" for key in aims]
    return choose(ask, say, "What is your aim?", aims, aims[0], labels)


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
    say(
        "The fingerprint is the covariance of the standardised features "
        f"({listing(list(features), 4)}). No tolerance has a published meaning: state yours."
    )
    return number(ask, say, "Within what share (0.10 = 10%)?", 0.10, 0.0, 1.0)


__all__ = [
    "AIMS",
    "AIM_TEXT",
    "Ask",
    "Say",
    "ask_aim",
    "ask_control",
    "ask_features",
    "ask_readout",
    "ask_tolerance",
    "ask_unit",
    "choose",
    "choose_many",
    "number",
    "text",
    "yes_no",
]

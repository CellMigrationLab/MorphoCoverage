"""The two kinds of caveat and how a section states them (private), written once.

The project owner's decision (2026-09-28): a caveat is either a **caution**
found in these data (a measured property, count or condition of this dataset
that should change how the answer is read, such as a non-normal readout, a
relation between field size and field mean, only three repeats, unstable
labels; or a missing declaration that kept a check from running, such as no
acquisition order for the order check) or a standing **assumption** of a
method, true of every dataset (the next-field bound is approximate; novelty
bounds hold at a number of fields fixed in advance; hold-outs are not
independent experiments; a fixed amount of effort says nothing about the
data). A caveat is a ``str``; a standing assumption is an ``Assumption``
(``assumption(text)``), marked where it is written, so every section, summary
and view tells the two apart by the one kind.

- ``caveat_lines``: every caveat once, the cautions as ⚠️ lines, then the
  assumptions under a plain "Assumptions:" heading (``assumption_lines``: one
  wrapped paragraph in the short form, one line each in the full form). An
  assumption is never dropped and never takes the ⚠️ marker.
- ``has_caution``: whether a text states a caution (a ⚠️ line); the
  assumptions never count, so they never raise a status
  (``retrospective.report_status.caveat_status``).
- ``pool_assumptions``: a section built from several results with its
  "Assumptions:" blocks as one, at its end, each assumption once.
- ``once``: items without repeats, each keeping its kind.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from ._style import CAUTION, check_detail, wrap


class Assumption(str):
    """A standing assumption of a method: a caveat that holds for every dataset.

    Not a caution about these data: ``caveat_lines`` lists it under
    "Assumptions:", never as a ⚠️ line, and it never raises a status. It is a
    ``str`` (it compares, joins and prints as its text); only its kind differs.
    """

    __slots__ = ()


def assumption(text: str) -> Assumption:
    """``text`` as a standing assumption (see ``Assumption``)."""
    return Assumption(text)


def is_assumption(text: object) -> bool:
    """Whether ``text`` is a standing assumption rather than a caution about these data."""
    return isinstance(text, Assumption)


def once(items: Iterable[str]) -> list[str]:
    """The items without repeats, in first-seen order (blank items dropped); an item keeps
    its kind (``Assumption``), and one given both as a caution and as an assumption is a
    caution (a caution is never hidden among the assumptions)."""
    kinds: dict[str, bool] = {}
    for item in items:
        text = item.strip()
        if text:
            kinds[text] = kinds.get(text, True) and is_assumption(item)
    return [Assumption(text) if standing else text for text, standing in kinds.items()]


# A first word that is a name or an identifier (siCTRL, drugA, pH3, dna_intensity): a
# capital would change it, so the sentence keeps its case.
_IDENTIFIER = re.compile(r"[a-z]\S*?[A-Z\d_]")


def _sentence(text: str, ends: tuple[str, ...] = (".", "?", "!", ")")) -> str:
    """``text`` with a capital first letter (unless its first word is a name, such as
    ``siCTRL``) and a closing full stop (unless it ends with one of ``ends``)."""
    first = text.split(maxsplit=1)[0] if text.strip() else ""
    keep = _IDENTIFIER.match(first) is not None
    sentence = text if keep else text[0].upper() + text[1:]
    return sentence if sentence.endswith(ends) else f"{sentence}."


ASSUMPTIONS = "Assumptions:"


def assumption_lines(
    assumptions: Iterable[str], indent: int | str = 2, *, detail: str = "short"
) -> list[str]:
    """The standing assumptions under a plain "Assumptions:" heading, each once.

    Short: one wrapped paragraph (``Assumptions: A. B.``); full: the heading,
    then one line (wrapped) per assumption. No marker: they hold for every
    dataset.
    """
    prefix = " " * indent if isinstance(indent, int) else indent
    # Run together in a paragraph, each needs its full stop, after a parenthesis too.
    sentences = [_sentence(text, (".", "?", "!")) for text in once(assumptions)]
    if not sentences:
        return []
    if check_detail(detail) == "short":
        return [wrap(f"{ASSUMPTIONS} {' '.join(sentences)}", prefix, hang=prefix + "  ")]
    items = [wrap(f"- {text}", prefix + "  ", hang=prefix + "    ") for text in sentences]
    return [f"{prefix}{ASSUMPTIONS}", *items]


def caveat_lines(
    caveats: Iterable[str], indent: int | str = 2, *, detail: str = "short"
) -> list[str]:
    """Every caveat once: the cautions about these data as wrapped ``⚠️`` lines, then the
    standing assumptions (``Assumption``) under "Assumptions:" (``assumption_lines``)."""
    prefix = " " * indent if isinstance(indent, int) else indent
    found = once(caveats)
    lines = [
        wrap(_sentence(text), f"{prefix}{CAUTION} ", hang=prefix + "   ")
        for text in found
        if not is_assumption(text)
    ]
    standing = [text for text in found if is_assumption(text)]
    return lines + assumption_lines(standing, prefix, detail=detail)


def _assumption_block(lines: list[str], start: int) -> tuple[int, list[str], bool] | None:
    """The "Assumptions:" block at ``lines[start]``: where it ends, its texts (a full block's
    items, or a short block's one paragraph) and whether it is a full block; or ``None``."""
    line = lines[start]
    pad = line[: len(line) - len(line.lstrip())]
    full = line.strip() == ASSUMPTIONS
    if not full and not line.lstrip().startswith(f"{ASSUMPTIONS} "):
        return None
    texts = [] if full else [line.strip().removeprefix(ASSUMPTIONS).strip()]
    end = start + 1
    while end < len(lines):
        following = lines[end]
        if full and following.startswith(f"{pad}  - "):
            texts.append(following.strip()[2:])
        elif following.startswith(pad + ("    " if full else "  ")) and following.strip():
            if not texts:
                break
            texts[-1] += " " + following.strip()
        else:
            break
        end += 1
    return end, texts, full


# Where one sentence of a short "Assumptions:" paragraph ends and the next begins.
_SENTENCE_END = re.compile(r"(?<=[.?!)])\s+(?=[A-Z'\d])")


def pool_assumptions(text: str) -> str:
    """``text`` (a section) with its "Assumptions:" blocks as one, at its end, each
    assumption once: a section built from several results states them once.

    The block takes the form (short paragraph or full list) and the indent of
    the first one found; a short paragraph already contained in an earlier one
    is dropped.
    """
    lines = text.splitlines()
    kept: list[str] = []
    items: list[str] = []
    form: tuple[str, bool] | None = None
    index = 0
    while index < len(lines):
        block = _assumption_block(lines, index)
        if block is None:
            if not (lines[index] == "" and kept and kept[-1] == ""):
                kept.append(lines[index])
            index += 1
            continue
        end, texts, full = block
        line = lines[index]
        form = form or (line[: len(line) - len(line.lstrip())], full)
        if not full:  # a paragraph of sentences: each sentence once
            texts = _SENTENCE_END.split(texts[0])
        items += [t for t in texts if t not in items]
        index = end
    if form is None:
        return text
    while kept and not kept[-1].strip():
        kept.pop()
    prefix, full = form
    if full:
        block_lines = assumption_lines([assumption(t) for t in items], prefix, detail="full")
    else:
        block_lines = [wrap(f"{ASSUMPTIONS} {' '.join(items)}", prefix, hang=prefix + "  ")]
    return "\n".join([*kept, *block_lines])


def has_caution(text: str) -> bool:
    """Whether ``text`` states a caution about these data (a ``⚠️`` line); the assumptions,
    printed without a marker, never count."""
    return any(line.lstrip().startswith(CAUTION) for line in text.splitlines())


__all__ = [
    "ASSUMPTIONS",
    "Assumption",
    "assumption",
    "assumption_lines",
    "caveat_lines",
    "has_caution",
    "is_assumption",
    "once",
    "pool_assumptions",
]

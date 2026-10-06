"""Report text fitted to its 100 columns (private; ``retrospective.report`` and
``workflows._results`` run every summary through ``fitted``; split out of ``_style``).

A prose line wider than 100 columns is wrapped under its own indent together with its
continuation lines, so no word is left alone on a line; table rows stay as they are (audit
10, A22).
"""

from __future__ import annotations

from ._style import _TABLE_GAP, WIDTH, Status, width, wrap


def fitted(text: str) -> str:
    """``text`` with every prose line wider than 100 columns wrapped under its own indent with
    its continuation lines, so no word is left alone; table rows as they are (audit 10, A22)."""
    source, lines, i = text.splitlines(), [], 0
    while i < len(source):
        line = source[i]
        i += 1
        if width(line) <= WIDTH or _TABLE_GAP.search(line.strip()):
            lines.append(line)
            continue
        pad = line[: len(line) - len(line.lstrip(" "))]
        deeper = i < len(source) and _indent_of(source[i]) > len(pad)
        body, hang = line[len(pad) :], pad if not deeper and _inner(source, i - 1) else None
        while not body.endswith(":") and i < len(source) and _continues(source[i], pad, hang):
            hang = source[i][: len(source[i]) - len(source[i].lstrip(" "))]
            body = f"{body} {source[i].strip()}"
            i += 1
        marker = next((x.marker for x in Status if body.startswith(f"{x.marker} ")), None)
        if marker:  # the marker in the indent, where its two columns are counted
            lines.append(wrap(body[len(marker) + 1 :], f"{pad}{marker} ", hang=hang or pad + "   "))
        else:
            lines.append(wrap(body, pad, hang=hang or pad + "  "))
    return "\n".join(lines)


def _indent_of(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _inner(source: list[str], k: int) -> bool:
    """Whether line ``k`` continues a paragraph: above it, past lines in its indent, the
    first other line is prose indented less (a paragraph's first line)."""
    starts = ("- ", *(f"{x.marker} " for x in Status))
    if source[k].lstrip(" ").startswith(starts):
        return False
    for above in reversed(source[:k]):
        if not above.strip() or _TABLE_GAP.search(above.strip()):
            return False
        if _indent_of(above) != _indent_of(source[k]):
            return _indent_of(above) < _indent_of(source[k])
        if above.lstrip(" ").startswith(starts):
            return False
    return False


def _continues(line: str, pad: str, hang: str | None) -> bool:
    """Whether ``line`` continues the prose line above, indented ``pad``: in ``hang``, or deeper
    than ``pad`` when no hang is known; neither a table row nor a new item or marker."""
    body = line.lstrip(" ")
    indent = len(line) - len(body)
    if not body or (indent != len(hang) if hang is not None else indent <= len(pad)):
        return False
    starts = ("- ", *(f"{x.marker} " for x in Status))
    return not _TABLE_GAP.search(body) and not body.startswith(starts)


__all__ = ["fitted"]

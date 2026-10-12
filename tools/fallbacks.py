"""Silent fallbacks: code that turns an error or a bad value into a quiet default.

Found (``ast`` only, names resolved through each module's imports, aliases included):

- ``swallowed error``: an ``except`` handler that does not raise (a ``raise`` inside a nested
  function does not count): it returns or assigns a value, or does ``pass``/``continue``/
  ``...``, or anything else and goes on;
- ``reported fallback``: the same, when the handler calls a reporter (``warnings.warn``,
  ``logging``, a ``warning``/``error``/``say`` call: the person is told);
- ``errors='ignore'`` and ``errors=<expression>`` (a mode that is not a constant);
- ``contextlib.suppress`` (under any alias);
- ``numpy.errstate ignore`` and ``warnings ignore`` (``simplefilter``/``filterwarnings``);
- ``errors='coerce'``: listed only (``REPORT_ONLY``), since a validator that coerces and then
  counts the missing values is the usual design.

A deliberate fallback carries its reason in a comment ``# fallback: <reason>`` on the
``except`` or call line, or alone on the line just above it; the guard counts only the
unmarked ones (``tools/baselines/fallbacks.json``, a ratchet that only shrinks).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import NamedTuple

import callgraph
import function_index as fi

REPORT_ONLY = ("errors='coerce'",)
REPORTERS = ("warn", "warning", "error", "exception", "info", "debug", "critical", "say")
MARKER = re.compile(r"#\s*fallback:\s*(\S.*)$")


class Fallback(NamedTuple):
    """One silent fallback: where it is, what it is and the reason marked on it."""

    module: str
    owner: str
    line: int
    kind: str
    reason: str | None

    @property
    def key(self) -> str:
        """``module:function`` (``<module>`` for module-level code)."""
        return f"{self.module}:{self.owner}"


def _target(package: callgraph.Package, module: str, call: ast.Call) -> str:
    found = package.expression(call.func, module, "")
    if found and found[0] in ("external", "builtin", "def"):
        return found[1]
    func = call.func
    return func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")


def _raises(handler: ast.ExceptHandler) -> bool:
    nodes = fi.own_nodes(handler)  # type: ignore[arg-type]
    return any(isinstance(n, ast.Raise) for n in nodes)


def _reports(package: callgraph.Package, module: str, handler: ast.ExceptHandler) -> bool:
    for node in fi.own_nodes(handler):  # type: ignore[arg-type]
        if isinstance(node, ast.Call):
            target = _target(package, module, node)
            last = target.rsplit(".", 1)[-1].rsplit(":", 1)[-1]
            if target.startswith("logging.") or last.lstrip("_") in REPORTERS:
                return True
    return False


def _ignores(call: ast.Call) -> bool:
    values = [*call.args[:1], *(k.value for k in call.keywords)]
    return any(isinstance(v, ast.Constant) and v.value == "ignore" for v in values)


def _call_kind(package: callgraph.Package, module: str, call: ast.Call) -> str | None:
    target = _target(package, module, call)
    if target in ("contextlib.suppress", "suppress"):
        return "contextlib.suppress"
    if target.endswith("errstate") and any(
        isinstance(k.value, ast.Constant) and k.value.value == "ignore" for k in call.keywords
    ):
        return "numpy.errstate ignore"
    if target in ("warnings.simplefilter", "warnings.filterwarnings") and _ignores(call):
        return "warnings ignore"
    return None


def _errors_kind(keyword: ast.keyword) -> str | None:
    if keyword.arg != "errors":
        return None
    value = keyword.value
    if isinstance(value, ast.Constant):
        return f"errors={value.value!r}" if value.value in ("ignore", "coerce") else None
    return f"errors={ast.unparse(value)}"


def _reason(lines: list[str], *numbers: int) -> str | None:
    """The reason of a ``# fallback:`` marker on one of ``numbers`` or just above the first."""
    first = min(numbers)
    candidates = [*numbers]
    if first >= 2 and lines[first - 2].strip().startswith("#"):
        candidates.append(first - 1)
    for number in candidates:
        found = MARKER.search(lines[number - 1])
        if found:
            return found.group(1).strip()
    return None


def module_fallbacks(
    package: callgraph.Package, module: str, tree: ast.Module, source: str
) -> list[Fallback]:
    """Every silent fallback of one module, in source order."""
    lines = source.splitlines()
    owner = {id(n): q for q, f in fi.walk_functions(tree) for n in ast.walk(f)}
    found: list[Fallback] = []
    for node in ast.walk(tree):
        where = owner.get(id(node), fi.MODULE_ROW)
        if isinstance(node, ast.ExceptHandler) and not _raises(node):
            kind = "reported fallback" if _reports(package, module, node) else "swallowed error"
            found.append(Fallback(module, where, node.lineno, kind, _reason(lines, node.lineno)))
        elif isinstance(node, ast.Call):
            kind = _call_kind(package, module, node)
            if kind:
                found.append(
                    Fallback(module, where, node.lineno, kind, _reason(lines, node.lineno))
                )
            for keyword in node.keywords:
                kind = _errors_kind(keyword)
                if kind:
                    numbers = (node.lineno, keyword.value.lineno)
                    reason = _reason(lines, *numbers)
                    found.append(Fallback(module, where, keyword.value.lineno, kind, reason))
    return sorted(found, key=lambda f: (f.line, f.kind))


def fallbacks(trees: list[tuple[str, ast.Module]] | None = None) -> list[Fallback]:
    """Every silent fallback of the package."""
    trees = fi.parsed() if trees is None else trees
    package = callgraph.Package(trees)
    sources = {fi.module_name(p): p.read_text(encoding="utf-8") for p in fi.modules()}
    out: list[Fallback] = []
    for module, tree in trees:
        out += module_fallbacks(package, module, tree, sources[module])
    return out


def unmarked(found: list[Fallback]) -> dict[str, int]:
    """``{"module:function": count}`` of the enforced fallbacks without a reason."""
    counts: dict[str, int] = {}
    for item in found:
        if item.reason is None and item.kind not in REPORT_ONLY:
            counts[item.key] = counts.get(item.key, 0) + 1
    return counts


def describe(item: Fallback) -> str:
    """One line for a person: where, what, and the reason when marked."""
    path = Path("src/morphocoverage", *item.module.split(".")).with_suffix(".py")
    if not (fi.ROOT / path).exists():
        path = path.with_suffix("") / "__init__.py"
    reason = f": {item.reason}" if item.reason else " (no reason marked)"
    return f"{path}:{item.line} {item.owner} {item.kind}{reason}"

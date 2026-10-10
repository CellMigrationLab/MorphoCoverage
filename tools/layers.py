"""The package's import layers and import cycles, from the source tree alone (ast only).

Layers, lowest first: the core modules (``morphocoverage/*.py``), ``strategies``,
``sampling_unit``, ``retrospective``, ``workflows``. A module imports its own layer or one
below it; the core modules sit beside ``strategies`` (the framework over the strategy
interface), and ``plots_pairwise`` reading ``sampling_unit`` is a recorded exception. Rules
(``RULES``) name what a layer must not import; ``cycles()`` finds import cycles between
modules. ``tests/test_function_guards.py`` fails on any violation or cycle beyond
``tools/baselines/layers.json`` (a ratchet, like the function checks).
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "morphocoverage"
PACKAGES = ("strategies", "sampling_unit", "retrospective", "workflows")

# The layer a module of ``key`` may not import (``core`` is every module outside a package).
RULES: dict[str, tuple[str, ...]] = {
    "core": ("retrospective", "workflows", "sampling_unit"),
    "strategies": ("sampling_unit", "retrospective", "workflows"),
    "sampling_unit": ("strategies", "retrospective", "workflows"),
    "retrospective": ("workflows",),
}


def module_name(path: Path) -> str:
    parts = list(path.relative_to(SOURCE).with_suffix("").parts)
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts) or "__init__"


def layer(module: str) -> str:
    head = module.split(".")[0]
    return head if head in PACKAGES else "core"


def import_graph() -> dict[str, set[str]]:
    """``{module: {modules of the package it imports}}`` (relative imports resolved)."""
    paths = sorted(p for p in SOURCE.rglob("*.py") if "__pycache__" not in p.parts)
    mods = {module_name(p): p for p in paths}
    graph: dict[str, set[str]] = {}
    for name, path in mods.items():
        package = name if path.name == "__init__.py" else ".".join(name.split(".")[:-1])
        found: set[str] = set()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.level:
                found |= _targets(node, package, mods)
        graph[name] = found - {name}
    return graph


def _targets(node: ast.ImportFrom, package: str, mods: dict[str, Path]) -> set[str]:
    base = package.split(".") if package else []
    if node.level > 1:
        base = base[: len(base) - (node.level - 1)]
    prefix = ".".join([*base, *([node.module] if node.module else [])])
    found = set()
    for candidate in [prefix, *(f"{prefix}.{a.name}" for a in node.names)]:
        parts = candidate.split(".")
        while parts and ".".join(parts) not in mods:
            parts.pop()
        if parts:
            found.add(".".join(parts))
    return found


def violations() -> list[str]:
    """``"importer -> imported"`` for every import a layer rule forbids."""
    out = []
    for module, imported in import_graph().items():
        banned = RULES.get(layer(module), ())
        out += [f"{module} -> {t}" for t in sorted(imported) if layer(t) in banned]
    return sorted(out)


def cycles() -> list[str]:
    """Each import cycle between modules, as ``a -> b -> ... -> a`` (strongly connected
    components of more than one module, smallest module name first)."""
    graph = import_graph()
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[str] = []
    on: set[str] = set()
    found: list[list[str]] = []
    counter = [0]

    def visit(v: str) -> None:
        index[v] = low[v] = counter[0]
        counter[0] += 1
        stack.append(v)
        on.add(v)
        for w in sorted(graph.get(v, ())):
            if w not in index:
                visit(w)
                low[v] = min(low[v], low[w])
            elif w in on:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            comp = []
            while True:
                w = stack.pop()
                on.discard(w)
                comp.append(w)
                if w == v:
                    break
            if len(comp) > 1:
                found.append(sorted(comp))

    for node in sorted(graph):
        if node not in index:
            visit(node)
    return sorted(" -> ".join(c) for c in found)


if __name__ == "__main__":
    print("violations:", *violations(), sep="\n  ")
    print("cycles:", *cycles(), sep="\n  ")

"""The package's import layers and import cycles, from the source tree alone (ast only).

Layers (``layer``): the core modules (``morphocoverage/*.py``, ``plots`` included),
``strategies``, ``sampling_unit``, ``retrospective`` and ``workflows``. ``RULES`` is the rule,
stated once here and in ``docs/common/architecture.md``:

- the core and ``strategies`` are the bottom: either may import the other (the core is the
  framework over the strategy interface), and neither imports ``sampling_unit``,
  ``retrospective`` or ``workflows``;
- ``sampling_unit`` imports the core only;
- ``retrospective`` imports the core, ``strategies`` and ``sampling_unit``;
- ``workflows`` imports anything.

Every import of the package is read: relative ones, absolute ones (``import morphocoverage.x``,
``from morphocoverage.x import y``) and ``importlib.import_module`` with a constant name. An
edge is ``runtime`` (executed on import: module level, class bodies, ``try``), ``local``
(inside a function) or ``typing`` (under ``if TYPE_CHECKING:``). ``from . import x`` is an edge
to the submodule ``x`` when there is one, else to the package.

``tests/test_function_guards.py`` fails on an import a rule forbids that
``tools/baselines/layers.json`` does not record, on any import cycle of runtime edges (no
baseline: Python cannot import it), and on a ``local`` or ``typing`` edge inside a cycle of the
whole graph (a back-edge that only defers the cycle) that the baseline does not record. The
baseline holds individual edges, ``"a -> b (kind)": reason``; ``--tighten`` drops the fixed.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "morphocoverage"
BASELINE = ROOT / "tools" / "baselines" / "layers.json"
PACKAGE = "morphocoverage"
PACKAGES = ("strategies", "sampling_unit", "retrospective", "workflows")
RUNTIME, LOCAL, TYPING = "runtime", "local", "typing"

# The layers a module of ``key`` may not import (``core`` is every module outside a package).
RULES: dict[str, tuple[str, ...]] = {
    "core": ("sampling_unit", "retrospective", "workflows"),
    "strategies": ("sampling_unit", "retrospective", "workflows"),
    "sampling_unit": ("strategies", "retrospective", "workflows"),
    "retrospective": ("workflows",),
}
Edges = dict[tuple[str, str], set[str]]


def module_name(path: Path, source: Path = SOURCE) -> str:
    """``retrospective.replay`` for ``retrospective/replay.py``; ``__init__`` for the root."""
    parts = list(path.relative_to(source).with_suffix("").parts)
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts) or "__init__"


def layer(module: str) -> str:
    """The layer of a module: its subpackage, or ``core``."""
    head = module.split(".")[0]
    return head if head in PACKAGES else "core"


def _resolve(dotted: str, names: tuple[str, ...], mods: set[str]) -> set[str]:
    """The modules an import of ``names`` from ``dotted`` (``""`` for the package) reaches: a
    name that is a submodule is that module, any other name the deepest existing module of
    ``dotted``."""
    found = set()
    for name in names or ("",):
        candidate = f"{dotted}.{name}".strip(".") if name else dotted
        if candidate in mods:
            found.add(candidate)
            continue
        parts = dotted.split(".") if dotted else []
        while parts and ".".join(parts) not in mods:
            parts.pop()
        found.add(".".join(parts) or "__init__")
    return found


def _absolute(name: str) -> str | None:
    if name == PACKAGE:
        return ""
    return name[len(PACKAGE) + 1 :] if name.startswith(f"{PACKAGE}.") else None


def _imported(node: ast.AST, package: str, mods: set[str]) -> set[str]:
    """The package's modules that one import statement or ``import_module`` call reaches."""
    if isinstance(node, ast.Import):
        targets = (_absolute(alias.name) for alias in node.names)
        return set().union(*(_resolve(t, (), mods) for t in targets if t is not None))
    if isinstance(node, ast.ImportFrom):
        names = tuple(alias.name for alias in node.names if alias.name != "*")
        if node.level:
            base = package.split(".") if package else []
            base = base[: len(base) - (node.level - 1)] if node.level > 1 else base
            dotted = ".".join([*base, *([node.module] if node.module else [])])
            return _resolve(dotted, names, mods)
        dotted = _absolute(node.module or "")
        return set() if dotted is None else _resolve(dotted, names, mods)
    if isinstance(node, ast.Call) and _is_import_module(node):
        argument = node.args[0] if node.args else None
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
            dotted = _absolute(argument.value)
            return set() if dotted is None else _resolve(dotted, (), mods)
    return set()


def _is_import_module(call: ast.Call) -> bool:
    target = call.func
    name = target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", "")
    return name == "import_module"


def _is_type_checking(test: ast.expr) -> bool:
    name = test.attr if isinstance(test, ast.Attribute) else getattr(test, "id", "")
    return name == "TYPE_CHECKING"


def _visit(node: ast.AST, kind: str, found: list[tuple[ast.AST, str]]) -> None:
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
            _visit(child, LOCAL if kind == RUNTIME else kind, found)
        elif isinstance(child, ast.If) and _is_type_checking(child.test):
            _visit(ast.Module(body=child.body, type_ignores=[]), TYPING, found)
            _visit(ast.Module(body=child.orelse, type_ignores=[]), kind, found)
        else:
            if isinstance(child, ast.Import | ast.ImportFrom | ast.Call):
                found.append((child, kind))
            _visit(child, kind, found)


def edges(source: Path = SOURCE) -> Edges:
    """``{(importer, imported): kinds}`` over the package under ``source``."""
    paths = sorted(p for p in source.rglob("*.py") if "__pycache__" not in p.parts)
    mods = {module_name(p, source): p for p in paths}
    known = set(mods)
    out: Edges = defaultdict(set)
    for name, path in mods.items():
        package = ".".join(path.parent.relative_to(source).parts)
        found: list[tuple[ast.AST, str]] = []
        _visit(ast.parse(path.read_text(encoding="utf-8")), RUNTIME, found)
        for node, kind in found:
            for target in _imported(node, package, known) - {name}:
                out[(name, target)].add(kind)
    return dict(out)


def import_graph(
    source: Path = SOURCE, kinds: tuple[str, ...] = (RUNTIME, LOCAL, TYPING)
) -> dict[str, set[str]]:
    """``{module: {modules it imports}}``, edges of the given kinds only."""
    graph: dict[str, set[str]] = defaultdict(set)
    for (a, b), found in edges(source).items():
        if found & set(kinds):
            graph[a].add(b)
    return dict(graph)


def _label(a: str, b: str, kinds: set[str]) -> str:
    kind = next(k for k in (RUNTIME, LOCAL, TYPING) if k in kinds)
    return f"{a} -> {b} ({kind})"


def violations(source: Path = SOURCE) -> list[str]:
    """``"importer -> imported (kind)"`` for every import a layer rule forbids (the strongest
    kind when there are several)."""
    return sorted(
        _label(a, b, kinds)
        for (a, b), kinds in edges(source).items()
        if layer(b) in RULES.get(layer(a), ())
    )


def _components(graph: dict[str, set[str]]) -> list[list[str]]:
    """The strongly connected components of more than one module (Tarjan)."""
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[str] = []
    on: set[str] = set()
    found: list[list[str]] = []

    def visit(v: str) -> None:
        index[v] = low[v] = len(index)
        stack.append(v)
        on.add(v)
        for w in sorted(graph.get(v, ())):
            if w not in index:
                visit(w)
                low[v] = min(low[v], low[w])
            elif w in on:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            component = []
            while True:
                w = stack.pop()
                on.discard(w)
                component.append(w)
                if w == v:
                    break
            if len(component) > 1:
                found.append(sorted(component))

    for node in sorted(graph):
        if node not in index:
            visit(node)
    return sorted(found)


def cycles(source: Path = SOURCE, kinds: tuple[str, ...] = (RUNTIME,)) -> list[str]:
    """Each import cycle of edges of ``kinds`` (runtime by default: these must not exist), as
    ``a -> b -> ...``: the modules of a strongly connected component, sorted."""
    return [" -> ".join(c) for c in _components(import_graph(source, kinds))]


def back_edges(source: Path = SOURCE) -> list[str]:
    """``"a -> b (local|typing)"`` for each deferred edge inside a cycle of the whole graph:
    an import that is not a runtime one and whose two modules import each other through
    some chain."""
    component = {m: i for i, c in enumerate(_components(import_graph(source))) for m in c}
    return sorted(
        _label(a, b, kinds)
        for (a, b), kinds in edges(source).items()
        if RUNTIME not in kinds and a in component and component[a] == component.get(b)
    )


def read_baseline(path: Path = BASELINE) -> dict[str, dict[str, str]]:
    """``{"violations": {edge: reason}, "back_edges": {edge: reason}}``."""
    if not path.exists():
        return {"violations": {}, "back_edges": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def tighten(path: Path = BASELINE) -> list[str]:
    """Drop the recorded edges that are gone; never adds. Returns the dropped edges."""
    baseline = read_baseline(path)
    current = {"violations": set(violations()), "back_edges": set(back_edges())}
    dropped = [e for kind, edges_ in baseline.items() for e in edges_ if e not in current[kind]]
    kept = {k: {e: r for e, r in v.items() if e in current[k]} for k, v in baseline.items()}
    path.write_text(json.dumps(kept, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return dropped


def main(argv: list[str] | None = None) -> int:
    """Print the violations, runtime cycles and back-edges; ``--tighten`` drops the fixed."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tighten", action="store_true", help="drop fixed baseline edges")
    args = parser.parse_args(argv)
    if args.tighten:
        print(*(tighten() or ["nothing to tighten"]), sep="\n")
    print("violations:", *violations(), sep="\n  ")
    print("runtime cycles:", *cycles(), sep="\n  ")
    print("back-edges:", *back_edges(), sep="\n  ")
    return 0


if __name__ == "__main__":
    sys.exit(main())

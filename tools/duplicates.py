"""Identical and structurally cloned functions, and repeated module constants (ast only).

Two tiers, both read from the function's body without its docstring, normalised on a deep copy
(``normalised``: the parsed tree is never changed):

- identical: the parameters (``self`` included) and the local names are anonymised; globals,
  callees, attributes and constants are kept, so two members of a group behave the same up to
  the names they give their values. ``tests/test_function_guards.py`` fails on an identical
  group of the package that ``tools/baselines/duplicates.json`` does not accept with a reason:
  ``{"<hash>:<members>": "reason"}``, keyed by the normalised hash so renaming a member keeps
  the entry, while a group that gains a member is a new key (``--accept``, ``--tighten``).
- structural: every name, argument and constant anonymised, attributes kept: the same shape
  with possibly other callees, globals or values. Report-only, candidates for a review.

Only bodies of at least ``MIN_NODES`` AST nodes are compared, and never a trivial one
(``pass``, ``...``, a constant return, ``raise NotImplementedError``). ``constants`` lists
upper-case module constants defined in two or more modules and numeric literals of three or
more significant digits used in two or more modules (report-only).

``python tools/duplicates.py`` writes ``docs/common/function_duplicates.json`` over the package
and ``data/`` (without ``data/simulations/decisions/candidates``; tests stay out). Copies the
AST cannot see (statements reordered, a helper inlined, one idea written twice) are proposed
on demand by ``tools/jscpd_review.py``, and the reviewed pairs are recorded in
``tools/baselines/duplicates_reviewed.json``.
"""

from __future__ import annotations

import argparse
import ast
import functools
import hashlib
import importlib.util
import json
import re
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
EXCLUDED = (DATA / "simulations" / "decisions" / "candidates",)
OUTPUT = ROOT / "docs" / "common" / "function_duplicates.json"
BASELINE = ROOT / "tools" / "baselines" / "duplicates.json"
MIN_NODES = 15
Function = ast.FunctionDef | ast.AsyncFunctionDef
Source = tuple[str, Path]

# Fields that differ between Python versions without a change of meaning.
_VERSION_FIELDS = frozenset({"type_comment", "kind", "type_params"})
_CONSTANT_NAME = re.compile(r"_?[A-Z][A-Z0-9_]*")


@functools.cache
def _index() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "function_index", ROOT / "tools/function_index.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def package_sources() -> list[Source]:
    """``(module name, path)`` of every module of the package, in a fixed order."""
    tool = _index()
    return [(tool.module_name(path), path) for path in tool.modules()]


def data_sources() -> list[Source]:
    """``(module name, path)`` of the scripts under ``data/``, the candidates excepted."""
    paths = sorted(
        p
        for p in DATA.rglob("*.py")
        if "__pycache__" not in p.parts and not any(p.is_relative_to(e) for e in EXCLUDED)
    )
    return [(module_for(path), path) for path in paths]


def module_for(path: Path) -> str:
    """The module name used in the reports: the package's own (``retrospective.replay``), or
    the dotted path from the repository root (``data.crick_idr.convert``), the file's stem
    outside the repository."""
    path = path.resolve()
    tool = _index()
    if path.is_relative_to(tool.SOURCE):
        return str(tool.module_name(path))
    if path.is_relative_to(ROOT):
        return ".".join(path.relative_to(ROOT).with_suffix("").parts)
    return path.stem


def stable_dump(node: object) -> str:
    """A serialisation of an AST that is the same on Python 3.11 to 3.13: fields by name,
    ``None`` and empty lists left out, the version-specific fields dropped."""
    if isinstance(node, ast.AST):
        fields = (
            f"{name}={stable_dump(value)}"
            for name, value in ast.iter_fields(node)
            if not (value is None or value == [] or name in _VERSION_FIELDS)
        )
        return f"{type(node).__name__}({','.join(fields)})"
    if isinstance(node, list):
        return "[" + ",".join(stable_dump(value) for value in node) + "]"
    return repr(node)


def _is_docstring(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def _body(function: Function) -> list[ast.stmt]:
    return [s for s in function.body if not _is_docstring(s)]


def local_names(function: Function) -> set[str]:
    """The parameters (``self`` and ``cls`` included) and the names the body binds, those it
    declares ``global`` or ``nonlocal`` excepted; nested functions' names and parameters
    count as local."""
    names: set[str] = set()
    declared: set[str] = set()
    for node in ast.walk(function):
        if isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store | ast.Del):
            names.add(node.id)
        elif isinstance(node, ast.ExceptHandler | ast.MatchAs | ast.MatchStar) and node.name:
            names.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            names.add(node.rest)
        elif node is not function and isinstance(node, Function | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Global | ast.Nonlocal):
            declared.update(node.names)
    return names - declared


class _Anonymised(ast.NodeTransformer):
    """Renames ``names`` (every name when ``None``) to ``_`` and, with ``constants``, replaces
    each constant by its type's name. Only ever applied to a deep copy."""

    def __init__(self, names: set[str] | None, constants: bool) -> None:
        self.names = names
        self.constants = constants

    def _hidden(self, name: str) -> str:
        return "_" if self.names is None or name in self.names else name

    def visit_Name(self, node: ast.Name) -> ast.AST:
        node.id = self._hidden(node.id)
        return node

    def visit_arg(self, node: ast.arg) -> ast.AST:
        node.arg = "_"
        return self.generic_visit(node)

    def _named(self, node: ast.AST) -> ast.AST:
        if getattr(node, "name", None):
            node.name = self._hidden(node.name)  # type: ignore[attr-defined]
        return self.generic_visit(node)

    visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = _named
    visit_ExceptHandler = visit_MatchAs = visit_MatchStar = _named

    def visit_MatchMapping(self, node: ast.MatchMapping) -> ast.AST:
        if node.rest:
            node.rest = self._hidden(node.rest)
        return self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if self.constants:
            node.value = type(node.value).__name__
        return node


def _copy(node: object) -> object:
    # A deep copy of the fields the hash reads (positions left out), several times faster
    # than ``copy.deepcopy`` on the package's 2600 functions.
    if isinstance(node, ast.AST):
        clone = type(node).__new__(type(node))
        for name, value in ast.iter_fields(node):
            setattr(clone, name, _copy(value))
        return clone
    if isinstance(node, list):
        return [_copy(value) for value in node]
    return node


def normalised(function: Function, structural: bool = False) -> list[ast.stmt]:
    """A deep copy of the body without its docstring, local names and parameters anonymised
    (every name and constant too when ``structural``). ``function`` is left unchanged."""
    transformer = _Anonymised(None if structural else local_names(function), structural)
    return [transformer.visit(_copy(statement)) for statement in _body(function)]


def digest(function: Function, structural: bool = False) -> str:
    """The hash of the normalised body: equal for the members of one group of that tier."""
    text = "".join(stable_dump(s) for s in normalised(function, structural))
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def size(function: Function) -> int:
    """The number of AST nodes in the body, docstring excluded."""
    return sum(1 for statement in _body(function) for _ in ast.walk(statement))


def trivial(function: Function) -> bool:
    """Whether the body is one statement that every stub shares: ``pass``, ``...``, a constant
    return or ``raise NotImplementedError``."""
    body = _body(function)
    if len(body) != 1:
        return not body
    only = body[0]
    if isinstance(only, ast.Pass):
        return True
    if isinstance(only, ast.Expr):
        return isinstance(only.value, ast.Constant) and only.value.value is Ellipsis
    if isinstance(only, ast.Return):
        return only.value is None or isinstance(only.value, ast.Constant)
    if isinstance(only, ast.Raise) and only.exc is not None:
        target = only.exc.func if isinstance(only.exc, ast.Call) else only.exc
        return isinstance(target, ast.Name) and target.id == "NotImplementedError"
    return False


def compared(sources: Iterable[Source]) -> dict[str, Function]:
    """``{"module:qualified name": node}`` of every function large enough to compare."""
    walk = _index().walk_functions
    out: dict[str, Function] = {}
    for module, path in sources:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for qualified, node in walk(tree):
            if size(node) >= MIN_NODES and not trivial(node):
                out[f"{module}:{qualified}"] = node
    return out


def _groups(functions: dict[str, Function], structural: bool) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for key, node in functions.items():
        groups[digest(node, structural)].append(key)
    return {d: sorted(members) for d, members in sorted(groups.items()) if len(members) > 1}


def find(sources: Sequence[Source] | None = None) -> dict[str, dict[str, list[str]]]:
    """``{"identical": {hash: members}, "structural": {hash: members}}`` over ``sources`` (the
    package by default); a structural group is listed only when its members are not all one
    identical group."""
    functions = compared(package_sources() if sources is None else sources)
    identical = _groups(functions, structural=False)
    exact = {key: d for d, members in identical.items() for key in members}
    structural = {
        d: members
        for d, members in _groups(functions, structural=True).items()
        if len({exact.get(m, m) for m in members}) > 1
    }
    return {"identical": identical, "structural": structural}


def exact_groups(sources: Sequence[Source] | None = None) -> dict[str, list[str]]:
    """The identical groups keyed as in the baseline (``"<hash>:<members>"``), the package by
    default."""
    functions = compared(package_sources() if sources is None else sources)
    return {f"{d}:{len(m)}": m for d, m in _groups(functions, structural=False).items()}


def _significant(value: float) -> int:
    mantissa = repr(abs(value)).split("e")[0].replace(".", "")
    return len(mantissa.strip("0"))


def _module_assignments(tree: ast.Module) -> Iterable[tuple[str, ast.expr]]:
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        yield from ((t.id, value) for t in targets if isinstance(t, ast.Name))


def constants(sources: Sequence[Source]) -> dict[str, dict[str, list[str]]]:
    """Report-only: ``{"names": {NAME: ["module = value", ...]}, "literals": {value: [...]}}``
    for an upper-case module constant assigned in two or more modules and a numeric literal
    of three or more significant digits (a float rounded to four digits, so ``1.4826`` and
    ``1.482602218505602`` meet) used in two or more modules."""
    names: dict[str, list[str]] = defaultdict(list)
    literals: dict[str, set[str]] = defaultdict(set)
    for module, path in sources:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for name, value in _module_assignments(tree):
            if _CONSTANT_NAME.fullmatch(name):
                names[name].append(f"{module} = {ast.unparse(value)[:80]}")
        for node in ast.walk(tree):
            number = node.value if isinstance(node, ast.Constant) else None
            if (
                isinstance(number, int | float)
                and not isinstance(number, bool)
                and _significant(number) >= 3
            ):
                key = f"{number:.4g}" if isinstance(number, float) else str(number)
                literals[key].add(f"{module}: {number!r}")
    return {
        "names": {
            name: sorted(entries)
            for name, entries in sorted(names.items())
            if len({e.split(" = ")[0] for e in entries}) > 1
        },
        "literals": {
            value: sorted(where)
            for value, where in sorted(literals.items())
            if len({w.split(": ")[0] for w in where}) > 1
        },
    }


def report(sources: Sequence[Source] | None = None) -> dict[str, object]:
    """The report written to ``docs/common/function_duplicates.json``: both tiers and the
    constants, over the package and ``data/`` by default."""
    sources = [*package_sources(), *data_sources()] if sources is None else sources
    groups = find(sources)
    return {
        "identical": sorted(groups["identical"].values()),
        "structural": sorted(groups["structural"].values()),
        "constants": constants(sources),
    }


def write(payload: dict[str, object], path: Path = OUTPUT) -> None:
    """Write the report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def read_baseline(path: Path = BASELINE) -> dict[str, str]:
    """The accepted identical groups, ``{"<hash>:<members>": reason}``."""
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _write_baseline(baseline: dict[str, str], path: Path) -> None:
    text = json.dumps(dict(sorted(baseline.items())), indent=1, ensure_ascii=False)
    path.write_text(text + "\n", encoding="utf-8")


def tighten(path: Path = BASELINE) -> list[str]:
    """Drop the accepted groups that were merged and record a group that lost members under
    its new key, reason kept; never adds. Returns what changed."""
    current = {key.split(":")[0]: len(m) for key, m in exact_groups().items()}
    baseline, changed = {}, []
    for key, reason in read_baseline(path).items():
        digest_, count = key.split(":")
        now = current.get(digest_, 0)
        if now == 0:
            changed.append(f"dropped {key}")
        elif now < int(count):
            baseline[f"{digest_}:{now}"] = reason
            changed.append(f"{key} -> {digest_}:{now}")
        else:
            baseline[key] = reason
    _write_baseline(baseline, path)
    return changed


def accept(key: str, reason: str, path: Path = BASELINE) -> str:
    """Accept the current identical group ``key`` (its hash, with or without ``:<members>``)
    with a reason; returns the recorded key."""
    if not reason.strip():
        raise ValueError("an accepted duplicate group needs a reason (--reason)")
    matches = [k for k in exact_groups() if k == key or k.split(":")[0] == key]
    if not matches:
        raise ValueError(f"no current identical group {key!r}: run `python tools/duplicates.py`")
    baseline = read_baseline(path)
    baseline[matches[0]] = reason.strip()
    _write_baseline(baseline, path)
    return matches[0]


def main(argv: list[str] | None = None) -> int:
    """Write the report; ``--tighten`` or ``--accept KEY --reason TEXT`` edit the baseline."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tighten", action="store_true", help="drop merged groups")
    parser.add_argument("--accept", metavar="KEY", help="accept an identical group")
    parser.add_argument("--reason", default="", help="why the accepted group stays")
    args = parser.parse_args(argv)
    if args.accept:
        print(f"accepted {accept(args.accept, args.reason)}")
        return 0
    if args.tighten:
        print(*(tighten() or ["nothing to tighten"]), sep="\n")
    payload = report()
    write(payload)
    groups = exact_groups()
    unaccepted = sorted(set(groups) - set(read_baseline()))
    counts = {k: len(v) for k, v in payload.items() if isinstance(v, list)}
    print(
        f"{counts['identical']} identical groups, {counts['structural']} structural groups "
        f"(package and data/); {len(unaccepted)} identical groups of the package not accepted"
    )
    for key in unaccepted:
        print(f"  {key}: {' | '.join(groups[key])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

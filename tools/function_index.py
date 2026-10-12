"""The function index and the mechanical function metrics, from the source tree alone.

``python tools/audit.py`` writes ``docs/common/function_index.jsonl``: one JSON line per
module, class and function of the package, keyed by ``module`` and ``name`` (the qualified
name, unique within a module: a property setter is ``name@setter``, a second definition of the
same name ``name#2``). Every row has ``kind`` (``module``, ``class``, ``function``,
``method``, ``property``, ``classmethod``, ``staticmethod`` or ``nested``), ``summary`` (the
first line of the docstring) and ``public``; class and function rows add ``decorators``,
``signature`` (a class's bases), ``hash`` and ``called_by``; function rows add the metrics
(``lines``, ``complexity``, ``arguments``, ``varargs``) and ``calls``. Line numbers are left
out on purpose: they move without a change of meaning and made every edit rewrite rows.

It parses with ``ast`` and imports nothing from the package, so it cannot hang or need the
environment, and it gives the same index on every Python version (``stable_dump``).

Metrics, like ``mccabe``/ruff ``C901`` on the function's own body (nested functions, classes
and lambdas are rows or expressions of their own and do not count):

- ``complexity``: one, plus one per ``if``/``elif``, ``for``, ``while``, ``except``, ``case``
  and conditional expression, plus one per extra operand of ``and``/``or``;
- ``lines``: from the first statement after the docstring to the end of the function, so a
  long docstring never pushes a function over the limit;
- ``arguments``: the named parameters, ``self`` and ``cls`` not counted; ``*args`` and
  ``**kwargs`` are listed in ``varargs`` and not counted.
"""

from __future__ import annotations

import ast
import hashlib
import json
import sys
from collections.abc import Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "morphocoverage"
INDEX = ROOT / "docs" / "common" / "function_index.jsonl"
BASELINES = ROOT / "tools" / "baselines"
MODULE_ROW = "<module>"

MAX_LINES = 60
MAX_COMPLEXITY = 15
MAX_ARGUMENTS = 8
_BRANCHES = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.ExceptHandler, ast.IfExp)
_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
# Fields whose presence or default differs between Python versions and carries no meaning.
_UNSTABLE_FIELDS = ("type_comment", "kind", "type_params")
Function = ast.FunctionDef | ast.AsyncFunctionDef
Definition = ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef


def modules() -> list[Path]:
    """Every module of the package, in a fixed order."""
    return sorted(p for p in SOURCE.rglob("*.py") if "__pycache__" not in p.parts)


def module_name(path: Path) -> str:
    """``retrospective.replay`` for ``src/morphocoverage/retrospective/replay.py``."""
    parts = list(path.relative_to(SOURCE).with_suffix("").parts)
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts) or "__init__"


def _decorator_suffix(node: Definition) -> str:
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Attribute) and decorator.attr in ("setter", "deleter"):
            return f"@{decorator.attr}"
    return ""


def walk_definitions(tree: ast.AST) -> Iterator[tuple[str, Definition, str]]:
    """``(unique qualified name, node, kind)`` of every class and function, nested included.

    A property setter or deleter is named ``name@setter``/``name@deleter``; any other repeated
    qualified name (a conditional definition, an overload) gets ``#2``, ``#3``, in order."""
    seen: dict[str, int] = {}

    def unique(name: str) -> str:
        seen[name] = seen.get(name, 0) + 1
        return name if seen[name] == 1 else f"{name}#{seen[name]}"

    def visit(node: ast.AST, prefix: str, parent: str) -> Iterator[tuple[str, Definition, str]]:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                name = unique(f"{prefix}{child.name}")
                yield name, child, "class"
                yield from visit(child, f"{name}.", "class")
            elif isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                name = unique(f"{prefix}{child.name}{_decorator_suffix(child)}")
                yield name, child, function_kind(child, parent)
                yield from visit(child, f"{name}.", "function")
            else:
                yield from visit(child, prefix, parent)

    yield from visit(tree, "", "module")


def walk_functions(tree: ast.AST) -> Iterator[tuple[str, Function]]:
    """``(unique qualified name, node)`` of every function and method, nested ones included."""
    for name, node, kind in walk_definitions(tree):
        if kind != "class":
            yield name, node  # type: ignore[misc]


def function_kind(node: Function, parent: str) -> str:
    """``function``, ``method``, ``property``, ``classmethod``, ``staticmethod`` or ``nested``
    (a function defined inside another function)."""
    if parent == "function":
        return "nested"
    if parent == "module":
        return "function"
    names = {ast.unparse(d).split("(")[0] for d in node.decorator_list}
    if names & {"property", "cached_property", "functools.cached_property"} or any(
        n.endswith((".setter", ".deleter")) for n in names
    ):
        return "property"
    for kind in ("classmethod", "staticmethod"):
        if kind in names:
            return kind
    return "method"


def own_nodes(node: Function, lambdas: bool = False) -> Iterator[ast.AST]:
    """The nodes of a function's own body: its signature, decorators, nested functions and
    classes excluded, and lambdas too unless ``lambdas``."""
    scopes = _SCOPES[:-1] if lambdas else _SCOPES

    def below(parent: ast.AST) -> Iterator[ast.AST]:
        for child in ast.iter_child_nodes(parent):
            if not isinstance(child, scopes):
                yield child
                yield from below(child)

    for statement in node.body:
        if not isinstance(statement, scopes):
            yield statement
            yield from below(statement)


def complexity(node: Function) -> int:
    """Cyclomatic complexity of the function's own body, counted like ``mccabe``."""
    total = 1
    for child in own_nodes(node):
        if isinstance(child, (*_BRANCHES, ast.match_case)):
            total += 1
        elif isinstance(child, ast.BoolOp):
            total += len(child.values) - 1
    return total


def _body(node: Definition | ast.Module) -> list[ast.stmt]:
    return [n for n in node.body if not (n is node.body[0] and _is_docstring(n))]


def code_lines(node: Function) -> int:
    """Lines from the first statement after the docstring to the end of the function."""
    body = _body(node)
    if not body or node.end_lineno is None:
        return 0
    return node.end_lineno - body[0].lineno + 1


def arguments(node: Function) -> int:
    """The number of named parameters, ``self`` and ``cls`` not counted (``varargs`` apart)."""
    args = node.args
    names = [a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)]
    return len([n for n in names if n not in ("self", "cls")])


def varargs(node: Function) -> list[str]:
    """``*args`` and ``**kwargs`` as written, counted apart from ``arguments``."""
    args = node.args
    return [
        *([f"*{args.vararg.arg}"] if args.vararg else []),
        *([f"**{args.kwarg.arg}"] if args.kwarg else []),
    ]


def signature(node: Function) -> str:
    """The function's signature as written (annotations kept; ``async`` marked)."""
    text = ast.unparse(node.args)
    returns = f" -> {ast.unparse(node.returns)}" if node.returns is not None else ""
    marker = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
    return f"{marker}({text}){returns}"


def stable_dump(node: object) -> str:
    """A dump of an AST that is the same on every Python version: no positions, fields that
    are ``None`` or empty left out, constants by ``repr``.

    ``ast.dump`` is not: 3.11, 3.12 and 3.13 differ in the defaults they print."""
    if isinstance(node, ast.AST):
        parts = [
            f"{name}={stable_dump(value)}"
            for name, value in ast.iter_fields(node)
            if name not in _UNSTABLE_FIELDS and value is not None and value != []
        ]
        return f"{type(node).__name__}({','.join(parts)})"
    if isinstance(node, list):
        return "[" + ",".join(stable_dump(v) for v in node) + "]"
    return repr(node)


def body_hash(node: Definition) -> str:
    """A hash of the body as written, docstring left out (names, constants and nested
    definitions included): equal only for literal copies, stable across Python versions."""
    text = "".join(stable_dump(n) for n in _body(node))
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def class_hash(node: ast.ClassDef) -> str:
    """A hash of a class's bases, decorators and statements other than its methods (fields,
    class attributes): what changes the class without changing a method row."""
    own = [n for n in _body(node) if not isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)]
    text = "".join(stable_dump(n) for n in [*node.bases, *node.keywords, *own])
    text += "".join(stable_dump(d) for d in node.decorator_list)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _is_docstring(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def first_line(node: Definition | ast.Module) -> str:
    """The first line of the docstring, ``""`` without one."""
    text = (ast.get_docstring(node) or "").strip()
    return text.splitlines()[0].strip() if text else ""


def _public(name: str, kind: str) -> bool:
    return kind != "nested" and not name.split(".")[-1].startswith("_")


def record(module: str, name: str, node: Definition, kind: str) -> dict[str, object]:
    """One index row of a class or function (``calls`` and ``called_by`` filled later)."""
    row: dict[str, object] = {
        "module": module,
        "name": name,
        "kind": kind,
        "decorators": [ast.unparse(d) for d in node.decorator_list],
        "summary": first_line(node),
        "public": _public(name, kind),
        "called_by": [],
    }
    if isinstance(node, ast.ClassDef):
        bases = [ast.unparse(b) for b in [*node.bases, *node.keywords]]
        row.update(signature=f"({', '.join(bases)})", hash=class_hash(node))
        return row
    row.update(
        signature=signature(node),
        lines=code_lines(node),
        complexity=complexity(node),
        arguments=arguments(node),
        varargs=varargs(node),
        calls=[],
        hash=body_hash(node),
    )
    return row


def parsed() -> list[tuple[str, ast.Module]]:
    """``(module name, tree)`` of every module, in a fixed order."""
    return [(module_name(p), ast.parse(p.read_text(encoding="utf-8"))) for p in modules()]


def build_index(trees: list[tuple[str, ast.Module]] | None = None) -> list[dict[str, object]]:
    """The index of the whole package: module, class and function rows, calls resolved."""
    trees = parsed() if trees is None else trees
    rows: list[dict[str, object]] = []
    for module, tree in trees:
        public = not any(part.startswith("_") for part in module.split("."))
        rows.append(
            {
                "module": module,
                "name": MODULE_ROW,
                "kind": "module",
                "summary": first_line(tree),
                "public": public or module == "__init__",
            }
        )
        rows += [record(module, q, n, k) for q, n, k in walk_definitions(tree)]
    _sibling("callgraph").fill_calls(rows, trees)
    return rows


def function_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """The rows of functions and methods (not modules or classes)."""
    return [r for r in rows if r["kind"] not in ("module", "class")]


def key(row: dict[str, object]) -> str:
    """``module:name``, the key of a row in the baselines and in ``calls``."""
    return f"{row['module']}:{row['name']}"


def row_text(row: dict[str, object]) -> str:
    """A row as it is written to the index (sorted keys, one line)."""
    return json.dumps(row, sort_keys=True, ensure_ascii=False)


def read_index(path: Path = INDEX) -> list[dict[str, object]]:
    """The committed index (empty when absent)."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_index(rows: list[dict[str, object]]) -> None:
    """Write the index, one row per line."""
    INDEX.parent.mkdir(parents=True, exist_ok=True)
    INDEX.write_text("".join(row_text(r) + "\n" for r in rows), encoding="utf-8")


def _sibling(name: str) -> object:
    """Another module of ``tools/`` (the folder is not a package)."""
    folder = str(Path(__file__).resolve().parent)
    if folder not in sys.path:
        sys.path.insert(0, folder)
    return __import__(name)


if __name__ == "__main__":
    print("run `python tools/audit.py` (it writes the index and checks the baselines)")

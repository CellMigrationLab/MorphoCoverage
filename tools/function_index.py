"""The function index and the mechanical function checks, from the source tree alone.

``python tools/function_index.py`` writes ``docs/common/function_index.jsonl`` (one JSON line
per function: module, qualified name, signature, the first line of its docstring, size,
complexity, arguments, callees and a hash of its normalised body) and the findings the
guards read. It parses with ``ast`` and imports nothing, so it cannot hang or need the
environment, and it is deterministic: line numbers are the only thing that moves without a
change of meaning, so the drift check ignores them.

Checks (``findings``): silent fallbacks (a handler that swallows an error, ``errors="ignore"``,
``contextlib.suppress``), size (lines), containment (complexity, arguments), missing public
docstrings. ``tests/test_function_guards.py`` fails on anything new beyond the baselines in
``tools/baselines/`` (a ratchet: the numbers only go down; ``--write-baselines`` records the
current state, and a removal is the only edit that should be made by hand).
"""

from __future__ import annotations

import argparse
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

MAX_LINES = 60
MAX_COMPLEXITY = 15
MAX_ARGUMENTS = 8
_BRANCHES = (
    ast.If,
    ast.For,
    ast.While,
    ast.ExceptHandler,
    ast.With,
    ast.BoolOp,
    ast.IfExp,
    ast.comprehension,
    ast.Assert,
)
Function = ast.FunctionDef | ast.AsyncFunctionDef


def modules() -> list[Path]:
    """Every module of the package, in a fixed order."""
    return sorted(p for p in SOURCE.rglob("*.py") if "__pycache__" not in p.parts)


def module_name(path: Path) -> str:
    """``retrospective.replay`` for ``src/morphocoverage/retrospective/replay.py``."""
    parts = list(path.relative_to(SOURCE).with_suffix("").parts)
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts) or "__init__"


def walk_functions(tree: ast.AST) -> Iterator[tuple[str, Function]]:
    """``(qualified name, node)`` of every function and method, nested ones included."""

    def visit(node: ast.AST, prefix: str) -> Iterator[tuple[str, Function]]:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                name = f"{prefix}{child.name}"
                yield name, child
                yield from visit(child, f"{name}.")
            elif isinstance(child, ast.ClassDef):
                yield from visit(child, f"{prefix}{child.name}.")
            else:
                yield from visit(child, prefix)

    yield from visit(tree, "")


def complexity(node: Function) -> int:
    """Cyclomatic complexity: one plus the branching constructs in the function."""
    return 1 + sum(isinstance(n, _BRANCHES) for n in ast.walk(node))


def arguments(node: Function) -> int:
    """The number of parameters, ``self`` and ``cls`` not counted."""
    args = node.args
    names = [a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)]
    return (
        len([n for n in names if n not in ("self", "cls")]) + bool(args.vararg) + bool(args.kwarg)
    )


def signature(node: Function) -> str:
    """The function's signature as written (annotations kept)."""
    text = ast.unparse(node.args)
    returns = f" -> {ast.unparse(node.returns)}" if node.returns is not None else ""
    return f"({text}){returns}"


def callees(node: Function) -> list[str]:
    """The names this function calls (attribute calls by their last name), sorted."""
    names = set()
    for call in (n for n in ast.walk(node) if isinstance(n, ast.Call)):
        target = call.func
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, ast.Attribute):
            names.add(target.attr)
    return sorted(names)


def body_hash(node: Function) -> str:
    """A hash of the body with its docstring and local names ignored: equal for copies."""
    body = [n for n in node.body if not _is_docstring(n)]
    text = "".join(ast.dump(n, annotate_fields=False) for n in body)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _is_docstring(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def first_line(node: Function) -> str:
    """The first line of the docstring, ``""`` without one."""
    text = ast.get_docstring(node) or ""
    return text.strip().splitlines()[0].strip() if text.strip() else ""


def record(module: str, name: str, node: Function) -> dict[str, object]:
    """One index line."""
    return {
        "module": module,
        "name": name,
        "signature": signature(node),
        "summary": first_line(node),
        "lines": node.end_lineno - node.lineno + 1 if node.end_lineno else 0,
        "line": node.lineno,
        "complexity": complexity(node),
        "arguments": arguments(node),
        "calls": callees(node),
        "hash": body_hash(node),
        "public": not name.split(".")[-1].startswith("_"),
    }


def build_index() -> list[dict[str, object]]:
    """The index of the whole package."""
    rows: list[dict[str, object]] = []
    for path in modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        name = module_name(path)
        rows += [record(name, qualified, node) for qualified, node in walk_functions(tree)]
    return rows


def silent_fallbacks(path: Path) -> list[tuple[str, int, str]]:
    """``(function, line, kind)`` of the silent fallbacks in a module: a handler that swallows
    the error (no ``raise``, only ``pass``/``continue``/a constant return), ``errors="ignore"``
    or ``"coerce"`` and ``contextlib.suppress``."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[tuple[str, int, str]] = []
    owner = {id(n): q for q, f in walk_functions(tree) for n in ast.walk(f)}
    for node in ast.walk(tree):
        where = owner.get(id(node), "<module>")
        if isinstance(node, ast.ExceptHandler) and _swallows(node):
            found.append((where, node.lineno, "swallowed error"))
        if (
            isinstance(node, ast.keyword)
            and node.arg == "errors"
            and isinstance(node.value, ast.Constant)
            and node.value.value in ("ignore", "coerce")
        ):
            found.append((where, node.value.lineno, f"errors={node.value.value!r}"))
        if isinstance(node, ast.Call) and _is_suppress(node):
            found.append((where, node.lineno, "contextlib.suppress"))
    return found


def _swallows(handler: ast.ExceptHandler) -> bool:
    if any(isinstance(n, ast.Raise) for b in handler.body for n in ast.walk(b)):
        return False
    return all(
        isinstance(b, ast.Pass | ast.Continue)
        or (isinstance(b, ast.Return) and (b.value is None or isinstance(b.value, ast.Constant)))
        for b in handler.body
    )


def _is_suppress(call: ast.Call) -> bool:
    target = call.func
    return (isinstance(target, ast.Name) and target.id == "suppress") or (
        isinstance(target, ast.Attribute) and target.attr == "suppress"
    )


def findings() -> dict[str, dict[str, int]]:
    """Every baseline-able finding: ``{kind: {"module:name": value}}``.

    ``lines``, ``complexity`` and ``arguments`` map a function to its size when it is over the
    limit; ``no_docstring`` maps a public function without a docstring to 1; ``fallbacks``
    counts the silent fallbacks of a function."""
    out: dict[str, dict[str, int]] = {
        "lines": {},
        "complexity": {},
        "arguments": {},
        "no_docstring": {},
        "fallbacks": {},
    }
    for row in build_index():
        key = f"{row['module']}:{row['name']}"
        for kind, limit in (
            ("lines", MAX_LINES),
            ("complexity", MAX_COMPLEXITY),
            ("arguments", MAX_ARGUMENTS),
        ):
            if int(row[kind]) > limit:  # type: ignore[call-overload]
                out[kind][key] = int(row[kind])  # type: ignore[call-overload]
        if row["public"] and not row["summary"]:
            out["no_docstring"][key] = 1
    for path in modules():
        counts: dict[str, int] = {}
        for where, _, _ in silent_fallbacks(path):
            key = f"{module_name(path)}:{where}"
            counts[key] = counts.get(key, 0) + 1
        out["fallbacks"].update(counts)
    return out


def index_key(rows: list[dict[str, object]]) -> list[tuple[object, ...]]:
    """What the drift check compares: everything but the line numbers."""
    return [(r["module"], r["name"], r["signature"], r["summary"], r["hash"]) for r in rows]


def read_index() -> list[dict[str, object]]:
    """The committed index (empty when absent)."""
    if not INDEX.exists():
        return []
    return [json.loads(line) for line in INDEX.read_text(encoding="utf-8").splitlines() if line]


def write_index(rows: list[dict[str, object]]) -> None:
    INDEX.parent.mkdir(parents=True, exist_ok=True)
    INDEX.write_text(
        "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )


def read_baseline(kind: str) -> dict[str, int]:
    path = BASELINES / f"{kind}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_baselines(current: dict[str, dict[str, int]]) -> None:
    BASELINES.mkdir(parents=True, exist_ok=True)
    for kind, values in current.items():
        text = json.dumps(dict(sorted(values.items())), indent=1, ensure_ascii=False)
        (BASELINES / f"{kind}.json").write_text(text + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--write-baselines", action="store_true", help="record the current findings"
    )
    args = parser.parse_args(argv)
    rows = build_index()
    write_index(rows)
    current = findings()
    if args.write_baselines:
        write_baselines(current)
    print(f"{len(rows)} functions indexed")
    for kind, values in current.items():
        print(f"  {kind}: {len(values)} over the baseline-able limit")
    return 0


if __name__ == "__main__":
    sys.exit(main())

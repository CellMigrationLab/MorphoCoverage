"""Exact and near-duplicate functions, from the source tree alone (ast only).

Exact: equal normalised bodies (``function_index.body_hash``), bodies of at least
``MIN_LINES`` lines. Near: the Jaccard similarity of the sets of consecutive-statement
shingles of two bodies of at least ``MIN_LINES`` lines reaches ``NEAR`` while the bodies
differ. Both are candidates for a human or model review (merge, keep with a reason), not
verdicts. ``tests/test_function_guards.py`` fails on a new exact group beyond
``tools/baselines/duplicates.json``; near duplicates are listed in
``docs/common/function_duplicates.json`` and only reported.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "common" / "function_duplicates.json"
MIN_LINES = 6
NEAR = 0.5
SHINGLE = 2


def _index():
    spec = importlib.util.spec_from_file_location(
        "function_index", ROOT / "tools/function_index.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Anonymise(ast.NodeTransformer):
    """Local names, arguments and literal values replaced by placeholders: copies with renamed
    variables or other constants still match."""

    def visit_Name(self, node: ast.Name) -> ast.AST:
        return ast.copy_location(ast.Name(id="_", ctx=node.ctx), node)

    def visit_arg(self, node: ast.arg) -> ast.AST:
        return ast.copy_location(ast.arg(arg="_"), node)

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        return ast.copy_location(ast.Constant(value=type(node.value).__name__), node)


def _dump(statement: ast.stmt) -> str:
    return ast.dump(_Anonymise().visit(statement), annotate_fields=False)


def shingles(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """Hashes of runs of ``SHINGLE`` consecutive statements, names normalised away."""
    body = [
        _dump(n)
        for n in node.body
        if not (
            isinstance(n, ast.Expr)
            and isinstance(n.value, ast.Constant)
            and isinstance(n.value.value, str)
        )
    ]
    if len(body) < SHINGLE:
        return {hashlib.md5("".join(body).encode()).hexdigest()}
    return {
        hashlib.md5("".join(body[i : i + SHINGLE]).encode()).hexdigest()
        for i in range(len(body) - SHINGLE + 1)
    }


def find() -> tuple[dict[str, list[str]], list[dict[str, object]]]:
    """``(exact, near)``: exact groups by hash (their ``module:name`` members) and near pairs."""
    tool = _index()
    sets: dict[str, set[str]] = {}
    sizes: dict[str, int] = {}
    hashes: dict[str, str] = {}
    for path in tool.modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for qualified, node in tool.walk_functions(tree):
            lines = (node.end_lineno or node.lineno) - node.lineno + 1
            if lines < MIN_LINES:
                continue
            key = f"{tool.module_name(path)}:{qualified}"
            sets[key], sizes[key], hashes[key] = shingles(node), lines, tool.body_hash(node)
    groups: dict[str, list[str]] = defaultdict(list)
    for key, digest in hashes.items():
        groups[digest].append(key)
    exact = {d: sorted(m) for d, m in groups.items() if len(m) > 1}
    near = _near(sets, hashes)
    return exact, near


def _near(sets: dict[str, set[str]], hashes: dict[str, str]) -> list[dict[str, object]]:
    # Candidates share at least one shingle: an inverted index keeps this far below all pairs.
    owners: dict[str, list[str]] = defaultdict(list)
    for key, shingle_set in sets.items():
        for s in shingle_set:
            owners[s].append(key)
    pairs: set[tuple[str, str]] = set()
    for members in owners.values():
        if 1 < len(members) <= 12:
            pairs.update(combinations(sorted(members), 2))
    out = []
    for a, b in sorted(pairs):
        if hashes[a] == hashes[b]:
            continue
        union = len(sets[a] | sets[b])
        score = len(sets[a] & sets[b]) / union if union else 0.0
        if score >= NEAR:
            out.append({"a": a, "b": b, "similarity": round(score, 3)})
    return sorted(out, key=lambda r: (-float(r["similarity"]), str(r["a"])))  # type: ignore[arg-type]


def write(exact: dict[str, list[str]], near: list[dict[str, object]]) -> None:
    payload = {"exact": sorted(exact.values()), "near": near}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def exact_groups() -> list[str]:
    """The exact groups as stable keys (``a | b``) for the baseline."""
    return sorted(" | ".join(g) for g in find()[0].values())


if __name__ == "__main__":
    e, n = find()
    write(e, n)
    print(f"{len(e)} exact groups, {len(n)} near pairs")

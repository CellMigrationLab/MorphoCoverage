"""The audit tools themselves (tools/): the index, its metrics, the call resolution, the
silent-fallback detector, the ratchet and the search, on small planted sources."""

from __future__ import annotations

import ast
import json
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import audit
import callgraph
import fallbacks
import function_index as fi
import ratchet
import search


def _tree(source: str) -> ast.Module:
    return ast.parse(textwrap.dedent(source))


def _function(source: str) -> ast.FunctionDef:
    node = _tree(source).body[0]
    assert isinstance(node, ast.FunctionDef)
    return node


def test_the_body_hash_is_the_same_on_every_python_version():
    # Pinned: computed once; ast.dump would give another digest on 3.11, 3.12 and 3.13.
    node = _function(
        '''
        def f(x: int, *rest, key=None) -> int:
            """Doc."""
            try:
                return [y for y in x if (z := y)] if key else {**x}
            except (ValueError, TypeError) as error:
                raise RuntimeError(f"{error!r:>3}") from None
        '''
    )
    assert fi.body_hash(node) == "5b7f0a4b44b70d92"


def test_the_hash_ignores_the_docstring_and_keeps_names():
    first = _function("def f(a):\n    'Doc.'\n    return a + 1\n")
    second = _function("def g(a):\n    return a + 1\n")
    renamed = _function("def g(b):\n    return b + 1\n")
    assert fi.body_hash(first) == fi.body_hash(second) != fi.body_hash(renamed)


def test_repeated_names_get_unique_keys_and_kinds():
    tree = _tree(
        """
        class C:
            @property
            def x(self): return 1
            @x.setter
            def x(self, value): pass
            @classmethod
            def make(cls): pass
            @staticmethod
            def plain(): pass
        if FLAG:
            def f(): pass
        else:
            def f():
                def inner(): pass
        """
    )
    found = [(name, kind) for name, _, kind in fi.walk_definitions(tree)]
    assert found == [
        ("C", "class"),
        ("C.x", "property"),
        ("C.x@setter", "property"),
        ("C.make", "classmethod"),
        ("C.plain", "staticmethod"),
        ("f", "function"),
        ("f#2", "function"),
        ("f#2.inner", "nested"),
    ]


def test_complexity_counts_like_mccabe_on_the_own_body():
    node = _function(
        """
        def f(a, b, c):
            if a and b and c:
                pass
            elif b:
                pass
            for x in a:
                while x:
                    break
            try:
                pass
            except ValueError:
                pass
            except TypeError:
                pass
            y = 1 if a else 2
            match a:
                case 1:
                    pass
                case _:
                    pass
            z = [v for v in a if v]
            with open(a):
                assert b
            def nested():
                if a:
                    pass
            return lambda: 1 if a else 2
        """
    )
    # 1 + if (1, plus 2 extra and-operands) + elif + for + while + 2 except + IfExp + 2 cases.
    assert fi.complexity(node) == 1 + 3 + 1 + 1 + 1 + 2 + 1 + 2


def test_lines_start_after_the_docstring_and_varargs_are_apart():
    node = _function(
        '''
        def f(self, a, b=1, *args, c, **kwargs):
            """A long docstring.

            Many lines.
            """
            x = a
            return x
        '''
    )
    assert fi.code_lines(node) == 2
    assert fi.arguments(node) == 3
    assert fi.varargs(node) == ["*args", "**kwargs"]


def _package() -> list[tuple[str, ast.Module]]:
    return [
        ("__init__", _tree("from .tools import helper\n")),
        ("tools", _tree("import numpy as np\n\ndef helper(x):\n    return np.mean(x)\n")),
        (
            "user",
            _tree(
                """
                from morphocoverage import helper as h
                from . import tools

                class Base:
                    def shared(self): pass

                class Child(Base):
                    def run(self, values, callback):
                        self.shared()
                        tools.helper(values)
                        values.sort(key=h)
                        callback()
                        return len(values)
                """
            ),
        ),
    ]


def test_calls_are_resolved_through_imports_and_self():
    trees = _package()
    rows = [r for module, tree in trees for r in _rows(module, tree)]
    callgraph.fill_calls(rows, trees)
    by_key = {fi.key(r): r for r in rows}
    assert by_key["user:Child.run"]["calls"] == [
        "?.callback",
        "?.sort",
        "len",
        "tools:helper",
        "user:Base.shared",
    ]
    assert by_key["tools:helper"]["calls"] == ["numpy.mean"]
    assert by_key["tools:helper"]["called_by"] == ["user:Child.run"]
    assert by_key["user:Base"]["called_by"] == ["user:Child"]


def _rows(module: str, tree: ast.Module) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [{"module": module, "name": fi.MODULE_ROW, "kind": "module"}]
    return rows + [fi.record(module, q, n, k) for q, n, k in fi.walk_definitions(tree)]


PLANTS = """
import contextlib
import warnings
from contextlib import suppress as quiet

import numpy as np
import pandas as pd


def returns_nan(x):
    try:
        return float(x)
    except ValueError:
        return np.nan


def logs(x):
    try:
        return float(x)
    except ValueError:
        warnings.warn("not a number")
        return None


def raise_in_nested(x):
    try:
        return float(x)
    except ValueError:
        def later():
            raise RuntimeError
        return later


def raises(x):
    try:
        return float(x)
    except ValueError as error:
        raise TypeError(x) from error


def marked(x):
    try:
        return float(x)
    except ValueError:  # fallback: text stays text
        return x


def marked_above(x):
    # fallback: closing twice is harmless
    with quiet(OSError):
        x.close()


def calls(s, mode):
    with contextlib.suppress(OSError):
        pass
    with np.errstate(divide="ignore"):
        pass
    warnings.simplefilter("ignore", UserWarning)
    pd.to_numeric(s, errors="coerce")
    pd.to_numeric(s, errors=mode)
    return s.drop(columns=["a"], errors="ignore")
"""


def _plants() -> list[fallbacks.Fallback]:
    tree = ast.parse(PLANTS)
    package = callgraph.Package([("plants", tree)])
    return fallbacks.module_fallbacks(package, "plants", tree, PLANTS)


def test_the_fallback_detector_finds_the_planted_kinds():
    found = {(f.owner, f.kind, f.reason) for f in _plants()}
    assert found == {
        ("returns_nan", "swallowed error", None),
        ("logs", "reported fallback", None),
        ("raise_in_nested", "swallowed error", None),
        ("marked", "swallowed error", "text stays text"),
        ("marked_above", "contextlib.suppress", "closing twice is harmless"),
        ("calls", "contextlib.suppress", None),
        ("calls", "numpy.errstate ignore", None),
        ("calls", "warnings ignore", None),
        ("calls", "errors='coerce'", None),
        ("calls", "errors=mode", None),
        ("calls", "errors='ignore'", None),
    }


def test_only_unmarked_enforced_fallbacks_count():
    assert fallbacks.unmarked(_plants()) == {
        "plants:returns_nan": 1,
        "plants:logs": 1,
        "plants:raise_in_nested": 1,
        "plants:calls": 5,
    }


@pytest.fixture
def baselines(tmp_path, monkeypatch) -> Path:
    monkeypatch.setattr(ratchet, "FOLDER", tmp_path)
    for kind in ratchet.KINDS:
        (tmp_path / f"{kind}.json").write_text("{}")
    return tmp_path


def _findings(**kinds: dict[str, int]) -> dict[str, dict[str, int]]:
    return {kind: kinds.get(kind, {}) for kind in ratchet.KINDS}


def test_tighten_lowers_drops_and_follows_a_moved_body_but_never_adds(baselines):
    ratchet.write_baseline("lines", {"m:long": 90, "m:fixed": 70, "m:old": 80})
    old = [{"module": "m", "name": "old", "kind": "function", "hash": "h1"}]
    new = [{"module": "n", "name": "new", "kind": "function", "hash": "h1"}]
    current = _findings(lines={"m:long": 75, "n:new": 80, "m:fresh": 99})
    ratchet.tighten(current, old, new)
    assert ratchet.read_baseline("lines") == {"m:long": 75, "n:new": 80}


def test_accept_needs_a_reason_and_records_it(baselines):
    current = _findings(lines={"m:f": 70})
    with pytest.raises(ValueError, match="--reason"):
        ratchet.accept("lines:m:f", " ", current)
    with pytest.raises(ValueError, match="nothing to accept"):
        ratchet.accept("lines:m:g", "why", current)
    ratchet.accept("lines:m:f", "a table of constants", current)
    assert ratchet.read_baseline("lines") == {"m:f": 70}
    (entry,) = json.loads((baselines / "accepted.json").read_text())
    assert entry["key"] == "m:f" and entry["value"] == 70 and entry["reason"]


def test_the_merge_base_check_admits_only_accepted_growth():
    base = _findings(lines={"m:a": 70, "m:old": 80})
    head = _findings(lines={"m:a": 72, "m:new": 80, "m:b": 65})
    problems = ratchet.compare(base, head, {"m:new": "m:old"}, [])
    assert problems == [
        "lines: m:a = 72 (raised from 70) without `--accept` reason",
        "lines: m:b = 65 (new entry) without `--accept` reason",
    ]
    accepted = [{"kind": "lines", "key": "m:b", "value": 65, "reason": "r", "date": "d"}]
    assert len(ratchet.compare(base, head, {"m:new": "m:old"}, accepted)) == 1


def test_write_baselines_is_refused():
    assert audit.main(["--write-baselines"]) == 2


def test_the_search_knows_abbreviations_and_spellings():
    rows = [
        {"module": "statistics", "name": "benjamini_hochberg", "kind": "function",
         "summary": "Adjusted p-values.", "public": True},
        {"module": "transforms", "name": "RobustStandardize", "kind": "class",
         "summary": "Centre on the median.", "public": True},
        {"module": "plots", "name": "draw", "kind": "function", "summary": "A figure.",
         "public": True},
    ]  # fmt: skip
    assert fi.key(search.find("BH", rows, embeddings=False)[0][0]) == (
        "statistics:benjamini_hochberg"
    )
    found = search.find("robust standardisation", rows, embeddings=False)[0]
    assert fi.key(found[0]) == "transforms:RobustStandardize"


def test_a_new_duplicate_or_layer_entry_needs_a_reason():
    base = ratchet._reasoned('{"h1:2": "kept"}', "duplicates")
    head = ratchet._reasoned('{"h1:2": "kept", "h2:2": "to merge", "h3:3": " "}', "duplicates")
    problems, admitted = ratchet.reasoned_changes(base, head)
    assert problems == ["duplicates: h3:3: new entry without a reason"]
    assert admitted == ["duplicates: h2:2 (to merge)"]
    layers = ratchet._reasoned('{"back_edges": {"a -> b (local)": "deferred"}}', "layers")
    assert layers == {"layers back_edges: a -> b (local)": "deferred"}

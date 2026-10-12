"""The duplicate finder, the jscpd review wrapper and the layer check (tools/), on planted code in
a temporary tree: what each tier catches, what it leaves to the on-demand review, and that the
tools never change the trees they read."""

from __future__ import annotations

import ast
import importlib.util
import json
import textwrap
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / f"tools/{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DUPLICATES = _load("duplicates")
REVIEW = _load("jscpd_review")
LAYERS = _load("layers")

HEAD = "import numpy as np\nfrom morphocoverage._checks import is_integer\n\n"
ORIGINAL = '''
def field_sample(fields, n, rng):
    """Draw n objects as whole fields in random order."""
    if not is_integer(n) or n < 0:
        raise ValueError(f"n must be a non-negative integer, not {n!r}")
    if n == 0:
        return np.array([], dtype=float)
    total = sum(len(f) for f in fields)
    if total < n:
        raise ValueError(f"asked for {n} objects from fields holding {total}")
    taken, count = [], 0
    for i in rng.permutation(len(fields)):
        field = fields[i]
        need = n - count
        if len(field) >= need:
            taken.append(rng.choice(field, size=need, replace=False))
            break
        taken.append(field)
        count += len(field)
    return np.concatenate(taken)
'''
_BODY = "\n".join("    " + line if line.strip() else line for line in ORIGINAL.splitlines()[3:])

# Planted copies of ORIGINAL and the tier each must land in (A3 of audit 12).
PLANTS: dict[str, tuple[str, str]] = {
    "verbatim": (ORIGINAL, "identical"),
    "annotations and another docstring": (
        ORIGINAL.replace(
            "(fields, n, rng):", "(fields: list, n: int, rng: object) -> object:"
        ).replace("Draw n objects", "Other words for: draw n objects"),
        "identical",
    ),
    "locals renamed": (
        ORIGINAL.replace("taken", "chosen").replace("count", "so_far").replace("need", "gap"),
        "identical",
    ),
    "function renamed": (ORIGINAL.replace("def field_sample", "def sample_fields"), "identical"),
    "parameters renamed": (
        ORIGINAL.replace("fields holding", "@")
        .replace("fields", "groups")
        .replace("@", "fields holding")
        .replace("rng", "generator"),
        "identical",
    ),
    "comment and blank line": (
        ORIGINAL.replace(
            "    taken, count", "    # walk the fields in random order\n\n    taken, count"
        ),
        "identical",
    ),
    "method with the same parameters": (
        "class Sampler:\n"
        + textwrap.indent(ORIGINAL.replace("(fields, n, rng)", "(self, fields, n, rng)"), "    "),
        "identical",
    ),
    "a constant changed": (ORIGINAL.replace("n < 0", "n < 1"), "structural"),
    "a message changed": (ORIGINAL.replace("asked for", "requested"), "structural"),
    "another global callee": (ORIGINAL.replace("is_integer(n)", "is_count(n)"), "structural"),
    "method reading self.rng": (
        "class Sampler:\n"
        + textwrap.indent(
            ORIGINAL.replace("(fields, n, rng)", "(self, fields, n)").replace("rng.", "self.rng."),
            "    ",
        ),
        "neither",
    ),
    "a logging line": (
        ORIGINAL.replace("    taken, count = [], 0\n", "    taken, count = [], 0\n    log(n)\n"),
        "neither",
    ),
    "statements reordered": (
        ORIGINAL.replace(
            "    if n == 0:\n        return np.array([], dtype=float)\n"
            "    total = sum(len(f) for f in fields)\n",
            "    total = sum(len(f) for f in fields)\n"
            "    if n == 0:\n        return np.array([], dtype=float)\n",
        ),
        "neither",
    ),
    "comprehension written as a loop": (
        ORIGINAL.replace(
            "    total = sum(len(f) for f in fields)\n",
            "    total = 0\n    for f in fields:\n        total += len(f)\n",
        ),
        "neither",
    ),
    "body wrapped in a with block": (
        "\ndef field_sample(fields, n, rng):\n"
        '    """X."""\n    with np.errstate(all="ignore"):\n' + _BODY + "\n",
        "neither",
    ),
    "another attribute called": (ORIGINAL.replace("rng.choice", "rng.integers"), "neither"),
}


def _write(folder: Path, name: str, source: str) -> tuple[str, Path]:
    path = folder / f"{name}.py"
    path.write_text(HEAD + source, encoding="utf-8")
    return name, path


def _tier(groups: dict[str, dict[str, list[str]]]) -> str:
    for tier in ("identical", "structural"):
        for members in groups[tier].values():
            modules = {m.split(":")[0] for m in members}
            if {"original", "plant"} <= modules:
                return tier
    return "neither"


@pytest.mark.parametrize("plant", sorted(PLANTS))
def test_each_planted_copy_lands_in_its_tier(tmp_path, plant):
    source, expected = PLANTS[plant]
    sources = [_write(tmp_path, "original", ORIGINAL), _write(tmp_path, "plant", source)]
    assert _tier(DUPLICATES.find(sources)) == expected


def test_normalising_never_changes_the_parsed_tree():
    tree = ast.parse(HEAD + ORIGINAL)
    before = ast.dump(tree, include_attributes=True)
    function = tree.body[-1]
    for structural in (False, True):
        DUPLICATES.normalised(function, structural)
        DUPLICATES.digest(function, structural)
    assert ast.dump(tree, include_attributes=True) == before
    # The finder twice on the same tree gives the same groups (no state left behind).
    assert DUPLICATES.digest(function) == DUPLICATES.digest(function)


def test_identical_keeps_globals_attributes_and_constants_and_hides_locals():
    def digest(source: str) -> str:
        return DUPLICATES.digest(ast.parse(textwrap.dedent(source)).body[0])

    base = "def f(x):\n    y = x.mean() + ALPHA\n    return round(y, 3)\n"
    assert digest(base) == digest(base.replace("x", "z").replace("y", "w"))
    for changed in ("ALPHA", "BETA"), ("mean", "median"), ("3", "4"), ("round", "floor"):
        assert digest(base) != digest(base.replace(*changed)), changed


def test_small_and_trivial_bodies_are_never_compared(tmp_path):
    stubs = '''
def a(x):
    """Doc."""
    raise NotImplementedError(type(x).__name__, x, len(x), str(x))


def b(x):
    """Doc."""
    raise NotImplementedError(type(x).__name__, x, len(x), str(x))


def c(x):
    return x + 1


def d(y):
    return y + 1
'''
    tree = ast.parse(stubs)
    assert DUPLICATES.size(tree.body[0]) >= DUPLICATES.MIN_NODES
    assert DUPLICATES.trivial(tree.body[0])
    assert DUPLICATES.size(tree.body[2]) < DUPLICATES.MIN_NODES
    path = tmp_path / "stubs.py"
    path.write_text(stubs, encoding="utf-8")
    assert DUPLICATES.find([("stubs", path)]) == {"identical": {}, "structural": {}}


def test_a_group_key_counts_its_members(tmp_path):
    sources = [_write(tmp_path, f"copy{i}", ORIGINAL) for i in range(13)]
    groups = DUPLICATES.exact_groups(sources)
    assert len(groups) == 1
    key, members = next(iter(groups.items()))
    assert key.endswith(":13") and len(members) == 13


def test_constants_report_repeated_names_and_literals(tmp_path):
    (tmp_path / "one.py").write_text("ALPHA = 0.05\nMAD = 1.4826\nx = 0.25\n", encoding="utf-8")
    (tmp_path / "two.py").write_text(
        'ALPHA = 0.05\ndef f(v):\n    """1.4826 in a docstring is text."""\n'
        "    return 1.482602218505602 * v + 0.25\n",
        encoding="utf-8",
    )
    found = DUPLICATES.constants([("one", tmp_path / "one.py"), ("two", tmp_path / "two.py")])
    assert found["names"] == {"ALPHA": ["one = 0.05", "two = 0.05"]}
    assert found["literals"] == {"1.483": ["one: 1.4826", "two: 1.482602218505602"]}


def test_tighten_drops_merged_groups_and_keeps_reasons(tmp_path):
    current = DUPLICATES.exact_groups()
    key = next(iter(current))
    digest_, count = key.split(":")
    baseline = tmp_path / "duplicates.json"
    baseline.write_text(
        json.dumps({"0" * 16 + ":2": "merged since", f"{digest_}:{int(count) + 1}": "kept"}),
        encoding="utf-8",
    )
    changed = DUPLICATES.tighten(baseline)
    assert json.loads(baseline.read_text(encoding="utf-8")) == {key: "kept"}
    assert len(changed) == 2
    with pytest.raises(ValueError, match="reason"):
        DUPLICATES.accept(key, " ", baseline)
    with pytest.raises(ValueError, match="no current identical group"):
        DUPLICATES.accept("f" * 16, "why", baseline)


# The jscpd review wrapper (jscpd itself never runs in the tests).


def _module(tmp_path: Path) -> Path:
    path = tmp_path / "planted.py"
    path.write_text(
        "X = 1\n\n\ndef outer():\n    a = 1\n\n    def inner():\n        return a\n\n"
        "    return inner\n\n\ndef other():\n    return 2\n",
        encoding="utf-8",
    )
    return path


def test_a_clone_span_maps_to_the_function_holding_most_of_it(tmp_path):
    path = _module(tmp_path)
    assert REVIEW.owner(path, 4, 10) == "planted:outer"
    assert REVIEW.owner(path, 7, 8) == "planted:outer.inner"
    assert REVIEW.owner(path, 1, 1) == "planted:<module>"
    assert REVIEW.owner(path, 9, 14) == "planted:other"


def test_candidates_drop_reviewed_pairs_and_pairs_inside_one_function(tmp_path):
    path = str(_module(tmp_path))

    def clone(a: tuple[int, int], b: tuple[int, int], **extra: object) -> dict:
        sides = [{"name": path, "start": s, "end": e} for s, e in (a, b)]
        return {"firstFile": sides[0], "secondFile": sides[1], "lines": 3, **extra}

    clones = {
        "token": [clone((4, 10), (13, 14)), clone((5, 5), (10, 10))],
        "ast": [clone((13, 14), (4, 10), similarity=0.85)],
        "semantic": [clone((4, 10), (4, 9), similarity=0.9)],
    }
    found = REVIEW.candidates(clones, known=set())
    assert [(c["a"], c["b"]) for c in found] == [("planted:other", "planted:outer")]
    assert found[0]["sources"] == ["jscpd-ast", "jscpd-token"]
    assert found[0]["similarity"] == 0.85
    assert REVIEW.candidates(clones, known={("planted:other", "planted:outer")}) == []


def test_recording_a_verdict_keeps_one_sorted_entry_per_pair(tmp_path):
    reviewed = tmp_path / "reviewed.json"
    REVIEW.record("m:b", "m:a", "jscpd-ast", "helper", "first", reviewed)
    REVIEW.record("m:a", "m:b", "jscpd-semantic", "intentional", "second", reviewed)
    rows = json.loads(reviewed.read_text(encoding="utf-8"))
    assert rows == [
        {
            "a": "m:a",
            "b": "m:b",
            "source": "jscpd-semantic",
            "verdict": "intentional",
            "reason": "second",
        }
    ]
    for bad in (
        {"source": "guess", "verdict": "merged", "reason": "r"},
        {"source": "audit12", "verdict": "maybe", "reason": "r"},
        {"source": "audit12", "verdict": "merged", "reason": " "},
    ):
        with pytest.raises(ValueError):
            REVIEW.record("m:a", "m:c", path=reviewed, **bad)


def test_a_pass_that_cannot_run_is_skipped_with_its_reason(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    clones, status = REVIEW.run_pass("semantic", [tmp_path], tmp_path)
    assert clones == [] and "not downloaded" in status
    monkeypatch.setattr(REVIEW.shutil, "which", lambda name: None)
    clones, status = REVIEW.run_pass("token", [tmp_path], tmp_path)
    assert clones == [] and "npx not found" in status


def test_the_reviewed_list_is_well_formed():
    rows = REVIEW.read_reviewed()
    keys = [(r["a"], r["b"]) for r in rows]
    assert all(a < b for a, b in keys) and len(set(keys)) == len(keys)
    for row in rows:
        assert set(row) == {"a", "b", "source", "verdict", "reason"}
        assert row["source"] in REVIEW.SOURCES and row["verdict"] in REVIEW.VERDICTS
        assert row["reason"].strip()


# The layer check.


def _package(tmp_path: Path, files: dict[str, str]) -> Path:
    source = tmp_path / "morphocoverage"
    for name, text in {"__init__.py": "", **files}.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text), encoding="utf-8")
    return source


def test_absolute_imports_and_import_module_are_read(tmp_path):
    source = _package(
        tmp_path,
        {
            "core.py": "from morphocoverage.workflows import w\n",
            "strategies/__init__.py": "",
            "strategies/s.py": "def f():\n    import morphocoverage.sampling_unit.u\n",
            "sampling_unit/__init__.py": "",
            "sampling_unit/u.py": (
                "import importlib\nM = importlib.import_module('morphocoverage.workflows.w')\n"
            ),
            "workflows/__init__.py": "",
            "workflows/w.py": "",
        },
    )
    assert LAYERS.violations(source) == [
        "core -> workflows.w (runtime)",
        "sampling_unit.u -> workflows.w (runtime)",
        "strategies.s -> sampling_unit.u (local)",
    ]


def test_runtime_cycles_fail_and_deferred_back_edges_are_listed(tmp_path):
    files = {
        "a.py": "from . import b\n",
        "b.py": "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from .a import X\n",
        "c.py": "from .d import Y\n",
        "d.py": "def f():\n    from .c import Z\n",
    }
    source = _package(tmp_path / "deferred", files)
    assert LAYERS.cycles(source) == []
    assert LAYERS.back_edges(source) == ["b -> a (typing)", "d -> c (local)"]
    files["b.py"] = "from .a import X\n"
    assert LAYERS.cycles(_package(tmp_path / "runtime", files)) == ["a -> b"]


def test_from_package_import_adds_the_package_only_for_a_name_that_is_not_a_module(tmp_path):
    source = _package(
        tmp_path,
        {
            "workflows/__init__.py": "def setup():\n    return 1\n",
            "workflows/guided.py": "from . import setup\n",
            "workflows/_tables.py": "from . import guided\n",
        },
    )
    graph = LAYERS.import_graph(source)
    assert graph["workflows._tables"] == {"workflows.guided"}
    assert graph["workflows.guided"] == {"workflows"}

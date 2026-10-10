"""The function index and its ratchets (tools/function_index.py): the index is current, and no
function is longer, more complex, wider, undocumented or more silent than the baselines allow."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "function_index", ROOT / "tools/function_index.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / f"tools/{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TOOL = _tool()
DUPLICATES = _load("duplicates")
LAYERS = _load("layers")
REGENERATE = "run `python tools/audit.py`"
RATCHET = "run `python tools/audit.py --write-baselines` only to record a REMOVAL"


@pytest.fixture(scope="module")
def current() -> dict[str, dict[str, int]]:
    return TOOL.findings()


def test_the_function_index_is_current():
    rows = TOOL.build_index()
    committed = TOOL.read_index()
    now, then = set(map(tuple, TOOL.index_key(rows))), set(map(tuple, TOOL.index_key(committed)))
    added, removed = sorted(now - then), sorted(then - now)
    assert not added and not removed, (
        f"docs/common/function_index.jsonl is out of date ({len(added)} new or changed, "
        f"{len(removed)} gone or changed; first: {(added or removed)[0][:2]}): {REGENERATE}"
    )


@pytest.mark.parametrize("kind", ["lines", "complexity", "arguments", "no_docstring", "fallbacks"])
def test_nothing_grows_beyond_its_baseline(current, kind):
    baseline = TOOL.read_baseline(kind)
    grown = {k: (baseline.get(k, 0), v) for k, v in current[kind].items() if v > baseline.get(k, 0)}
    assert not grown, f"{kind}: new or larger (was, now): {dict(list(grown.items())[:8])}"


@pytest.mark.parametrize("kind", ["lines", "complexity", "arguments", "no_docstring", "fallbacks"])
def test_a_baseline_never_holds_what_was_fixed(current, kind):
    stale = {k: v for k, v in TOOL.read_baseline(kind).items() if current[kind].get(k, 0) < v}
    assert not stale, (
        f"{kind}: fixed or shrunk, tighten the baseline ({RATCHET}): {list(stale)[:8]}"
    )


def _baseline_file(name: str):
    import json

    return json.loads((TOOL.BASELINES / f"{name}.json").read_text(encoding="utf-8"))


def test_no_new_exact_duplicate_functions():
    new = sorted(set(DUPLICATES.exact_groups()) - set(_baseline_file("duplicates")))
    assert not new, f"new exact duplicates (merge them, or explain): {new[:5]}"


def test_a_duplicate_baseline_never_holds_what_was_merged():
    stale = sorted(set(_baseline_file("duplicates")) - set(DUPLICATES.exact_groups()))
    assert not stale, f"merged: tighten the baseline ({RATCHET}): {stale[:5]}"


def test_no_import_crosses_a_layer_it_must_not():
    allowed = set(_baseline_file("layers")["violations"])
    new = sorted(set(LAYERS.violations()) - allowed)
    assert not new, f"a layer imports one it must not (tools/layers.py RULES): {new[:5]}"
    stale = sorted(allowed - set(LAYERS.violations()))
    assert not stale, f"fixed, tighten the baseline ({RATCHET}): {stale[:5]}"


def test_no_new_import_cycle():
    allowed = set(_baseline_file("layers")["cycles"])
    new = sorted(set(LAYERS.cycles()) - allowed)
    assert not new, f"new import cycle between modules: {new[:3]}"
    stale = sorted(allowed - set(LAYERS.cycles()))
    assert not stale, f"cycle gone, tighten the baseline ({RATCHET}): {stale[:3]}"

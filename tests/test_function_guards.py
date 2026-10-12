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


@pytest.fixture(scope="module")
def identical() -> dict[str, list[str]]:
    return DUPLICATES.exact_groups()


def test_no_new_identical_function_bodies(identical):
    accepted = DUPLICATES.read_baseline()
    new = {k: v for k, v in identical.items() if k not in accepted}
    assert not new, (
        "identical function bodies (merge them, or accept with a reason: "
        f"`python tools/duplicates.py --accept HASH --reason ...`): {list(new.values())[:5]}"
    )


def test_no_accepted_duplicate_group_gains_a_member(identical):
    count = {key.split(":")[0]: len(members) for key, members in identical.items()}
    grown = {
        key: count[key.split(":")[0]]
        for key in DUPLICATES.read_baseline()
        if count.get(key.split(":")[0], 0) > int(key.split(":")[1])
    }
    assert not grown, f"an accepted identical group gained a member (now): {grown}"


def test_every_accepted_duplicate_group_has_a_reason_and_still_exists(identical):
    accepted = DUPLICATES.read_baseline()
    assert all(reason.strip() for reason in accepted.values())
    stale = sorted(set(accepted) - set(identical))
    assert not stale, f"merged or shrunk, run `python tools/duplicates.py --tighten`: {stale[:5]}"


def test_no_import_crosses_a_layer_it_must_not():
    allowed = LAYERS.read_baseline()["violations"]
    found = LAYERS.violations()
    new = sorted(set(found) - set(allowed))
    assert not new, f"a layer imports one it must not (tools/layers.py RULES): {new[:5]}"
    stale = sorted(set(allowed) - set(found))
    assert not stale, f"fixed, run `python tools/layers.py --tighten`: {stale[:5]}"


def test_no_runtime_import_cycle():
    found = LAYERS.cycles()
    assert not found, f"modules import each other at import time (no baseline): {found[:3]}"


def test_no_new_deferred_import_inside_a_cycle():
    allowed = LAYERS.read_baseline()["back_edges"]
    found = LAYERS.back_edges()
    new = sorted(set(found) - set(allowed))
    assert not new, (
        "a function-local or TYPE_CHECKING import closes an import cycle (break the cycle, "
        f"or record the edge with a reason in tools/baselines/layers.json): {new[:5]}"
    )
    stale = sorted(set(allowed) - set(found))
    assert not stale, f"gone, run `python tools/layers.py --tighten`: {stale[:5]}"

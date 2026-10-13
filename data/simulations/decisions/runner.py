"""Runs a module of the benchmark: every design point, every replicate, every method on the
same dataset (docs/common/validation.md, "How the benchmark is run").

- **Seeds.** Replicate ``i`` of point ``p`` of module ``m`` is simulated from
  ``SeedSequence([seed, crc32(m), crc32(p), i])``: a dataset depends on nothing else, so a
  resumed or parallel run gives the same numbers, and each method gets its own child seed
  for its own randomness (permutations, bootstraps).
- **Paired.** Every method of the module runs on the same dataset; the rows carry the
  replicate, so differences between methods are paired.
- **Chunks.** The replicates of a point are cut into chunks of ``--chunk``; each chunk is
  written to ``<out>/<module>/chunks/<point>__<first>_<last>.csv`` once complete (written to
  a temporary name, then renamed). A run stopped at any time (a Colab timeout) loses at most
  the chunks in progress, one per worker; running again skips every chunk on disk.
- **Failures and runtime.** A method that raises, or returns a missing or non-finite
  primary output, is recorded as failed with its message; the runtime of every call is
  recorded.
- **Sizes.** ``smoke`` (one replicate of the first point of each kind and alternative
  class, small settings: a test of the code), ``pilot`` (1% of the full replicates of every
  point, at least 2: timing, memory, convergence and generator sanity only, never a
  result), ``full`` (the counts fixed by the protocol, ``FULL``).

Usage, from the repository root (each module script calls ``main``)::

    PYTHONPATH=src:data/simulations python -m decisions.p1_repeat_inference \\
        --out results --workers 4 --chunk 100 --size full
    PYTHONPATH=src:data/simulations python -m decisions.p1_repeat_inference summarise --out results
"""

from __future__ import annotations

import argparse
import importlib
import math
import os
import sys
import time
import zlib
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# The replicate counts fixed by the protocol (docs/common/validation.md, "Selection rule").
FULL = {"core_null": 10_000, "core": 1_000, "stress": 1_000, "screening": 250}
PILOT_SHARE = 0.01
# Development runs (smoke, pilot) use seed 0; the confirmatory runs a new seed, fixed here
# before any of them (docs/common/validation.md).
DEVELOPMENT_SEED = 0
CONFIRMATORY_SEED = 20261006
KINDS = ("core", "stress", "screening")
# The bookkeeping columns of every row; no method output may take one of these names.
RESERVED = frozenset(
    {
        "module",
        "point",
        "kind",
        "null",
        "alternative",
        "replicate",
        "seed",
        "method",
        "question",
        "failed",
        "error",
        "runtime",
    }
)
SIZES = ("smoke", "pilot", "full")


@dataclass(frozen=True)
class Point:
    """One design point: its ``params`` (design and outcome keys), its layer (``kind``:
    core, stress or screening), whether the calibration gate reads it (``null``) and the
    alternative class whose power it enters (``alternative``; a P4 point can be both)."""

    name: str
    kind: str
    params: dict[str, Any] = field(default_factory=dict)
    null: bool = True
    alternative: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")


@dataclass(frozen=True)
class Method:
    """A method under comparison: ``run(data, context) -> outputs`` (a dict of numbers), its
    ``role`` (``ours``: the package today; ``candidate``; ``comparator``; ``reference``), the
    ``question`` it answers (one selection per question), its ``simplicity`` rank for ties
    (lower is simpler), the ``primary`` output that must be finite and, for a slow
    comparator, the replicates of a point it runs on (``max_replicates``)."""

    name: str
    run: Callable[[Any, dict[str, Any]], dict[str, float]]
    role: str = "candidate"
    question: str = "main"
    simplicity: int = 1
    primary: str = "p"
    max_replicates: int | None = None


@dataclass(frozen=True)
class Rule:
    """The selection rule of one question (docs/common/validation.md): the
    gate metric on the gated points and its bound (``binomial``: simultaneous one-sided
    Clopper-Pearson; ``mean``: normal; an empty ``gate``: none), the interval coverage
    gate, outputs that must always be 1 (``hard``), the objective over the alternative
    ``classes`` of this question (empty: every class) and the tie margin. The gated points
    are the core points marked ``null`` that are not an alternative of this question, or
    every ``null`` core point with ``gate_alternatives`` (P4, P5)."""

    question: str = "main"
    classes: tuple[str, ...] = ()
    gate_alternatives: bool = False
    gate: str = "reject"
    gate_kind: str = "binomial"
    bound: float = 0.060
    coverage: bool = True
    coverage_bound: float = 0.93
    hard: tuple[str, ...] = ()
    objective: str = "reject"
    maximise: bool = True
    tie: float = 0.02
    failure_bound: float = 0.01


@dataclass(frozen=True)
class Module:
    """A module: its name (``p1``), the simulation of one dataset from a point's params, the
    points, the methods and the rules."""

    name: str
    title: str
    simulate: Callable[[dict[str, Any], np.random.Generator], Any]
    points: tuple[Point, ...]
    methods: tuple[Method, ...]
    rules: tuple[Rule, ...] = (Rule(),)
    alpha: float = 0.05


def replicates(point: Point, size: str) -> int:
    """The replicates of ``point`` at ``size`` (see the module)."""
    if size == "smoke":
        return 1
    # A gated core point needs the calibration gate's 10,000 replicates (a P4 point is gated
    # and enters a class's objective too).
    key = "core_null" if point.kind == "core" and point.null else point.kind
    full = FULL[key]
    return full if size == "full" else max(2, math.ceil(PILOT_SHARE * full))


def smoke_points(points: Sequence[Point]) -> list[Point]:
    """The points of a smoke run: the first core null point, the first point of each
    alternative class, the first stress point and the first screening point."""
    chosen, seen = [], set()
    for point in points:
        key = (point.kind, point.alternative or ("null" if point.null else ""))
        if key not in seen:
            seen.add(key)
            chosen.append(point)
    return chosen


def seed_sequence(seed: int, module: str, point: str, replicate: int) -> np.random.SeedSequence:
    """The seed of one replicate (see the module)."""
    return np.random.SeedSequence(
        [seed, zlib.crc32(module.encode()), zlib.crc32(point.encode()), replicate]
    )


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def run_replicate(
    module: Module, point: Point, replicate: int, seed: int, smoke: bool
) -> list[dict[str, Any]]:
    """Every method of ``module`` on replicate ``replicate`` of ``point``: one row each."""
    sequence = seed_sequence(seed, module.name, point.name, replicate)
    data_seed, *method_seeds = sequence.spawn(1 + len(module.methods))
    data = module.simulate(point.params, np.random.default_rng(data_seed))
    rows = []
    for method, method_seed in zip(module.methods, method_seeds, strict=True):
        if method.max_replicates is not None and replicate >= method.max_replicates:
            continue
        context = {
            "params": point.params,
            "rng": np.random.default_rng(method_seed),
            "smoke": smoke,
            "alpha": module.alpha,
            "replicate": replicate,
        }
        start = time.perf_counter()
        error = ""
        try:
            outputs = dict(method.run(data, context))
            clash = set(outputs) & RESERVED
            if clash:
                raise ValueError(f"outputs may not be named {sorted(clash)}")
            if not _finite(outputs.get(method.primary)):
                error = f"no finite {method.primary}"
        except Exception as caught:  # noqa: BLE001 - every failure is a result, recorded
            outputs, error = {}, f"{type(caught).__name__}: {caught}"[:200]
        rows.append(
            {
                "module": module.name,
                "point": point.name,
                "kind": point.kind,
                "null": point.null,
                "alternative": point.alternative or "",
                "replicate": replicate,
                "seed": seed,
                "method": method.name,
                "question": method.question,
                "failed": int(bool(error)),
                "error": error,
                "runtime": time.perf_counter() - start,
                **outputs,
            }
        )
    return rows


def _chunk_path(out: Path, module: str, point: str, first: int, last: int) -> Path:
    return out / module / "chunks" / f"{point}__{first:06d}_{last:06d}.csv"


def plan(
    module: Module, size: str, chunk: int, kinds: Iterable[str]
) -> list[tuple[Point, int, int]]:
    """The chunks of a run: ``(point, first, last)`` replicates, last excluded."""
    points = smoke_points(module.points) if size == "smoke" else list(module.points)
    units = []
    for point in points:
        if point.kind not in kinds:
            continue
        total = replicates(point, size)
        for first in range(0, total, chunk):
            units.append((point, first, min(total, first + chunk)))
    return units


def _work(args: tuple[str, str, int, int, int, bool, str]) -> tuple[str, float]:
    """One chunk in a worker: the script's module, simulated and written (see the module)."""
    script, point_name, first, last, seed, smoke, out = args
    module = importlib.import_module(script).MODULE
    point = next(p for p in module.points if p.name == point_name)
    start = time.perf_counter()
    rows = []
    for replicate in range(first, last):
        rows += run_replicate(module, point, replicate, seed, smoke)
    path = _chunk_path(Path(out), module.name, point.name, first, last)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    os.replace(temporary, path)
    return point_name, time.perf_counter() - start


def _write_points(module: Module, out: Path) -> None:
    """``<out>/<module>/points.csv``: every point's layer, gate, class and parameters."""
    import json

    folder = Path(out) / module.name
    folder.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "point": p.name,
            "kind": p.kind,
            "null": p.null,
            "alternative": p.alternative or "",
            "params": json.dumps(p.params, sort_keys=True, default=str),
        }
        for p in module.points
    ]
    pd.DataFrame(rows).to_csv(folder / "points.csv", index=False)


def _single_thread() -> None:
    """One linear-algebra thread per worker process (the workers fill the CPUs)."""
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"


def run(
    script: str,
    out: Path,
    size: str = "smoke",
    workers: int | None = None,
    chunk: int = 100,
    seed: int | None = None,
    kinds: Iterable[str] = KINDS,
    progress: Callable[[str], None] | None = print,
) -> int:
    """Run the module of ``script`` (``decisions.p1_repeat_inference``) into ``out`` (see the
    module); returns the chunks written by this call."""
    if size not in SIZES:
        raise ValueError(f"size must be one of {SIZES}")
    module = importlib.import_module(script).MODULE
    seed = (CONFIRMATORY_SEED if size == "full" else DEVELOPMENT_SEED) if seed is None else seed
    chunk = 1 if size == "smoke" else max(1, int(chunk))
    units = plan(module, size, chunk, set(kinds))
    _write_points(module, out)
    todo = [u for u in units if not _chunk_path(out, module.name, u[0].name, u[1], u[2]).exists()]
    if progress:
        progress(
            f"{module.name}: {len(units) - len(todo)} of {len(units)} chunks on disk, "
            f"{len(todo)} to run ({size}, seed {seed})"
        )
    jobs = [(script, p.name, a, b, seed, size == "smoke", str(out)) for p, a, b in todo]
    workers = max(1, workers or os.cpu_count() or 1)
    start, done = time.perf_counter(), 0
    if workers == 1 or len(jobs) <= 1:
        results: Iterable[tuple[str, float]] = map(_work, jobs)
        pool = None
    else:
        pool = get_context("spawn").Pool(workers, initializer=_single_thread)
        results = pool.imap_unordered(_work, jobs)
    try:
        for _point, _seconds in results:
            done += 1
            if progress:
                elapsed = time.perf_counter() - start
                left = elapsed / done * (len(jobs) - done)
                progress(
                    f"{module.name}: {done}/{len(jobs)} chunks, {elapsed / 60:.1f} min, "
                    f"about {left / 60:.1f} min left"
                )
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return done


def main(script: str, argv: Sequence[str] | None = None) -> None:
    """The command line of a module script: ``run`` (default) or ``summarise``."""
    parser = argparse.ArgumentParser(
        prog=script, description=importlib.import_module(script).__doc__
    )
    parser.add_argument("command", nargs="?", default="run", choices=("run", "summarise"))
    parser.add_argument("--out", type=Path, default=Path("decisions_results"))
    parser.add_argument("--size", choices=SIZES, default=None)
    parser.add_argument("--smoke", action="store_true", help="the same as --size smoke")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--chunk", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--kinds", default=",".join(KINDS), help="layers: core,stress,screening")
    args = parser.parse_args(argv)
    if args.command == "summarise":
        from .summary import summarise

        written = summarise(script, args.out)
        print("\n".join(str(p) for p in written))
        return
    size = "smoke" if args.smoke else (args.size or "smoke")
    kinds = [k for k in args.kinds.split(",") if k]
    run(script, args.out, size, args.workers, args.chunk, args.seed, kinds)


if __name__ == "__main__":  # pragma: no cover - the module scripts call ``main``
    main(sys.argv[1])

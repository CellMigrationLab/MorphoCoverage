"""Propose duplicate functions with jscpd for an agent to review: on demand, never in CI.

``python tools/jscpd_review.py`` runs jscpd 5.4.1 (``npx -y jscpd@5.4.1``) over the package and
``data/`` (without ``data/simulations/decisions/candidates``; tests stay out) in three passes,
each skipped with a message when it cannot run:

- ``token``: copied runs of at least ``MIN_TOKENS`` tokens (``jscpd-token``);
- ``ast``: functions whose normalised syntax trees share at least ``SIMILARITY`` of their
  subtrees (``--similarity``, ``jscpd-ast``);
- ``semantic``: functions that do the same thing written differently, by the local embedding
  model (CodeRankEmbed) within Python only (``--semantic --semantic-scope same``,
  ``jscpd-semantic``); only when the model is already downloaded (once, 548 MB:
  ``npx -y jscpd@5.4.1 --semantic-download``); it takes about half an hour on the package.

Each clone's file and line spans are mapped to the functions they cover (``module:qualname``,
``module:<module>`` for module-level code). Pairs inside one function, pairs already reviewed in
``tools/baselines/duplicates_reviewed.json`` and pairs of one identical group accepted in
``tools/baselines/duplicates.json`` are dropped; the rest are written to ``OUTPUT`` (under the
ignored ``build/``) for review, function by function. Record each verdict with ``--record``:

    python tools/jscpd_review.py --record MODULE:A MODULE:B --source jscpd-ast \\
        --verdict helper --reason "the shared loop moved to statistics.bh_within"

Verdicts: ``merged`` (a true duplicate, merged), ``helper`` (a shared part moved to a helper),
``overlap-kept`` (functional overlap, kept apart for the reason given) and ``intentional``
(kept on purpose, with the reason). Later runs report only the new pairs.
"""

from __future__ import annotations

import argparse
import ast
import datetime
import functools
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
REVIEWED = ROOT / "tools" / "baselines" / "duplicates_reviewed.json"
OUTPUT = ROOT / "build" / "duplicate_candidates.json"
JSCPD = ("npx", "-y", "jscpd@5.4.1")
MIN_TOKENS = 50
SIMILARITY = 0.8
SOURCES = ("jscpd-token", "jscpd-ast", "jscpd-semantic", "audit12", "ast-guard")
VERDICTS = ("merged", "helper", "overlap-kept", "intentional")
PASSES: dict[str, tuple[str, ...]] = {
    "token": ("--min-tokens", str(MIN_TOKENS)),
    "ast": ("--similarity", str(SIMILARITY), "--kind", "similar,ast"),
    "semantic": ("--semantic", "--semantic-scope", "same", "--kind", "semantic"),
}


@functools.cache
def _duplicates() -> ModuleType:
    spec = importlib.util.spec_from_file_location("duplicates", ROOT / "tools/duplicates.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def default_paths() -> list[Path]:
    """The package and ``data/``."""
    return [ROOT / "src" / "morphocoverage", ROOT / "data"]


def model_downloaded() -> bool:
    """Whether jscpd's local embedding model is in its cache (never downloaded here)."""
    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "jscpd" / "models"
    return cache.is_dir() and any("CodeRankEmbed" in p.name for p in cache.iterdir())


def run_pass(name: str, paths: Iterable[Path], folder: Path) -> tuple[list[dict], str]:
    """``(clones, status)`` of one pass; no clones and the reason when it cannot run."""
    if shutil.which(JSCPD[0]) is None:
        return [], "skipped: npx not found (install Node.js to run jscpd)"
    if name == "semantic" and not model_downloaded():
        return [], (
            "skipped: the local model is not downloaded "
            f"(once, 548 MB: `{' '.join(JSCPD)} --semantic-download`)"
        )
    output = folder / name
    command = [
        *JSCPD,
        *map(str, paths),
        *("--format", "python", "--reporters", "json", "--output", str(output)),
        *("--absolute", "--no-colors"),
        *("--ignore", "**/data/simulations/decisions/candidates/**,**/__pycache__/**"),
        *PASSES[name],
    ]
    done = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    report = output / "jscpd-report.json"
    if done.returncode != 0 or not report.exists():
        tail = (done.stderr or done.stdout).strip().splitlines()[-3:]
        return [], f"skipped: jscpd exited with {done.returncode}: {' / '.join(tail)}"
    clones = json.loads(report.read_text(encoding="utf-8"))["duplicates"]
    if name == "ast":  # the exact copies of this pass are the token pass's
        clones = [c for c in clones if c.get("kind") != "exact"]
    return clones, f"ran: {len(clones)} clones"


@functools.cache
def _functions(path: Path) -> tuple[str, list[tuple[str, int, int]]]:
    """The module name and ``(qualified name, first line, last line)`` of each function."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    walk = _duplicates()._index().walk_functions
    spans = [(q, n.lineno, n.end_lineno or n.lineno) for q, n in walk(tree)]
    return _duplicates().module_for(path), spans


def owner(path: Path, start: int, end: int) -> str:
    """``module:qualname`` of the function that holds most of the lines ``start``-``end`` (the
    innermost on a tie), ``module:<module>`` when no function does."""
    module, spans = _functions(path.resolve())
    best: tuple[int, int, str] = (0, 0, "<module>")
    for name, first, last in spans:
        overlap = min(end, last) - max(start, first) + 1
        if overlap > 0 and (overlap, -(last - first)) > best[:2]:
            best = (overlap, -(last - first), name)
    return f"{module}:{best[2]}"


def pair_key(a: str, b: str) -> tuple[str, str]:
    """The key of a pair: its two members sorted."""
    return (a, b) if a <= b else (b, a)


def read_reviewed(path: Path = REVIEWED) -> list[dict[str, str]]:
    """The reviewed pairs."""
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def known_pairs(reviewed: list[dict[str, str]]) -> set[tuple[str, str]]:
    """The reviewed pairs and every pair of an accepted identical group."""
    known = {pair_key(r["a"], r["b"]) for r in reviewed}
    accepted = _duplicates().read_baseline()
    for key, members in _duplicates().exact_groups().items():
        if key in accepted:
            known |= {pair_key(a, b) for a in members for b in members if a < b}
    return known


def candidates(
    clones: dict[str, list[dict]], known: set[tuple[str, str]]
) -> list[dict[str, object]]:
    """The new pairs of functions with how and where they were found: the AST and semantic
    similarity when a pass gives one, else ``None`` (a token clone is a copied run, not a
    score of the whole functions); the most similar first, then the longest copied run."""
    sources: dict[tuple[str, str], set[str]] = defaultdict(set)
    where: dict[tuple[str, str], list[str]] = defaultdict(list)
    similarity: dict[tuple[str, str], float] = {}
    lines: dict[tuple[str, str], int] = defaultdict(int)
    for name, items in clones.items():
        for clone in items:
            sides = [clone["firstFile"], clone["secondFile"]]
            key = pair_key(*(owner(Path(s["name"]), s["start"], s["end"]) for s in sides))
            if key[0] == key[1] or key in known:
                continue
            sources[key].add(f"jscpd-{name}")
            where[key].append(
                " | ".join(f"{_relative(s['name'])}:{s['start']}-{s['end']}" for s in sides)
            )
            lines[key] = max(lines[key], int(clone.get("lines", 0)))
            if clone.get("similarity") is not None:
                similarity[key] = max(float(clone["similarity"]), similarity.get(key, 0.0))
    order = sorted(where, key=lambda k: (-similarity.get(k, 0.0), -lines[k], k))
    return [
        {
            "a": key[0],
            "b": key[1],
            "sources": sorted(sources[key]),
            "similarity": similarity.get(key),
            "lines": lines[key],
            "where": where[key],
        }
        for key in order
    ]


def _relative(name: str) -> str:
    path = Path(name).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else name


def record(
    a: str, b: str, source: str, verdict: str, reason: str, path: Path = REVIEWED
) -> dict[str, str]:
    """Add (or replace) the verdict on one pair in the reviewed list, kept sorted."""
    if source not in SOURCES:
        raise ValueError(f"source must be one of {', '.join(SOURCES)}, not {source!r}")
    if verdict not in VERDICTS:
        raise ValueError(f"verdict must be one of {', '.join(VERDICTS)}, not {verdict!r}")
    if not reason.strip():
        raise ValueError("a reviewed pair needs a reason (--reason)")
    first, second = pair_key(a, b)
    entry = {"a": first, "b": second, "source": source, "verdict": verdict, "reason": reason}
    kept = [r for r in read_reviewed(path) if pair_key(r["a"], r["b"]) != (first, second)]
    rows = sorted([*kept, entry], key=lambda r: (r["a"], r["b"]))
    path.write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return entry


def main(argv: list[str] | None = None) -> int:
    """Run the passes and write the candidates, or ``--record`` one verdict."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--passes", default=",".join(PASSES), help="token,ast,semantic")
    parser.add_argument("--paths", nargs="*", type=Path, help="default: the package and data/")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--record", nargs=2, metavar=("A", "B"), help="record a verdict")
    parser.add_argument("--source", default="jscpd-semantic", choices=SOURCES)
    parser.add_argument("--verdict", choices=VERDICTS)
    parser.add_argument("--reason", default="")
    args = parser.parse_args(argv)
    if args.record:
        if not args.verdict:
            parser.error("--record needs --verdict and --reason")
        print(json.dumps(record(*args.record, args.source, args.verdict, args.reason)))
        return 0
    names = [p for p in args.passes.split(",") if p]
    unknown = sorted(set(names) - set(PASSES))
    if unknown:
        parser.error(f"unknown passes {unknown}; choose from {', '.join(PASSES)}")
    paths = args.paths or default_paths()
    clones, status = {}, {}
    with tempfile.TemporaryDirectory() as folder:
        for name in names:
            clones[name], status[name] = run_pass(name, paths, Path(folder))
            print(f"{name}: {status[name]}")
    found = candidates(clones, known_pairs(read_reviewed()))
    payload = {
        "date": datetime.datetime.now(datetime.UTC).date().isoformat(),
        "passes": status,
        "record_with": "python tools/jscpd_review.py --record A B --source S --verdict V "
        "--reason TEXT",
        "candidates": found,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", "utf-8")
    print(f"{len(found)} new pairs to review: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

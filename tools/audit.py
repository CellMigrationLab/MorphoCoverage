"""Regenerate the function index and the duplicate list, and keep the ratchets.

``python tools/audit.py`` rewrites ``docs/common/function_index.jsonl`` and
``docs/common/function_duplicates.json`` and prints the counts. The baselines in
``tools/baselines/`` only tighten:

- ``--tighten`` lowers them to the current values and drops what was fixed (never adds);
- ``--accept KIND:module:name --reason "..."`` admits one new or larger entry, recorded in
  ``tools/baselines/accepted.json``;
- ``--rename OLD NEW --reason "..."`` records a function moved with edits
  (``renames.json``), so its entry follows it; an unchanged body is followed by itself;
- ``--check-against REF`` fails when a baseline holds an entry that is new or larger than at
  the merge base with ``REF`` and was not accepted, or when a limit changed (CI);
- ``--fallbacks`` lists every silent fallback with its marked reason;
- ``--find "query"`` searches the index (run it before writing a helper).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import duplicates
import fallbacks
import function_index
import layers
import ratchet
import search


def _arguments(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tighten", action="store_true", help="lower the baselines, never add")
    parser.add_argument("--accept", metavar="KIND:module:name", help="admit one entry")
    parser.add_argument("--rename", nargs=2, metavar=("OLD", "NEW"), help="record a move")
    parser.add_argument("--reason", default="", help="why (with --accept or --rename)")
    parser.add_argument("--check-against", metavar="REF", help="compare with the merge base")
    parser.add_argument("--fallbacks", action="store_true", help="list the silent fallbacks")
    parser.add_argument("--find", metavar="QUERY", help="search the index")
    parser.add_argument("--write-baselines", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def _tighten_lists(exact: dict[str, list[str]]) -> None:
    """The duplicate groups and layer findings recorded: only the ones still there kept."""
    folder = function_index.BASELINES
    groups = sorted(" | ".join(g) for g in exact.values())
    kept = [g for g in json.loads((folder / "duplicates.json").read_text()) if g in groups]
    found = {"violations": layers.violations(), "cycles": layers.cycles()}
    recorded = json.loads((folder / "layers.json").read_text())
    tightened = {name: [v for v in recorded[name] if v in found[name]] for name in found}
    for name, value in (("duplicates", kept), ("layers", tightened)):
        text = json.dumps(value, indent=1, ensure_ascii=False) + "\n"
        (folder / f"{name}.json").write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Run the command line; returns the exit status."""
    args = _arguments(argv)
    if args.write_baselines:
        print(
            "--write-baselines is gone: it admitted anything. Use --tighten to record a fix, "
            '--accept KIND:module:name --reason "..." to admit one entry.',
            file=sys.stderr,
        )
        return 2
    if args.find:
        print(search.show(args.find))
        return 0
    if args.check_against:
        problems = ratchet.check_against(args.check_against)
        print("\n".join(problems) or f"baselines only tightened against {args.check_against}")
        return 1 if problems else 0
    old_rows = function_index.read_index()
    trees = function_index.parsed()
    rows = function_index.build_index(trees)
    function_index.write_index(rows)
    exact, near = duplicates.find()
    duplicates.write(exact, near)
    found = fallbacks.fallbacks(trees)
    current = ratchet.findings(rows, found)
    if args.fallbacks:
        print("\n".join(fallbacks.describe(f) for f in found))
    functions = len(function_index.function_rows(rows))
    print(f"{functions} functions; {len(exact)} exact duplicate groups, {len(near)} near pairs")
    print(f"{len(layers.violations())} layer violations, {len(layers.cycles())} import cycles")
    for kind, values in current.items():
        print(f"  {kind}: {len(values)}")
    try:
        if args.accept:
            print(ratchet.accept(args.accept, args.reason, current))
        if args.rename:
            if not args.reason.strip():
                raise ValueError("--rename needs --reason: what changed besides the name")
            print(ratchet.record_rename(*args.rename, args.reason.strip()))
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2
    if args.tighten:
        for kind, (before, after) in ratchet.tighten(current, old_rows, rows).items():
            print(f"  {kind} baseline: {before} -> {after} entries")
        _tighten_lists(exact)
    return 0


if __name__ == "__main__":
    sys.exit(main())

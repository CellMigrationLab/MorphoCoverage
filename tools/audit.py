"""Regenerate every generated audit file and, on request, record the baselines.

``python tools/audit.py`` rewrites ``docs/common/function_index.jsonl`` and
``docs/common/function_duplicates.json`` and prints the counts; ``--write-baselines`` also
records the current function findings, exact duplicate groups, layer violations and import
cycles as the ratchets ``tests/test_function_guards.py`` enforces (use it to record a
removal, never to accept something new without a reason).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write-baselines", action="store_true")
    args = parser.parse_args(argv)
    index, duplicates, layers = _load("function_index"), _load("duplicates"), _load("layers")
    rows = index.build_index()
    index.write_index(rows)
    exact, near = duplicates.find()
    duplicates.write(exact, near)
    current = index.findings()
    print(f"{len(rows)} functions; {len(exact)} exact duplicate groups, {len(near)} near pairs")
    print(f"{len(layers.violations())} layer violations, {len(layers.cycles())} import cycles")
    for kind, values in current.items():
        print(f"  {kind}: {len(values)}")
    if args.write_baselines:
        index.write_baselines(current)
        extra = {
            "duplicates": duplicates.exact_groups(),
            "layers": {"violations": layers.violations(), "cycles": layers.cycles()},
        }
        for name, value in extra.items():
            path = index.BASELINES / f"{name}.json"
            path.write_text(
                json.dumps(value, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())

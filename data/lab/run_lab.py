"""Run MorphoCoverage on the lab datasets the way a user would, and store what it says.

For each dataset (all by default, or the ids given), with the declaration taken from its
card (readout, features, control, observation unit):

- ``data_check.txt``: ``check_tables``;
- ``report_short.txt`` / ``report_full.txt``: ``report(..., integrity=True)``, the default
  rules, the default definition of new on the declared features;
- ``run.txt``: the package version, the settings and the run time.

Results go to ``data/lab/results/<id>/`` and are read against the card's hypotheses in
``data/lab/FINDINGS.md``. Usage, from the repository root::

    PYTHONPATH=src python data/lab/run_lab.py [id ...]
"""

from __future__ import annotations

import subprocess
import sys
import time
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import catalog  # the catalog lives next to this script

import morphocoverage
from morphocoverage import check_tables
from morphocoverage.workflows import Setup, default_novelty, report


def _version() -> str:
    """The package version and the commit the run was made from."""
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=HERE,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    return f"{getattr(morphocoverage, '__version__', 'unversioned')} at commit {commit or '?'}"


def setup_for(data: catalog.LabDataset) -> Setup:
    """The declaration a user would write from the card."""
    features = tuple(data.feature_cols)
    return Setup(
        readout=data.card["readout"],
        features=features,
        control=data.control,
        novelty=default_novelty(features),
        observation_unit=data.observation_unit,
        random_state=0,
        n_jobs=4,
    )


def run(dataset_id: str) -> None:
    data = catalog.load(dataset_id)
    out = catalog.RESULTS / dataset_id
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    check = check_tables(
        *data.tables,
        feature_cols=data.feature_cols,
        observation_unit=data.observation_unit,
        control=data.control,
    )
    (out / "data_check.txt").write_text(check.summary() + "\n")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        setup = setup_for(data)
        result = report(*data.tables, setup, integrity=True)
    (out / "report_short.txt").write_text(result.summary() + "\n")
    (out / "report_full.txt").write_text(result.summary(detail="full") + "\n")
    seconds = time.perf_counter() - start
    lines = [
        f"dataset: {dataset_id}",
        f"morphocoverage: {_version()}",
        f"readout: {setup.readout}; features: {len(setup.features)}; control: {data.control}",
        "novelty: default_novelty(features); rules: the defaults; random_state 0",
        f"run time: {seconds:.0f} s",
        "warnings:",
        *([f"  {w.category.__name__}: {w.message}" for w in caught] or ["  none"]),
    ]
    (out / "run.txt").write_text("\n".join(lines) + "\n")
    print(f"{dataset_id}: {seconds:.0f} s", flush=True)


def main(argv: list[str]) -> None:
    for dataset_id in argv or catalog.ids():
        run(dataset_id)


if __name__ == "__main__":
    main(sys.argv[1:])

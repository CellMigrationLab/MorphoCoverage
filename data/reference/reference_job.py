"""Run one reference job: a report, short and full, its warnings and its page.

Called by ``reference.py`` in a subprocess, from the root of the tree under test (a ``git
archive`` export or a copy of the working tree), with ``PYTHONPATH=<tree>/src``::

    python reference_job.py <kind> <name> <out_dir> [<owner jobs file>]

kinds: ``lab``, ``synthetic``, ``morphobricks``, ``scenario`` (the committed data) and ``owner``
(a job of the owner's jobs file, see README.md). Nothing written depends on the clock, so two
runs of the same code give the same files. A failure is part of the output: ``record.json``
holds its stage, type and message, ``traceback.txt`` its traceback.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import traceback
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import pandas as pd

TREE = Path.cwd()


@dataclass
class Job:
    """What one job reports on: the tables, the declaration, the page's title and extras."""

    objects: pd.DataFrame
    fovs: pd.DataFrame
    setup: Any
    title: str
    feature_tables: dict[str, Any] | None = None
    extra_files: dict[str, pd.DataFrame] = field(default_factory=dict)
    scan: dict[str, Any] | None = None


def _package() -> Any:
    """The workflows of the tree under test, refused when imported from anywhere else."""
    import morphocoverage
    from morphocoverage import workflows

    source = Path(morphocoverage.__file__).resolve()
    if not source.is_relative_to(TREE.resolve() / "src"):
        raise ImportError(f"morphocoverage imported from {source}, not from {TREE}/src")
    return workflows


def lab(name: str) -> Job:
    """A lab dataset of ``data/lab``, its card's readout and control."""
    workflows = _package()
    sys.path.insert(0, str(TREE / "data" / "lab"))
    import catalog

    data = catalog.load(name)
    features = tuple(data.feature_cols)
    setup = workflows.Setup(
        readout=data.card["readout"],
        features=features,
        control=data.control,
        novelty=workflows.default_novelty(features),
        observation_unit=data.observation_unit,
        random_state=0,
        n_jobs=1,
    )
    objects, fovs = data.tables
    return Job(objects, fovs, setup, f"MorphoCoverage report: {name}")


def synthetic(name: str) -> Job:
    """The constructed dataset of ``data/synthetic``, with rules and a rare target."""
    workflows = _package()
    from morphocoverage import FixedFOVStrategy, RareTarget, ScalarPrecisionStrategy

    folder = TREE / "data" / "synthetic"
    features = ("area", "intensity", "eccentricity")
    setup = workflows.Setup(
        readout="area",
        features=features,
        control={"condition": "control"},
        observation_unit="cell",
        acquisition_order_col="acquisition_order",
        stage_cols=("stage_x", "stage_y"),
        rules=(FixedFOVStrategy(8), ScalarPrecisionStrategy("area", 0.05, "fov", "upper_95")),
        novelty=workflows.default_novelty(features),
        rare=RareTarget(
            detection_probability=0.8,
            min_fov_occurrences=2,
            phenotype="rare",
            object_prevalence=0.01,
        ),
        rare_label="rare",
        random_state=0,
        n_jobs=1,
    )
    objects = pd.read_csv(folder / "synthetic_cells.csv")
    fovs = pd.read_csv(folder / "synthetic_fovs.csv")
    return Job(objects, fovs, setup, f"MorphoCoverage report: {name}")


def morphobricks(name: str) -> Job:
    """The fiberfeatures table of ``data/morphobricks``, its fields from its own rows."""
    workflows = _package()
    table = pd.read_csv(TREE / "data" / "morphobricks" / "fiberfeatures_patches.csv")
    field_columns = ["condition", "biological_repeat", "technical_repeat", "image_file"]
    features = (
        "nematic_order",
        "linearity_mean",
        "entropy_mean",
        "bundle_evidence_mean",
        "pixel_category_network_fraction",
    )
    setup = workflows.Setup(
        readout="nematic_order",
        features=features,
        control={"condition": "condition_2"},
        observation_unit="patch",
        novelty=workflows.default_novelty(features),
        random_state=0,
        n_jobs=1,
    )
    fovs = table[field_columns].drop_duplicates()
    return Job(table, fovs, setup, "MorphoCoverage report: fiberfeatures patches")


def scenario(name: str) -> Job:
    """A report scenario of ``data/scenarios``, declared as its ``scenario.json`` says."""
    workflows = _package()
    sys.path.insert(0, str(TREE / "data" / "scenarios"))
    import make_scenarios

    loaded = make_scenarios.load(name)
    setup = workflows.Setup(**{**loaded.setup, "n_jobs": 1})
    return Job(loaded.objects, loaded.fovs, setup, f"Scenario {name}", loaded.feature_tables)


def owner_jobs(path: Path) -> dict[str, Callable[[], dict[str, Any]]]:
    """The ``JOBS`` of an owner's jobs file: name -> a function returning a job's fields."""
    spec = importlib.util.spec_from_file_location("reference_owner_jobs", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"{path} cannot be loaded as a Python file")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    jobs = getattr(module, "JOBS", None)
    if not isinstance(jobs, dict):
        raise TypeError(f"{path} defines no JOBS dictionary (name -> function)")
    return jobs


def owner(name: str, jobs_file: Path | None) -> Job:
    """A job of the owner's jobs file, its fields checked by ``Job``."""
    if jobs_file is None:
        raise ValueError("an owner job needs the owner's jobs file")
    _package()
    jobs = owner_jobs(jobs_file)
    if name not in jobs:
        raise KeyError(f"{name!r} is not in the JOBS of {jobs_file}: {sorted(jobs)}")
    return Job(**jobs[name]())


BUILT_IN: dict[str, Callable[[str], Job]] = {
    "lab": lab,
    "synthetic": synthetic,
    "morphobricks": morphobricks,
    "scenario": scenario,
}


def _caught(record: dict[str, Any], key: str, call: Callable[[], Any]) -> Any:
    """``call()`` with every warning recorded under ``key``, in order."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = call()
    record[key] = [f"{item.category.__name__}: {item.message}" for item in caught]
    return value


def _write(job: Job, out: Path, record: dict[str, Any]) -> None:
    """The job's extra files and scan, its report (short and full) and its page."""
    workflows = _package()
    attach = []
    for file_name, frame in job.extra_files.items():
        frame.to_csv(out / file_name, index=False)
        attach.append(out / file_name)
    if job.scan is not None:
        record["stage"] = "scan"
        scan = workflows.scan_design(job.objects, job.fovs, **job.scan)
        (out / "scan.txt").write_text(scan.summary() + "\n")
        attach.insert(0, out / "scan.txt")
    record["stage"] = "report"
    result = _caught(
        record,
        "warnings",
        lambda: workflows.report(
            job.objects, job.fovs, job.setup, integrity=True, feature_tables=job.feature_tables
        ),
    )
    (out / "report_short.txt").write_text(result.summary() + "\n")
    (out / "report_full.txt").write_text(result.summary(detail="full") + "\n")
    record["stage"] = "page"
    _caught(
        record,
        "page_warnings",
        lambda: workflows.html_report(
            result,
            job.objects,
            job.fovs,
            out,
            title=job.title,
            attach=attach,
            feature_tables=job.feature_tables,
        ),
    )


def run(kind: str, name: str, out: Path, jobs_file: Path | None) -> dict[str, Any]:
    """Run one job into ``out``; its record, the failure (if any) included."""
    out.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {"kind": kind, "name": name, "status": "ok", "stage": "setup"}
    try:
        job = owner(name, jobs_file) if kind == "owner" else BUILT_IN[kind](name)
        _write(job, out, record)
    except Exception as error:  # noqa: BLE001 - a job's failure is part of its output
        record["status"] = f"failed at {record['stage']}: {type(error).__name__}: {error}"
        (out / "traceback.txt").write_text(traceback.format_exc())
    del record["stage"]
    (out / "record.json").write_text(json.dumps(record, indent=1) + "\n")
    return record


def main() -> None:
    """The command line: kind, name, output folder, and the owner's jobs file if any."""
    kind, name, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    jobs_file = Path(sys.argv[4]) if len(sys.argv) > 4 else None
    record = run(kind, name, out, jobs_file)
    print(f"{kind} {name}: {record['status']}", flush=True)


if __name__ == "__main__":
    main()

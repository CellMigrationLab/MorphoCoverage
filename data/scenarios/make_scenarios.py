"""Small acquisitions for every kind of input a user gives MorphoCoverage, with the answers.

Each scenario is one folder under ``data/scenarios/<name>/``: the FOV table (``fovs.csv``),
the observation table (``observations.csv``), each other feature table (``<table>.csv``)
and ``scenario.json``, which holds the ``Setup`` arguments as plain JSON, the feature
tables with their tags, and the truth: the planted effects, the null conditions and
features, the analyses that must run, the questions that must say "not applicable" and
why, or the refusal expected. ``tests/test_scenarios_matrix.py`` runs every folder and
checks the truth; ``README.md`` lists them.

Three kinds of scenario (``scenario_axes.py``, ``scenario_edges.py``):

- ``axes``: the tables (explainable only, two tagged explainable tables, one or two
  embeddings, a mix) x the controls (none, one, several) x normalisation x the unit (cells,
  one image per field) x the wells x the target (tolerances, a fingerprint on the
  explainable features or on an embedding, a labelled rare phenotype, a positive control),
  every pair of values in at least one scenario;
- ``edge``: designs the axes do not reach (two repeats, one condition, a screen, ...);
- ``mistake``: what users get wrong, and what the package does with it now.

The truth (``scenario.json``'s ``truth``) holds:

- ``outcome``: ``"runs"`` or ``"refused"``; a refusal names where (``"setup"`` or
  ``"report"``), the exception and a part of its message;
- ``results``: the analyses that must run (keys of ``Report.results``), ``absent`` those
  that must not;
- ``not_applicable``: by question, a part of the reason ``Report.not_assessed`` or the
  question's headline gives; ``headlines``: a part of a question's headline;
- ``contrast_kinds``, ``separation_spaces``: the kinds of comparison of the contrast table,
  the feature spaces scored for separation;
- ``untested_across``, ``untested_whole`` (a part of the status), ``left_out``,
  ``compared``, ``tested_features``, ``balanced``: what the tests across repeats, the
  whole-table test, the discovery and the cell-level screen did;
- ``overview``: parts of the dataset overview;
- ``planted``, ``null_conditions``, ``null_features``, ``whole_table_planted``: the
  conditions shifted, by feature or table, and those not, checked only where the design
  makes the answer near-certain (a strong effect, five repeats or more);
- ``html``: whether the test also builds the page, and ``table_column`` whether its CSVs
  name each feature's table.

The tables are built from a latent effect per observation: a condition's shift (on a log
scale), a repeat's, a well's and a field's, and the observation's own noise. ``area`` is
``10 exp(effect + noise)``, so a shift of 0.8 multiplies it by 2.2; ``ecc`` never depends
on the effect (a null feature). Everything is seeded: running this again writes the same
bytes, which the test checks.

Usage, from the repository root::

    python data/scenarios/make_scenarios.py
"""

from __future__ import annotations

import json
import shutil
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SEED = 20261006
REPEATS = tuple(f"2026_0{1 + i // 4}_{5 + 7 * (i % 4):02d}" for i in range(8))
KEY = ["condition", "biological_repeat", "technical_repeat", "image_file", "label_id"]
DECIMALS = 4


@dataclass(frozen=True)
class Design:
    """The layout of one acquisition.

    ``shifts`` are the conditions' effects on the latent scale (0 when not named);
    ``absent`` the (condition, repeat index) pairs never imaged; ``fields`` the fields per
    condition and repeat (per well with wells), one number or one per repeat; ``cells``
    the observations per field (1: one image per field); ``wells`` the technical repeats
    per condition and repeat (0: no ``technical_repeat`` column).
    """

    conditions: tuple[str, ...]
    repeats: int = 4
    shifts: Mapping[str, float] = field(default_factory=dict)
    absent: frozenset[tuple[str, int]] = frozenset()
    fields: int | tuple[int, ...] = 3
    cells: int = 5
    wells: int = 0
    repeat_sd: float = 0.1
    field_sd: float = 0.05
    seed: int = SEED


@dataclass
class Scenario:
    """One folder: the tables, the Setup as plain JSON, and the truth."""

    name: str
    kind: str
    checks: str
    objects: pd.DataFrame
    fovs: pd.DataFrame
    setup: dict[str, Any]
    truth: dict[str, Any]
    tables: dict[str, tuple[pd.DataFrame, str | None]] = field(default_factory=dict)
    axes: dict[str, Any] = field(default_factory=dict)
    planes: dict[str, Any] | None = None


def acquisition(design: Design) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """The observation key, the FOV table and each observation's latent effect.

    The observation table holds only the key here (``KEY``, ``technical_repeat`` when
    there are wells); the feature columns come from ``observation_columns`` and the other
    tables from ``table``.
    """
    rng = np.random.default_rng(design.seed)
    fields = design.fields
    per_repeat = fields if isinstance(fields, tuple) else (fields,) * design.repeats
    rows: list[dict[str, Any]] = []
    fov_rows: list[dict[str, Any]] = []
    effects: list[float] = []
    for r, repeat in enumerate(REPEATS[: design.repeats]):
        repeat_effect = rng.normal(0, design.repeat_sd)
        for c, condition in enumerate(design.conditions):
            if (condition, r) in design.absent:
                continue
            for w in range(max(design.wells, 1)):
                well_effect = rng.normal(0, design.field_sd) if design.wells else 0.0
                for f in range(per_repeat[r]):
                    key = {"condition": condition, "biological_repeat": repeat}
                    if design.wells:
                        # Wells are named on the plate, so each condition has its own.
                        key["technical_repeat"] = f"W{c * design.wells + w + 1:02d}"
                    key["image_file"] = f"field_{f + 1:02d}.tif"
                    fov_rows.append(dict(key))
                    effect = (
                        design.shifts.get(condition, 0.0)
                        + repeat_effect
                        + well_effect
                        + rng.normal(0, design.field_sd)
                    )
                    for label in range(1, design.cells + 1):
                        rows.append({**key, "label_id": label})
                        effects.append(effect)
    return pd.DataFrame(rows), pd.DataFrame(fov_rows), np.asarray(effects)


def _round(frame: pd.DataFrame) -> pd.DataFrame:
    """Floats rounded, so the CSVs are small and read back as written."""
    floats = frame.select_dtypes("float").columns
    return frame.assign(**{c: frame[c].round(DECIMALS) for c in floats})


def observation_columns(
    keys: pd.DataFrame,
    effect: np.ndarray,
    columns: Sequence[str],
    seed: int,
    rare_conditions: Sequence[str] = (),
) -> pd.DataFrame:
    """The observation table: ``keys`` and the named feature columns.

    ``area`` (positive, the readout) and ``length`` and ``intensity_mean`` follow the
    effect; ``ecc`` does not (null); ``count`` is a count with many zeros that follows the
    effect; ``velocity`` is signed, a drift of -0.5 plus the effect; ``rare`` labels
    5% of observations, 25% in ``rare_conditions``.
    """
    rng = np.random.default_rng(seed)
    n = len(keys)
    makers: dict[str, Callable[[], np.ndarray]] = {
        "area": lambda: 10 * np.exp(effect + rng.normal(0, 0.2, n)),
        "length": lambda: 5 + effect + rng.normal(0, 0.5, n),
        "intensity_mean": lambda: 100 * np.exp(0.5 * effect + rng.normal(0, 0.2, n)),
        "ecc": lambda: rng.uniform(0, 1, n),
        "count": lambda: rng.poisson(0.4 * np.exp(effect)).astype(float),
        "velocity": lambda: effect - 0.5 + rng.normal(0, 0.3, n),
        "rare": lambda: (
            rng.uniform(size=n) < np.where(keys["condition"].isin(rare_conditions), 0.25, 0.05)
        ).astype(int),
    }
    values = {name: makers[name]() for name in columns}
    return _round(keys.assign(**values))


def table(
    keys: pd.DataFrame,
    effect: np.ndarray,
    columns: Sequence[str],
    seed: int,
    *,
    affected: int = 2,
    baseline: float = 0.0,
    reverse: bool = True,
) -> pd.DataFrame:
    """Another feature table for the same observations: ``keys`` and ``columns``.

    The first ``affected`` columns follow the effect, the others are noise, around
    ``baseline`` (an explainable measurement such as a length is positive, an embedding's
    coordinate is not). The rows are reversed by default: a table is matched by its key,
    never by its row order.
    """
    rng = np.random.default_rng(seed)
    n = len(keys)
    values = {
        name: baseline + rng.normal(0, 1, n) + (2 * effect if j < affected else 0.0)
        for j, name in enumerate(columns)
    }
    frame = _round(keys.assign(**values))
    return frame.iloc[::-1].reset_index(drop=True) if reverse else frame


def dino(keys: pd.DataFrame, effect: np.ndarray, seed: int, width: int = 8) -> pd.DataFrame:
    """An embedding: ``width`` columns ``d0`` ... named as a model writes them."""
    return table(keys, effect, [f"d{j}" for j in range(width)], seed)


def final_names(
    tables: Mapping[str, tuple[pd.DataFrame, str | None]], name: str, observed: Sequence[str]
) -> list[str]:
    """Table ``name``'s feature columns as the report names them: a column also found in
    another table, or among the ``observed`` columns, is prefixed with its table's name."""
    frame = tables[name][0]
    others = set(observed)
    for other, (frame_other, _) in tables.items():
        if other != name:
            others |= set(frame_other.columns)
    return [f"{name}: {c}" if c in others else c for c in frame.columns if c not in (*KEY, "z")]


def covariance(columns: Sequence[str], tolerance: float = 0.2) -> dict[str, Any]:
    """A fingerprint criterion as plain JSON: the covariance of ``columns`` within a
    tolerance (``setup_arguments`` builds the ``OutcomeCriterion``)."""
    return {"metric": "covariance", "columns": list(columns), "tolerance": tolerance}


RARE_TARGET = {
    "detection_probability": 0.8,
    "min_fov_occurrences": 2,
    "phenotype": "rare",
    "object_prevalence": 0.1,
}


def setup_arguments(spec: Mapping[str, Any]) -> dict[str, Any]:
    """The ``Setup`` keyword arguments from ``scenario.json``'s ``setup``.

    Plain JSON stands for the objects: ``fingerprint`` is a list of ``covariance``
    criteria, ``rare`` the ``RareTarget`` fields, ``positive`` the ``PositiveControl``
    fields, ``novelty: "default"`` the ``default_novelty`` of the features, and
    ``feature_tables`` lists of columns.
    """
    from morphocoverage import RareTarget
    from morphocoverage.multivariate_outcomes import CovarianceError
    from morphocoverage.retrospective.criteria import OutcomeCriterion
    from morphocoverage.workflows import PositiveControl, default_novelty

    arguments = dict(spec)
    for name in ("features", "intensity_features"):
        if name in arguments:
            arguments[name] = tuple(arguments[name])
    if "fingerprint" in arguments:
        arguments["fingerprint"] = tuple(
            OutcomeCriterion(
                CovarianceError(tuple(c["columns"]), True), c["tolerance"], "lower_is_better"
            )
            for c in arguments["fingerprint"]
        )
    if "rare" in arguments:
        arguments["rare"] = RareTarget(**arguments["rare"])
    if "positive" in arguments:
        arguments["positive"] = PositiveControl(**arguments["positive"])
    if arguments.get("novelty") == "default":
        arguments["novelty"] = default_novelty(arguments["features"])
    for name in ("feature_tables", "explainable_tables"):
        if arguments.get(name):
            arguments[name] = {k: tuple(v) for k, v in arguments[name].items()}
    return arguments


@dataclass
class Loaded:
    """A scenario as a user holds it: the tables read from CSV, and its declaration."""

    name: str
    objects: pd.DataFrame
    fovs: pd.DataFrame
    setup: dict[str, Any]
    feature_tables: dict[str, Any] | None
    truth: dict[str, Any]
    spec: dict[str, Any]


def load(name: str, root: Path = HERE) -> Loaded:
    """Read scenario ``name``: its tables as ``report`` takes them, the ``Setup`` keyword
    arguments (``setup_arguments``), and the truth. A scenario with ``planes`` is split per
    plane with ``planes_as_features`` first, as the documentation asks."""
    folder = root / name
    spec = json.loads((folder / "scenario.json").read_text())
    objects = pd.read_csv(folder / "observations.csv")
    fovs = pd.read_csv(folder / "fovs.csv")
    if spec.get("planes"):
        from morphocoverage import planes_as_features

        objects = planes_as_features(objects, fovs, **spec["planes"])[0]
    tables: dict[str, Any] = {}
    for table_name, entry in spec["feature_tables"].items():
        frame = pd.read_csv(folder / entry["file"])
        tables[table_name] = frame if entry["tag"] is None else (frame, entry["tag"])
    return Loaded(
        name,
        objects,
        fovs,
        setup_arguments(spec["setup"]),
        tables or None,
        spec["truth"],
        spec,
    )


def names(root: Path = HERE) -> list[str]:
    """The scenarios written under ``root``, in order."""
    return sorted(p.parent.name for p in root.glob("*/scenario.json"))


def _csv(frame: pd.DataFrame, path: Path) -> None:
    """One table, written the same way every time."""
    frame.to_csv(path, index=False, lineterminator="\n")


def write(scenario: Scenario, root: Path) -> None:
    """Write one scenario's folder (see the module)."""
    folder = root / scenario.name
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    _csv(scenario.fovs, folder / "fovs.csv")
    _csv(scenario.objects, folder / "observations.csv")
    entries = {}
    for name, (frame, tag) in scenario.tables.items():
        file = f"{name}.csv"
        _csv(frame, folder / file)
        entries[name] = {"file": file, "tag": tag}
    spec = {
        "name": scenario.name,
        "kind": scenario.kind,
        "checks": scenario.checks,
        "axes": scenario.axes,
        "planes": scenario.planes,
        "setup": scenario.setup,
        "feature_tables": entries,
        "truth": scenario.truth,
    }
    (folder / "scenario.json").write_text(json.dumps(spec, indent=1) + "\n")


def _given(scenario: Scenario) -> str:
    """An edge or mistake scenario's design in a few words, from its tables."""
    fovs = scenario.fovs
    counts = {
        "condition": fovs["condition"].nunique(),
        "repeat": fovs["biological_repeat"].nunique(),
        "field": len(fovs),
        "row": len(scenario.objects),
    }
    text = ", ".join(f"{n} {word}{'' if n == 1 else 's'}" for word, n in counts.items())
    tables = [f"{name} ({tag or 'non-explainable'})" for name, (_, tag) in scenario.tables.items()]
    return text + (f"; {', '.join(tables)}" if tables else "")


def readme_rows(scenarios: Sequence[Scenario]) -> str:
    """The README's table: one row per scenario."""
    lines = ["| Scenario | Kind | Input | What it checks |", "| --- | --- | --- | --- |"]
    for s in scenarios:
        given = ", ".join(f"{k} {v}" for k, v in s.axes.items()) if s.axes else _given(s)
        lines.append(f"| `{s.name}` | {s.kind} | {given} | {s.checks} |")
    return "\n".join(lines)


def main(root: Path = HERE) -> list[Scenario]:
    """Write every scenario under ``root`` and the README's table (see the module)."""
    sys.path.insert(0, str(HERE))
    try:
        from scenario_axes import SCENARIOS as AXES_SCENARIOS
        from scenario_edges import SCENARIOS as EDGE_SCENARIOS
    finally:
        sys.path.remove(str(HERE))
    scenarios = [build() for build in (*AXES_SCENARIOS, *EDGE_SCENARIOS)]
    for scenario in scenarios:
        write(scenario, root)
    readme = (HERE / "README.md").read_text() if (HERE / "README.md").exists() else ""
    start, end = "<!-- scenarios -->", "<!-- /scenarios -->"
    if start in readme and root == HERE:
        head, rest = readme.split(start, 1)
        tail = rest.split(end, 1)[1]
        table_text = readme_rows(scenarios)
        (HERE / "README.md").write_text(f"{head}{start}\n{table_text}\n{end}{tail}")
    return scenarios


if __name__ == "__main__":
    written = main()
    print(f"{len(written)} scenarios written to {HERE}")

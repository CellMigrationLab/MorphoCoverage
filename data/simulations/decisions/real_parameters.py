"""Summary statistics of real feature tables, to anchor the benchmark's designs in data
(docs/common/validation.md, "Real-data anchoring" and its provenance table).

For each dataset: the conditions, repeats, wells per condition and repeat, fields per well,
observations per field and the share of fields without any (when the FOV table lists every
field), and the control-relative density of each condition (its median observations per
field over the control's). For each declared feature, on the control's observations: the
variance shares of the biological repeat, the well (when the table names wells), the field
and the observation (``statistics.nested_variance_components``, method of moments, a
negative component reported as 0; with one observation per field the field holds the
residual), the CV, skewness, excess kurtosis and the share of positive values. For the
declared features together: the correlation spectrum's effective rank (participation ratio)
and its first eigenvalue's share. For an embedding table: the effective rank and anisotropy
of its covariance and the repeat centroid shifts (each repeat's control centroid from the
mean of them, in within-group SDs).

Only summaries are written, never a row of the tables. The owner's tables outside the
repository are read from a path on the command line and never copied:

    PYTHONPATH=src:data/simulations python -m decisions.real_parameters \\
        --table NAME OBJECTS.csv FOVS.csv --control CTRL --features a,b,c \\
        [--embedding-prefix dino_] --out summary.csv

Without ``--table``, the committed datasets: the lab's (``data/lab``) and idr0028's
development tables (``data/crick_idr``; idr0139, held out, is left out unless
``--include-held-out``: its role is never to choose a setting), written to
``results/real_parameters.csv`` beside this file. Feature columns are always declared
(the card's, the dataset's, ``--features`` or ``--embedding-prefix``), never inferred.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from morphocoverage._repeat_alias import refuse_repeat_alias
from morphocoverage.metadata import CONDITION_COLUMN, REPEAT_COLUMN, TECHNICAL_REPEAT_COLUMN
from morphocoverage.statistics import nested_variance_components

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1]
FIELDS = ("position_id", "image_file")
OUTPUT = HERE / "results" / "real_parameters.csv"


def _row(dataset: str, scope: str, name: str, statistic: str, value: Any) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "scope": scope,
        "name": name,
        "statistic": statistic,
        "value": value,
    }


def _quantiles(values: np.ndarray) -> dict[str, float]:
    if not len(values):
        return {}
    q10, q50, q90 = np.quantile(values, [0.1, 0.5, 0.9])
    return {
        "median": float(q50),
        "q10": float(q10),
        "q90": float(q90),
        "mean": float(values.mean()),
    }


def _field(table: pd.DataFrame) -> str:
    """The FOV column: ``position_id``, else ``image_file`` (one file one position)."""
    return next(c for c in FIELDS if c in table)


def _field_key(table: pd.DataFrame) -> list[str]:
    keys = [c for c in (CONDITION_COLUMN, REPEAT_COLUMN, TECHNICAL_REPEAT_COLUMN) if c in table]
    return [*keys, _field(table)]


def design_rows(
    dataset: str, objects: pd.DataFrame, fovs: pd.DataFrame, control: str
) -> list[dict[str, Any]]:
    """The dataset-level statistics (see the module)."""
    rows = [_row(dataset, "dataset", "", "observations", len(objects))]
    key = [c for c in _field_key(fovs) if c in objects]
    counts = objects.groupby(key).size()
    every = fovs.set_index(key).index.unique()
    per_field = counts.reindex(every, fill_value=0)
    rows.append(_row(dataset, "dataset", "", "fields", len(per_field)))
    rows.append(_row(dataset, "dataset", "", "empty_field_share", float((per_field == 0).mean())))
    rows += [
        _row(dataset, "dataset", "", f"observations_per_field_{k}", v)
        for k, v in _quantiles(per_field.to_numpy(dtype=float)).items()
    ]
    for column in (CONDITION_COLUMN, REPEAT_COLUMN):
        rows.append(_row(dataset, "dataset", "", f"{column}s", fovs[column].nunique()))
    group = [CONDITION_COLUMN, REPEAT_COLUMN]
    if TECHNICAL_REPEAT_COLUMN in fovs and fovs[TECHNICAL_REPEAT_COLUMN].notna().any():
        wells = fovs.groupby(group)[TECHNICAL_REPEAT_COLUMN].nunique()
        fields_per_well = fovs.groupby([*group, TECHNICAL_REPEAT_COLUMN]).size()
        rows += [
            _row(dataset, "dataset", "", f"wells_per_condition_repeat_{k}", v)
            for k, v in _quantiles(wells.to_numpy(dtype=float)).items()
        ]
        rows += [
            _row(dataset, "dataset", "", f"fields_per_well_{k}", v)
            for k, v in _quantiles(fields_per_well.to_numpy(dtype=float)).items()
        ]
    density = per_field.groupby(level=CONDITION_COLUMN).median()
    if control in density.index and density[control] > 0:
        for condition, value in density.items():
            if condition != control:
                rows.append(
                    _row(
                        dataset,
                        "condition",
                        str(condition),
                        "density_vs_control",
                        float(value / density[control]),
                    )
                )
    return rows


def _codes(frame: pd.DataFrame, levels: Sequence[str]) -> list[np.ndarray]:
    """Unique unit numbers at each nested level (each level's key includes its parents')."""
    return [
        frame.groupby(list(levels[: i + 1]), sort=False).ngroup().to_numpy()
        for i in range(len(levels))
    ]


def feature_rows(
    dataset: str, objects: pd.DataFrame, control: str, features: Sequence[str]
) -> list[dict[str, Any]]:
    """The per-feature statistics on the control's observations (see the module)."""
    rows: list[dict[str, Any]] = []
    mine = objects[objects[CONDITION_COLUMN] == control]
    field = _field(mine)
    candidates = [(REPEAT_COLUMN, "repeat")]
    if TECHNICAL_REPEAT_COLUMN in mine and mine[TECHNICAL_REPEAT_COLUMN].notna().all():
        candidates.append((TECHNICAL_REPEAT_COLUMN, "well"))
    candidates.append((field, "field"))
    for feature in features:
        frame = mine.dropna(subset=[feature])
        values = frame[feature].to_numpy(dtype=float)
        if len(values) < 3 or np.ptp(values) == 0:
            continue
        # A level with as many units as the one above it (one well per repeat) is that level;
        # one with as many units as observations (one observation per field) is the residual:
        # merged, the share named with both.
        columns, labels, units, residual = [], [], 1, "residual"
        for column, label in candidates:
            count = frame.groupby([*columns, column]).ngroups
            if count >= len(frame):
                residual = f"{label}_and_residual"
                break
            if count > units:
                columns.append(column)
                labels.append(label)
                units = count
            elif labels:
                labels[-1] = f"{labels[-1]}_and_{label}"
        names = [*labels, residual]
        try:
            parts = nested_variance_components(values, _codes(frame, columns))
            parts = np.clip(parts, 0, None)
        except ValueError:
            parts = np.full(len(names), np.nan)
        total = parts.sum()
        for label, part in zip(names, parts, strict=True):
            rows.append(
                _row(
                    dataset,
                    "feature",
                    feature,
                    f"share_{label}",
                    float(part / total) if total > 0 else np.nan,
                )
            )
        mean = values.mean()
        rows += [
            _row(
                dataset,
                "feature",
                feature,
                "cv",
                float(values.std(ddof=1) / abs(mean)) if mean else np.nan,
            ),
            _row(dataset, "feature", feature, "skewness", float(stats.skew(values))),
            _row(dataset, "feature", feature, "excess_kurtosis", float(stats.kurtosis(values))),
            _row(dataset, "feature", feature, "positive_share", float((values > 0).mean())),
        ]
    if len(features) > 1:
        complete = mine[list(features)].dropna()
        varies = complete.loc[:, complete.std() > 0]
        if varies.shape[1] > 1 and len(varies) > 2:
            eigen = np.clip(
                np.linalg.eigvalsh(np.corrcoef(varies.to_numpy(dtype=float).T)), 0, None
            )
            rows.append(
                _row(
                    dataset,
                    "features",
                    "",
                    "correlation_effective_rank",
                    float(eigen.sum() ** 2 / np.square(eigen).sum()),
                )
            )
            rows.append(
                _row(
                    dataset,
                    "features",
                    "",
                    "correlation_first_share",
                    float(eigen.max() / eigen.sum()),
                )
            )
    return rows


def embedding_rows(
    dataset: str, objects: pd.DataFrame, control: str, columns: Sequence[str]
) -> list[dict[str, Any]]:
    """Effective rank, anisotropy and repeat centroid shifts of an embedding table."""
    mine = objects[objects[CONDITION_COLUMN] == control].dropna(subset=list(columns))
    values = mine[list(columns)].to_numpy(dtype=float)
    eigen = np.clip(np.linalg.eigvalsh(np.cov(values.T)), 0, None)
    rows = [
        _row(
            dataset,
            "embedding",
            "",
            "effective_rank",
            float(eigen.sum() ** 2 / np.square(eigen).sum()),
        ),
        _row(dataset, "embedding", "", "anisotropy", float(eigen.max() / eigen.sum())),
        _row(dataset, "embedding", "", "dimensions", len(columns)),
    ]
    repeats = mine[REPEAT_COLUMN].to_numpy()
    centres = np.array([values[repeats == r].mean(axis=0) for r in pd.unique(repeats)])
    within = values - centres[pd.factorize(repeats)[0]]
    sd = float(np.sqrt(np.square(within).sum(axis=1).mean()))
    shifts = np.linalg.norm(centres - centres.mean(axis=0), axis=1) / sd if sd > 0 else []
    rows += [
        _row(dataset, "embedding", "", f"repeat_shift_{k}", v)
        for k, v in _quantiles(np.asarray(shifts, dtype=float)).items()
    ]
    return rows


def summarise(
    dataset: str,
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    control: str,
    features: Sequence[str],
    embedding: Sequence[str] = (),
) -> pd.DataFrame:
    """Every statistic of one dataset, one row each (see the module)."""
    # The former name ``repeat`` is refused, as the package refuses it (owner ruling 14).
    refuse_repeat_alias(objects=objects, fovs=fovs)
    rows = design_rows(dataset, objects, fovs, control)
    rows += feature_rows(dataset, objects, control, features)
    if embedding:
        rows += embedding_rows(dataset, objects, control, embedding)
    return pd.DataFrame(rows)


def committed(include_held_out: bool = False) -> pd.DataFrame:
    """The statistics of the committed lab and Crick/IDR datasets."""
    frames = []
    sys.path.insert(0, str(DATA / "lab"))
    try:
        catalog = importlib.import_module("catalog")
        for name in catalog.ids():
            found = catalog.load(name)
            objects, fovs = found.tables
            control = found.control["condition"]
            frames.append(summarise(f"lab/{name}", objects, fovs, control, found.feature_cols))
    finally:
        sys.path.remove(str(DATA / "lab"))
    sys.path.insert(0, str(DATA / "crick_idr"))
    try:
        datasets = importlib.import_module("datasets")
        for name in datasets.NAMES:
            found = datasets.load(name)
            if found.role == "held_out" and not include_held_out:
                continue
            features = list(found.features) or [found.readout]
            frames.append(
                summarise(f"crick_idr/{name}", found.objects, found.fovs, found.control, features)
            )
    finally:
        sys.path.remove(str(DATA / "crick_idr"))
    return pd.concat(frames, ignore_index=True)


def main(argv: Sequence[str] | None = None) -> None:
    """The command line (see the module)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", nargs=3, metavar=("NAME", "OBJECTS", "FOVS"))
    parser.add_argument("--control")
    parser.add_argument("--features", default="")
    parser.add_argument("--embedding-prefix", default="")
    parser.add_argument("--include-held-out", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.table:
        name, objects_path, fovs_path = args.table
        objects, fovs = pd.read_csv(objects_path), pd.read_csv(fovs_path)
        features = [f for f in args.features.split(",") if f]
        embedding = [
            c
            for c in objects.columns
            if args.embedding_prefix and c.startswith(args.embedding_prefix)
        ]
        if not features and not embedding:
            raise SystemExit("declare the feature columns: --features or --embedding-prefix")
        frame = summarise(name, objects, fovs, args.control, features, embedding)
    else:
        frame = committed(args.include_held_out)
    out = args.out or OUTPUT
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False, float_format="%.6g")
    print(f"{len(frame)} statistics written to {out}")


if __name__ == "__main__":
    main()

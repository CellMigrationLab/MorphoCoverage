"""Retrospective check: does a well mean from whole FOVs behave like one from random objects?

For every well of the complete source plates, draw ``k`` whole FOVs at random
(what a microscope acquires) and compare the error of their mean against the
full well with the error expected from the same number of objects drawn at
random from the whole well (the assumption behind object-count rules:
spatially uniform samples, or random object sampling). The random-object
error uses the exact simple-random-sampling variance with finite-population
correction, so only FOV draws are simulated.

Metrics are each paper's own: ``Fascin_Ratio`` for idr0139 and
``YAPTAZ_Ratio`` for idr0028, both nuclear mean / (nuclear mean +
cytoplasmic mean) intensity, read from the stored complete plates
(``datasets.load("idr0139_plate")`` and ``"idr0028_screenB_plates"``).
Usage::

    python data/crick_idr/fov_vs_objects.py

The summary tables are written to ``results/fov_vs_objects_*.csv`` with the
source commit, so the numbers in ``docs/preliminary_findings.md`` (entry 1)
come from tracked files.

Exploratory and retrospective: the full-well mean is the reference, and only
the imaged fields of each well are known.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from convert import SOURCE_COMMIT
from datasets import RESULTS, load

N_DRAWS = 400
SEED = 0
MIN_OBJECTS = 50


def compare(objects: pd.DataFrame, fovs: pd.DataFrame, metric: str, ks, rng) -> pd.DataFrame:
    """Relative RMSE of the well mean from k random FOVs against the same number of objects."""
    rows = []
    key = ["biological_repeat", "technical_repeat"]
    grouped = objects.groupby(["biological_repeat", "image_file"])[metric]
    sums_all, counts_all = grouped.sum(), grouped.count()
    variances = objects.groupby(key)[metric].var()
    for (plate, well), well_fovs in fovs.groupby(key):
        index = pd.MultiIndex.from_arrays([well_fovs["biological_repeat"], well_fovs["image_file"]])
        sums = sums_all.reindex(index, fill_value=0).to_numpy()
        counts = counts_all.reindex(index, fill_value=0).to_numpy()
        total = counts.sum()
        if total < MIN_OBJECTS:
            continue
        mean, variance = sums.sum() / total, variances[(plate, well)]
        for k in ks:
            if k >= len(index):
                continue
            draws = np.array([rng.choice(len(index), k, replace=False) for _ in range(N_DRAWS)])
            n = counts[draws].sum(axis=1)
            ok = n > 0
            error_fovs = sums[draws].sum(axis=1)[ok] / n[ok] - mean
            mse_objects = (variance / n[ok] * (1 - n[ok] / total)).mean()
            rows.append(
                {
                    "plate": plate,
                    "well": well,
                    "k_fovs": k,
                    "objects": n[ok].mean(),
                    "rel_rmse_fovs": np.sqrt((error_fovs**2).mean()) / mean,
                    "rel_rmse_random_objects": np.sqrt(mse_objects) / mean,
                }
            )
    result = pd.DataFrame(rows)
    result["ratio"] = result["rel_rmse_fovs"] / result["rel_rmse_random_objects"]
    return result


def summarize(result: pd.DataFrame) -> pd.DataFrame:
    return result.groupby("k_fovs").agg(
        wells=("well", "size"),
        median_objects=("objects", "median"),
        rel_rmse_fovs=("rel_rmse_fovs", "median"),
        rel_rmse_random_objects=("rel_rmse_random_objects", "median"),
        ratio_median=("ratio", "median"),
        ratio_q25=("ratio", lambda s: s.quantile(0.25)),
        ratio_q75=("ratio", lambda s: s.quantile(0.75)),
        wells_ratio_above_1_5=("ratio", lambda s: (s > 1.5).mean()),
        wells_fovs_above_5pct=("rel_rmse_fovs", lambda s: (s > 0.05).mean()),
        wells_fovs_above_2_5pct=("rel_rmse_fovs", lambda s: (s > 0.025).mean()),
        wells_objects_above_5pct=("rel_rmse_random_objects", lambda s: (s > 0.05).mean()),
        wells_objects_above_2_5pct=("rel_rmse_random_objects", lambda s: (s > 0.025).mean()),
    )


def main() -> None:
    rng = np.random.default_rng(SEED)
    RESULTS.mkdir(exist_ok=True)
    for name, dataset, ks in (
        ("idr0139_fascin", "idr0139_plate", [1, 2, 3]),
        ("idr0028_screenB_yaptaz", "idr0028_screenB_plates", [1, 2, 4, 8, 14]),
    ):
        data = load(dataset)
        table = summarize(compare(data.objects, data.fovs, data.readout, ks, rng))
        print(f"{data.description}, {data.readout}")
        print(table.to_string(float_format="{:.3g}".format))
        table.assign(source_commit=SOURCE_COMMIT, n_draws=N_DRAWS, seed=SEED).to_csv(
            RESULTS / f"fov_vs_objects_{name}.csv", float_format="%.6g"
        )


if __name__ == "__main__":
    main()

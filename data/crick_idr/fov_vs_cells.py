"""Retrospective check: does a well mean from whole FOVs behave like one from random cells?

For every well of the full source plates, draw ``k`` whole FOVs at random (what
a microscope acquires) and compare the error of their mean against the full
well with the error expected from the same number of cells drawn at random
from the whole well (the assumption behind cell-number rules: spatially
uniform samples, or random cell sampling). The random-cell error uses the exact
simple-random-sampling variance with finite-population correction, so only
FOV draws are simulated.

Metrics are each paper's own: Fascin_Ratio for idr0139 and YAPTAZ_Ratio for
idr0028, both nuclear mean / (nuclear mean + cytoplasmic mean) intensity.

This uses the complete plates, not the committed subset, so it downloads about
2 GB. Usage::

    python data/crick_idr/fov_vs_cells.py RAW_DIR

Exploratory and retrospective: the full-well mean is the reference, and only
the imaged fields of each well are known.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from convert import IDR0028_PLATES, IDR0139_PLATE, fetch

N_DRAWS = 400
SEED = 0


def _cells(raw_dir: Path, prefix: str, image_col: str, well_of, protein: str) -> tuple:
    images = pd.read_csv(raw_dir / f"{prefix}_Image.csv", usecols=["ImageNumber", image_col])
    images["well"] = well_of(images[image_col])
    nuclei = pd.read_csv(
        raw_dir / f"{prefix}_Nuclei.csv",
        usecols=["ImageNumber", "ObjectNumber", f"Intensity_MeanIntensity_{protein}"],
    ).rename(columns={f"Intensity_MeanIntensity_{protein}": "nuc"})
    cytoplasm = pd.read_csv(
        raw_dir / f"{prefix}_Cytoplasm.csv",
        usecols=["ImageNumber", "Parent_Nuclei", f"Intensity_MeanIntensity_{protein}"],
    ).rename(columns={"Parent_Nuclei": "ObjectNumber", f"Intensity_MeanIntensity_{protein}": "cyto"})
    cells = nuclei.merge(cytoplasm, on=["ImageNumber", "ObjectNumber"], validate="one_to_one")
    cells["metric"] = cells["nuc"] / (cells["nuc"] + cells["cyto"])
    cells = cells.merge(images[["ImageNumber", "well"]], on="ImageNumber", validate="many_to_one")
    return cells, images


def compare(cells: pd.DataFrame, images: pd.DataFrame, ks, rng) -> pd.DataFrame:
    """Relative RMSE of the well mean from k random FOVs versus the same number of random cells."""
    rows = []
    for well, fovs in images.groupby("well"):
        ids = fovs["ImageNumber"].to_numpy()
        in_well = cells[cells["well"].eq(well)]
        sums = in_well.groupby("ImageNumber")["metric"].sum().reindex(ids, fill_value=0).to_numpy()
        counts = in_well.groupby("ImageNumber").size().reindex(ids, fill_value=0).to_numpy()
        total = counts.sum()
        if total < 50:
            continue
        mean, variance = sums.sum() / total, in_well["metric"].var()
        for k in ks:
            if k >= len(ids):
                continue
            draws = np.array([rng.choice(len(ids), k, replace=False) for _ in range(N_DRAWS)])
            n = counts[draws].sum(axis=1)
            ok = n > 0
            error_fovs = sums[draws].sum(axis=1)[ok] / n[ok] - mean
            mse_cells = (variance / n[ok] * (1 - n[ok] / total)).mean()
            rows.append(
                {
                    "well": well,
                    "k_fovs": k,
                    "cells": n[ok].mean(),
                    "rel_rmse_fovs": np.sqrt((error_fovs**2).mean()) / mean,
                    "rel_rmse_random_cells": np.sqrt(mse_cells) / mean,
                }
            )
    result = pd.DataFrame(rows)
    result["ratio"] = result["rel_rmse_fovs"] / result["rel_rmse_random_cells"]
    return result


def summarize(result: pd.DataFrame) -> pd.DataFrame:
    return result.groupby("k_fovs").agg(
        wells=("well", "size"),
        median_cells=("cells", "median"),
        rel_rmse_fovs=("rel_rmse_fovs", "median"),
        rel_rmse_random_cells=("rel_rmse_random_cells", "median"),
        ratio_median=("ratio", "median"),
        ratio_q25=("ratio", lambda s: s.quantile(0.25)),
        ratio_q75=("ratio", lambda s: s.quantile(0.75)),
        wells_ratio_above_1_5=("ratio", lambda s: (s > 1.5).mean()),
    )


def main(raw_dir: Path) -> None:
    fetch(raw_dir)
    rng = np.random.default_rng(SEED)
    cells, images = _cells(raw_dir, "idr0139", "FileName_DNA",
                           lambda s: s.str.split("_").str[1], "Fascin")
    print(f"idr0139 plate {IDR0139_PLATE}, Fascin_Ratio")
    print(summarize(compare(cells, images, [1, 2, 3], rng)).to_string(float_format="{:.3g}".format))

    results = []
    for plate in IDR0028_PLATES:
        cells, images = _cells(
            raw_dir, f"idr0028_{plate}", "Metadata_FileLocation",
            lambda s: s.str.rsplit("/", n=1).str[-1].str.split("-").str[0], "YAPTAZ",
        )
        results.append(compare(cells, images, [1, 2, 4, 8, 14], rng).assign(plate=plate))
    print("idr0028 screen B plates 1A, 2A, 2B, YAPTAZ_Ratio")
    print(summarize(pd.concat(results)).to_string(float_format="{:.3g}".format))


if __name__ == "__main__":
    main(Path(sys.argv[1]))

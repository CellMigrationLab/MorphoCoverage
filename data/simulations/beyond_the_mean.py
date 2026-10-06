"""Beyond the mean: does the comparison of the distributions hold its false positives when the
condition and its control were imaged in different fields? (owner, 2026-10-06, audit 8)

``sampling_unit.distribution`` compares, per biological repeat against the control of the
same repeat, the spread (log SD ratio) and the 10%, 50% and 90% quantile shifts (in control
SDs), each tested across the repeats (a one-sample t-test on the per-repeat values,
Benjamini-Hochberg over the features and statistics). On the cells pooled over a side's
fields, the field design reads as an effect: a control imaged in one field of 60 cells has
no field-to-field variation in its spread, three fields of 20 do. The package reads both
sides on field-balanced subsamples (the same number of fields per side and repeat, the
smaller side's, and of cells per field, the smallest field's, averaged over draws). This
simulates the null (no condition effect, 5 independent normal features) over designs:

- 3 and 6 biological repeats, each shifting both sides alike (SD ``REPEAT_SD``);
- each field shifted by its own normal offset (SD 0.5 or 1.0 cell SDs);
- the designs of ``DESIGNS``: equal sides, a control of one field (60 cells, or 6), a
  condition that lost cells, and uneven random field and cell counts.

Each setting is ``N_REPLICATES`` datasets; for each statistic the share of tests with
p <= 0.05, and the share of datasets with any q <= 0.05 over the 20 tests (``any_q``), for
the package's balanced test (``balanced``) and, for comparison, the same test on the pooled
cells (``pooled``, the rule before audit 8). Results: ``results/beyond_the_mean.csv``.
Usage, from the repository root::

    PYTHONPATH=src python data/simulations/beyond_the_mean.py [workers]
"""

from __future__ import annotations

import sys
import zlib
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from morphocoverage.sampling_unit.distribution import QUANTILES, STATISTICS, distribution_contrast
from morphocoverage.statistics import benjamini_hochberg, paired_t

N_REPLICATES = 400
FEATURES = 5
REPEAT_SD = 0.3
REPEATS = (3, 6)
FIELD_SDS = (0.5, 1.0)
# (control fields, control cells per field, condition fields, condition cells per field);
# None: 2 to 10 fields a side, 5 to 40 cells each, drawn per side and repeat.
DESIGNS = {
    "equal": (3, 20, 3, 20),
    "control one field of 60": (1, 60, 3, 20),
    "control one field of 6": (1, 6, 3, 20),
    "condition lost cells": (3, 20, 3, 5),
    "uneven": None,
}


def _dataset(
    repeats: int, field_sd: float, design: str, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The observations and FOVs of one null dataset."""
    rng = np.random.default_rng(seed)
    counts = DESIGNS[design]
    rows, fields = [], []
    for r in range(repeats):
        shift = rng.normal(0, REPEAT_SD, FEATURES)
        for side, index in (("ctrl", 0), ("cond", 2)):
            n_fields = counts[index] if counts else int(rng.integers(2, 11))
            for f in range(n_fields):
                cells = counts[index + 1] if counts else int(rng.integers(5, 41))
                name = f"{side}_{r}_{f}"
                fields.append({"condition": side, "biological_repeat": f"R{r}", "image_file": name})
                values = 10 + shift + rng.normal(0, field_sd, FEATURES)
                values = values + rng.normal(size=(cells, FEATURES))
                rows += [
                    {"condition": side, "biological_repeat": f"R{r}", "image_file": name,
                     "label_id": k, **{f"f{j}": v[j] for j in range(FEATURES)}}
                    for k, v in enumerate(values)
                ]  # fmt: skip
    return pd.DataFrame(rows), pd.DataFrame(fields)


def _pooled(objects: pd.DataFrame) -> pd.DataFrame:
    """The same statistics on the pooled cells of each side and repeat (the rule before
    audit 8), tested across the repeats and corrected over the features and statistics."""
    rows = []
    for j in range(FEATURES):
        per_repeat = []
        for _, block in objects.groupby("biological_repeat"):
            treated = block.loc[block["condition"] == "cond", f"f{j}"].to_numpy()
            control = block.loc[block["condition"] == "ctrl", f"f{j}"].to_numpy()
            sd = control.std(ddof=1)
            shifts = (np.quantile(treated, QUANTILES) - np.quantile(control, QUANTILES)) / sd
            per_repeat.append([np.log(treated.std(ddof=1) / sd), *shifts])
        values = np.array(per_repeat)
        rows += [{"statistic": s, "p": paired_t(values[:, i])} for i, s in enumerate(STATISTICS)]
    tests = pd.DataFrame(rows)
    tests["q"] = benjamini_hochberg(tests["p"].to_numpy())
    return tests


def _setting(arguments: tuple) -> dict:
    repeats, field_sd, design = arguments
    hits = {f"{m}_{s}": 0 for m in ("balanced", "pooled") for s in STATISTICS}
    tests = {key: 0 for key in hits}
    family = {"balanced": 0, "pooled": 0}
    for replicate in range(N_REPLICATES):
        seed = zlib.crc32(repr((repeats, field_sd, design, replicate)).encode())
        objects, fovs = _dataset(repeats, field_sd, design, seed)
        found = {
            "balanced": distribution_contrast(
                objects,
                fovs,
                features=[f"f{j}" for j in range(FEATURES)],
                control={"condition": "ctrl"},
                fov_col="image_file",
                observation_unit="cell",
                random_state=replicate,
            ).tests,
            "pooled": _pooled(objects),
        }
        for method, result in found.items():
            family[method] += bool((result["q"] <= 0.05).any())
            for statistic in STATISTICS:
                p = result.loc[result["statistic"] == statistic, "p"].dropna()
                hits[f"{method}_{statistic}"] += int((p <= 0.05).sum())
                tests[f"{method}_{statistic}"] += len(p)
    row = {"repeats": repeats, "field_sd": field_sd, "design": design}
    for key, hit in hits.items():
        row[key] = hit / tests[key] if tests[key] else np.nan
    for method, hit in family.items():
        rate = hit / N_REPLICATES
        row[f"{method}_any_q"] = rate
        row[f"{method}_any_q_se"] = float(np.sqrt(rate * (1 - rate) / N_REPLICATES))
    return row


def main(workers: int = 1) -> pd.DataFrame:
    """Every setting (see the module)."""
    settings = list(product(REPEATS, FIELD_SDS, DESIGNS))
    with Pool(workers) as pool:
        rows = pool.map(_setting, settings)
    table = pd.DataFrame(rows)
    out = Path(__file__).parent / "results" / "beyond_the_mean.csv"
    table.round(4).to_csv(out, index=False)
    return table


if __name__ == "__main__":
    print(main(int(sys.argv[1]) if len(sys.argv) > 1 else 1).to_string())

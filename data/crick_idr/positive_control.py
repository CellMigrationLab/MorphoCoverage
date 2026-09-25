"""Positive control: does "would one more FOV show something new?" find a phenotype that is there?

Semi-synthetic, on the committed idr0028 subset. A negative-control (MOCK)
well is the host population. Candidate "next FOVs" are judged against ``k``
host FOVs drawn at random, with the definition of new used by the
sufficiency report (``SupportNoveltyEstimator.novel_fovs_against``:
standardized features, fitted on the ``k`` FOVs only):

- **host**: the host's remaining FOVs. Nothing new is there by construction,
  so the fraction judged new is the false-alarm rate.
- **whole FOV**: every FOV of a knockdown well on the same plate (siYAP,
  siLATS1 or siARAP2). The whole field is a different condition, a strong
  positive control.
- **rare cells**: a remaining host FOV with ``s`` cells of one knockdown FOV
  added. A small group of cells unlike the host, the case the question is
  meant to catch.

Rates are pooled over hosts, draws and candidates. Every draw uses a fixed
seed. The truth is the donor label, not a biological claim that the donor
cells differ; the donor wells' mean YAP/TAZ ratio differs from MOCK by about
1.1-1.4 SD (siYAP, siLATS1) or 0.3-0.5 SD (siARAP2). Adding donor cells to a host FOV
mixes cells imaged in different fields, so the rare-cell case is an
approximation of a rare state, not a measured one.

Usage::

    python data/crick_idr/positive_control.py

writes ``data/crick_idr/results/positive_control_idr0028.csv``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from morphocoverage import validate_input_tables
from morphocoverage._parallel import map_units, worker_pool
from morphocoverage.strategies import SupportNoveltyEstimator
from morphocoverage.transforms import Standardize

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SEED = 0
DRAWS = 20
FOVS_SEEN = (5, 10, 20)
EPSILON_SCALES = (0.5, 0.75, 1.0)
CLUSTER_SIZES = (1, 2, 3)
RARE_CELLS = (3, 5, 10)
# Host (MOCK) and donor (knockdown) wells on the same plate: (plate, host, donors).
DESIGN = (
    ("1A", "B6", ("M23", "I23")),
    ("2A", "K16", ("N23", "J23")),
    ("2B", "P13", ("C13", "C17")),
)


def _load():
    objects = pd.read_csv(HERE / "idr0028_screenB_nuclei.csv.gz", dtype={"repeat": str})
    fovs = pd.read_csv(HERE / "idr0028_screenB_fovs.csv.gz", dtype={"repeat": str})
    features = [c for c in objects.columns if c.startswith(("Nuclei_", "Cytoplasm_"))]
    return objects, fovs, features


def _fov_cells(objects, fovs, plate, well, features):
    """Feature matrix of each FOV of one well, in FOV-table order (empty FOVs included)."""
    names = fovs.loc[(fovs["repeat"] == plate) & (fovs["well"] == well), "image_file"]
    by_file = dict(tuple(objects.groupby("image_file")))
    empty = np.empty((0, len(features)))
    return [by_file[n][features].to_numpy(float) if n in by_file else empty for n in names]


def _table(blocks, features, prefix):
    """Validated input with one FOV per block of cells, named ``prefix0``, ``prefix1``, ..."""
    names = [f"{prefix}{j}" for j in range(len(blocks))]
    kept = [(name, block) for name, block in zip(names, blocks, strict=True) if len(block)]
    # One frame for all blocks: building and concatenating a frame per block
    # dominated the runtime of the controls.
    objects = pd.DataFrame(np.vstack([block for _, block in kept]), columns=features)
    objects["image_file"] = np.repeat([name for name, _ in kept], [len(b) for _, b in kept])
    return validate_input_tables(
        objects, fovs=pd.DataFrame({"image_file": names}), feature_cols=features
    )


def _spiked(host_blocks, donor_blocks, size, rng):
    """Host FOVs, each with ``size`` cells of a random donor FOV added."""
    if not any(len(donor) >= size for donor in donor_blocks):
        # Redrawing below would never end; the draws themselves are unchanged.
        raise ValueError(f"no donor FOV has {size} cells to add")
    spiked = []
    for block in host_blocks:
        donor = donor_blocks[rng.integers(len(donor_blocks))]
        while len(donor) < size:
            donor = donor_blocks[rng.integers(len(donor_blocks))]
        cells = donor[rng.choice(len(donor), size=size, replace=False)]
        spiked.append(np.vstack([block, cells]))
    return spiked


def _judge(seen_blocks, candidates, k, features, scale):
    """``(setting, judged new, candidates)`` for one draw; no randomness."""
    seen = _table(seen_blocks, features, "seen")
    tables = {key: _table(blocks, features, "new") for key, blocks in candidates.items()}
    counts = []
    for fraction in EPSILON_SCALES:
        for m in CLUSTER_SIZES:
            novelty = SupportNoveltyEstimator(fraction * scale, 1, Standardize(), 2, m)
            for (kind, donor, size), table in tables.items():
                flags = novelty.novel_fovs_against(seen, table)
                counts.append(((fraction, m, k, kind, donor, size), int(np.sum(flags)), len(flags)))
    return counts


def run(n_jobs: int = -1) -> pd.DataFrame:
    """Rates at which each kind of candidate FOV is judged new, per setting.

    Every random choice is drawn here, in the same order as a serial run; only
    the novelty checks, which draw nothing, go to ``n_jobs`` worker processes
    (every CPU by default), so the table does not depend on ``n_jobs``.
    """
    objects, fovs, features = _load()
    rng = np.random.default_rng(SEED)
    scale = np.sqrt(len(features))
    tallies: dict[tuple, list[int]] = {}
    with worker_pool(n_jobs) as pool:
        for plate, host_well, donor_wells in DESIGN:
            host = _fov_cells(objects, fovs, plate, host_well, features)
            donors = {w: _fov_cells(objects, fovs, plate, w, features) for w in donor_wells}
            for k in FOVS_SEEN:
                jobs = []
                for _ in range(DRAWS):
                    order = rng.permutation(len(host))
                    rest = [host[i] for i in order[k:]]
                    candidates = {("host", "-", 0): rest}
                    for well, blocks in donors.items():
                        candidates[("whole_fov", well, 0)] = blocks
                        for size in RARE_CELLS:
                            spiked = _spiked(rest, blocks, size, rng)
                            candidates[("rare_cells", well, size)] = spiked
                    jobs.append(([host[i] for i in order[:k]], candidates, k, features, scale))
                for draw in map_units(_judge, jobs, n_jobs, pool):
                    for key, new, total in draw:
                        count = tallies.setdefault(key, [0, 0])
                        count[0] += new
                        count[1] += total
    rows = [
        {
            "epsilon_scale": fraction,
            "cluster_size": m,
            "fovs_seen": k,
            "candidate": kind,
            "donor_well": donor,
            "added_cells": size,
            "judged_new": new,
            "candidates": total,
            "rate": new / total,
        }
        for (fraction, m, k, kind, donor, size), (new, total) in sorted(tallies.items())
    ]
    return pd.DataFrame(rows)


def main() -> None:
    """Run the control and write the results table."""
    table = run()
    RESULTS.mkdir(exist_ok=True)
    table.to_csv(RESULTS / "positive_control_idr0028.csv", index=False, float_format="%.4f")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()

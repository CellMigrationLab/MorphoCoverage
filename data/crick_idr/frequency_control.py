"""Frequency control: are the next-FOV probability and its bound right when a rare state is there?

Semi-synthetic, on the committed idr0028 subset, extending
``positive_control.py``. A MOCK well is the host acquisition (28 FOVs). A rare
state is planted in a known fraction ``p`` of its FOVs by adding 5 siYAP cells
from the same plate to each, a state the positive control finds in about 95%
of FOVs. The planted FOVs are either **spread** (each FOV independently with
probability ``p``) or **clustered** (the same expected number, taken as the
FOVs nearest a random FOV by stage position).

Each planted acquisition is replayed in a random order. After ``k`` FOVs the
estimator (the sufficiency report's definition of new) predicts the chance
that the next FOV shows something new, with its 95% upper bound. Two truths:

- **biological**: the chance that the next FOV shows the planted state while
  none of the first ``k`` FOVs does, (planted FOVs left) / (FOVs left), and 0
  once the state has been seen;
- **operational**: the fraction of the remaining FOVs that the definition
  judges new against the first ``k``, as in ``morphocoverage.calibration``;
  it adds host false alarms and missed detections to the biological truth.

A stopping replay also stops at the first ``k >= 2`` whose bound is at most
``TARGET`` and records whether the planted state had been seen by then.

Usage::

    python data/crick_idr/frequency_control.py

writes ``results/frequency_control_idr0028.csv`` (per prefix length) and
``results/frequency_control_stops_idr0028.csv`` (per stop).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from positive_control import HERE, RESULTS, _fov_cells, _load, _table

from morphocoverage import ObjectiveConfig, SamplingObjective
from morphocoverage._parallel import map_units, worker_pool
from morphocoverage.strategies import SupportNoveltyEstimator
from morphocoverage.transforms import Standardize

SEED = 1
REPLICATES = 50
FREQUENCIES = (0.05, 0.1, 0.2)
LAYOUTS = ("spread", "clustered")
FOVS_SEEN = (5, 10, 20)
PLANTED_CELLS = 5
TARGET = 0.25
# Host (MOCK) and siYAP donor wells on the same plate.
DESIGN = (("1A", "B6", "M23"), ("2A", "K16", "N23"))
POPULATION = ObjectiveConfig(SamplingObjective.POPULATION)
NOVELTY = SupportNoveltyEstimator(1.0 * np.sqrt(30), 1, Standardize(), 2, 3)
PREDICTED, UPPER = "lofo_novel_fov_fraction", "lofo_novel_fov_upper"


def _planted_fovs(p, layout, positions, rng):
    """Boolean mask of the FOVs that receive the rare state."""
    n = len(positions)
    if layout == "spread":
        return rng.random(n) < p
    count = rng.binomial(n, p)
    mask = np.zeros(n, dtype=bool)
    if count:
        centre = positions[rng.integers(n)]
        mask[np.argsort(np.linalg.norm(positions - centre, axis=1), kind="stable")[:count]] = True
    return mask


def _acquisition(host, donor, planted, rng):
    """Host FOVs with ``PLANTED_CELLS`` donor cells added to the planted ones."""
    if planted.any() and not any(len(source) >= PLANTED_CELLS for source in donor):
        # Redrawing below would never end; the draws themselves are unchanged.
        raise ValueError(f"no donor FOV has {PLANTED_CELLS} cells to plant")
    blocks = []
    for block, plant in zip(host, planted, strict=True):
        if plant:
            source = donor[rng.integers(len(donor))]
            while len(source) < PLANTED_CELLS:
                source = donor[rng.integers(len(donor))]
            cells = source[rng.choice(len(source), size=PLANTED_CELLS, replace=False)]
            block = np.vstack([block, cells])
        blocks.append(block)
    return blocks


def _prefix(blocks, planted, order, k, features):
    """Prediction, bound and both truths after the first ``k`` FOVs of ``order``."""
    seen, rest = order[:k], order[k:]
    visible = _table([blocks[i] for i in seen], features, "seen")
    evidence = NOVELTY.estimate(visible, objective=POPULATION)
    if evidence.quantities[PREDICTED].availability.value != "available":
        return None
    unseen_state = not planted[seen].any()
    biological = float(planted[rest].mean()) if unseen_state else 0.0
    judged = NOVELTY.novel_fovs_against(visible, _table([blocks[i] for i in rest], features, "new"))
    return {
        "predicted": evidence.value(PREDICTED),
        "bound": evidence.value(UPPER),
        "biological": biological,
        "operational": float(judged.mean()),
        "state_seen": not unseen_state,
    }


def _replay(blocks, planted, order, features):
    """``(k, row)`` for every prefix length of one planted acquisition; no randomness."""
    rows = []
    for k in range(2, len(blocks)):
        row = _prefix(blocks, planted, order, k, features)
        if row is not None:
            rows.append((k, row))
    return rows


def run(n_jobs: int = -1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-prefix predictions against both truths, and one row per stopping replay.

    Every random choice is drawn here, in the same order as a serial run; only
    the replays, which draw nothing, go to ``n_jobs`` worker processes (every
    CPU by default, each with one BLAS thread), so the tables do not depend on ``n_jobs``.
    """
    objects, fovs, features = _load()
    rng = np.random.default_rng(SEED)
    prefixes, stops = [], []
    with worker_pool(n_jobs) as pool:
        for plate, host_well, donor_well in DESIGN:
            host = _fov_cells(objects, fovs, plate, host_well, features)
            donor = _fov_cells(objects, fovs, plate, donor_well, features)
            table = fovs[(fovs["repeat"] == plate) & (fovs["well"] == host_well)]
            positions = table[["stage_x_um", "stage_y_um"]].to_numpy(float)
            for p in FREQUENCIES:
                for layout in LAYOUTS:
                    # One batch of replicates at a time keeps memory bounded.
                    keys, jobs = [], []
                    for replicate in range(REPLICATES):
                        planted = _planted_fovs(p, layout, positions, rng)
                        blocks = _acquisition(host, donor, planted, rng)
                        order = rng.permutation(len(blocks))
                        key = {"plate": plate, "frequency": p, "layout": layout}
                        key.update({"replicate": replicate, "planted": int(planted.sum())})
                        keys.append(key)
                        jobs.append((blocks, planted, order, features))
                    for key, rows in zip(keys, map_units(_replay, jobs, n_jobs, pool), strict=True):
                        stop = None
                        for k, row in rows:
                            if k in FOVS_SEEN:
                                prefixes.append({**key, "fovs_seen": k, **row})
                            if stop is None and row["bound"] <= TARGET:
                                stop = {**key, "stop_fov": k, **row}
                        stops.append(stop or {**key, "stop_fov": None, "state_seen": None})
    return pd.DataFrame(prefixes), pd.DataFrame(stops)


def main() -> None:
    """Run the control and write both tables."""
    prefixes, stops = run()
    RESULTS.mkdir(exist_ok=True)
    prefixes.to_csv(RESULTS / "frequency_control_idr0028.csv", index=False, float_format="%.4f")
    stops.to_csv(RESULTS / "frequency_control_stops_idr0028.csv", index=False, float_format="%.4f")
    summary = prefixes.assign(
        held_bio=prefixes["bound"] >= prefixes["biological"],
        held_op=prefixes["bound"] >= prefixes["operational"],
    ).groupby(["frequency", "layout", "fovs_seen"])[
        ["predicted", "biological", "operational", "bound", "held_bio", "held_op", "state_seen"]
    ]
    print(summary.mean().round(3).to_string())
    print(HERE)


if __name__ == "__main__":
    main()

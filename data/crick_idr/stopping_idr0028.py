"""When would a stopping rule have stopped, and was it enough? Replayed idr0028 acquisitions.

Roadmap Steps 2 and 3 (``docs/proposals/microscopist_report_roadmap.md``). For
each well (28 FOVs), 10 FOVs are held back as a fixed reference per split (5
splits, the same reservoirs as ``reliability_idr0028.py`` for its first five
splits). The other 18 are replayed in the recorded order and 10 random orders;
after every FOV each rule decides, and the prefix is judged against the
held-back FOVs on the YAP/TAZ nuclear ratio of Marcotti et al.

Rules:

- the JCB averaging-CV rule on cells, CV of the mean at most 5% (published
  parameterization, ``resampling_unit="object"``);
- the same rule with whole FOVs as the resampling unit;
- fixed rules of 4 and 8 FOVs.

What you wanted to preserve: the well mean within 5%, and within 2.5%.
idr0028 is a development dataset.

Usage::

    python data/crick_idr/stopping_idr0028.py

writes ``results/stopping_calibration_idr0028.csv`` and
``results/adequacy_idr0028.csv``.
"""

from __future__ import annotations

import time

from datasets import RESULTS, load
from datasets import YAPTAZ_RATIO as RATIO

from morphocoverage import (
    FixedFOVStrategy,
    ObjectiveConfig,
    SamplingObjective,
    ScalarPrecisionStrategy,
)
from morphocoverage.outcomes import RelativeMeanError
from morphocoverage.retrospective import (
    ALL_CRITERIA,
    OutcomeCriterion,
    adequacy_points,
    replay_acquisitions,
    stopping_calibration,
)

CRITERIA = (
    OutcomeCriterion(RelativeMeanError(RATIO), 0.05, "lower_is_better", "the well mean within 5%"),
    OutcomeCriterion(
        RelativeMeanError(RATIO), 0.025, "lower_is_better", "the well mean within 2.5%"
    ),
)
STRATEGIES = (
    ScalarPrecisionStrategy(RATIO, 0.05, "object", "point"),
    ScalarPrecisionStrategy(RATIO, 0.05, "fov", "point"),
    FixedFOVStrategy(4),
    FixedFOVStrategy(8),
)


def main() -> None:
    """Replay, then write the calibration and adequacy tables."""
    objects, fovs = load("idr0028_screenB").tables
    started = time.perf_counter()
    replay = replay_acquisitions(
        objects,
        fovs,
        feature_cols=[RATIO],
        strategies=list(STRATEGIES),
        criteria=list(CRITERIA),
        objective=ObjectiveConfig(SamplingObjective.POPULATION),
        reservoir_fraction=0.35,
        acquisition_order_col="acquisition_order",
        n_splits=5,
        random_state=0,
        min_fovs=2,
        incremental=True,
        n_jobs=-1,
        group_cols=["condition", "repeat", "technical_repeat"],
    )
    seconds = time.perf_counter() - started
    calibration = stopping_calibration(replay)
    adequacy = adequacy_points(replay)
    RESULTS.mkdir(exist_ok=True)
    calibration.table.to_csv(
        RESULTS / "stopping_calibration_idr0028.csv", index=False, float_format="%.4f"
    )
    realized = adequacy.realized[adequacy.realized["criterion"] == ALL_CRITERIA]
    points = adequacy.apparent.merge(
        realized.drop(columns=["criterion"]),
        on=["condition", "repeat", "technical_repeat", "split", "order_id", "order", "stream_fovs"],
    )
    points.to_csv(RESULTS / "adequacy_idr0028.csv", index=False)
    print(f"replay: {seconds:.0f} s")
    print(calibration.summary())
    print()
    print(adequacy.summary())


if __name__ == "__main__":
    main()

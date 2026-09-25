"""Held-out check on idr0139: do the idr0028 conclusions about the well mean transfer?

On the development data (12 wells x 28 fields) the well mean of the YAP/TAZ
ratio was within 5% of held-back fields from 2 fields in every well, and the
JCB cell-number rule stopped safely at 2 fields (``docs/preliminary_findings.md``,
entries 4 and 5). This script asks the same questions of a dataset those
conclusions were never tuned on: the 10 wells x 4 fields of idr0139 and the
paper's fascin ratio, with the registry refusing any claim of tuning on it.

The dataset is shallow: each hold-out keeps 2 fields as the reference and
leaves 2 to draw from, so only 1 and 2 fields can be judged, against a
2-field reference whose own error is the ceiling. That is enough to refute
"one or two fields are enough" but not to say how many are.

Outputs, in ``results/``: ``heldout_reliability_idr0139.csv``,
``heldout_ceiling_idr0139.csv``, ``heldout_protocol_idr0139.csv`` and
``heldout_calibration_idr0139.csv``.

Usage::

    python data/crick_idr/heldout_idr0139.py
"""

from __future__ import annotations

from datasets import FASCIN_RATIO, GROUPS, RESULTS, load, registry

from morphocoverage import (
    FixedFOVStrategy,
    ObjectiveConfig,
    SamplingObjective,
    ScalarPrecisionStrategy,
)
from morphocoverage.benchmark import AcquisitionOrder
from morphocoverage.outcomes import RelativeMeanError
from morphocoverage.retrospective import (
    OutcomeCriterion,
    SamplingHierarchy,
    protocol_reliability,
    replay_acquisitions,
    sampling_reliability,
    stopping_calibration,
)

CRITERIA = [
    OutcomeCriterion(
        RelativeMeanError(FASCIN_RATIO), 0.05, "lower_is_better", "the well mean within 5%"
    ),
    OutcomeCriterion(
        RelativeMeanError(FASCIN_RATIO), 0.025, "lower_is_better", "the well mean within 2.5%"
    ),
]


def main() -> None:
    """Run the held-out reliability, protocol and calibration; print and write them."""
    datasets = registry()
    assert datasets["idr0139"].role.value == "held_out"
    # A run that claimed to be tuned on the held-out data would be refused here.
    datasets.check_tuning(["idr0028_screenB"])
    objects, fovs = load("idr0139").tables
    RESULTS.mkdir(exist_ok=True)

    reliability = sampling_reliability(
        objects,
        fovs,
        feature_cols=[FASCIN_RATIO],
        criteria=CRITERIA,
        fov_counts=[1, 2],
        reservoir_fraction=0.5,
        n_splits=6,
        draws=10,
        random_state=0,
        group_cols=list(GROUPS),
        observation_unit="nucleus",
    )
    reliability.curve.to_csv(
        RESULTS / "heldout_reliability_idr0139.csv", index=False, float_format="%.4f"
    )
    reliability.ceiling.to_csv(
        RESULTS / "heldout_ceiling_idr0139.csv", index=False, float_format="%.4f"
    )
    print(reliability.summary(), "\n")

    protocol = protocol_reliability(
        reliability, SamplingHierarchy("image_file", "well", "repeat"), threshold=0.9, strata=[]
    )
    protocol.table.to_csv(
        RESULTS / "heldout_protocol_idr0139.csv", index=False, float_format="%.4f"
    )
    print(protocol.summary(), "\n")

    replay = replay_acquisitions(
        objects,
        fovs,
        feature_cols=[FASCIN_RATIO],
        strategies=[
            ScalarPrecisionStrategy(FASCIN_RATIO, 0.05, "object", "upper_95"),
            ScalarPrecisionStrategy(FASCIN_RATIO, 0.025, "object", "upper_95"),
            FixedFOVStrategy(2),
        ],
        criteria=CRITERIA,
        objective=ObjectiveConfig(SamplingObjective.POPULATION),
        reservoir_fraction=0.5,
        orders=[AcquisitionOrder.recorded(), *(AcquisitionOrder.permuted(s) for s in range(5))],
        acquisition_order_col="acquisition_order",
        n_splits=6,
        min_fovs=1,
        group_cols=list(GROUPS),
        observation_unit="nucleus",
    )
    calibration = stopping_calibration(replay)
    calibration.table.to_csv(
        RESULTS / "heldout_calibration_idr0139.csv", index=False, float_format="%.4f"
    )
    print(calibration.summary())


if __name__ == "__main__":
    main()

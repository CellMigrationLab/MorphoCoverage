"""Roadmap Steps 4-8 and 10 on the committed idr0028 subset: protocol to the full report.

Each part answers one microscopist's question on the 12 wells (28 FOVs each)
and prints its plain-language summary; tables go to ``results/``. idr0028 is a
development dataset: nothing here is a protocol.

- **Step 4, protocol** (``protocol_idr0028.csv``): from the Step 1 curves
  (``results/reliability_idr0028.csv``, run ``reliability_idr0028.py`` first),
  how many wells are adequately represented at each number of fields, a well
  counting when at least 90% of random acquisitions preserved what was wanted.
  Wells of all conditions are pooled explicitly; the independent unit is the
  plate (``repeat``).
- **Step 5, inspection** (``inspection_idr0028.csv``): fields to reopen per
  well, for the mean and the 95th percentile of the YAP/TAZ ratio, new states
  (support novelty on all 30 standardized features, ``epsilon = sqrt(30)``,
  clusters of 3), observation counts and differences from stage neighbours.
- **Step 6, controls** (``control_idr0028.csv``): siYAP wells as the declared
  positive control, MOCK wells as the reference, paired by plate, after 10
  reference fields; also 5 siYAP cells planted into MOCK fields.
- **Step 7, effort** (``effort_idr0028.csv``): variance components of the mean
  YAP/TAZ ratio per condition (observation, field, well) and projected
  standard errors at the same total number of fields. Wells and plates are
  confounded here (one well per plate in MOCK), so the plate is not a level.
- **Step 8, order** (``order_efficiency_idr0028.csv``): recorded (serpentine
  raster), random and dispersed orders of the same positions, with the stage
  travel needed to preserve the well mean within 2.5% reliably, and the
  distance below which neighbouring fields are alike.
- **Step 10, the report** (``microscopist_report_idr0028.txt``,
  ``sampling_profile_idr0028.csv``): all of the above, question by question.

Usage::

    python data/crick_idr/report_steps_idr0028.py
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from idr0028_data import RATIO, RESULTS, load

from morphocoverage import FixedFOVStrategy, ObjectiveConfig, SamplingObjective
from morphocoverage.benchmark import AcquisitionOrder
from morphocoverage.outcomes import RelativeMeanError
from morphocoverage.retrospective import (
    OutcomeCriterion,
    ReliabilityResult,
    RepresentationControl,
    SamplingHierarchy,
    Summary,
    adequacy_points,
    effort_projection,
    inspect_fovs,
    microscopist_report,
    order_efficiency,
    protocol_reliability,
    redundancy_distance,
    replay_acquisitions,
    representation_control,
    stopping_calibration,
)
from morphocoverage.strategies import SupportNoveltyEstimator
from morphocoverage.transforms import Standardize

GROUPS = ["condition", "repeat", "well"]
STAGE = ("stage_x_um", "stage_y_um")
LABELS = {
    "relative_mean_error:YAPTAZ_Ratio <= 0.05": "the well mean of YAP/TAZ ratio within 5%",
    "relative_mean_error:YAPTAZ_Ratio <= 0.025": "the well mean of YAP/TAZ ratio within 2.5%",
    "covariance_error:YAPTAZ_Ratio <= 0.1": "the variance of YAP/TAZ ratio within 10%",
}


def _timed(label, run):
    started = time.perf_counter()
    result = run()
    print(f"[{label}: {time.perf_counter() - started:.0f} s]")
    return result


def reliability_from_tables() -> ReliabilityResult:
    """The Step 1 curves and their ceilings, as ``reliability_idr0028.py`` wrote them.

    The ceiling table travels with the curve: without it the summary cannot say
    that a flat curve (the variance within 10%) is capped by the held-back
    fields rather than by the fields acquired.
    """
    curve = pd.read_csv(RESULTS / "reliability_idr0028.csv", dtype={"repeat": str})
    ceiling = pd.read_csv(RESULTS / "reliability_ceiling_idr0028.csv", dtype={"repeat": str})
    settings = {"labels": LABELS, "n_splits": int(ceiling["splits"].max())}
    return ReliabilityResult(curve, pd.DataFrame(), tuple(GROUPS), settings, ceiling)


def protocol():
    per_sample = reliability_from_tables()
    return protocol_reliability(
        per_sample,
        SamplingHierarchy("image_file", "well", "repeat"),
        threshold=0.9,
        strata=[],
    )


def inspection(objects, fovs, features):
    novelty = SupportNoveltyEstimator(np.sqrt(len(features)), 1, Standardize(), 5, 3)
    # The ratio's summaries, counts and stage neighbours; new states in all 30 features.
    ratio_view = inspect_fovs(
        objects,
        fovs,
        feature_cols=[RATIO],
        summaries=[Summary(RATIO, "mean"), Summary(RATIO, 0.95)],
        stage_cols=STAGE,
        group_cols=GROUPS,
    )
    novelty_view = inspect_fovs(
        objects, fovs, feature_cols=features, novelty=novelty, count_ratio=None, group_cols=GROUPS
    )
    return ratio_view, novelty_view


def controls(objects, fovs, features):
    novelty = SupportNoveltyEstimator(np.sqrt(len(features)), 1, Standardize(), 2, 3)
    control = RepresentationControl(
        positive={"condition": ["YAP"]}, reference={"condition": ["MOCK"]}, planted_cells=5
    )
    return representation_control(
        objects,
        fovs,
        feature_cols=features,
        novelty=novelty,
        control=control,
        pair_cols=["repeat"],
        fovs_seen=10,
        draws=20,
        n_jobs=-1,
    )


def effort(objects, fovs):
    # Image names and well names repeat across plates, so the plate goes into the
    # identity of a field and of a well (it is not a variance level here: in MOCK
    # every plate holds one well, so plate and well cannot be separated).
    ids = {
        "position_id": lambda t: t["repeat"] + "/" + t["image_file"],
        "well_id": lambda t: t["repeat"] + "/" + t["well"],
    }
    objects, fovs = objects.assign(**ids), fovs.assign(**ids)
    return effort_projection(
        objects,
        fovs,
        feature=RATIO,
        hierarchy=SamplingHierarchy("position_id", "well_id"),
        strata=["condition"],
        n_boot=500,
    )


def order(objects, fovs):
    criterion = OutcomeCriterion(
        RelativeMeanError(RATIO), 0.025, "lower_is_better", "the well mean within 2.5%"
    )
    orders = [
        AcquisitionOrder.recorded(),
        *(AcquisitionOrder.permuted(s) for s in range(5)),
        *(AcquisitionOrder.dispersed(s) for s in range(5)),
    ]
    replay = replay_acquisitions(
        objects,
        fovs,
        feature_cols=[RATIO],
        strategies=[FixedFOVStrategy(8)],
        criteria=[criterion],
        objective=ObjectiveConfig(SamplingObjective.POPULATION),
        reservoir_fraction=0.35,
        orders=orders,
        acquisition_order_col="acquisition_order",
        stage_cols=STAGE,
        n_splits=5,
        incremental=True,
        group_cols=GROUPS,
        n_jobs=-1,
    )
    redundancy = redundancy_distance(
        objects,
        fovs,
        feature_cols=[RATIO],
        stage_cols=STAGE,
        bin_edges=[0, 400, 600, 900, 1300, 1800, 2600],
        group_cols=GROUPS,
    )
    return replay, redundancy


def main() -> None:
    """Run every part, print its summary and write its tables."""
    objects, fovs, features = load()
    RESULTS.mkdir(exist_ok=True)

    protocol_result = _timed("protocol", protocol)
    protocol_result.table.to_csv(RESULTS / "protocol_idr0028.csv", index=False, float_format="%.4f")
    print(protocol_result.summary(), "\n")

    ratio_view, novelty_view = _timed("inspection", lambda: inspection(objects, fovs, features))
    listed = pd.concat([ratio_view.fovs, novelty_view.fovs], ignore_index=True)
    keep = [c for c in listed.columns if c != "value"] + ["value"]
    listed[keep].to_csv(RESULTS / "inspection_idr0028.csv", index=False, float_format="%.4f")
    print(ratio_view.summary(), "\n")
    print(novelty_view.summary(), "\n")

    control_result = _timed("controls", lambda: controls(objects, fovs, features))
    control_result.table.to_csv(RESULTS / "control_idr0028.csv", index=False, float_format="%.4f")
    print(control_result.summary(), "\n")

    effort_result = _timed("effort", lambda: effort(objects, fovs))
    effort_result.components.to_csv(
        RESULTS / "effort_idr0028.csv", index=False, float_format="%.6g"
    )
    effort_result.comparisons.to_csv(
        RESULTS / "effort_comparisons_idr0028.csv", index=False, float_format="%.4f"
    )
    print(effort_result.summary(), "\n")

    replay, redundancy = _timed("order", lambda: order(objects, fovs))
    adequacy = adequacy_points(replay)
    efficiency = order_efficiency(replay, adequacy)
    efficiency.table.to_csv(
        RESULTS / "order_efficiency_idr0028.csv", index=False, float_format="%.1f"
    )
    redundancy.to_csv(RESULTS / "redundancy_idr0028.csv", index=False, float_format="%.6g")
    print(efficiency.summary(), "\n")
    print(redundancy.to_string(index=False))

    # Step 10: every answer in one document (Step 1 curves reloaded from their tables).
    report = microscopist_report(
        controls=control_result,
        calibration=stopping_calibration(replay),
        adequacy=adequacy,
        reliability=reliability_from_tables(),
        protocol=protocol_result,
        inspection=novelty_view,
        effort=effort_result,
        efficiency=efficiency,
        redundancy=redundancy,
    )
    (RESULTS / "microscopist_report_idr0028.txt").write_text(report.summary() + "\n")
    report.profile.to_csv(RESULTS / "sampling_profile_idr0028.csv", index=False)


if __name__ == "__main__":
    main()

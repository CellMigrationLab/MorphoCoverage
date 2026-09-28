"""How many fields do you need? Sampling reliability on the committed idr0028 subset.

Question P1 of ``docs/questions.md``. For each
well (28 FOVs), 10 FOVs are held back as a fixed reference per split; random
sets of ``k`` of the other 18 FOVs are judged against it, over 10 splits and 50
draws per ``k``. The criteria are provisional presets
applied to the YAP/TAZ nuclear ratio of Marcotti et al. (nuclear mean /
(nuclear + cytoplasmic mean intensity)):

- the well mean within 5% (the JCB target for independent samples);
- the well mean within 2.5% (the stricter JCB target);
- the variance of the ratio within 10% (a distribution criterion).

idr0028 is registered as a development dataset: it is used to build and check
these analyses, not to set a protocol.

Usage::

    python data/crick_idr/reliability_idr0028.py

writes ``results/reliability_idr0028.csv`` and ``results/reliability_ceiling_idr0028.csv``.
"""

from __future__ import annotations

from datasets import RESULTS, load
from datasets import YAPTAZ_RATIO as RATIO

from morphocoverage.multivariate_outcomes import CovarianceError
from morphocoverage.outcomes import RelativeMeanError
from morphocoverage.retrospective import OutcomeCriterion, sampling_reliability

CRITERIA = (
    OutcomeCriterion(
        RelativeMeanError(RATIO),
        0.05,
        "lower_is_better",
        "the well mean of YAP/TAZ ratio within 5%",
    ),
    OutcomeCriterion(
        RelativeMeanError(RATIO),
        0.025,
        "lower_is_better",
        "the well mean of YAP/TAZ ratio within 2.5%",
    ),
    OutcomeCriterion(
        CovarianceError((RATIO,), False),
        0.10,
        "lower_is_better",
        "the variance of YAP/TAZ ratio within 10%",
    ),
)
FOV_COUNTS = (2, 4, 6, 8, 12, 15, 18)


def main() -> None:
    """Run the curves and write the tables."""
    objects, fovs = load("idr0028_screenB").tables
    result = sampling_reliability(
        objects,
        fovs,
        feature_cols=[RATIO],
        criteria=list(CRITERIA),
        fov_counts=list(FOV_COUNTS),
        reservoir_fraction=0.35,
        n_splits=10,
        draws=50,
        random_state=0,
        n_jobs=-1,
        group_cols=["condition", "repeat", "technical_repeat"],
    )
    RESULTS.mkdir(exist_ok=True)
    result.curve.to_csv(RESULTS / "reliability_idr0028.csv", index=False, float_format="%.4f")
    result.ceiling.to_csv(
        RESULTS / "reliability_ceiling_idr0028.csv", index=False, float_format="%.4f"
    )
    print(result.summary())


if __name__ == "__main__":
    main()

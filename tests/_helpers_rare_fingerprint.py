"""The rare-phenotype screen and its Setup, shared by the report and page tests."""

import numpy as np
import pandas as pd

from morphocoverage import RareTarget
from morphocoverage.multivariate_outcomes import QuantileError
from morphocoverage.retrospective import OutcomeCriterion
from morphocoverage.workflows import Setup, default_novelty

FEATURES = ("area", "dna", "ratio")


def _screen(seed=0, narrow=0.6, mitotic=(0.02, 0.10)):
    """siCTRL, narrow (the same mean, a narrower spread) and arrest (more mitoses), 3
    repeats, 2 wells each, 6 fields of 40 cells; mitoses are bright and small."""
    rng = np.random.default_rng(seed)
    rows, fovs = [], []
    for r, repeat in enumerate(("2026_01_05", "2026_01_12", "2026_01_19")):
        batch = rng.normal(0, 0.1, 3)
        for c, condition in enumerate(("siCTRL", "narrow", "arrest")):
            for w in range(2):
                well = f"{'ABC'[c]}{w + 1}"
                for f in range(6):
                    image = f"{condition}_{repeat}_{well}_{f}.tif"
                    z = rng.normal(0, 1, (40, 3)) + batch
                    if condition == "narrow":
                        z = z * narrow
                    share = mitotic[1] if condition == "arrest" else mitotic[0]
                    marked = rng.random(40) < share
                    z[marked] = rng.normal([-2.5, 4.0, 2.5], 0.3, (marked.sum(), 3))
                    for i in range(40):
                        rows.append(
                            {
                                "condition": condition,
                                "biological_repeat": repeat,
                                "repeat_group": repeat,
                                "technical_repeat": well,
                                "image_file": image,
                                "label_id": i + 1,
                                "area": 200 + 20 * z[i, 0],
                                "dna": 1000 + 100 * z[i, 1],
                                "ratio": 0.4 + 0.05 * z[i, 2],
                                "mitotic": bool(marked[i]),
                            }
                        )
                    fovs.append(
                        {
                            "condition": condition,
                            "biological_repeat": repeat,
                            "repeat_group": repeat,
                            "technical_repeat": well,
                            "image_file": image,
                        }
                    )
    return pd.DataFrame(rows), pd.DataFrame(fovs)


def _setup(**changes):
    settings = {
        "readout": "area",
        "features": FEATURES,
        "control": {"condition": "siCTRL"},
        "fingerprint": (
            OutcomeCriterion(
                QuantileError(FEATURES, True, quantiles=(0.1, 0.5, 0.9)),
                0.1,
                "lower_is_better",
                "the 10-90% quantiles within 0.1 SD",
            ),
        ),
        "rare": RareTarget(
            object_prevalence=0.02,
            detection_probability=0.95,
            min_fov_occurrences=3,
            phenotype="mitotic (pH3)",
        ),
        "rare_label": "mitotic",
        "novelty": default_novelty(FEATURES),
        "observation_unit": "cell",
        "random_state": 0,
    }
    return Setup(**{**settings, **changes})

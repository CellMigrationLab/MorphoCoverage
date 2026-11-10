"""The tables the tests of the HTML report share."""

import numpy as np
import pandas as pd


def _tables(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for repeat in ("r1", "r2", "r3"):
        for condition, shift in (("ctrl", 0.0), ("drug", 2.0)):
            for field in range(5):
                for label in range(6):
                    rows.append(
                        {
                            "condition": condition,
                            "biological_repeat": repeat,
                            "repeat_group": repeat,
                            "image_file": f"{condition}_{repeat}_{field}.tif",
                            "label_id": label + 1,
                            "area": rng.normal(20 + shift, 2),
                            "speed": rng.normal(5, 1),
                            "aspect": rng.normal(2, 0.2),
                        }
                    )
    objects = pd.DataFrame(rows)
    fovs = objects[["condition", "biological_repeat", "image_file"]].drop_duplicates()
    return objects, fovs

"""Builders shared by the sampling-unit tests: conditions, technical repeats, repeat designs."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _tables(shift=1.0, n_repeats=3, cells=30, seed=0, unbalanced=False):
    """One well per condition and repeat: 'area' differs in every repeat, 'noise' does not,
    'flat' holds one value."""
    rng = np.random.default_rng(seed)
    rows = []
    for repeat in range(n_repeats):
        for condition, effect in (("ctrl", 0.0), ("drug", shift)):
            n = cells * (4 if unbalanced and repeat == 0 else 1)
            for field in range(n):
                rows.append(
                    {
                        "condition": condition,
                        "biological_repeat": f"r{repeat}",
                        "position_id": f"{condition}-{repeat}-{field}",
                        "area": rng.normal(10 + effect, 1),
                        "noise": rng.normal(5, 1),
                        "aspect": rng.normal(2, 0.3),
                        "flat": 1.0,
                    }
                )
    objects = pd.DataFrame(rows)
    fovs = objects[["condition", "biological_repeat", "position_id"]].drop_duplicates()
    return objects, fovs


def _repeat_tables(conditions, repeats, fields=6, cells=20, effect=None, seed=0, shift=None):
    """``conditions`` per repeat, ``fields`` fields of ``cells`` cells each; ``effect`` the
    mean of each condition, ``shift`` a multiplier of each repeat's values."""
    rng = np.random.default_rng(seed)
    rows, fovs = [], []
    for repeat in repeats:
        for condition in conditions[repeat]:
            for field in range(fields):
                key = {
                    "condition": condition,
                    "biological_repeat": repeat,
                    "position_id": f"{condition}_{repeat}_{field}",
                }
                fovs.append(key)
                mean = (effect or {}).get(condition, 10.0) * (shift or {}).get(repeat, 1.0)
                rows += [{**key, "x": v} for v in rng.normal(mean, 1.0, cells)]
    return pd.DataFrame(rows), pd.DataFrame(fovs)


def _three(effect=None, seed=0):
    repeats = ["r1", "r2", "r3"]
    return _repeat_tables({r: ["ctrl", "drug"] for r in repeats}, repeats, effect=effect, seed=seed)


def _two_conditions(
    shift: float, field_sd: float, seed: int = 3, n_fields: int = 12, n_wells: int = 2
):
    """Control and treated wells in two repeats; ``shift`` moves the treated condition's ``x``.

    ``y`` never differs between the conditions.
    """
    rng = np.random.default_rng(seed)
    objects, fovs = [], []
    for repeat in ("R1", "R2"):
        for condition, well_shift in (("control", 0.0), ("treated", shift)):
            for well in [f"w{i + 1}" for i in range(n_wells)]:
                for j in range(n_fields):
                    name = f"{repeat}_{condition}_{well}_f{j}"
                    fovs.append(
                        {
                            "condition": condition,
                            "biological_repeat": repeat,
                            "well": well,
                            "image_file": name,
                        }
                    )
                    offset = well_shift + rng.normal(0, field_sd)
                    for _ in range(25):
                        objects.append(
                            {
                                "condition": condition,
                                "biological_repeat": repeat,
                                "well": well,
                                "image_file": name,
                                "x": 10 + offset + rng.normal(),
                                "y": rng.normal(),
                            }
                        )
    return pd.DataFrame(objects), pd.DataFrame(fovs)


CONTRAST_GROUPS = ["condition", "biological_repeat", "well"]


def _technical(objects, fovs):
    """The same tables with the well named as the MorphoBricks technical repeat."""
    rename = {"well": "technical_repeat"}
    return objects.rename(columns=rename), fovs.rename(columns=rename)

"""A synthetic design like the ECM dataset (owner, 2026-10-05), for the comparison tests.

Two matrices (``VEC``, ``THBS5``) by four siRNA levels (none, ``SiNC``, ``SiCol6a1``,
``SiFBN2``): eight conditions, ``VEC``, ``VEC_SiNC``, ..., ``THBS5_SiFBN2``, in six biological
repeats ``R1``-``R6``, two wells of three fields each, unevenly: ``THBS5_SiFBN2`` was not
imaged in R6 and ``VEC`` not in R5. ``effects`` adds a log shift to a condition's ``area``
and its value to ``order``.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

MATRICES = ("VEC", "THBS5")
SIRNAS = ("none", "SiNC", "SiCol6a1", "SiFBN2")
REPEATS = tuple(f"R{r}" for r in range(1, 7))
MISSING = (("THBS5_SiFBN2", "R6"), ("VEC", "R5"))


def name(matrix: str, sirna: str) -> str:
    """``VEC`` for no siRNA, else ``VEC_SiNC``."""
    return matrix if sirna == "none" else f"{matrix}_{sirna}"


CONDITIONS = tuple(name(m, s) for m in MATRICES for s in SIRNAS)


def design() -> dict[str, dict[str, str]]:
    """Each condition with its factor levels."""
    return {name(m, s): {"matrix": m, "sirna": s} for m in MATRICES for s in SIRNAS}


def tables(
    seed: int = 0,
    effects: Mapping[str, float] | None = None,
    missing: tuple = MISSING,
    cells: int = 8,
    repeat_sd: float = 0.1,
    effect_by_repeat: Mapping[str, Mapping[str, float]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The observation and FOV tables (see the module); ``effect_by_repeat`` adds a
    repeat-specific log shift to a condition (``{condition: {repeat: shift}}``)."""
    rng = np.random.default_rng(seed)
    effects = dict(effects or {})
    by_repeat = effect_by_repeat or {}
    objects, fovs = [], []
    for repeat in REPEATS:
        shift = rng.normal(0, repeat_sd)
        for condition in CONDITIONS:
            if (condition, repeat) in missing:
                continue
            effect = effects.get(condition, 0.0) + by_repeat.get(condition, {}).get(repeat, 0.0)
            for well in range(2):
                technical = f"{condition}_{repeat}_w{well}"
                well_shift = rng.normal(0, 0.05)
                for field in range(3):
                    fov = f"{technical}_f{field}"
                    fovs.append(
                        {
                            "condition": condition,
                            "biological_repeat": repeat,
                            "technical_repeat": technical,
                            "fov_id": fov,
                        }
                    )
                    level = 10 * np.exp(shift + well_shift + effect)
                    for _ in range(cells):
                        objects.append(
                            {
                                "condition": condition,
                                "biological_repeat": repeat,
                                "technical_repeat": technical,
                                "fov_id": fov,
                                "area": level * np.exp(rng.normal(0, 0.2)),
                                "order": rng.normal(1 + effect, 0.3),
                            }
                        )
    return pd.DataFrame(objects), pd.DataFrame(fovs)

"""Is the predicted chance of a new phenotype in the next FOV right? Check it by shuffling.

``SupportNoveltyEstimator`` predicts, from the FOVs imaged so far, the chance
that one more representative FOV shows a phenotype not yet seen
(``lofo_novel_fov_fraction``) and an upper confidence bound on that chance
(``lofo_novel_fov_upper``). A deeply sampled dataset lets both be checked
against what actually happened, without any simulation model:

1. within one analysis group, pick ``k`` of the acquired FOVs at random;
2. predict from those ``k`` alone, exactly as during acquisition;
3. judge every other FOV of the group against the same ``k``, with the same
   definition of new; the fraction judged new is what the next FOV would
   actually have shown, for a next FOV drawn at random from the rest;
4. repeat for many random subsets and several ``k``.

``calibrate_next_fov_novelty`` reports, per group and ``k``, the mean
prediction, the mean actual rate, the mean bound and how often the bound held
(was at least the actual rate). A well-calibrated prediction has mean
prediction close to the mean actual rate; a valid 95% bound holds in about
95% of draws or more.

Limits:

- Random subsets treat the group's FOVs as exchangeable, as a
  representative (random-position) acquisition does. They do not test drift
  along the recorded acquisition order.
- The actual rate is measured on the group's remaining FOVs only. With few
  remaining FOVs it is noisy, which lowers how often the bound appears to
  hold; ``remaining_fovs`` says how many there were.
- Neighbouring FOVs are often correlated, which the bound ignores; this is
  one thing the check can reveal.
- The check is for this dataset and these settings (``epsilon``, transform,
  ``min_novel_observations``); it does not transfer to other settings.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .evaluator import iter_analysis_groups, objects_in_fovs, restrict_input
from .evidence import Availability
from .objectives import ObjectiveConfig, SamplingObjective
from .schema import validate_input_tables
from .strategies.support_novelty import FOV_QUANTITY, UPPER_QUANTITY, SupportNoveltyEstimator

_POPULATION = ObjectiveConfig(SamplingObjective.POPULATION)

CALIBRATION_COLUMNS = (
    "fovs_imaged",
    "remaining_fovs",
    "draws",
    "unavailable_draws",
    "predicted",
    "actual",
    "bound",
    "bound_held",
)


def _check_settings(novelty, fov_counts, draws, random_state) -> list[int]:
    """Validate the calibration settings; return the sorted distinct FOV counts."""
    if not isinstance(novelty, SupportNoveltyEstimator):
        raise TypeError("novelty must be a SupportNoveltyEstimator")
    counts = list(fov_counts)
    if not counts:
        raise ValueError("give at least one FOV count")
    for value, label in [*((c, "fov_counts") for c in counts), (draws, "draws")]:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{label} must hold positive integers")
    if min(counts) < novelty.min_fovs:
        raise ValueError(
            f"fov_counts must be at least the estimator's min_fovs ({novelty.min_fovs})"
        )
    if isinstance(random_state, bool) or not isinstance(random_state, int):
        raise TypeError("random_state must be an integer seed")
    return sorted(set(counts))


def _draw(data, novelty, chosen, rest):
    """Prediction, bound and actual novel rate for one random subset, or None."""
    visible_fovs, new_fovs = data.fovs.iloc[chosen], data.fovs.iloc[rest]
    visible = restrict_input(data, objects_in_fovs(data, visible_fovs), visible_fovs)
    evidence = novelty.estimate(visible, objective=_POPULATION)
    predicted, bound = evidence.quantities[FOV_QUANTITY], evidence.quantities[UPPER_QUANTITY]
    if predicted.availability is not Availability.AVAILABLE:
        return None
    new = restrict_input(data, objects_in_fovs(data, new_fovs), new_fovs)
    novel = novelty.novel_fovs_against(visible, new)
    if novel is None:
        return None
    return predicted.value, bound.value, float(novel.mean())


def calibrate_next_fov_novelty(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    novelty: SupportNoveltyEstimator,
    fov_counts: Sequence[int],
    draws: int,
    random_state: int,
    fov_col: str | None = None,
    group_cols: Sequence[str] | None = None,
    observation_unit: str | None = None,
) -> pd.DataFrame:
    """Predicted versus actual chance that the next FOV shows a new phenotype.

    One row per analysis group and FOV count ``k`` in ``fov_counts``, from
    ``draws`` random subsets of ``k`` acquired FOVs (see the module
    docstring). Columns: the group columns, then

    - ``fovs_imaged`` (``k``) and ``remaining_fovs`` (the FOVs the actual rate
      was measured on);
    - ``draws`` used and ``unavailable_draws`` (subsets on which the
      prediction was undefined, for example too few FOVs with objects);
    - ``predicted``: mean predicted chance; ``actual``: mean fraction of the
      remaining FOVs that did show something new;
    - ``bound``: mean upper bound; ``bound_held``: fraction of draws in which
      the bound was at least the actual rate.

    Means are ``None`` when no draw was usable. FOV counts not smaller than a
    group's FOV count are skipped for that group. ``fovs`` must list every
    acquired FOV; FOVs without observations can be drawn and count as not new.
    """
    counts = _check_settings(novelty, fov_counts, draws, random_state)
    data = validate_input_tables(
        objects,
        fovs=fovs,
        feature_cols=feature_cols,
        fov_col=fov_col,
        group_cols=group_cols,
        observation_unit=observation_unit,
    )
    rng = np.random.default_rng(random_state)
    rows = []
    for group, group_data in iter_analysis_groups(data):
        total = len(group_data.fovs)
        for k in (c for c in counts if c < total):
            results = []
            for _ in range(draws):
                order = rng.permutation(total)
                results.append(_draw(group_data, novelty, order[:k], order[k:]))
            usable = np.array([r for r in results if r is not None], dtype=float)
            row = {**group, "fovs_imaged": k, "remaining_fovs": total - k}
            row.update({"draws": len(usable), "unavailable_draws": draws - len(usable)})
            if len(usable):
                predicted, bound, actual = usable.T
                row.update(
                    {
                        "predicted": float(predicted.mean()),
                        "actual": float(actual.mean()),
                        "bound": float(bound.mean()),
                        "bound_held": float(np.mean(bound >= actual)),
                    }
                )
            else:
                row.update(dict.fromkeys(("predicted", "actual", "bound", "bound_held")))
            rows.append(row)
    return pd.DataFrame(rows, columns=[*data.group_cols, *CALIBRATION_COLUMNS])


def pool_calibration(table: pd.DataFrame) -> pd.DataFrame:
    """Calibration over all groups, per FOV count, weighting groups by usable draws."""
    rows = []
    for k, runs in table.groupby("fovs_imaged", sort=True):
        runs = runs[runs["draws"] > 0]
        weights = runs["draws"].to_numpy(dtype=float)
        row = {
            "fovs_imaged": int(k),
            "groups": len(runs),
            "draws": int(weights.sum()),
        }
        for column in ("predicted", "actual", "bound", "bound_held"):
            values = runs[column].to_numpy(dtype=float)
            row[column] = float(np.average(values, weights=weights)) if len(runs) else None
        rows.append(row)
    return pd.DataFrame(rows)


__all__ = ["calibrate_next_fov_novelty", "pool_calibration"]

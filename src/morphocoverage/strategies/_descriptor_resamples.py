"""The resample arithmetic of the descriptor-spread rule (``stabilization``).

Pure functions of plain arrays: the paper's descriptors of many equally sized
object resamples, or of many resamples of whole FOVs, at once.

Shared-private: every name here is used by ``stabilization``; they are private
to the package and may change with it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _describe_many(samples: np.ndarray, descriptors: Sequence[str]) -> dict[str, np.ndarray]:
    """Descriptors for each row of a 2-D array of equally sized samples."""
    q1, median, q3 = np.percentile(samples, [25, 50, 75], axis=1)
    available = {
        "mean": samples.mean(axis=1),
        "median": median,
        "sd": samples.std(axis=1, ddof=1),
        "iqr": q3 - q1,
    }
    return {name: available[name] for name in descriptors}


# Upper bound on values held at once when resampling whole FOVs.
_MAX_RESAMPLED_VALUES = 5_000_000


def _describe_fov_resamples(
    grouped: np.ndarray, counts: np.ndarray, draws: np.ndarray, descriptors: Sequence[str]
) -> dict[str, np.ndarray]:
    """Descriptors of each resample of whole FOVs, without a Python loop over resamples.

    ``grouped`` holds the values ordered by FOV, ``counts`` the values per FOV
    and ``draws`` one row of FOV indices per resample. Each resample is the
    concatenation of its FOVs' values, as in ``describe``; quantiles use the
    same linear interpolation as ``np.percentile``. Resamples with fewer than
    2 values give NaN.
    """
    n_resamples = len(draws)
    starts = np.cumsum(counts) - counts
    chosen = draws.ravel()
    lengths = counts[chosen]
    # Index of every value of every chosen FOV, and the resample it belongs to.
    offsets = np.repeat(starts[chosen] - (np.cumsum(lengths) - lengths), lengths)
    members = offsets + np.arange(lengths.sum())
    values = grouped[members]
    resample = np.repeat(np.repeat(np.arange(n_resamples), draws.shape[1]), lengths)
    sizes = np.bincount(resample, minlength=n_resamples)
    defined = sizes >= 2
    safe = np.maximum(sizes, 1)

    mean = np.bincount(resample, weights=values, minlength=n_resamples) / safe
    deviations = np.bincount(
        resample, weights=(values - mean[resample]) ** 2, minlength=n_resamples
    )
    sd = np.sqrt(deviations / np.maximum(sizes - 1, 1))

    # Values sorted within each resample: one integer sort on (resample, value rank).
    by_value = np.argsort(grouped, kind="stable")
    rank = np.empty_like(by_value)
    rank[by_value] = np.arange(len(grouped))
    keys = np.sort(resample * len(grouped) + rank[members])
    # A trailing NaN keeps indexing valid for resamples made only of empty FOVs.
    ordered = np.append(grouped[by_value][keys % max(1, len(grouped))], np.nan)
    first = np.cumsum(sizes) - sizes

    def quantile(q: float) -> np.ndarray:
        """Linearly interpolated quantile of every resample (``np.percentile``'s default)."""
        position = q * (safe - 1)
        low = np.floor(position).astype(np.int64)
        high = np.minimum(low + 1, safe - 1)
        a, b = ordered[first + low], ordered[first + high]
        return a + (b - a) * (position - low)

    available = {
        "mean": lambda: mean,
        "sd": lambda: sd,
        "median": lambda: quantile(0.5),
        "iqr": lambda: quantile(0.75) - quantile(0.25),
    }
    return {name: np.where(defined, available[name](), np.nan) for name in descriptors}


def _chunk_sizes(total: int, chunk: int) -> list[int]:
    """``total`` split into consecutive parts of at most ``chunk``."""
    return [min(chunk, total - start) for start in range(0, total, chunk)]

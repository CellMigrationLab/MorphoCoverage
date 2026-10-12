"""Were the fields a random sample of the well? A check of the acquisition order that flags.

Every stopping rule reads the fields seen so far as a random sample of the
well's positions: the per-field mean-precision rule as exchangeable fields, the
per-object rule as independent objects, the rare-incidence rule as
exchangeable fields, the novelty bound as independent fields. A raster that
tiles the well row by row breaks that when the readout follows a spatial
trend (a gradient, an edge effect, an empty region) or drifts over the
acquisition: the first rows then look settled, and a rule can say stop after
one or two of them while the well's mean is far away (audit 4, P2-1, P2-2,
P4-2).

``check_acquisition_order`` looks for that in the fields seen so far, and only
in them, so it gives the same answer live and in a replay of the same prefix:

- with an acquisition-order column, a trend of the field means of the readout
  against acquisition order (Spearman's rank correlation, two-sided);
- with stage columns, spatial autocorrelation of the field means between
  neighbouring fields (Moran's I; neighbours are fields within 1.5 times the
  median nearest-neighbour distance, so the eight neighbours of a grid
  position; one-sided permutation p-value from ``PERMUTATIONS`` permutations
  with a fixed seed, over the fields in a canonical order, so it never depends
  on the order in which the fields arrived);
- the two tests share ``ALPHA`` (Bonferroni: each at ``ALPHA`` divided by the
  number of tests run), so random fields raise the flag in about ``ALPHA`` of
  wells or fewer; a test needs ``MIN_FIELDS`` fields with a mean;
- with stage columns, the stage extent the fields have covered, which is what
  a rare-state rule's stop rests on: a state confined to positions not yet
  visited cannot have been seen.

It **flags, never vetoes**: a significant trend is printed as a caution beside
every stop (``StrategyResult.explain``, ``decision_view``, the live line and
the report's stop lines), and the rules' decisions are unchanged. A check that
finds nothing does not show the fields to be representative (a raster over a
gradient is flagged only once the trend is visible above the noise), so the
assumption is still stated beside it; without either column it is stated
alone (``RANDOM_SAMPLE_ASSUMPTION``). Random or dispersed acquisition orders
remove the problem instead of flagging it (``docs/live/stopping_rules.md``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse, spatial, stats

from ._caveats import assumption
from ._stage_coordinates import stage_matrix
from ._wording import count, percent
from .schema import ValidatedInput, fov_values
from .statistics import ALPHA

RANDOM_SAMPLE_ASSUMPTION = (
    "treats the fields as a random sample of the well's positions; a raster over a "
    "spatial trend can stop early"
)
RARE_ASSUMPTION = (
    "treats the fields as exchangeable: a rare state confined to positions not yet "
    "visited cannot have been seen"
)
# A stop's caveat when no order check ran: a missing check is a caution about these data.
NOT_CHECKED = "the acquisition order was not checked"
PERMUTATIONS = 999
MIN_FIELDS = 4
NEIGHBOUR_FACTOR = 1.5
SEED = 20260928


@dataclass(frozen=True)
class OrderCheck:
    """The acquisition-order check of the fields seen so far (see the module).

    ``n_fields`` counts the fields with a mean of ``readout`` (fields without
    objects, or with only missing values, have none); ``positions`` holds the
    stage position of every field seen, empty ones included, in acquisition
    order. A test that was not run has ``None`` and its reason in ``reasons``.
    """

    readout: str
    n_fields: int
    order_col: str | None = None
    stage_cols: tuple[str, str] | None = None
    spearman_rho: float | None = None
    spearman_p: float | None = None
    moran_i: float | None = None
    moran_p: float | None = None
    positions: tuple[tuple[float, float], ...] = ()
    reasons: tuple[str, ...] = ()
    alpha: float = ALPHA

    @property
    def level(self) -> float:
        """Each test's level: ``alpha`` split over the tests run (Bonferroni), so the flag
        is raised in at most about ``alpha`` of random samples."""
        tests = sum(p is not None for p in (self.spearman_p, self.moran_p))
        return self.alpha / max(1, tests)

    @property
    def trend(self) -> bool:
        """Whether the field means trend with acquisition order (Spearman p below ``level``)."""
        return self.spearman_p is not None and self.spearman_p < self.level

    @property
    def autocorrelated(self) -> bool:
        """Whether neighbouring fields have alike means (Moran's I p below ``level``)."""
        return self.moran_p is not None and self.moran_p < self.level

    @property
    def flagged(self) -> bool:
        """Whether either test found the fields not to be a random sample."""
        return self.trend or self.autocorrelated

    def extent_share(self, n_fields: int) -> float | None:
        """Share of the stage extent of every field seen that the first ``n_fields`` covered.

        The extent is the bounding box of the positions, widened by one field
        spacing (the median nearest-neighbour distance), so that a row of a
        grid covers its share of the rows rather than no area; ``None`` with
        fewer than two distinct positions.
        """
        every = np.array(self.positions, dtype=float).reshape(-1, 2)
        if len(np.unique(every, axis=0)) < 2:
            return None
        nearest = spatial.cKDTree(every).query(every, k=2)[0][:, 1]
        spacing = float(np.median(nearest[nearest > 0])) if np.any(nearest > 0) else 0.0
        first = every[: max(1, int(n_fields))]
        whole = np.ptp(every, axis=0) + spacing
        part = np.ptp(first, axis=0) + spacing
        return float(np.prod(part) / np.prod(whole))

    def span(self) -> str | None:
        """The stage region the fields seen cover, in stage units, or ``None`` without positions."""
        if not self.positions or self.stage_cols is None:
            return None
        every = np.array(self.positions, dtype=float)
        low, high = every.min(axis=0), every.max(axis=0)
        return ", ".join(
            f"{column} {low[i]:g} to {high[i]:g}" for i, column in enumerate(self.stage_cols)
        )

    def _found(self) -> list[str]:
        found = []
        if self.trend:
            found.append(
                f"the field means of {self.readout} trend with acquisition order (Spearman rho "
                f"{self.spearman_rho:.2f}, p = {_p(self.spearman_p)})"
            )
        if self.autocorrelated:
            found.append(
                f"neighbouring fields have alike means of {self.readout} (Moran's I "
                f"{self.moran_i:.2f}, permutation p = {_p(self.moran_p)})"
            )
        return found

    def _clear(self) -> list[str]:
        """The tests that flagged nothing, as one phrase with the threshold they were held to:
        a p of 0.037 is not "no trend", only not flagged after Bonferroni (audit 10, A21)."""
        names, values = [], []
        if self.spearman_p is not None and not self.trend:
            names.append("trend with acquisition order")
            values.append(f"Spearman p = {_p(self.spearman_p)}")
        if self.moran_p is not None and not self.autocorrelated:
            names.append("spatial autocorrelation")
            values.append(f"Moran's I permutation p = {_p(self.moran_p)}")
        if not names:
            return []
        threshold = f"flagged at the Bonferroni threshold of {self.level:g}"
        return [f"no {' or '.join(names)} {threshold} ({'; '.join(values)})"]

    def caution(self, *, rare: bool = False, stop_after: int | None = None) -> str:
        """The sentence printed beside a stop: what was flagged, or the assumption it rests on.

        ``rare`` adds what a rare-state rule's stop rests on: the stage region
        the fields cover (the share of the acquired fields' extent covered by
        the first ``stop_after`` fields, in a report) and exchangeable fields.
        """
        fields = count(self.n_fields, "field")
        if self.flagged:
            # Alike neighbours bias a mean only when the positions were not drawn at
            # random (a raster, a region); a trend with the order biases it as acquired.
            condition = "" if self.trend else "unless the positions were chosen at random, "
            text = (
                f"caution: {' and '.join(self._found())} over {fields}, so {condition}the "
                "fields may not represent the well and this stop can be early (a flag, not a "
                "veto; image fields in a random or dispersed order)"
            )
        else:
            tested = self._clear()
            head = f"order check: {' and '.join(tested)} over {fields}" if tested else ""
            reasons = "; ".join(self.reasons)
            head = "; ".join(part for part in (head, reasons and f"not tested: {reasons}") if part)
            text = f"{head}; still {RANDOM_SAMPLE_ASSUMPTION}" if head else RANDOM_SAMPLE_ASSUMPTION
        if rare:
            text += f"; {self._coverage(stop_after)}"
        return text

    def _coverage(self, stop_after: int | None) -> str:
        """Where a rare-state stop has looked: the stage extent covered, then the assumption."""
        if stop_after is not None:
            share = self.extent_share(stop_after)
            if share is not None:
                return (
                    f"the {count(stop_after, 'field')} up to the recorded stop covered "
                    f"{percent(share)} of the stage extent of the acquired fields; it "
                    f"{RARE_ASSUMPTION}"
                )
        span = self.span()
        if span is not None:
            return f"the fields so far cover the stage at {span}; it {RARE_ASSUMPTION}"
        return f"it {RARE_ASSUMPTION}"


def order_caveat(text: str) -> str:
    """A stop's order caveat (``OrderCheck.caution``) by its kind (``_style.Assumption``).

    When the check ran and found nothing (``order check: no trend ...; still
    treats ...``), the random-sample assumption is a standing one: an
    ``Assumption``. A flag (``caution: ...``) is a caution about these data, and
    so is the assumption stated alone or with ``not tested`` only: no test ran,
    and the sentence says so (``NOT_CHECKED``).
    """
    if text.startswith("order check:"):
        return assumption(text.replace("; still treats", "; every stop still treats", 1))
    if text.startswith(("treats", "it treats")):
        return f"{NOT_CHECKED}, so every stop {text.removeprefix('it ')}"
    if text.startswith("not tested:"):
        return f"{NOT_CHECKED} ({text})"
    return text


def _p(value: float | None) -> str:
    """A p-value with two significant digits."""
    return "undefined" if value is None else f"{value:.2g}"


def _field_means(data: ValidatedInput, readout: str) -> np.ndarray:
    """The mean of ``readout`` in each field of ``data.fovs`` (NaN without a finite value)."""
    values = data.objects[readout].to_numpy(dtype=float)
    return np.array([part.mean() if len(part) else np.nan for part in fov_values(data, values)])


def _order_ranks(fovs: pd.DataFrame, order_col: str | None) -> np.ndarray:
    """Each field's rank in acquisition order: its column, else the order the fields arrived."""
    if order_col is None or order_col not in fovs.columns:
        return np.arange(len(fovs), dtype=float)
    return fovs[order_col].reset_index(drop=True).rank(method="average").to_numpy(dtype=float)


def _spearman(means: np.ndarray, ranks: np.ndarray) -> tuple[float | None, float | None, str]:
    """Spearman's rho and its two-sided p-value over the fields with a mean, or why not."""
    keep = np.isfinite(means)
    if keep.sum() < MIN_FIELDS:
        return None, None, f"fewer than {MIN_FIELDS} fields with a mean"
    if np.ptp(means[keep]) == 0:
        return None, None, "the field means do not vary"
    rho, p = stats.spearmanr(ranks[keep], means[keep])
    return float(rho), float(p), ""


def _positions(fovs: pd.DataFrame, stage_cols: tuple[str, str]) -> np.ndarray | None:
    """The stage position of each field (NaN where a coordinate is missing), or ``None``."""
    if any(column not in fovs.columns for column in stage_cols):
        return None
    return stage_matrix(fovs, stage_cols, missing="keep")


def _moran(means: np.ndarray, positions: np.ndarray) -> tuple[float | None, float | None, str]:
    """Moran's I of the field means between neighbours and its permutation p-value, or why not.

    Fields are put in a canonical order (by position, then mean) before the
    permutations are drawn, so the p-value depends on the fields seen, never on
    the order in which they arrived.
    """
    keep = np.isfinite(means) & np.isfinite(positions).all(axis=1)
    if np.isfinite(means).sum() < MIN_FIELDS:
        return None, None, f"fewer than {MIN_FIELDS} fields with a mean"
    if keep.sum() < MIN_FIELDS:
        return None, None, f"fewer than {MIN_FIELDS} fields with a mean and a stage position"
    values, where = means[keep], positions[keep]
    canonical = np.lexsort((values, where[:, 1], where[:, 0]))
    values, where = values[canonical], where[canonical]
    z = values - values.mean()
    if not np.any(z):
        return None, None, "the field means do not vary"
    tree = spatial.cKDTree(where)
    nearest = tree.query(where, k=2)[0][:, 1]
    reach = NEIGHBOUR_FACTOR * float(np.median(nearest))
    pairs = tree.query_pairs(reach, output_type="ndarray")
    if not len(pairs):
        return None, None, "no two fields are stage neighbours"
    n = len(z)
    weights = sparse.coo_matrix(
        (
            np.ones(2 * len(pairs)),
            (np.r_[pairs[:, 0], pairs[:, 1]], np.r_[pairs[:, 1], pairs[:, 0]]),
        ),
        shape=(n, n),
    ).tocsr()

    def moran(z_values: np.ndarray) -> np.ndarray:
        # One row per arrangement: n / S0 * z'Wz / z'z, S0 the number of ordered neighbour pairs.
        cross = (np.asarray(weights @ z_values.T).T * z_values).sum(axis=-1)
        return n / (2 * len(pairs)) * cross / (z_values**2).sum(axis=-1)

    observed = float(moran(z[None, :])[0])
    rng = np.random.default_rng(SEED)
    shuffled = rng.permuted(np.tile(z, (PERMUTATIONS, 1)), axis=1)
    as_large = int((moran(shuffled) >= observed - 1e-12).sum())
    return observed, (1 + as_large) / (PERMUTATIONS + 1), ""


def check_acquisition_order(
    data: ValidatedInput,
    readout: str,
    *,
    order_col: str | None = None,
    stage_cols: Sequence[str] | None = None,
) -> OrderCheck:
    """The acquisition-order check of the fields in ``data`` (one analysis group; see the module).

    ``data.fovs`` lists the fields in acquisition order (a visible prefix, live
    or replayed); ``order_col`` names the acquisition-order column (without it
    in the FOV table, the order of the rows is the order), ``stage_cols`` the
    two stage-coordinate columns. With neither, nothing is tested and the
    caution is the assumption alone.
    """
    if readout not in data.objects.columns:
        raise ValueError(f"the readout {readout!r} is not a column of the observations")
    if stage_cols is not None:
        if isinstance(stage_cols, str) or len(tuple(stage_cols)) != 2:
            raise TypeError("stage_cols must be two column names")
        stage_cols = tuple(stage_cols)
    means = _field_means(data, readout)
    reasons: list[str] = []
    found: dict[str, object] = {}
    if order_col is not None:
        rho, p, reason = _spearman(means, _order_ranks(data.fovs, order_col))
        found.update(spearman_rho=rho, spearman_p=p)
        if reason:
            reasons.append(reason)
    positions: tuple[tuple[float, float], ...] = ()
    if stage_cols is not None:
        where = _positions(data.fovs, stage_cols)
        if where is None:
            reasons.append("the stage columns are not in the FOV table")
        else:
            located = where[np.isfinite(where).all(axis=1)]
            positions = tuple((float(x), float(y)) for x, y in located)
            moran_i, moran_p, reason = _moran(means, where)
            found.update(moran_i=moran_i, moran_p=moran_p)
            if reason and reason not in reasons:
                reasons.append(reason)
    return OrderCheck(
        readout=readout,
        n_fields=int(np.isfinite(means).sum()),
        order_col=order_col,
        stage_cols=stage_cols,
        positions=positions,
        reasons=tuple(reasons),
        **found,
    )


__all__ = [
    "ALPHA",
    "NOT_CHECKED",
    "PERMUTATIONS",
    "RANDOM_SAMPLE_ASSUMPTION",
    "RARE_ASSUMPTION",
    "OrderCheck",
    "check_acquisition_order",
    "order_caveat",
]

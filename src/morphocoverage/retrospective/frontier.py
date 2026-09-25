"""The benchmark score: fields saved against a fixed number of fields, at matched risk.

Every stopping rule, whatever it computes, decides when to stop. On replayed
acquisitions scored against held-back fields (``replay_acquisitions``), a rule
at one setting has two numbers:

- **risk**: the share of replayed acquisitions whose acquisition, ended where
  the rule stopped (or at the end of the stream if it never stopped), did not
  preserve the declared outcome in the held-back fields; acquisitions whose
  outcome could not be judged are left out of the share and counted;
- **effort**: the mean number of fields acquired, the stop or the whole
  stream when the rule never stopped.

Sweeping a rule's threshold traces a curve of effort against risk. Acquiring
a fixed number of fields ``k`` traces the **baseline curve**, which needs no
data at all; it is computed from the same replays for ``k = 1`` up to the
longest stream. A rule is worth having only where it lies below that curve,
which it can do only by giving easy samples fewer fields and hard ones more.

The **score** of a rule family at a risk level ``alpha`` is

    fields saved = (fewest mean fields at which the fixed rule's risk <= alpha)
                 - (fewest mean fields at which the family's risk <= alpha),

in fields, so 0 is the null result (as good as a fixed count) and a negative
value is worse. Its interval resamples the declared **independent units**
(samples or repeats, ``unit_cols``) with replacement; hold-outs, orders and
draws describe each unit and never narrow it. With fewer than two units there
is no interval.

Also reported:

- ``baseline_risk_at_same_effort``: the fixed rule's risk at the family's
  chosen mean effort (linear interpolation between whole fields), the mirror
  of the score;
- the chosen setting is the cheapest from which risk stays within ``alpha``
  at every setting with more effort, so a lucky dip in a noisy curve is not
  taken as safe (the rule of ``protocol_reliability``);
- **adaptivity**: the Spearman correlation, over replayed acquisitions,
  between where the family stopped and where held-back fields confirmed the
  outcome lastingly (sustained realized adequacy; not reached counts as one
  beyond the stream). A rule that saves fields by stopping later on harder
  samples has a positive correlation; one that saves them by chance does not;
- ``transfer``: a setting chosen on a development dataset to meet ``alpha``,
  and the risk that same setting gives on a held-out dataset. That is the
  prospective test: a threshold is only as good as its risk on data it was
  not chosen on.

Families are built from strategies replayed with ``family`` (one setting per
strategy) or, for a single-threshold rule, with ``sweep``, which re-decides
the stored evidence of one replayed configuration at many thresholds through
``redecide`` (the replay must keep its results: ``keep_results=True``).
Families from several replays can be scored together when the replays share
reservoirs and orders, for example a population and a rare-objective replay
with the same ``random_state`` and splits.

This is a retrospective benchmark. Its tolerances, thresholds and risk
levels must be fixed before it is run, and a performance claim belongs to a
held-out dataset (``BenchmarkRegistry``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ..policies import ThresholdPolicy, redecide
from ._text import percent
from .adequacy import adequacy_points
from .criteria import Preservation
from .replay import ReplayResult

BASELINE = "fixed fields"
_RUN = ["split", "order_id"]
FAMILY_COLUMNS = ("family", "setting")


def _run_keys(replay: ReplayResult) -> list[str]:
    return [*replay.group_cols, *_RUN]


def _with_streams(replay: ReplayResult, decisions: pd.DataFrame) -> pd.DataFrame:
    """Attach each run's stream and reservoir sizes, so replays can be matched later."""
    keys = _run_keys(replay)
    streams = replay.streams[[*keys, "order", "stream_fovs", "reservoir_fovs"]]
    columns = [*keys, "fovs", "decision", *FAMILY_COLUMNS]
    return decisions[columns].merge(streams, on=keys, how="left", validate="many_to_one")


def family(replay: ReplayResult, name: str, settings: Mapping[str, Any]) -> pd.DataFrame:
    """Decisions of replayed strategies as one family: ``settings`` maps config_id to setting."""
    if not isinstance(replay, ReplayResult):
        raise TypeError("replay must be a ReplayResult")
    if not settings:
        raise ValueError("a family needs at least one configuration")
    missing = set(settings) - set(replay.decisions["config_id"])
    if missing:
        raise ValueError(f"configurations not in the replay: {sorted(missing)}")
    rows = replay.decisions[replay.decisions["config_id"].isin(list(settings))].copy()
    rows["family"] = name
    rows["setting"] = rows["config_id"].map(dict(settings))
    return _with_streams(replay, rows)


def sweep(
    replay: ReplayResult,
    *,
    config_id: str,
    quantity: str,
    thresholds: Sequence[float],
    name: str,
) -> pd.DataFrame:
    """One configuration's stored evidence re-decided at each threshold, as one family.

    Each stored ``StrategyResult`` is re-decided with ``ThresholdPolicy(quantity,
    threshold)``; ``redecide`` equals running the same estimators with that
    policy, so no estimator is run again.
    """
    if not isinstance(replay, ReplayResult):
        raise TypeError("replay must be a ReplayResult")
    if "result" not in replay.decisions.columns:
        raise ValueError("the replay kept no results; run it with keep_results=True")
    thresholds = list(thresholds)
    if not thresholds:
        raise ValueError("give at least one threshold")
    rows = replay.decisions[replay.decisions["config_id"] == config_id]
    if rows.empty:
        raise ValueError(f"configuration {config_id!r} is not in the replay")
    parts = []
    for threshold in thresholds:
        policy = ThresholdPolicy(quantity, threshold)
        decided = rows.drop(columns=["result"]).copy()
        decisions = []
        for result in rows["result"]:
            outcome = redecide(result, [policy], name=name).decision
            decisions.append(None if outcome is None else outcome.value)
        decided["decision"] = decisions
        decided["family"] = name
        decided["setting"] = float(threshold)
        parts.append(decided)
    return _with_streams(replay, pd.concat(parts, ignore_index=True))


def _statuses(replay: ReplayResult, criterion: str) -> dict[tuple, np.ndarray]:
    """Per run, the criterion's status after 1..stream fields (index 0 is one field)."""
    keys = _run_keys(replay)
    outcomes = replay.outcomes[replay.outcomes["criterion"] == criterion]
    if outcomes.empty:
        raise ValueError(f"criterion {criterion!r} was not assessed in the replay")
    table = {}
    for key, rows in outcomes.groupby(keys, sort=False, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        table[key] = rows.sort_values("fovs")["status"].to_numpy()
    return table


def _family_runs(decisions: pd.DataFrame, keys, statuses, streams) -> list[dict]:
    """One row per run and setting: where the family stopped and the status there."""
    rows = []
    group_keys = [*keys, "family", "setting"]
    for key, own in decisions.groupby(group_keys, sort=False, dropna=False):
        run = tuple(key[: len(keys)])
        if run not in streams:
            raise ValueError(f"run {run} is not a run of the scored replay")
        stream, order = streams[run]
        if int(own["stream_fovs"].iloc[0]) != stream:
            raise ValueError("families come from replays with different streams")
        stops = own.loc[own["decision"] == "stop", "fovs"]
        stopped = len(stops) > 0
        effort = int(stops.min()) if stopped else stream
        rows.append(
            {
                **dict(zip(group_keys, key, strict=True)),
                "order": order,
                "stopped": stopped,
                "effort": effort,
                "stream_fovs": stream,
                "status": statuses[run][effort - 1],
            }
        )
    return rows


def _baseline_runs(keys, statuses, streams) -> list[dict]:
    """The fixed rule at every k from 1 to the longest stream, on every run."""
    longest = max(stream for stream, _ in streams.values())
    rows = []
    for run, (stream, order) in streams.items():
        for k in range(1, longest + 1):
            effort = min(k, stream)
            rows.append(
                {
                    **dict(zip(keys, run, strict=True)),
                    "family": BASELINE,
                    "setting": k,
                    "order": order,
                    "stopped": k <= stream,
                    "effort": effort,
                    "stream_fovs": stream,
                    "status": statuses[run][effort - 1],
                }
            )
    return rows


def _curve(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (name, setting), own in runs.groupby(["family", "setting"], sort=False):
        defined = own["status"] != Preservation.UNDEFINED.value
        failed = own["status"] == Preservation.NOT_PRESERVED.value
        rows.append(
            {
                "family": name,
                "setting": setting,
                "runs": len(own),
                "effort": float(own["effort"].mean()),
                "risk": float(failed.sum() / defined.sum()) if defined.any() else None,
                "judged": int(defined.sum()),
                "never_stopped": float((~own["stopped"]).mean()),
            }
        )
    return pd.DataFrame(rows)


def _from_here_on(risks: np.ndarray, alpha: float) -> int | None:
    """Index of the first setting (by effort) from which every later risk is within alpha.

    The same "from here on" rule as the protocol's field count: a lucky dip in
    a noisy curve, followed by worse settings, is not a safe choice. Settings
    whose risk could not be judged (NaN) neither start nor break the run.
    """
    chosen = None
    for i, risk in enumerate(risks):
        if np.isnan(risk):
            continue
        if risk <= alpha + 1e-12:
            chosen = i if chosen is None else chosen
        else:
            chosen = None
    return chosen


def _needed(curve: pd.DataFrame, alpha: float) -> pd.Series | None:
    """The cheapest setting from which risk stays at most ``alpha``, or None."""
    ordered = curve.sort_values(["effort", "setting"], kind="stable")
    index = _from_here_on(ordered["risk"].astype(float).to_numpy(), alpha)
    return None if index is None else ordered.iloc[index]


def _interpolate_baseline(baseline: pd.DataFrame, effort: float) -> float | None:
    points = baseline.dropna(subset=["risk"]).sort_values("effort")
    if points.empty:
        return None
    return float(np.interp(effort, points["effort"], points["risk"]))


def _unit_ids(runs: pd.DataFrame, unit_cols: Sequence[str]) -> pd.Series:
    if not unit_cols:
        return pd.Series("all", index=runs.index)
    return runs[list(unit_cols)].astype(str).agg(" / ".join, axis=1)


def _bootstrap(runs, unit_cols, alphas, n_boot, rng) -> dict[tuple[str, float], np.ndarray]:
    """Fields saved per family and alpha in each resample of independent units."""
    runs = runs.assign(
        unit=_unit_ids(runs, unit_cols),
        defined=runs["status"] != Preservation.UNDEFINED.value,
        failed=runs["status"] == Preservation.NOT_PRESERVED.value,
    )
    units = sorted(runs["unit"].unique())
    if len(units) < 2:
        return {}
    configs = list(runs.groupby(["family", "setting"], sort=False).groups)
    index = {u: i for i, u in enumerate(units)}
    shape = (len(configs), len(units))
    count, effort, defined, failed = (np.zeros(shape) for _ in range(4))
    for c, ((name, setting), own) in enumerate(runs.groupby(["family", "setting"], sort=False)):
        per_unit = own.groupby("unit")[["effort", "defined", "failed"]].agg(["sum", "count"])
        for unit, row in per_unit.iterrows():
            u = index[unit]
            count[c, u] = row[("effort", "count")]
            effort[c, u] = row[("effort", "sum")]
            defined[c, u] = row[("defined", "sum")]
            failed[c, u] = row[("failed", "sum")]
    picks = rng.integers(0, len(units), size=(n_boot, len(units)))
    weights = np.zeros((n_boot, len(units)))
    np.add.at(weights, (np.arange(n_boot)[:, None], picks), 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_effort = (weights @ effort.T) / (weights @ count.T)
        risk = (weights @ failed.T) / (weights @ defined.T)
    families = {}
    for c, (name, _) in enumerate(configs):
        families.setdefault(name, []).append(c)
    saved = {}
    for alpha in alphas:
        needed = {}
        for name, columns in families.items():
            cols = np.array(columns)
            best = np.full(n_boot, np.nan)
            for b in range(n_boot):
                # Settings in the order of their effort in this resample.
                order = np.argsort(mean_effort[b, cols], kind="stable")
                index = _from_here_on(risk[b, cols][order], alpha)
                if index is not None:
                    best[b] = mean_effort[b, cols][order][index]
            needed[name] = best
        for name in families:
            if name != BASELINE:
                saved[(name, alpha)] = needed[BASELINE] - needed[name]
    return saved


def _adaptivity(runs: pd.DataFrame, realized: pd.DataFrame, keys) -> float | None:
    """Spearman correlation of the stop with sustained realized adequacy, over runs."""
    merged = runs.merge(realized, on=keys, how="inner")
    if len(merged) < 3 or merged["effort"].nunique() < 2 or merged["needed"].nunique() < 2:
        return None
    return float(stats.spearmanr(merged["effort"], merged["needed"]).statistic)


@dataclass(frozen=True)
class FrontierResult:
    """The effort-risk curves, the score per family and risk level, and every run.

    - ``curve``: per family and setting, ``runs``, mean ``effort`` (fields),
      ``risk`` (share of judged runs not preserved), ``judged`` runs and the
      share ``never_stopped``. The baseline family is ``"fixed fields"``.
    - ``scores``: per family and ``alpha``, the chosen ``setting``, its
      ``effort`` and ``risk``, the baseline's ``baseline_effort``, the
      ``fields_saved`` with ``saved_low`` / ``saved_high`` and
      ``saved_reached`` (the share of unit resamples in which both reached
      ``alpha``), ``baseline_risk_at_same_effort`` and ``adaptivity``.
    - ``runs``: one row per run, family and setting.
    """

    curve: pd.DataFrame
    scores: pd.DataFrame
    runs: pd.DataFrame
    settings: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """The score in sentences, per risk level."""
        return frontier_summary(self)


def frontier(
    replay: ReplayResult,
    families: Sequence[pd.DataFrame],
    *,
    criterion: str,
    alphas: Sequence[float],
    unit_cols: Sequence[str],
    n_boot: int = 2000,
    random_state: int = 0,
    label: str | None = None,
) -> FrontierResult:
    """Score rule families against the fixed-field rule at each risk level in ``alphas``.

    ``criterion`` is the name of an ``OutcomeCriterion`` assessed in
    ``replay``. ``unit_cols`` are the group columns of the independent unit
    (``[]`` treats all runs as one unit, so there is no interval). Families
    come from ``family`` or ``sweep`` on this replay, or on replays with the
    same streams. ``label`` is the plain wording of the criterion.
    """
    if not isinstance(replay, ReplayResult):
        raise TypeError("replay must be a ReplayResult")
    alphas = [float(a) for a in alphas]
    if not alphas or not all(0 < a < 1 for a in alphas):
        raise ValueError("alphas must be risk levels between 0 and 1")
    missing = [c for c in unit_cols if c not in replay.group_cols]
    if missing:
        raise ValueError(f"unit columns {missing} are not group columns of the replay")
    if isinstance(n_boot, bool) or not isinstance(n_boot, int) or n_boot < 1:
        raise ValueError("n_boot must be a positive integer")
    keys = _run_keys(replay)
    statuses = _statuses(replay, criterion)
    streams = {
        tuple(row[k] for k in keys): (int(row["stream_fovs"]), row["order"])
        for _, row in replay.streams.iterrows()
    }
    rows = _baseline_runs(keys, statuses, streams)
    names = set()
    for decisions in families:
        own = decisions["family"].unique()
        if BASELINE in own:
            raise ValueError(f"{BASELINE!r} is the baseline's name; name families otherwise")
        overlap = names & set(own)
        if overlap:
            raise ValueError(f"family names used twice: {sorted(overlap)}")
        names |= set(own)
        rows += _family_runs(decisions, keys, statuses, streams)
    runs = pd.DataFrame(rows)
    curve = _curve(runs)
    realized = adequacy_points(replay).realized
    realized = realized[realized["criterion"] == criterion]
    # Not reached within the stream counts as one field beyond it.
    needed = realized["sustained_realized"].astype("Float64")
    fill = realized["stream_fovs"].astype(float) + 1
    realized = realized.assign(needed=needed.fillna(fill).astype(float))[[*keys, "needed"]]
    rng = np.random.default_rng(random_state)
    boot = _bootstrap(runs, unit_cols, alphas, n_boot, rng)
    baseline_curve = curve[curve["family"] == BASELINE]
    scores = []
    for alpha in alphas:
        base = _needed(baseline_curve, alpha)
        for name in [BASELINE, *sorted(names)]:
            own_curve = curve[curve["family"] == name]
            chosen = _needed(own_curve, alpha)
            row = {
                "family": name,
                "alpha": alpha,
                "setting": None if chosen is None else chosen["setting"],
                "effort": None if chosen is None else float(chosen["effort"]),
                "risk": None if chosen is None else float(chosen["risk"]),
                "baseline_setting": None if base is None else base["setting"],
                "baseline_effort": None if base is None else float(base["effort"]),
            }
            reached = chosen is not None and base is not None
            row["fields_saved"] = float(base["effort"] - chosen["effort"]) if reached else None
            draws = boot.get((name, alpha))
            # An interval only around a score that exists.
            if reached and draws is not None and np.isfinite(draws).any():
                finite = draws[np.isfinite(draws)]
                row["saved_low"] = float(np.quantile(finite, 0.025))
                row["saved_high"] = float(np.quantile(finite, 0.975))
                row["saved_reached"] = float(np.isfinite(draws).mean())
            else:
                row["saved_low"] = row["saved_high"] = row["saved_reached"] = None
            row["baseline_risk_at_same_effort"] = (
                None if chosen is None else _interpolate_baseline(baseline_curve, chosen["effort"])
            )
            if chosen is not None and name != BASELINE:
                own_runs = runs[(runs["family"] == name) & (runs["setting"] == chosen["setting"])]
                row["adaptivity"] = _adaptivity(own_runs[[*keys, "effort"]], realized, keys)
            else:
                row["adaptivity"] = None
            scores.append(row)
    units = int(_unit_ids(runs, unit_cols).nunique())
    settings = {
        **dict(replay.settings),
        "criterion": criterion,
        "label": label or replay.settings.get("labels", {}).get(criterion, criterion),
        "alphas": alphas,
        "unit_cols": list(unit_cols),
        "units": units,
        "n_boot": n_boot,
        "random_state_frontier": random_state,
    }
    return FrontierResult(curve, pd.DataFrame(scores), runs, settings)


def transfer(development: FrontierResult, held_out: FrontierResult, alpha: float) -> pd.DataFrame:
    """Settings chosen on ``development`` to meet ``alpha``, and their risk on ``held_out``.

    One row per family: the chosen setting and its development risk and
    effort, then the held-out risk and effort of the same setting (``None``
    when the held-out dataset did not run it). For the fixed rule the setting
    is the number of fields.
    """
    rows = []
    scores = development.scores[np.isclose(development.scores["alpha"], alpha)]
    if scores.empty:
        raise ValueError(f"alpha {alpha} was not scored on the development dataset")
    for _, row in scores.iterrows():
        entry = {
            "family": row["family"],
            "alpha": float(alpha),
            "setting": row["setting"],
            "development_risk": row["risk"],
            "development_effort": row["effort"],
            "held_out_risk": None,
            "held_out_effort": None,
            "held_out_judged": None,
        }
        if row["setting"] is not None and not pd.isna(row["setting"]):
            curve = held_out.curve
            match = curve[(curve["family"] == row["family"]) & (curve["setting"] == row["setting"])]
            if len(match):
                entry["held_out_risk"] = match["risk"].iloc[0]
                entry["held_out_effort"] = float(match["effort"].iloc[0])
                entry["held_out_judged"] = int(match["judged"].iloc[0])
        rows.append(entry)
    return pd.DataFrame(rows)


def _fields(value: float) -> str:
    value = round(float(value), 1)
    if value == 1:
        return "1 field"
    return f"{value:.0f} fields" if value.is_integer() else f"{value:.1f} fields"


def _number(value: float) -> str:
    """One decimal, without a negative zero."""
    value = round(float(value), 1)
    return f"{0.0 if value == 0 else value:.1f}"


def frontier_summary(result: FrontierResult) -> str:
    """ "At most 5% too-early stops: a fixed count needs ..., this rule saves ..." in sentences."""
    label = result.settings.get("label", result.settings.get("criterion"))
    units = result.settings.get("units", 0)
    blocks = []
    for alpha, rows in result.scores.groupby("alpha", sort=True):
        lines = [f"For {label}, stopping too early in at most {percent(alpha)} of acquisitions:"]
        base = rows[rows["family"] == BASELINE].iloc[0]
        if base["effort"] is None or pd.isna(base["effort"]):
            lines.append(
                "  a fixed number of fields never got there within the fields available, "
                "so no rule can be scored against it at this level."
            )
            blocks.append("\n".join(lines))
            continue
        lines.append(
            f"  a fixed number of fields needed {_fields(int(base['setting']))} "
            f"(risk {percent(base['risk'])})."
        )
        for row in rows[rows["family"] != BASELINE].itertuples(index=False):
            if row.effort is None or pd.isna(row.effort):
                lines.append(f"  {row.family}: never got there at any setting tried.")
                continue
            saved = row.fields_saved
            verdict = (
                f"saves {_number(saved)} fields"
                if saved > 0.05
                else f"costs {_number(-saved)} more fields"
                if saved < -0.05
                else "does as well as the fixed count, no better"
            )
            interval = (
                ""
                if row.saved_low is None or pd.isna(row.saved_low)
                else f", {percent(0.95)} interval {_number(row.saved_low)} to "
                f"{_number(row.saved_high)}"
            )
            adaptive = (
                ""
                if row.adaptivity is None or pd.isna(row.adaptivity)
                else f"; its stops follow the fields each acquisition needed with Spearman "
                f"{row.adaptivity:.2f}"
            )
            lines.append(
                f"  {row.family} at {row.setting:g}: {_fields(row.effort)} on average "
                f"(risk {percent(row.risk)}); it {verdict}{interval}{adaptive}."
            )
        lines.append(
            f"  Intervals resample the {units} independent units; hold-outs and orders are "
            "not independent experiments. Risk is judged on held-back fields."
        )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


__all__ = [
    "BASELINE",
    "FrontierResult",
    "family",
    "frontier",
    "frontier_summary",
    "sweep",
    "transfer",
]

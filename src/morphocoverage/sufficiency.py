"""Here is an analysed dataset: did we sample enough, and would one more FOV show something new?

``sufficiency_report`` answers two questions for each analysis group
(condition x repeat by default) of a finished acquisition, given the
observation table and the FOV table of every acquired position:

1. **Did we sample enough?** (``enough``) For each criterion (a strategy with
   its thresholds): is it met by all the acquired FOVs, after how many FOVs
   was it first met when the FOVs are replayed in the declared acquisition
   orders, and so how many FOVs were imaged beyond that point? A criterion not
   met by the acquired FOVs says more FOVs were needed by that criterion.
2. **Would one more FOV show something new?** (``next_fov``, when a
   ``novelty`` definition is given) The estimated chance that one more
   representative FOV shows a phenotype not seen in the other FOVs, with an
   upper confidence bound, and how many more FOVs would be needed before the
   bound could fall below ``novelty_target``. ``new_observations`` lists the
   observations counted as new so they can be inspected in the images, and
   ``calibration`` (optional) checks the prediction against this dataset by
   shuffling (``morphocoverage.calibration``).

``summary()`` puts both answers into sentences, one block per group.

Reading it:

- Every answer holds only for the stated criterion, thresholds and
  definition of new. Thresholds without a published meaning are the user's
  choice, so compare several (pass the same strategy with different
  thresholds) rather than read one verdict as a fact.
- "New" is new at the chosen resolution in the chosen features: at least
  ``min_novel_observations`` observations with fewer than ``min_other_fovs``
  other FOVs within ``epsilon``. Look at ``new_observations`` to see whether
  that is biology (a rare cell state) or technique (debris, segmentation
  errors, a focus problem). Too fine a resolution calls almost every FOV new.
- With few FOVs the bound stays high whatever the data look like: with no
  new FOV among ``k``, the 95% bound is ``1 - 0.05 ** (1 / k)``, so claiming
  "below 10%" needs at least 29 FOVs.
- Random orders treat FOVs as exchangeable; the recorded order keeps drift
  and neighbour effects. Declare both to see the difference.
- This is self-assessment of one dataset. It says what each criterion
  concludes, not that the criterion is right; that needs held-out deep data
  (``docs/benchmarking.md``). The calibration check is the part that tests a
  prediction against this dataset itself.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .benchmark import (
    AcquisitionOrder,
    BenchmarkDataset,
    BenchmarkRegistry,
    records_to_frame,
    run_benchmark,
)
from .bounds import extra_trials_for_bound
from .calibration import calibrate_next_fov_novelty
from .evaluator import iter_analysis_groups
from .evidence import Availability, Direction
from .framework import run_strategies
from .objectives import ObjectiveConfig, SamplingObjective
from .results import StrategyResult
from .schema import fov_positions, validate_input_tables
from .strategies.base import SamplingStrategy
from .strategies.support_novelty import (
    FOV_QUANTITY,
    UPPER_QUANTITY,
    SupportNoveltyEstimator,
    SupportNoveltyStrategy,
)

_ROW_ORDER = "_report_row_order"
_POPULATION = ObjectiveConfig(SamplingObjective.POPULATION)


@dataclass(frozen=True)
class CalibrationSettings:
    """How to check the new-phenotype prediction by shuffling; see ``morphocoverage.calibration``.

    ``fov_counts`` are the numbers of FOVs to predict from, ``draws`` the
    random subsets per count and group, ``random_state`` the seed.
    """

    fov_counts: tuple[int, ...]
    draws: int
    random_state: int


@dataclass(frozen=True)
class SufficiencyReport:
    """The answers, and the detailed tables behind them; see the module docstring.

    - ``enough``: did we sample enough? One row per group and criterion.
    - ``next_fov``: would one more FOV show something new? One row per group
      (empty without a ``novelty`` definition).
    - ``new_observations``: the observations counted as new, with their FOV
      and metadata columns, for inspection.
    - ``calibration``: predicted versus actual chance of a new phenotype, per
      group and number of FOVs (empty unless requested).
    - ``final``: every criterion's decision and evidence (``q.<name>``) on the
      full acquisition. ``stops``: the stop-FOV distribution over orders.
    """

    enough: pd.DataFrame
    next_fov: pd.DataFrame
    new_observations: pd.DataFrame
    calibration: pd.DataFrame
    final: pd.DataFrame
    stops: pd.DataFrame
    group_cols: tuple[str, ...] = ()
    novelty_definition: str | None = None

    def summary(self) -> str:
        """Both answers in plain sentences, one block per analysis group."""
        blocks = []
        keys = list(self.group_cols)
        tables = [table[keys] for table in (self.enough, self.next_fov) if not table.empty]
        groups = pd.concat(tables).drop_duplicates() if tables else pd.DataFrame()
        # Without group columns there is one pooled group, which has no key.
        rows = list(groups.itertuples(index=False, name=None)) if keys else [()] * bool(tables)
        for values in rows:
            group = dict(zip(keys, values, strict=True))
            label = ", ".join(f"{k}={v}" for k, v in group.items()) or "all FOVs (pooled)"
            lines = [f"{label}:"]
            criteria = _rows_of(self.enough, group)
            if len(criteria):
                lines.append("  Did we sample enough?")
                lines += [f"  - {answer}" for answer in criteria["answer"]]
            outlook = _rows_of(self.next_fov, group)
            if len(outlook):
                lines.append("  Would one more FOV show something new?")
                lines += [f"  - {line}" for line in outlook["answer"].iloc[0].split("\n")]
            blocks.append("\n".join(lines))
        if self.novelty_definition:
            blocks.append(f"'New' means {self.novelty_definition}.")
        return "\n\n".join(blocks)


def _rows_of(table: pd.DataFrame, group: Mapping[str, object]) -> pd.DataFrame:
    """The rows of ``table`` belonging to one analysis group."""
    mask = pd.Series(True, index=table.index)
    for column, value in group.items():
        mask &= table[column].eq(value)
    return table.loc[mask]


def _policy_text(policy: Mapping, result: StrategyResult) -> str:
    """One decision policy as a readable rule, such as ``averaging_cv <= 0.05``."""
    parameters = policy["parameters"]
    quantity = result.evidence.quantities.get(parameters.get("quantity"))
    if quantity is not None and set(parameters) == {"quantity", "threshold"}:
        better = "<=" if quantity.direction is Direction.LOWER_IS_BETTER else ">="
        return f"{parameters['quantity']} {better} {parameters['threshold']:g}"
    if {"effort", "target"} <= set(parameters):
        return f"{parameters['effort']} >= {parameters['target']:g}"
    return f"{policy['name']} {json.dumps(parameters, sort_keys=True)}"


def _policies(result: StrategyResult) -> str:
    """The decision policies and their settings, compactly."""
    policies = result.configuration.get("policies", [])
    return "; ".join(_policy_text(policy, result) for policy in policies)


def _final_row(result: StrategyResult, group_cols: Sequence[str]) -> dict:
    """One row of the full-acquisition table."""
    row = {column: result.group.get(column) for column in group_cols}
    row.update(
        {
            "strategy": result.strategy,
            "config_id": result.config_id,
            "policies": _policies(result),
            "n_fovs": result.n_fovs,
            "n_observations": result.n_observations,
            "verdict": None if result.decision is None else result.decision.value,
            "undefined_reason": result.evidence.diagnostics.get("undefined_reason"),
        }
    )
    for name, quantity in result.evidence.quantities.items():
        available = quantity.availability is Availability.AVAILABLE
        row[f"q.{name}"] = quantity.value if available else None
    return row


def _stop_summary(records: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Stop-FOV distribution per group and configuration."""
    rows = []
    for key, runs in records.groupby(keys, sort=False, dropna=False):
        stopped = runs[runs["stopped"]]
        stop = stopped["n_fovs"].to_numpy(dtype=float)
        spare = (stopped["total_fovs"] - stopped["n_fovs"]).to_numpy(dtype=float)
        row = dict(zip(keys, key if isinstance(key, tuple) else (key,), strict=True))
        row.update(
            {
                "acquired_fovs": int(runs["total_fovs"].iloc[0]),
                "orders": len(runs),
                "never_stopped": int((~runs["stopped"]).sum()),
                "stop_fov_median": float(np.median(stop)) if len(stop) else None,
                "stop_fov_p10": float(np.quantile(stop, 0.1)) if len(stop) else None,
                "stop_fov_p90": float(np.quantile(stop, 0.9)) if len(stop) else None,
                "spare_fovs_median": float(np.median(spare)) if len(spare) else None,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _enough_answer(final: Mapping, stops: Mapping) -> str:
    """One criterion's answer to 'did we sample enough?' as a sentence."""
    criterion = f"{final['strategy']} ({final['policies']})"
    acquired, orders = final["n_fovs"], stops["orders"]
    if pd.isna(final["verdict"]):
        return f"{criterion}: undefined for these data ({final['undefined_reason']})."
    if final["verdict"] == "continue":
        head = f"{criterion}: no, not met by the {acquired} acquired FOVs"
    else:
        head = f"{criterion}: yes, met by the {acquired} acquired FOVs"
    if pd.isna(stops["stop_fov_median"]):
        return f"{head}; not met at any point in {orders} replayed orders."
    first = (
        f"first met after {stops['stop_fov_median']:g} FOVs "
        f"(10-90%: {stops['stop_fov_p10']:g}-{stops['stop_fov_p90']:g}) over {orders} orders"
    )
    never = stops["never_stopped"]
    missing = f", never in {never}" if never else ""
    return f"{head}; {first}{missing}."


def _enough_table(final: pd.DataFrame, stops: pd.DataFrame, group_cols: list[str]):
    """Question 1, one row per group and criterion."""
    keys = [*group_cols, "config_id"]
    merged = final.merge(stops, on=[*keys, "strategy"], how="left", validate="one_to_one")
    rows = []
    for _, row in merged.iterrows():
        verdict = row["verdict"]
        enough = {"stop": "yes", "continue": "no"}.get(verdict, "undefined")
        rows.append(
            {
                **{column: row[column] for column in group_cols},
                "criterion": row["strategy"],
                "settings": row["policies"],
                "config_id": row["config_id"],
                "fovs_acquired": row["n_fovs"],
                "sampled_enough": enough,
                "enough_after_fovs": row["stop_fov_median"],
                "enough_after_p10": row["stop_fov_p10"],
                "enough_after_p90": row["stop_fov_p90"],
                "fovs_beyond_enough": row["spare_fovs_median"],
                "orders": row["orders"],
                "orders_never_enough": row["never_stopped"],
                "answer": _enough_answer(row, row),
            }
        )
    return pd.DataFrame(rows)


def novelty_definition(novelty: SupportNoveltyEstimator) -> str:
    """What counts as a FOV showing something new, in words."""
    m, others = novelty.min_novel_observations, novelty.min_other_fovs
    space = "raw feature units" if novelty.transform is None else novelty.transform.name
    return (
        f"at least {m} observation{'s' if m > 1 else ''} with fewer than {others} other "
        f"FOV{'s' if others > 1 else ''} within distance {novelty.epsilon:g} ({space})"
    )


def _percent(value: float) -> str:
    """A probability as a short percentage."""
    return f"{100 * value:.0f}%" if value >= 0.01 or value == 0 else f"{100 * value:.1g}%"


def _more_fovs(count: int) -> str:
    """``1 more FOV``, ``3 more FOVs``."""
    return f"{count} more FOV" if count == 1 else f"{count} more FOVs"


def _next_fov_answer(row: Mapping, target: float | None, calibrated: Mapping | None) -> str:
    """Question 2 for one group, as lines of text."""
    if pd.isna(row["chance_new"]):
        return f"Not estimable: {row['reason']}."
    confidence = _percent(row["confidence"])
    chance = (
        f"Chance: {_percent(row['chance_new'])} (at most {_percent(row['chance_new_at_most'])} "
        f"with {confidence} confidence); {row['fovs_with_new']} of {row['fovs_imaged']} FOVs "
        "showed something the others did not."
    )
    lines = [chance]
    if target is not None:
        goal = f"To claim below {_percent(target)} with {confidence} confidence"
        if row["chance_new_at_most"] <= target:
            lines.append(f"{goal}: already reached.")
        else:
            best = row["extra_fovs_if_nothing_new"]
            rate = row["extra_fovs_at_current_rate"]
            at_rate = (
                "never at the current rate of new FOVs"
                if pd.isna(rate)
                else f"{_more_fovs(rate)} if new FOVs keep appearing at the current rate"
            )
            lines.append(
                f"{goal}: at least {_more_fovs(best)} if none shows anything new; {at_rate}."
            )
    if calibrated is not None:
        predicted, actual = _percent(calibrated["predicted"]), _percent(calibrated["actual"])
        held = _percent(calibrated["bound_held"])
        lines.append(
            f"Check on this dataset: predicting from {int(calibrated['fovs_imaged'])} random FOVs, "
            f"predicted {predicted}, actual {actual}; "
            f"the bound held in {held} of {int(calibrated['draws'])} draws."
        )
    return "\n".join(lines)


def _next_fov_tables(data, novelty, target):
    """Question 2 per group, and the observations counted as new."""
    rows, flagged = [], []
    features = set(data.feature_cols)
    for group, group_data in iter_analysis_groups(data):
        evidence = novelty.estimate(group_data, objective=_POPULATION)
        chance, upper = evidence.quantities[FOV_QUANTITY], evidence.quantities[UPPER_QUANTITY]
        k = len(group_data.fovs)
        row = {**group, "fovs_imaged": k, "confidence": novelty.confidence}
        if chance.availability is not Availability.AVAILABLE:
            row.update(
                dict.fromkeys(("fovs_with_new", "chance_new", "chance_new_at_most"), None),
                extra_fovs_if_nothing_new=None,
                extra_fovs_at_current_rate=None,
                reason=chance.reason,
            )
            rows.append(row)
            continue
        new_fovs = int(evidence.diagnostics["novel_fovs"])
        row.update(
            {
                "fovs_with_new": new_fovs,
                "chance_new": chance.value,
                "chance_new_at_most": upper.value,
                "reason": None,
            }
        )
        for column, keep_rate in (
            ("extra_fovs_if_nothing_new", False),
            ("extra_fovs_at_current_rate", True),
        ):
            row[column] = (
                None
                if target is None
                else extra_trials_for_bound(
                    new_fovs, k, target, novelty.confidence, keep_rate=keep_rate
                )
            )
        rows.append(row)
        flags = novelty.novel_observation_flags(group_data)
        if flags is not None and flags.any():
            positions = fov_positions(group_data, group_data.fovs)
            counts = np.bincount(positions, weights=flags, minlength=k)
            # Only observations of FOVs that count as new, so one outlier stays out
            # when several new observations are required.
            counted = flags & (counts[positions] >= novelty.min_novel_observations)
            keep = [c for c in group_data.objects.columns if c not in features]
            flagged.append(group_data.objects.loc[counted, keep])
    new_observations = pd.concat(flagged, ignore_index=True) if flagged else pd.DataFrame()
    table = pd.DataFrame(rows)
    for column in ("fovs_with_new", "extra_fovs_if_nothing_new", "extra_fovs_at_current_rate"):
        table[column] = table[column].astype("Int64")  # counts, missing as <NA>
    return table, new_observations


def _check_report_inputs(strategies, orders, novelty, target, calibration):
    """Validate the combination of questions asked."""
    if not strategies and novelty is None:
        raise ValueError("give at least one strategy or a novelty definition")
    if not orders or not all(isinstance(order, AcquisitionOrder) for order in orders):
        raise ValueError("orders must be a non-empty sequence of AcquisitionOrder")
    if novelty is not None and not isinstance(novelty, SupportNoveltyEstimator):
        raise TypeError("novelty must be a SupportNoveltyEstimator")
    if target is not None:
        if novelty is None:
            raise ValueError("novelty_target needs a novelty definition")
        if isinstance(target, bool) or not 0 < target < 1:
            raise ValueError("novelty_target must be a probability between 0 and 1")
    if calibration is not None:
        if novelty is None:
            raise ValueError("calibration needs a novelty definition")
        if not isinstance(calibration, CalibrationSettings):
            raise TypeError("calibration must be CalibrationSettings")


def sufficiency_report(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    strategies: Iterable[SamplingStrategy],
    objective: ObjectiveConfig,
    orders: Sequence[AcquisitionOrder],
    novelty: SupportNoveltyEstimator | None = None,
    novelty_target: float | None = None,
    calibration: CalibrationSettings | None = None,
    acquisition_order_col: str | None = None,
    fov_col: str | None = None,
    group_cols: Sequence[str] | None = None,
    observation_unit: str | None = None,
) -> SufficiencyReport:
    """Did we sample enough, and would one more FOV show something new? Per analysis group.

    ``fovs`` must list every acquired FOV, including FOVs without
    observations. ``strategies`` are the criteria for "enough". ``orders``
    are the replay orders, for example ``[AcquisitionOrder.recorded()] +
    [AcquisitionOrder.permuted(s) for s in range(20)]``; the recorded order
    needs ``acquisition_order_col``, and without that column only permuted
    orders are allowed, since the table's row order is not an acquisition
    record.

    ``novelty`` defines what counts as new and turns on the second question.
    With ``novelty_target``, the report also says how many more FOVs are
    needed before the chance of something new can be claimed below the target,
    and adds that claim as a criterion for "enough" (a ``SupportNoveltyStrategy``
    on the upper bound). ``calibration`` checks the prediction on this
    dataset by shuffling. Other arguments are those of ``run_strategies``.
    """
    strategies, orders = list(strategies), list(orders)
    _check_report_inputs(strategies, orders, novelty, novelty_target, calibration)
    if novelty is not None and novelty_target is not None:
        strategies.append(
            SupportNoveltyStrategy(
                novelty.epsilon,
                novelty_target,
                novelty.min_other_fovs,
                novelty.transform,
                novelty.min_fovs,
                novelty.min_novel_observations,
                UPPER_QUANTITY,
                novelty.confidence,
            )
        )
    if acquisition_order_col is None:
        if any(order.kind == "recorded" for order in orders):
            raise ValueError("the recorded order needs acquisition_order_col")
        if _ROW_ORDER in fovs.columns:
            raise ValueError(f"the FOV table already has a column {_ROW_ORDER!r}")
        # Only a tie-breaking base for permutations, never replayed as recorded.
        fovs = fovs.assign(**{_ROW_ORDER: np.arange(len(fovs))})
        acquisition_order_col = _ROW_ORDER

    columns = {
        "feature_cols": list(feature_cols),
        "fov_col": fov_col,
        "group_cols": None if group_cols is None else list(group_cols),
        "observation_unit": observation_unit,
    }
    data = validate_input_tables(objects, fovs=fovs, **columns)
    resolved_groups = list(data.group_cols)

    final, stops, enough = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if strategies:
        final, stops = _criteria_tables(
            objects,
            fovs,
            strategies,
            objective,
            orders,
            acquisition_order_col,
            columns,
            resolved_groups,
        )
        enough = _enough_table(final, stops, resolved_groups)

    next_fov, new_observations, calibrated = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if novelty is not None:
        next_fov, new_observations = _next_fov_tables(data, novelty, novelty_target)
        if calibration is not None:
            calibrated = calibrate_next_fov_novelty(
                objects,
                fovs,
                novelty=novelty,
                fov_counts=calibration.fov_counts,
                draws=calibration.draws,
                random_state=calibration.random_state,
                **columns,
            )
        next_fov["answer"] = [
            _next_fov_answer(
                row, novelty_target, _largest_calibrated(calibrated, row, resolved_groups)
            )
            for _, row in next_fov.iterrows()
        ]
    return SufficiencyReport(
        enough=enough,
        next_fov=next_fov,
        new_observations=new_observations,
        calibration=calibrated,
        final=final,
        stops=stops,
        group_cols=tuple(resolved_groups),
        novelty_definition=None if novelty is None else novelty_definition(novelty),
    )


def _largest_calibrated(calibrated: pd.DataFrame, row: Mapping, group_cols: list[str]):
    """The group's calibration row with the most FOVs and usable draws, or None."""
    if calibrated.empty:
        return None
    rows = _rows_of(calibrated, {c: row[c] for c in group_cols})
    rows = rows[rows["draws"] > 0]
    return None if rows.empty else rows.sort_values("fovs_imaged").iloc[-1]


def _criteria_tables(
    objects, fovs, strategies, objective, orders, acquisition_order_col, columns, group_cols
):
    """Decisions on the full acquisition, and the stop-FOV distribution over orders."""
    results = run_strategies(
        objects, fovs=fovs, strategies=strategies, objective=objective, **columns
    )
    final = pd.DataFrame([_final_row(result, group_cols) for result in results])
    dataset = BenchmarkDataset(
        name="sufficiency_report",
        role="development",
        load=lambda: (objects, fovs),
        feature_cols=tuple(columns["feature_cols"]),
        acquisition_order_col=acquisition_order_col,
        fov_col=columns["fov_col"],
        group_cols=None if columns["group_cols"] is None else tuple(columns["group_cols"]),
        observation_unit=columns["observation_unit"],
    )
    records = records_to_frame(
        run_benchmark(
            BenchmarkRegistry([dataset]),
            dataset.name,
            strategies=strategies,
            metrics=[],
            objective=objective,
            orders=orders,
        )
    )
    group_keys = [f"group.{column}" for column in group_cols]
    stops = _stop_summary(records, [*group_keys, "strategy", "config_id"])
    stops = stops.rename(columns={f"group.{c}": c for c in group_cols})
    return final, stops


__all__ = [
    "CalibrationSettings",
    "SufficiencyReport",
    "novelty_definition",
    "sufficiency_report",
]

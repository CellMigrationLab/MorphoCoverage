"""Benchmark protocol: declared dataset roles, acquisition orders and one record per run.

Nothing here selects a winner. It fixes how runs are recorded so that
strategies can be compared on strategy-independent outcome metrics
(``outcomes.py``), with tuning kept apart from the data used to claim
success:

- **Dataset roles.** Every dataset is registered as ``development`` or
  ``held_out``. Configurations may be tuned only on development data; a run
  declares what it was tuned on (``tuned_on``), which must be development
  datasets. A run on a dataset it was tuned on is recorded as in-sample.
  Claims of performance belong to held-out or out-of-sample runs.
  ``leave_one_dataset_out`` gives folds when data are scarce.
- **Acquisition order.** The recorded order, or a random permutation of whole
  FOVs with its seed, drawn per analysis group (``permuted_fovs``), so a
  group's order depends on its own FOVs only. A permutation treats the FOVs of
  a group as exchangeable, which the recorded order need not be (stage drift,
  neighbouring fields); both are recorded so the difference stays visible.
- **Stop.** A strategy stops at the first prefix whose decision is ``stop``
  (with ``stop_on="rule"``, whose ``rule_decision`` is ``stop``: the rule's own
  verdict, whatever the objective's other components say). The record keeps
  ``stop_on``, and ``final_decision`` is the decision it read, so a stopped
  run's final decision is ``stop``. A run that never stops is recorded with
  ``stopped=False``, its final decision, and outcomes and effort at the full
  acquisition.
- **Reference.** Outcomes compare the stop with the full group by default.
  That reference contains the prefix, so errors shrink to zero as the prefix
  grows. With ``reservoir=ReservoirSplit(...)`` each group is split first:
  strategies replay the stream only and outcomes are scored against the
  held-out reservoir (``reservoir.py``). The stream is a thinned acquisition:
  its recorded order skips the held-out FOVs, so neighbouring stream FOVs are
  further apart than at the microscope, and a recorded-order replay is not an
  exact replay of the acquisition.
- **Record.** One ``BenchmarkRecord`` per dataset, analysis group, order and
  strategy, with effort at the stop (FOVs, objects, area), outcomes,
  estimator compute times, and whole-decision latency (median and maximum
  per step, and the total up to the stop; see ``results.Timing``).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from ._areas import area_problem
from ._checks import check_seed, is_integer
from ._parallel import worker_pool
from ._seeds import group_rng
from .data_check import validate_input_tables
from .evaluator import iter_analysis_groups
from .objectives import ObjectiveConfig
from .outcomes import OutcomeMetric
from .reservoir import ReservoirSplit
from .results import Decision, StrategyResult
from .schema import ValidatedInput, normalize_observation_unit
from .sequential import SequentialResult, _Acquisition, _evaluate_prefixes
from .strategies.base import SamplingStrategy

_PERMUTED_ORDER_COL = "_benchmark_order"


class DatasetRole(str, Enum):
    """Whether a dataset may be used for tuning."""

    DEVELOPMENT = "development"
    HELD_OUT = "held_out"


@dataclass(frozen=True)
class BenchmarkDataset:
    """A deeply sampled acquisition declared for benchmarking.

    ``load`` returns ``(objects, fovs)``; it is called for each run so that
    datasets are only read when used. The column arguments are passed to the
    framework's validation unchanged. ``area_col``, if given, is an FOV-table
    column of imaged area, summed into the effort at the stop.
    """

    name: str
    role: DatasetRole
    load: Callable[[], tuple[pd.DataFrame, pd.DataFrame]] = field(compare=False)
    feature_cols: tuple[str, ...]
    acquisition_order_col: str
    fov_col: str | None = None
    group_cols: tuple[str, ...] | None = None
    area_col: str | None = None
    observation_unit: str | None = None
    description: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("a dataset needs a non-empty name")
        object.__setattr__(self, "role", DatasetRole(self.role))
        if isinstance(self.feature_cols, str):
            raise TypeError("feature_cols must be a sequence of column names, not a string")
        object.__setattr__(self, "feature_cols", tuple(self.feature_cols))
        if self.group_cols is not None:
            if isinstance(self.group_cols, str):
                raise TypeError("group_cols must be a sequence of column names, not a string")
            object.__setattr__(self, "group_cols", tuple(self.group_cols))
        if not callable(self.load):
            raise TypeError("load must be a callable returning (objects, fovs)")
        if self.observation_unit is not None:
            object.__setattr__(
                self, "observation_unit", normalize_observation_unit(self.observation_unit)
            )


class BenchmarkRegistry:
    """The datasets of one benchmark, each with a declared role."""

    def __init__(self, datasets: Iterable[BenchmarkDataset] = ()) -> None:
        self._datasets: dict[str, BenchmarkDataset] = {}
        for dataset in datasets:
            self.register(dataset)

    def register(self, dataset: BenchmarkDataset) -> None:
        """Add a dataset; names are unique and a role cannot be changed later."""
        if not isinstance(dataset, BenchmarkDataset):
            raise TypeError("only BenchmarkDataset instances can be registered")
        if dataset.name in self._datasets:
            raise ValueError(f"dataset {dataset.name!r} is already registered")
        self._datasets[dataset.name] = dataset

    def __getitem__(self, name: str) -> BenchmarkDataset:
        try:
            return self._datasets[name]
        except KeyError:
            raise KeyError(
                f"dataset {name!r} is not registered; known: {sorted(self._datasets)}"
            ) from None

    def __contains__(self, name: object) -> bool:
        return name in self._datasets

    def names(self, role: DatasetRole | str | None = None) -> list[str]:
        """Registered names, optionally only those with ``role``, in registration order."""
        if role is None:
            return list(self._datasets)
        role = DatasetRole(role)
        return [name for name, d in self._datasets.items() if d.role is role]

    def check_tuning(self, tuned_on: Iterable[str]) -> tuple[str, ...]:
        """Validate that every tuning dataset is registered for development."""
        if isinstance(tuned_on, str):
            raise TypeError("tuned_on must be a sequence of dataset names, not a string")
        tuned = tuple(tuned_on)
        for name in tuned:
            if self[name].role is not DatasetRole.DEVELOPMENT:
                raise ValueError(
                    f"dataset {name!r} is held out; configurations must not be tuned on it"
                )
        return tuned


def leave_one_dataset_out(
    registry: BenchmarkRegistry,
) -> Iterator[tuple[str, tuple[str, ...]]]:
    """Folds over development datasets: (evaluated dataset, datasets to tune on)."""
    names = registry.names(DatasetRole.DEVELOPMENT)
    for name in names:
        yield name, tuple(other for other in names if other != name)


@dataclass(frozen=True)
class AcquisitionOrder:
    """The recorded acquisition order, a seeded random permutation of whole FOVs, or a
    seeded spatially dispersed order (greedy farthest point on stage coordinates).

    A dispersed order needs stage coordinates, so it is replayed with
    ``morphocoverage.retrospective.replay_acquisitions(stage_cols=...)``. A
    seed is a non-negative integer (numpy integers accepted); a negative one
    is refused here rather than later by numpy.
    """

    kind: str
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.kind == "recorded":
            if self.seed is not None:
                raise ValueError("the recorded order takes no seed")
        elif self.kind in ("permuted", "dispersed"):
            if not is_integer(self.seed):
                raise TypeError(f"a {self.kind} order needs an integer seed")
            object.__setattr__(self, "seed", check_seed(self.seed, f"a {self.kind} order's seed"))
        else:
            raise ValueError("kind must be 'recorded', 'permuted' or 'dispersed'")

    @classmethod
    def recorded(cls) -> AcquisitionOrder:
        """The order in the dataset's acquisition-order column."""
        return cls("recorded")

    @classmethod
    def permuted(cls, seed: int) -> AcquisitionOrder:
        """A uniformly random order of whole FOVs, reproducible from ``seed``."""
        return cls("permuted", seed)

    @classmethod
    def dispersed(cls, seed: int) -> AcquisitionOrder:
        """Greedy farthest-point order on stage coordinates from a start chosen by ``seed``."""
        return cls("dispersed", seed)

    def apply(
        self, groups: Iterable[tuple[Mapping[str, Any], ValidatedInput]], order_col: str
    ) -> tuple[pd.DataFrame, str]:
        """The FOV rows of the analysis ``groups`` with the order to replay, and its column.

        A permuted order is drawn group by group with ``permuted_fovs`` (seeded
        from the order's seed and the group's identity), so a group's order
        never depends on the other groups or on the row order of the table. The
        recorded order keeps ``order_col``.
        """
        if self.kind == "dispersed":
            raise ValueError(
                "a dispersed order needs stage coordinates; replay it with "
                "morphocoverage.retrospective.replay_acquisitions(stage_cols=...)"
            )
        groups = list(groups)
        tables = [group_data.fovs for _, group_data in groups]
        if self.kind == "recorded":
            return pd.concat(tables, ignore_index=True), order_col
        if any(_PERMUTED_ORDER_COL in table.columns for table in tables):
            raise ValueError(f"the FOV table already has a column {_PERMUTED_ORDER_COL!r}")
        permuted = [
            permuted_fovs(group_data.fovs, group_data.fov_key, self.seed, group).assign(
                **{_PERMUTED_ORDER_COL: np.arange(len(group_data.fovs))}
            )
            for group, group_data in groups
        ]
        return pd.concat(permuted, ignore_index=True), _PERMUTED_ORDER_COL


def permuted_fovs(
    fovs: pd.DataFrame,
    fov_key: Sequence[str],
    random_state: int,
    group: Mapping[str, Any],
    *more: int,
) -> pd.DataFrame:
    """One analysis group's FOV rows in a seeded uniformly random order of whole FOVs.

    The one permutation behind every permuted acquisition order: the benchmark's
    (``AcquisitionOrder.apply``, seeded from the order's seed) and the
    retrospective replay's (seeded from its ``random_state``, the order's seed
    and the split). The rows are put in FOV-key order first and the generator
    is seeded with ``group_rng(random_state, group, *more)``, so the order
    depends only on which FOVs the group holds, never on other groups or on the
    table's row order. Shared-private: used by ``retrospective.replay``, not
    part of the public API.
    """
    canonical = fovs.sort_values(list(fov_key), kind="stable")
    rng = group_rng(random_state, group, *more)
    return canonical.iloc[rng.permutation(len(canonical))]


@dataclass(frozen=True)
class BenchmarkRecord:
    """One strategy run on one analysis group of one dataset under one acquisition order."""

    dataset: str
    role: DatasetRole
    in_sample: bool
    tuned_on: tuple[str, ...]
    group: Mapping[str, Any]
    order: AcquisitionOrder
    objective: ObjectiveConfig
    strategy: str
    config_id: str
    estimator_ids: tuple[str, ...]
    policy_ids: tuple[str, ...]
    stopped: bool
    stop_step: int | None
    final_decision: Decision | None
    n_fovs: int
    n_observations: int
    observation_unit: str
    observation_unit_declared: bool
    area: float | None
    total_fovs: int
    total_observations: int
    total_area: float | None
    outcomes: Mapping[str, float | None]
    outcome_ids: Mapping[str, str]
    compute_seconds_to_stop: float
    max_compute_seconds: float
    incremental: bool
    decision_seconds_to_stop: float
    median_decision_seconds: float
    max_decision_seconds: float
    reference: str = "full"
    reservoir_fovs: int | None = None
    reservoir: Mapping[str, Any] | None = None
    stop_on: str = "decision"

    @property
    def n_objects(self) -> int:
        """Backward-compatible alias for n_observations."""
        return self.n_observations

    @property
    def total_objects(self) -> int:
        """Backward-compatible alias for total_observations."""
        return self.total_observations

    def as_row(self) -> dict[str, Any]:
        """A flat dict: group columns as ``group.<col>``, outcomes as ``outcome.<name>``."""
        row: dict[str, Any] = {
            "dataset": self.dataset,
            "role": self.role.value,
            "in_sample": self.in_sample,
            "tuned_on": ",".join(self.tuned_on),
            "order": self.order.kind,
            "order_seed": self.order.seed,
            "objective": self.objective.objective.value,
            "strategy": self.strategy,
            "config_id": self.config_id,
            "estimator_ids": ",".join(self.estimator_ids),
            "policy_ids": ",".join(self.policy_ids),
            "stop_on": self.stop_on,
            "stopped": self.stopped,
            "stop_step": self.stop_step,
            "final_decision": None if self.final_decision is None else self.final_decision.value,
            "n_fovs": self.n_fovs,
            "n_observations": self.n_observations,
            "observation_unit": self.observation_unit,
            "observation_unit_declared": self.observation_unit_declared,
            "n_objects": self.n_objects,
            "area": self.area,
            "total_fovs": self.total_fovs,
            "total_observations": self.total_observations,
            "total_objects": self.total_objects,
            "total_area": self.total_area,
            "compute_seconds_to_stop": self.compute_seconds_to_stop,
            "max_compute_seconds": self.max_compute_seconds,
            "incremental": self.incremental,
            "decision_seconds_to_stop": self.decision_seconds_to_stop,
            "median_decision_seconds": self.median_decision_seconds,
            "max_decision_seconds": self.max_decision_seconds,
            "reference": self.reference,
            "reservoir_fovs": self.reservoir_fovs,
        }
        for key, value in (self.reservoir or {}).items():
            row[f"reservoir.{key}"] = value
        row.update({f"group.{column}": value for column, value in self.group.items()})
        row.update({f"outcome.{name}": value for name, value in self.outcomes.items()})
        return row


def records_to_frame(records: Iterable[BenchmarkRecord]) -> pd.DataFrame:
    """One row per record (see ``BenchmarkRecord.as_row``)."""
    return pd.DataFrame([record.as_row() for record in records])


def _group_key(group: Mapping[str, Any]) -> tuple:
    """Hashable identity of an analysis group, for matching replay results."""
    return tuple(group.items())


def _area(data: ValidatedInput, area_col: str | None) -> float | None:
    """Total imaged area of the FOVs, or ``None`` when the dataset declares no area column.

    Areas are checked by ``_areas.area_problem``, the check
    ``ImagedAreaEstimator`` uses too: a field with no recorded area, or an
    infinite, zero or negative one, raises here (summing it would misstate the
    effort at the stop), and so do areas that are not numbers.
    """
    if area_col is None:
        return None
    problem = area_problem(data.fovs, area_col, data.fov_key)
    if problem is not None:
        raise ValueError(problem)
    return float(data.fovs[area_col].to_numpy(dtype=float).sum())


def _record_run(
    history: Sequence[SequentialResult],
    full: ValidatedInput,
    scored_against: ValidatedInput,
    metrics: Sequence[OutcomeMetric],
    dataset: BenchmarkDataset,
    stop_on: str = "decision",
    **provenance: Any,
) -> BenchmarkRecord:
    """Find the stop in one strategy's history on one group and score it against ``scored_against``.

    ``full`` is everything the replay could reveal (the group, or its stream).
    ``stop_on`` is ``"decision"`` or ``"rule"`` (see ``run_benchmark``); the
    recorded ``final_decision`` is the decision it reads. The stop prefix is
    scored in acquisition order, as the replay revealed it.
    """

    def read(result: StrategyResult) -> Decision | None:
        return result.decision if stop_on == "decision" else result.rule_decision

    stop_index = next(
        (i for i, item in enumerate(history) if read(item.result) is Decision.STOP), None
    )
    last_index = len(history) - 1 if stop_index is None else stop_index
    last = history[last_index]
    rank = {fov: i for i, fov in enumerate(last.acquired_fovs)}
    fov_rank = np.array(
        [
            rank.get(row, -1)
            for row in full.fovs.loc[:, list(full.fov_key)].itertuples(index=False, name=None)
        ],
        dtype=np.int64,
    )
    acquired = np.flatnonzero(fov_rank >= 0)
    stop_fovs = full.fovs.iloc[acquired[np.argsort(fov_rank[acquired], kind="stable")]]
    stopped = _Acquisition(full, stop_fovs).prefix(len(stop_fovs))
    times = [item.result.compute_seconds for item in history]
    decisions = [item.result.timing.total_decision_seconds for item in history]
    used_times = times[: last_index + 1]
    used_decisions = decisions[: last_index + 1]
    configuration = last.result.configuration
    return BenchmarkRecord(
        dataset=dataset.name,
        role=dataset.role,
        group=dict(last.result.group),
        objective=last.result.objective,
        strategy=last.result.strategy,
        config_id=last.result.config_id,
        estimator_ids=tuple(e["config_id"] for e in configuration["estimators"]),
        policy_ids=tuple(p["config_id"] for p in configuration["policies"]),
        stopped=stop_index is not None,
        stop_step=None if stop_index is None else last.step,
        final_decision=read(last.result),
        stop_on=stop_on,
        n_fovs=len(stopped.fovs),
        n_observations=len(stopped.objects),
        observation_unit=stopped.observation_unit,
        observation_unit_declared=stopped.observation_unit_declared,
        area=_area(stopped, dataset.area_col),
        total_fovs=len(full.fovs),
        total_observations=len(full.objects),
        total_area=_area(full, dataset.area_col),
        outcomes={metric.name: metric.evaluate(scored_against, stopped) for metric in metrics},
        outcome_ids={metric.name: metric.config_id for metric in metrics},
        compute_seconds_to_stop=float(sum(used_times)),
        max_compute_seconds=float(max(used_times)),
        decision_seconds_to_stop=float(sum(used_decisions)),
        median_decision_seconds=float(np.median(used_decisions)),
        max_decision_seconds=float(max(used_decisions)),
        **provenance,
    )


def run_benchmark(
    registry: BenchmarkRegistry,
    dataset_name: str,
    *,
    strategies: Iterable[SamplingStrategy],
    metrics: Iterable[OutcomeMetric],
    objective: ObjectiveConfig,
    orders: Iterable[AcquisitionOrder] = (AcquisitionOrder.recorded(),),
    tuned_on: Iterable[str] = (),
    min_fovs: int = 1,
    incremental: bool = False,
    reservoir: ReservoirSplit | None = None,
    n_jobs: int = 1,
    stop_on: str = "decision",
) -> list[BenchmarkRecord]:
    """Replay every strategy on one registered dataset and record one run per group and order.

    ``tuned_on`` names the datasets the strategies' settings were chosen on;
    they must be development datasets. Metrics see the full group and the stop
    prefix only, so they are identical for every strategy. ``incremental`` is
    passed to the replay; decisions are the same either way, only the recorded
    latencies differ. With ``reservoir``, strategies replay each group's stream
    and outcomes are scored against its held-out reservoir; effort totals
    then refer to the stream. ``n_jobs`` replays the analysis groups of each
    strategy and order in that many worker processes (``-1``: every CPU);
    records are the same for every value (see ``morphocoverage._parallel`` on
    the BLAS thread count) apart from the recorded latencies. Latencies from
    ``n_jobs > 1`` are measured on workers sharing the CPUs, one BLAS thread
    each, so they are not the time a live decision would take. ``stop_on``
    is ``"decision"`` (the objective's decision, the default) or ``"rule"``
    (the rule's own verdict, ``StrategyResult.rule_decision``), so a table of
    rule verdicts and its stop points use the same decision; each record keeps
    ``stop_on``, and its ``final_decision`` is the decision ``stop_on`` reads.
    """
    if stop_on not in ("decision", "rule"):
        raise ValueError(f"stop_on must be 'decision' or 'rule', not {stop_on!r}")
    tuned = registry.check_tuning(tuned_on)
    dataset = registry[dataset_name]
    strategy_list = list(strategies)
    metric_list = list(metrics)
    names = [metric.name for metric in metric_list]
    if len(set(names)) != len(names):
        raise ValueError(
            f"outcome metric names must be distinct: {names}; give two settings of one "
            "metric different names with label=, e.g. MMDError(..., label='mmd_wide')"
        )
    if dataset.area_col is not None and dataset.area_col in dataset.feature_cols:
        raise ValueError("the area column is acquisition metadata, not a phenotype feature")

    objects, fovs = dataset.load()
    columns = {
        "feature_cols": list(dataset.feature_cols),
        "fov_col": dataset.fov_col,
        "group_cols": None if dataset.group_cols is None else list(dataset.group_cols),
        "observation_unit": dataset.observation_unit,
    }
    data = validate_input_tables(objects, fovs=fovs, **columns)
    if dataset.area_col is not None and dataset.area_col not in data.fovs.columns:
        raise ValueError(f"area_col {dataset.area_col!r} is not a column of the FOV table")
    _area(data, dataset.area_col)  # refuses unusable areas before any replay
    full_groups = {
        _group_key(group): group_data for group, group_data in iter_analysis_groups(data)
    }
    references = full_groups
    replay_tables = (objects, fovs)
    reference_fields: dict[tuple, dict[str, Any]] = {key: {} for key in full_groups}
    if reservoir is not None:
        if not isinstance(reservoir, ReservoirSplit):
            raise TypeError("reservoir must be a ReservoirSplit")
        splits = {key: reservoir.split(dict(key), group) for key, group in full_groups.items()}
        full_groups = {key: stream for key, (stream, _) in splits.items()}
        references = {key: held for key, (_, held) in splits.items()}
        replay_tables = (
            pd.concat([stream.objects for stream in full_groups.values()]),
            pd.concat([stream.fovs for stream in full_groups.values()]),
        )
        reference_fields = {
            key: {
                "reference": "reservoir",
                "reservoir_fovs": len(held.fovs),
                "reservoir": reservoir.parameters,
            }
            for key, held in references.items()
        }

    records: list[BenchmarkRecord] = []
    # One pool for every order and strategy, so workers start once.
    with worker_pool(n_jobs) as pool:
        for order in orders:
            if order.kind == "recorded":
                # The loaded tables (or the streams) are replayed as they are.
                ordered_fovs, order_col = replay_tables[1], dataset.acquisition_order_col
            else:
                ordered_fovs, order_col = order.apply(
                    [(dict(key), group) for key, group in full_groups.items()],
                    dataset.acquisition_order_col,
                )
            for strategy in strategy_list:
                replay = _evaluate_prefixes(
                    replay_tables[0],
                    ordered_fovs,
                    strategies=[strategy],
                    acquisition_order_col=order_col,
                    objective=objective,
                    min_fovs=min_fovs,
                    incremental=incremental,
                    n_jobs=n_jobs,
                    pool=pool,
                    **columns,
                )
                histories: dict[tuple, list[SequentialResult]] = {}
                for item in replay:
                    histories.setdefault(_group_key(item.result.group), []).append(item)
                for key, history in histories.items():
                    records.append(
                        _record_run(
                            history,
                            full_groups[key],
                            references[key],
                            metric_list,
                            dataset,
                            stop_on,
                            in_sample=dataset.name in tuned,
                            tuned_on=tuned,
                            order=order,
                            incremental=incremental,
                            **reference_fields[key],
                        )
                    )
    return records


__all__ = [
    "AcquisitionOrder",
    "BenchmarkDataset",
    "BenchmarkRecord",
    "BenchmarkRegistry",
    "DatasetRole",
    "leave_one_dataset_out",
    "records_to_frame",
    "run_benchmark",
]

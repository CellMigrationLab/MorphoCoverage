"""The benchmark's code at a tiny size (docs/common/validation.md; data/simulations/decisions):
the generators have the properties they claim, the runner is seeded, paired and resumable,
each module script runs end to end with its summary, the real-data summaries and the Colab
notebook's code path work. No result of the benchmark is computed here."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
DECISIONS = ROOT / "data" / "simulations" / "decisions"
sys.path.insert(0, str(ROOT / "data" / "simulations"))

from decisions import colab, runner, summary
from decisions.candidates import hd_tests, multiplicity, sequential_rules
from decisions.design import Design, layout
from decisions.embedding import EmbeddingSpec
from decisions.embedding import generate as embed
from decisions.populations import PopulationSpec
from decisions.populations import generate as populate
from decisions.real_parameters import summarise as real_summary
from decisions.scalar import Scalar, generate
from decisions.sequential import schedule, sequence

SCRIPTS = list(colab.MODULES.values())

# --- The generators -------------------------------------------------------------------------


def test_the_layout_holds_the_designed_fields_wells_and_conditions():
    design = Design(
        repeats=3, wells=2, fields=4, presence=(("control", "treated"),) * 2 + (("control",),)
    )
    fields = layout(design, np.random.default_rng(0)).fields
    assert fields.groupby(["biological_repeat", "condition"]).size().to_dict() == {
        ("R1", "control"): 8, ("R1", "treated"): 8, ("R2", "control"): 8,
        ("R2", "treated"): 8, ("R3", "control"): 8,
    }  # fmt: skip
    fixed = layout(Design(cells=1.0, fixed_cells=True), np.random.default_rng(0)).fields
    assert (fixed["n_cells"] == 1).all()


def test_unrecorded_wells_add_their_effect_but_leave_no_column():
    data = generate(
        Design(wells=2, wells_recorded=False), Scalar(icc_well=0.2), np.random.default_rng(1)
    )
    assert "technical_repeat" not in data.objects and "technical_repeat" not in data.fovs
    assert data.layout.fields["well"].nunique() == 16


def test_the_scalar_truth_is_the_declared_fold_change_and_the_simulation_finds_it():
    spec = Scalar(distribution="lognormal", effect=1.0)
    data = generate(Design(repeats=60, fields=10), spec, np.random.default_rng(2))
    truth = data.truth["x0", "treated", "cell", "ratio"]
    assert truth == pytest.approx(np.sqrt(np.log1p(0.3**2)), rel=1e-6)
    means = data.objects.groupby("condition")["x0"].mean()
    assert np.log(means["treated"] / means["control"]) == pytest.approx(truth, abs=0.03)
    assert len(data.conditional) == data.layout.fields["well"].nunique()


def test_informative_cluster_size_separates_the_cell_and_field_weighted_truths():
    spec = Scalar(distribution="normal", scale="difference", effect=1.0, icc_field=0.3)
    data = generate(Design(informative_size=0.8), spec, np.random.default_rng(3))
    cell = data.truth["x0", "treated", "cell", "difference"]
    field = data.truth["x0", "treated", "field", "difference"]
    assert cell == pytest.approx(field)  # an additive effect: the same difference
    # A fold change: the crowded fields' higher values enter the cell-weighted difference.
    spec = Scalar(distribution="lognormal", scale="ratio", effect=1.0, icc_field=0.3)
    data = generate(Design(informative_size=0.8), spec, np.random.default_rng(3))
    assert (
        data.truth["x0", "treated", "cell", "difference"]
        > data.truth["x0", "treated", "field", "difference"]
    )


def test_the_embedding_shift_has_the_declared_length_and_the_rotation_batch_moves_repeats():
    spec = EmbeddingSpec(dims=16, shift=0.5)
    data = embed(Design(), spec, np.random.default_rng(4))
    assert data.truth["shift_norm"] == pytest.approx(0.5 * spec.within_sd)
    assert len(data.columns) == 16 and not data.truth["mean_null"]
    covariance = embed(
        Design(), EmbeddingSpec(shift=1.0, shift_kind="covariance"), np.random.default_rng(4)
    )
    assert covariance.truth["mean_null"] and not covariance.truth["distribution_null"]


def test_the_populations_have_their_prevalence_and_a_batch_only_group_one_repeat():
    data = populate(Design(repeats=3, fields=20, cells=30.0), PopulationSpec(prevalence=0.2),
                    np.random.default_rng(5))  # fmt: skip
    assert 0.1 < data.truth["labels"].mean() < 0.3
    batch = populate(Design(repeats=3), PopulationSpec(structure="batch"), np.random.default_rng(5))
    labelled = batch.objects.loc[batch.truth["labels"] == 1, "biological_repeat"]
    assert set(labelled) <= {"R1"}


def test_a_sequence_interleaves_the_wells_and_its_looks_end_at_the_last_field():
    design = Design(repeats=1, conditions=("control",), wells=2, fields=6)
    found = sequence(design, Scalar(), np.random.default_rng(6))
    assert [w[-2:] for w in found.wells[:4]] == ["w1", "w2", "w1", "w2"]
    assert schedule(12, 5, 5) == [5, 10, 12]
    assert found.truth > 0


# --- The candidates -------------------------------------------------------------------------


def test_the_sign_flip_u_statistic_cannot_go_below_two_in_two_to_the_n():
    deltas = np.ones((4, 5)) + np.random.default_rng(7).normal(scale=0.01, size=(4, 5))
    assert hd_tests.sign_flip_u(deltas) == pytest.approx(2 / 16)


def test_the_multiplicity_procedures_reject_nothing_under_a_flat_null_and_nest():
    p = np.linspace(0.2, 1, 30)
    families = np.repeat(np.arange(3), 10)
    for rejected in (
        multiplicity.bh(p),
        multiplicity.by(p),
        multiplicity.hierarchical(p, families),
    ):
        assert not rejected.any()
    strong = np.r_[np.full(5, 1e-6), p[5:]]
    assert set(np.flatnonzero(multiplicity.by(strong))) <= set(
        np.flatnonzero(multiplicity.bh(strong))
    )


def test_the_spending_levels_spend_at_most_alpha_over_the_looks():
    levels = sequential_rules.spending_levels([5, 10, 15, 20], 20, 0.05)
    assert sum(1 - level for level in levels) == pytest.approx(0.05, abs=1e-9)


# --- The runner, the module scripts and their summaries ------------------------------------


def test_a_replicate_is_a_pure_function_of_its_seed_and_every_method_reads_it():
    module = importlib.import_module("decisions.p5_multiple_testing")
    point = module.MODULE.points[0]
    first = runner.run_replicate(module.MODULE, point, 3, 0, True)
    second = runner.run_replicate(module.MODULE, point, 3, 0, True)
    drop = ["runtime"]
    assert pd.DataFrame(first).drop(columns=drop).equals(pd.DataFrame(second).drop(columns=drop))
    assert {r["method"] for r in first} == {m.name for m in module.MODULE.methods}


@pytest.mark.parametrize("script", SCRIPTS)
def test_each_module_runs_at_smoke_size_resumes_and_summarises(script, tmp_path):
    written = runner.run(script, tmp_path, "smoke", workers=1, progress=None)
    assert written > 0
    assert runner.run(script, tmp_path, "smoke", workers=1, progress=None) == 0  # resumed
    module = importlib.import_module(script).MODULE
    rows = summary.load(tmp_path, module.name)
    assert not rows.empty and set(rows["method"]) == {m.name for m in module.methods}
    files = summary.summarise(script, tmp_path)
    assert all(f.exists() for f in files)
    assert "Winner" in files[1].read_text()
    assert (tmp_path / module.name / "points.csv").exists()


def test_the_workers_give_the_rows_of_a_serial_run(tmp_path):
    script = "decisions.p5_multiple_testing"
    runner.run(script, tmp_path / "serial", "smoke", workers=1, progress=None)
    runner.run(script, tmp_path / "pool", "smoke", workers=2, progress=None)
    serial = summary.load(tmp_path / "serial", "p5").drop(columns="runtime")
    pooled = summary.load(tmp_path / "pool", "p5").drop(columns="runtime")
    keys = ["point", "replicate", "method"]
    pd.testing.assert_frame_equal(
        serial.sort_values(keys).reset_index(drop=True),
        pooled.sort_values(keys).reset_index(drop=True),
    )


def test_the_selection_rule_gates_ranks_and_breaks_ties_by_simplicity():
    rows = []
    for method, rate, power in (
        ("simple", 0.05, 0.80),
        ("complex", 0.05, 0.81),
        ("loose", 0.09, 0.95),
    ):
        for replicate in range(4000):
            rng = np.random.default_rng(replicate)
            rows.append({"point": "null", "kind": "core", "null": True, "alternative": "",
                         "replicate": replicate, "method": method, "question": "main", "failed": 0,
                         "runtime": 0.0, "reject": float(rng.random() < rate)})  # fmt: skip
            rows.append({"point": "alt", "kind": "core", "null": False, "alternative": "a",
                         "replicate": replicate, "method": method, "question": "main", "failed": 0,
                         "runtime": 0.0, "reject": float(rng.random() < power)})  # fmt: skip
    frame = pd.DataFrame(rows)
    methods = tuple(runner.Method(n, lambda d, c: {}, simplicity=s)
                    for n, s in (("simple", 1), ("complex", 3), ("loose", 1)))  # fmt: skip
    module = runner.Module("test", "test", lambda p, r: None, (), methods)
    chosen = summary.select(frame, summary.point_table(frame), module, runner.Rule(coverage=False))
    chosen = chosen.set_index("method")
    assert not chosen.loc["loose", "eligible"]  # 9% false positives
    assert chosen.loc["complex", "tied"] and chosen.loc["simple", "winner"]


# --- Real-data summaries and the notebook ---------------------------------------------------


def test_the_real_data_summaries_cover_the_committed_datasets_and_never_the_held_out_one():
    table = pd.read_csv(DECISIONS / "results" / "real_parameters.csv")
    datasets = set(table["dataset"])
    assert {"lab/cct8_cdm_morphology", "crick_idr/idr0028_screenB"} <= datasets
    assert not any("idr0139" in d for d in datasets)
    assert {"share_residual", "cv", "observations_per_field_median"} <= set(table["statistic"])


def test_the_real_data_summary_reads_a_given_table_and_an_embedding():
    data = embed(Design(repeats=3), EmbeddingSpec(dims=6), np.random.default_rng(8))
    found = real_summary(
        "given", data.objects, data.fovs, "control", ["e0", "e1"], list(data.columns)
    )
    assert {"effective_rank", "repeat_shift_median", "share_field"} <= set(found["statistic"])


def test_the_real_data_summary_refuses_the_former_repeat_name():
    """A given table naming the biological repeat ``repeat`` is refused, not renamed."""
    data = embed(Design(repeats=3), EmbeddingSpec(dims=6), np.random.default_rng(8))
    former = {"biological_repeat": "repeat"}
    with pytest.raises(ValueError, match="'biological_repeat'"):
        real_summary(
            "given",
            data.objects.rename(columns=former),
            data.fovs.rename(columns=former),
            "control",
            ["e0", "e1"],
        )


def _cells() -> list[str]:
    notebook = json.loads((DECISIONS / "benchmark_colab.ipynb").read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert not cell["outputs"] and cell["execution_count"] is None
    return ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]


def test_the_notebook_is_thin_guards_colab_and_runs_locally_at_smoke_size(tmp_path, monkeypatch):
    cells = _cells()
    for source in cells:
        assert source.startswith("# @title") and "@param" not in source
        for line in source.splitlines():
            if "/content" in line or "google.colab import" in line:
                assert "google.colab' in sys.modules" in source.replace('"', "'"), line
    namespace: dict[str, object] = {}
    monkeypatch.chdir(ROOT)
    exec(cells[0], namespace)  # noqa: S102 - the notebook's own code, locally
    monkeypatch.chdir(tmp_path)
    for source in cells[1:3]:
        exec(source, namespace)  # noqa: S102
    namespace.update(MODULES=["P0", "P5"], SIZE="smoke", WORKERS=1)
    monkeypatch.setattr(colab, "run_p0", lambda out: 0)  # P0 is this suite's own test file
    for source in cells[3:]:
        exec(source, namespace)  # noqa: S102
    assert (tmp_path / "decisions_results" / "p5" / "summary.md").exists()
    assert (tmp_path / "decisions_results.zip").exists()


def _colab_clone(monkeypatch, tmp_path, clone_code: int) -> tuple[list, list, RuntimeError | None]:
    """Run the notebook's first cell as on Colab with a fake token, ``subprocess.run`` only
    recorded (the clone answering ``clone_code``); the commands, their environments and the
    error raised, if any."""
    import subprocess
    import types

    token = "ghp_FAKE0123456789"
    userdata = types.SimpleNamespace(get=lambda name: token)
    colab_module = types.ModuleType("google.colab")
    colab_module.userdata = userdata
    monkeypatch.setitem(sys.modules, "google", types.ModuleType("google"))
    monkeypatch.setitem(sys.modules, "google.colab", colab_module)
    monkeypatch.setattr(sys, "path", list(sys.path))
    commands, environments = [], []

    def run(command, **kwargs):
        commands.append(list(map(str, command)))
        environments.append(kwargs.get("env") or {})
        if "clone" in command:
            return subprocess.CompletedProcess(command, clone_code, "", "fatal: no access\n")
        url = "https://github.com/CellMigrationLab/MorphoCoverage.git\n"
        return subprocess.CompletedProcess(command, 0, url, "")

    monkeypatch.setattr(subprocess, "run", run)
    error = None
    try:
        exec(_cells()[0], {})  # noqa: S102 - the notebook's own code, its commands recorded
    except RuntimeError as raised:  # fallback: the test inspects the error itself
        error = raised
    for text in [*map(" ".join, commands), str(error or "")]:
        assert token not in text, text
    return commands, environments, error


def test_the_colab_clone_keeps_the_token_out_of_urls_commands_and_errors_591(monkeypatch, tmp_path):
    # The token was in the clone URL: in the command, the clone's .git/config and the
    # CalledProcessError of a failed clone.
    commands, environments, error = _colab_clone(monkeypatch, tmp_path, 0)
    assert error is None
    clone = next(c for c in commands if "clone" in c)
    assert "https://github.com/CellMigrationLab/MorphoCoverage.git" in clone
    assert environments[commands.index(clone)]["GIT_ASKPASS"]
    assert any("remote.origin.url" in c for c in commands)  # the clone's remote is checked
    commands, _, error = _colab_clone(monkeypatch, tmp_path, 128)
    assert isinstance(error, RuntimeError) and "git clone failed" in str(error)
    assert not any("pip" in c for c in commands)


def test_a_failed_p0_stops_the_run_before_any_other_module_520(tmp_path, monkeypatch):
    # run discarded P0's exit code and went on to P1-P5 as if the gate had passed.
    calls = []
    monkeypatch.setattr(colab, "run_p0", lambda out: 1)
    monkeypatch.setattr(runner, "run", lambda *args: calls.append(args))
    with pytest.raises(RuntimeError, match="P0"):
        colab.run(["P0", "P5"], "smoke", tmp_path)
    assert calls == []
    monkeypatch.setattr(colab, "run_p0", lambda out: 0)
    colab.run(["P0", "P5"], "smoke", tmp_path)
    assert len(calls) == 1


def _semi_inputs(fields_per_repeat: int = 4, values: np.ndarray | None = None):
    meta = pd.DataFrame(
        {
            "biological_repeat": np.repeat(["R1", "R2"], fields_per_repeat * 3),
            "position_id": np.repeat(np.arange(2 * fields_per_repeat), 3).astype(str),
        }
    )
    if values is None:
        values = np.random.default_rng(0).normal(size=(len(meta), 3))
    return values, meta


def test_semi_synthetic_splits_every_repeat_and_shifts_by_the_requested_amount_524():
    from decisions.embedding import semi_synthetic

    values, meta = _semi_inputs()
    _, labelled, vector = semi_synthetic(values, meta, 0.5, np.random.default_rng(1))
    centred = values - values.mean(axis=0)
    within = np.sqrt(np.square(centred).sum(axis=1).mean())
    assert np.linalg.norm(vector) == pytest.approx(0.5 * within)
    assert (labelled.groupby("biological_repeat")["condition"].nunique() == 2).all()
    # Two exactly opposite rows: every draw of two different rows cancels, never a zero shift.
    opposite = np.tile([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], (12, 1))
    _, _, vector = semi_synthetic(opposite, meta, 0.5, np.random.default_rng(1))
    assert np.linalg.norm(vector) == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("inputs", "shift", "match"),
    [
        (_semi_inputs(fields_per_repeat=1), 0.5, "2 fields"),  # no treated field at all
        (_semi_inputs(values=np.ones((24, 3))), 0.5, "vary"),  # a zero shift asked as 0.5
        ((np.ones((5, 3)), _semi_inputs()[1]), 0.5, "rows"),
        ((np.full((24, 3), np.nan), _semi_inputs()[1]), 0.5, "finite"),
    ],
)
def test_semi_synthetic_refuses_what_it_cannot_split_or_shift_524(inputs, shift, match):
    # One field per repeat gave no treated field; constant values a zero shift for 0.5.
    from decisions.embedding import semi_synthetic

    with pytest.raises(ValueError, match=match):
        semi_synthetic(*inputs, shift, np.random.default_rng(1))

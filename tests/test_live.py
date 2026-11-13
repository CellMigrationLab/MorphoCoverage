"""The live workflow (``workflows.live``) and its folder's manifest (``workflows._manifest``)."""

from __future__ import annotations

import dataclasses
import shutil
import warnings

import numpy as np
import pandas as pd
import pytest
from _helpers_results import _flat, _nucleus_setup, _updates, _well
from _helpers_workflows import AREA_SETUP, READOUT, _setup, _sictrl_tables, _well_acquisition

from morphocoverage import validate_input_tables
from morphocoverage._style import width
from morphocoverage.multivariate_outcomes import CovarianceError
from morphocoverage.results import Decision
from morphocoverage.retrospective import OutcomeCriterion
from morphocoverage.strategies import FixedFOVStrategy
from morphocoverage.workflows import LiveFolder, LiveTables, Setup, live_session, write_fov
from morphocoverage.workflows.live import MANIFEST, print_update


def test_i4_livetables_key_is_the_fov_identity_of_the_validator():
    objects, fovs = _well_acquisition()
    # The same image_file names recur in every condition, repeat and well.
    assert fovs.duplicated("image_file").any()
    setup = _setup(rules=(FixedFOVStrategy(4),), group_cols=("condition",))
    ordered = fovs.sort_values(["biological_repeat", "acquisition_order"]).reset_index(drop=True)
    tables = LiveTables(live_session(setup))
    parts = [objects.merge(ordered.iloc[[i]][KEY], on=KEY) for i in range(len(ordered))]
    for i in range(len(ordered)):
        tables.update(pd.concat(parts[: i + 1], ignore_index=True), ordered[: i + 1])
    assert tables.key == ("condition", "biological_repeat", "technical_repeat", "image_file")
    assert [u.n_objects for u in tables.updates] == [len(p) for p in parts]
    # All at once, too.
    at_once = LiveTables(live_session(setup))
    at_once.update(pd.concat(parts, ignore_index=True), ordered)
    assert len(at_once.updates) == len(ordered)


def test_i10_sentences_use_the_observation_unit_and_need_a_control_for_unusual_ones():
    setup = _setup(observation_unit="nucleus")
    assert "over all nuclei" in setup.declaration()[0]
    without = " ".join(_setup(observation_unit="cell", control=None).declaration())
    assert "unusual cells are not assessed without a control" in without
    assert "for unusual" not in without
    objects, fovs = _well_acquisition()
    session = live_session(_setup(observation_unit="nucleus", rules=(FixedFOVStrategy(4),)))
    first = fovs.iloc[[0]]
    waiting = objects.merge(fovs.iloc[[0, 1]][KEY], on=KEY)
    update = LiveTables(session).update(waiting, first)[0]
    assert "nucleus rows wait for their FOV rows" in update.summary()


def test_i11_watch_until_stop_follows_the_objective_and_stop_on_rule_each_verdict(tmp_path):
    objects, fovs = _well_acquisition()
    fingerprint = (
        OutcomeCriterion(CovarianceError(["feature_a", "feature_b"], True), 0.5, "lower_is_better"),
    )
    setup = _setup(rules=(FixedFOVStrategy(2),), fingerprint=fingerprint)
    group = fovs[(fovs["condition"] == "control") & (fovs["biological_repeat"] == "R1")]
    for _, row in group.sort_values("acquisition_order").iterrows():
        write_fov(tmp_path, objects.merge(row.to_frame().T[KEY], on=KEY), row.to_dict())
    seen = LiveFolder(tmp_path, live_session(setup)).watch(
        interval=0.01, timeout=1, until_stop=True, on_update=None
    )
    # The objective stays unsupported (no rule decides the fingerprint), so the watch
    # does not stop (owner, 2026-10-05, #85) and reads every field.
    assert len(seen) == len(group)
    seen = LiveFolder(tmp_path, live_session(setup)).watch(
        interval=0.01, timeout=5, until_stop=True, on_update=None, stop_on="rule"
    )
    last = seen[-1].results[0]
    # With stop_on="rule", the rule stops at the second field of each well (wells
    # alternate: w0, w1, w0, w1), and watch returns once both wells stop (audit 4, P2-5).
    assert last.decision != Decision.STOP and last.rule_decision == Decision.STOP
    assert len(seen) == 4 < len(group)


KEY = ["condition", "biological_repeat", "technical_repeat", "image_file"]


def test_i2_livetables_waits_for_the_first_fov_row_before_fixing_its_key():
    objects, fovs = _well_acquisition()
    setup = _setup(rules=(FixedFOVStrategy(4),))
    ordered = fovs.sort_values(["condition", "biological_repeat", "acquisition_order"])
    ordered = ordered.reset_index(drop=True)
    parts = [objects.merge(ordered.iloc[[i]][KEY], on=KEY) for i in range(len(ordered))]
    tables = LiveTables(live_session(setup))
    for i in range(len(ordered)):
        grown = pd.concat(parts[: i + 1], ignore_index=True)
        tables.update(grown, ordered[:i])  # the field's objects arrive first ...
        tables.update(grown, ordered[: i + 1])  # ... then its FOV row
    assert tables.key == ("condition", "biological_repeat", "technical_repeat", "image_file")
    assert [u.n_objects for u in tables.updates] == [len(p) for p in parts]


def test_i14_livetables_key_is_built_from_the_validator_resolvers():
    objects, fovs = _well_acquisition()
    one = fovs[(fovs["condition"] == "control") & (fovs["biological_repeat"] == "R1")].copy()
    mine = objects.merge(one[KEY], on=KEY)
    # A producer that writes every column leaves the optional ones empty.
    one["technical_repeat"], mine["technical_repeat"] = None, None
    one["position_id"], mine["position_id"] = None, None
    tables = LiveTables(live_session(_setup(rules=(FixedFOVStrategy(4),), stage_cols=None)))
    assert len(tables.update(mine, one)) == len(one)
    data = validate_input_tables(mine, fovs=one, feature_cols=[READOUT])
    assert tables.key == data.fov_key == ("condition", "biological_repeat", "image_file")


def test_i16_one_observation_reads_in_the_singular():
    fov = pd.DataFrame({"position_id": ["P1"]})
    one = pd.DataFrame({"position_id": ["P1"], READOUT: [1.0]})
    session = live_session(Setup(readout=READOUT, rules=(FixedFOVStrategy(4),)))
    (update,) = LiveTables(session).update(one, fov)
    assert "P1: 1 object (read in" in update.summary()


SETUP = Setup(readout="x", observation_unit="cell")


def _field(well: str, index: int, n: int = 5, shift: float = 0.0):
    """One field's objects and metadata, deterministic."""
    values = 100.0 + shift + np.arange(n, dtype=float) + index
    metadata = {
        "condition": "MOCK",
        "biological_repeat": "R1",
        "technical_repeat": well,
        "image_file": f"{well}_f{index:03d}.tif",
    }
    return pd.DataFrame({"x": values}), metadata


def _folder_session(folder, rule_fovs: int = 2) -> LiveFolder:
    return LiveFolder(folder, live_session(SETUP, rules=(FixedFOVStrategy(rule_fovs),)))


def _fields_in(live: LiveFolder) -> int:
    return sum(len(group._parts) for group in live.session._groups.values())


def test_p2_5_watch_until_stop_waits_for_every_group(tmp_path):
    # Two wells acquired in turn; the rule stops each well at its second field.
    for index in range(3):
        for well in ("A01", "B01"):
            write_fov(tmp_path, *_field(well, index))
    seen = _folder_session(tmp_path).watch(
        interval=0.01, timeout=5, until_stop=True, on_update=None
    )
    # A01 stops at update 3, B01 still continues there; both stop at update 4.
    assert len(seen) == 4
    assert seen[-1].fov_metadata["technical_repeat"] == "B01"
    latest = _folder_session(tmp_path).watch(
        interval=0.01, timeout=5, until_stop="latest_group", on_update=None
    )
    assert len(latest) == 3 and latest[-1].fov_metadata["technical_repeat"] == "A01"
    with pytest.raises(ValueError, match="until_stop"):
        _folder_session(tmp_path).watch(interval=0.01, until_stop="first_group")


def test_p2_6_a_rewritten_or_truncated_manifest_raises(tmp_path):
    for index in range(5):
        write_fov(tmp_path, *_field("A01", index, shift=0.0))
    live = _folder_session(tmp_path, rule_fovs=50)
    assert len(live.poll()) == 5
    # The acquisition restarted in the same folder: same header, new fields, and
    # soon more bytes than were read.
    shutil.rmtree(tmp_path)
    tmp_path.mkdir()
    for index in range(3):
        write_fov(tmp_path, *_field("A01", 100 + index))
    with pytest.raises(ValueError, match="shorter than the part already read"):
        live.poll()
    for index in range(3, 12):
        write_fov(tmp_path, *_field("A01", 100 + index))
    with pytest.raises(ValueError, match="rewritten, truncated or replaced after 5 fields"):
        live.poll()
    assert _fields_in(live) == 5
    (tmp_path / MANIFEST).unlink()
    with pytest.raises(ValueError, match="was removed"):
        live.poll()


def test_p2_7_objects_appended_twice_are_warned_about():
    objects, metadata = _field("A01", 0)
    objects = objects.assign(image_file=metadata["image_file"], label_id=np.arange(1, 6))
    fovs = pd.DataFrame([metadata])
    tables = LiveTables(live_session(SETUP))
    # The extractor retried and appended the field's objects twice before its FOV row.
    with pytest.warns(UserWarning, match="exact copy of another row"):
        update = tables.update(pd.concat([objects, objects], ignore_index=True), fovs)
    assert update[0].n_objects == 10  # counted as they are, but no longer silently
    session = live_session(SETUP)
    relabelled = objects.assign(label_id=[1, 1, 2, 3, 4])
    with pytest.warns(UserWarning, match="label_id appears twice in the same field"):
        session.add_fov(relabelled, metadata)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        live_session(SETUP).add_fov(objects, metadata)  # a clean field: no warning


def test_p2_8_reused_file_names_are_refused_before_any_field_is_added():
    rng = np.random.default_rng(0)
    well_a = pd.DataFrame({"image_file": "f001.tif", "x": rng.normal(100, 10, 30)})
    well_b = pd.DataFrame({"image_file": "f001.tif", "x": rng.normal(200, 10, 30)})
    fovs = pd.DataFrame(
        [
            {
                "condition": "MOCK",
                "biological_repeat": "R1",
                "technical_repeat": w,
                "image_file": "f001.tif",
            }
            for w in ("A01", "B01")
        ]
    )
    tables = LiveTables(live_session(SETUP))
    with pytest.raises(ValueError, match="is reused across different metadata values"):
        tables.update(pd.concat([well_a, well_b], ignore_index=True), fovs)
    assert not tables.session._groups and not tables.updates
    # Field by field: the same reason, not "append a field's objects before its FOV row".
    by_field = LiveTables(live_session(SETUP))
    by_field.update(well_a, fovs.iloc[:1])
    with pytest.raises(ValueError, match="is reused across different metadata values"):
        by_field.update(pd.concat([well_a, well_b], ignore_index=True), fovs)
    visible, _ = next(iter(by_field.session._groups.values())).visible()
    assert len(visible.objects) == 30  # well A holds only its own objects


def test_p2_10_watch_returns_with_the_session_at_the_stop(tmp_path):
    for index in range(6):
        write_fov(tmp_path, *_field("A01", index))
    live = _folder_session(tmp_path)
    seen = live.watch(interval=0.01, timeout=5, until_stop=True, on_update=None)
    # The backlog is not consumed past the stop: the returned view is the session.
    assert len(seen) == 2 and _fields_in(live) == 2 and len(live.updates) == 2
    assert len(live.poll()) == 4 and _fields_in(live) == 6
    limited = _folder_session(tmp_path, rule_fovs=50)
    assert len(limited.watch(interval=0.01, timeout=5, max_fovs=3, on_update=None)) == 3
    assert _fields_in(limited) == 3


def test_p2_11_a_manifest_line_written_twice_is_skipped(tmp_path):
    for index in range(4):
        write_fov(tmp_path, *_field("A01", index))
    manifest = tmp_path / MANIFEST
    lines = manifest.read_text().splitlines(keepends=True)
    manifest.write_text("".join(lines) + lines[-1])  # the 4th field's line written twice
    for index in range(4, 10):
        write_fov(tmp_path, *_field("A01", index))
    live = _folder_session(tmp_path, rule_fovs=50)
    with pytest.warns(UserWarning, match="line 6 repeats line 5 exactly"):
        assert len(live.poll()) == 10
    assert _fields_in(live) == 10
    write_fov(tmp_path, *_field("A01", 20))
    # The same field again with another object file: raises, naming both lines.
    objects, metadata = _field("A01", 3)
    write_fov(tmp_path, objects, metadata, name="retry")
    with pytest.raises(ValueError, match="line 14 lists the field of line 5"):
        live.poll()
    assert _fields_in(live) == 11


def test_h6_live_identifiers_keep_their_leading_zeros(tmp_path):
    objects = pd.DataFrame({"position_id": ["001"] * 5, "x": np.arange(5.0) + 1})
    write_fov(
        tmp_path, objects, {"condition": "A", "biological_repeat": "01", "position_id": "001"}
    )
    update = LiveFolder(tmp_path, live_session(Setup(readout="x"))).poll()[0]
    assert update.n_objects == 5 and update.fov_metadata["position_id"] == "001"


def test_h11_a_field_written_to_an_empty_manifest_gets_its_header(tmp_path):
    (tmp_path / MANIFEST).write_text("")
    write_fov(tmp_path, pd.DataFrame({"x": [1.0]}), {"image_file": "a.tif"})
    assert (tmp_path / MANIFEST).read_text().splitlines()[0] == "image_file,objects_file"


def test_h14_h16_live_tables_keep_integers_and_say_the_unit():
    fovs = pd.DataFrame({"plate": [1, 2], "image_file": [10, 11]})
    objects = pd.DataFrame({"plate": [1, 1], "image_file": [10, 10], "x": [1.0, 2.0]})
    session = live_session(Setup(readout="x", observation_unit="nucleus", group_cols=("plate",)))
    update = LiveTables(session).update(objects, fovs)[0]
    assert isinstance(update.fov_metadata["plate"], int)
    assert "2 nuclei" in update.summary()


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"interval": 0}, "interval must be a finite number of seconds above 0, not 0"),
        ({"until_stop": "x"}, "until_stop must be False, True, 'all_groups' or 'latest_group'"),
        ({"stop_on": "x"}, "stop_on must be 'decision' or 'rule', not 'x'"),
    ],
)
def test_watch_refuses_its_arguments(tmp_path, arguments, message):
    with pytest.raises(ValueError) as raised:
        LiveFolder(tmp_path, live_session(AREA_SETUP)).watch(on_update=None, **arguments)
    assert str(raised.value) == message


def test_live_tables_refuse_tables_that_shrink_or_late_objects():
    objects, fovs = _sictrl_tables()
    with pytest.raises(TypeError) as raised:
        LiveTables(live_session(AREA_SETUP)).update(objects.to_dict(), fovs)
    assert str(raised.value) == "objects and fovs must be pandas DataFrames"
    tables = LiveTables(live_session(AREA_SETUP))
    tables.update(objects.iloc[:6], fovs.iloc[:2])
    with pytest.raises(ValueError) as raised:
        tables.update(objects.iloc[:3], fovs.iloc[:2])
    assert str(raised.value) == "the tables only grow: rows already read were removed"
    with pytest.raises(ValueError) as raised:
        tables.update(pd.concat([objects.iloc[:6], objects.iloc[:1]]), fovs.iloc[:2])
    assert str(raised.value) == (
        "objects appended for field(s) already added [('siCTRL', '2026_01_01', "
        "'siCTRL_0.tif')]; append a field's objects before its FOV row"
    )


def test_live_tables_hold_objects_until_their_field():
    objects, fovs = _sictrl_tables()
    tables = LiveTables(live_session(AREA_SETUP))
    assert tables.update(objects.iloc[:3], fovs.iloc[:0]) == []
    assert tables.pending_objects == 3
    updates = tables.update(objects.iloc[:6], fovs.iloc[:1])
    assert [u.fov_metadata["image_file"] for u in updates] == ["siCTRL_0.tif"]
    assert [(u.n_objects, u.pending_objects) for u in updates] == [(3, 3)]
    assert tables.pending_objects == 3 and len(tables.updates) == 1


def _live_folder(folder):
    objects, fovs = _sictrl_tables()
    for _, field in fovs.iterrows():
        mine = objects.loc[objects["image_file"] == field["image_file"], ["label_id", "area"]]
        write_fov(folder, mine, dict(field))
    return LiveFolder(folder, live_session(AREA_SETUP))


def test_watch_hands_each_update_over_and_stops_on_ctrl_c(tmp_path):
    seen = []
    updates = _live_folder(tmp_path / "a").watch(interval=0.01, timeout=0.05, on_update=seen.append)
    names = ["siCTRL_0.tif", "siCTRL_1.tif", "A_0.tif", "A_1.tif"]
    assert [u.fov_metadata["image_file"] for u in seen] == names
    assert [u.fov_metadata["image_file"] for u in updates] == names

    def interrupt(update):
        raise KeyboardInterrupt

    kept = _live_folder(tmp_path / "b").watch(interval=0.01, on_update=interrupt)
    assert [u.fov_metadata["image_file"] for u in kept] == ["siCTRL_0.tif"]


def test_a_stop_the_order_check_flags_is_marked_to_look_at():
    setup = _nucleus_setup(acquisition_order_col="order", stage_cols=("stage_x", "stage_y"))
    updates = _updates(setup, *_well(drift=0.02))
    flagged = [u for u in updates if u.order_check.flagged and u.summary().count("stop:")]
    assert flagged
    short = _flat(flagged[0].summary())
    assert short.startswith("⚠️ field ") and "but the order check flags these fields" in short
    assert short.count("(a flag, not a veto") == 1
    calm = [u for u in updates if not u.order_check.flagged and "— stop:" in u.summary()]
    assert all(u.summary().startswith("✅") for u in calm)


def test_the_text_is_a_pure_function_of_the_update_and_fits_the_page():
    first = _updates(_nucleus_setup(), *_well())
    second = _updates(_nucleus_setup(), *_well())
    for a, b in zip(first, second, strict=True):
        a, b = (dataclasses.replace(u, read_seconds=0.0) for u in (a, b))
        for detail in ("short", "full"):
            assert a.summary(detail) == b.summary(detail)
        assert all(width(line) <= 100 for line in a.summary().splitlines())


def test_detail_is_checked_and_print_update_passes_it(capsys):
    update = _updates(_nucleus_setup(), *_well(n_fields=2))[-1]
    with pytest.raises(ValueError, match="short"):
        update.summary(detail="long")
    print_update(update, detail="full")
    assert capsys.readouterr().out.strip() == update.summary("full")
    print_update(update)
    assert capsys.readouterr().out.strip() == update.summary()


def _pointed(folder, target: str):
    """One field written by write_fov, its manifest line pointed at ``target`` instead."""
    write_fov(folder, *_field("A01", 0))
    manifest = folder / MANIFEST
    # Only files: iterdir order is filesystem-dependent, and a test may have made a
    # sub-folder of objects/ first (which, picked here, made the replace a silent no-op).
    written = [path for path in (folder / "objects").iterdir() if path.is_file()]
    assert len(written) == 1
    relative = written[0].relative_to(folder).as_posix()
    manifest.write_text(manifest.read_text().replace(relative, target))
    return _folder_session(folder)


@pytest.mark.parametrize("where", ["parent", "absolute", "symlink"])
def test_an_object_file_outside_the_folder_is_refused_before_it_is_read(tmp_path, where):
    """../, an absolute path or a symbolic link out of the folder is not this acquisition's
    field (#360): refused naming the line, nothing added, the line read again next poll."""
    folder, outside = tmp_path / "live", tmp_path / "other_run.csv"
    folder.mkdir()
    pd.DataFrame({"x": [1.0, 2.0]}).to_csv(outside, index=False)
    target = {"parent": "../other_run.csv", "absolute": str(outside)}.get(where, "objects/link.csv")
    live = _pointed(folder, target)
    if where == "symlink":
        (folder / "objects" / "link.csv").symlink_to(outside)
    read = []
    live.reader = lambda path: read.append(path) or pd.read_csv(path)
    for _ in range(2):
        with pytest.raises(ValueError, match="line 2 names the object file .* not inside"):
            live.poll()
    assert read == [] and _fields_in(live) == 0
    opted = LiveFolder(folder, live.session, outside_files=True)
    assert len(opted.poll()) == 1 and _fields_in(opted) == 1


def test_a_missing_object_file_outside_the_folder_is_named_when_opted_in(tmp_path):
    """outside_files=True lifts the folder check only: a missing file is still named."""
    folder = tmp_path / "live"
    folder.mkdir()
    live = _pointed(folder, "../gone.csv")
    opted = LiveFolder(folder, live.session, outside_files=True)
    with pytest.raises(FileNotFoundError, match="gone.csv, which does not exist"):
        opted.poll()
    assert _fields_in(opted) == 0


def test_a_nested_relative_object_file_inside_the_folder_is_read(tmp_path):
    folder = tmp_path / "live"
    (folder / "objects" / "well").mkdir(parents=True)
    nested = folder / "objects/well/f.csv"
    pd.DataFrame({"x": [1.0, 2.0, 3.0]}).to_csv(nested, index=False)
    live = _pointed(folder, "objects/well/f.csv")
    (update,) = live.poll()
    assert update.n_objects == 3  # the nested file was read, not the file write_fov wrote (5)


@pytest.mark.parametrize(
    "controls",
    [
        {"interval": 0},
        {"interval": float("inf")},
        {"interval": float("nan")},
        {"interval": True},
        {"timeout": -1},
        {"timeout": float("nan")},
        {"timeout": False},
        {"max_fovs": -1},
        {"max_fovs": 1.5},
        {"max_fovs": True},
    ],
)
def test_invalid_watch_controls_consume_no_field(tmp_path, controls):
    """Every control is checked before the first poll (#360): no field is added."""
    write_fov(tmp_path, *_field("A01", 0))
    live = _folder_session(tmp_path)
    with pytest.raises(ValueError, match="must be"):
        live.watch(**{"interval": 0.01, "timeout": 1, **controls}, on_update=None)
    assert _fields_in(live) == 0 and live.updates == []


def test_max_fovs_zero_adds_no_field(tmp_path):
    write_fov(tmp_path, *_field("A01", 0))
    live = _folder_session(tmp_path)
    assert live.watch(interval=0.01, timeout=1, max_fovs=0, on_update=None) == []
    assert _fields_in(live) == 0 and len(live.poll()) == 1

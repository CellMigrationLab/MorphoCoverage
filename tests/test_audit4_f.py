"""Regressions for the fourth audit, group F: live robustness (findings P2-5 to P2-11).

One test per finding, named after it; each fails on the code before the fix.
"""

from __future__ import annotations

import shutil
import warnings

import numpy as np
import pandas as pd
import pytest

from morphocoverage import FixedFOVStrategy
from morphocoverage.workflows import LiveFolder, LiveTables, Setup, live_session, write_fov
from morphocoverage.workflows.live import MANIFEST

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

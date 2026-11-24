"""The page's last section, "Plan the next experiment" (owner, 2026-10-02): the reader picks
what the next experiment should get right, or what the microscope should watch for, and the
code it gives runs on the session's ``result``, ``objects`` and ``fovs``."""

import html
import re
import warnings
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
from _helpers_rare_fingerprint import _screen, _setup

from morphocoverage import ObjectiveConfig, workflows
from morphocoverage.strategies import RareIncidenceStrategy
from morphocoverage.workflows import live_session, report
from morphocoverage.workflows._html_next import fov_counts
from morphocoverage.workflows._html_plan import plan_card


@pytest.fixture(scope="module")
def plain():
    """No fingerprint and no rare target: the populations' group is the one to plan for."""
    objects, fovs = _screen()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = report(objects, fovs, _setup(fingerprint=(), rare=None, rare_label=None))
    return objects, fovs, result


@pytest.fixture(scope="module")
def declared():
    """The mitotic label and the fingerprint declared."""
    objects, fovs = _screen()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = report(objects, fovs, _setup())
    return objects, fovs, result


def _choices(markup, name):
    """``{option: panel markup}`` of the picker ``name`` (the last panel runs on to the end of
    the markup, which the tests only search)."""
    block = markup.split(f'<select id="{name}-pick">', 1)[1].split('<div class="choose">')[0]
    listed = block.split("</select>", 1)[0]
    options = [html.unescape(o) for o in re.findall(r"<option[^>]*>(.*?)</option>", listed)]
    panels = re.split(rf'<div class="choice-panel" id="{name}-panel-\d+"[^>]*>', block)[1:]
    return dict(zip(options, panels, strict=True))


def _code(panel):
    return html.unescape(re.search(r"<pre><code>(.*?)</code></pre>", panel, re.DOTALL).group(1))


class _Plan:
    def summary(self):
        return "the plan"


def _fake_optimise(seen):
    def optimise(objects, fovs, setup, **settings):
        seen.update(objects=objects, setup=setup, **settings)
        return _Plan()

    return optimise


class _Folder:
    """``LiveFolder`` that keeps the session it is given instead of watching a folder."""

    seen: ClassVar[dict] = {}

    def __init__(self, folder, session):
        _Folder.seen = {"folder": folder, "session": session}

    def watch(self, **settings):
        _Folder.seen["watch"] = settings


def test_the_readout_comes_first_and_its_code_runs(plain, monkeypatch):
    objects, fovs, result = plain
    seen = {}
    monkeypatch.setattr(workflows, "optimise", _fake_optimise(seen))
    choices = _choices(plan_card(result, objects, fovs, str), "plan-optimise")
    first = next(iter(choices))
    assert first.startswith(f"The mean of {result.setup.readout} in every well")
    exec(_code(choices[first]), {"result": result, "objects": objects, "fovs": fovs})  # noqa: S102
    assert seen["setup"].readout == result.setup.readout
    assert seen["fov_counts"] == fov_counts(fovs) and seen["reservoir_fraction"] == 0.35


def test_a_group_is_planned_where_it_is_rarest_with_its_rule(plain, monkeypatch):
    objects, fovs, result = plain
    seen = {}
    monkeypatch.setattr(workflows, "optimise", _fake_optimise(seen))
    choices = _choices(plan_card(result, objects, fovs, str), "plan-optimise")
    name, panel = next((o, p) for o, p in choices.items() if " seen in at least 3 fields" in o)
    group = name.split()[0]
    words = re.sub(r"<[^>]+>", " ", panel)
    assert "Once" in words and "validated on the images" in words and "rarest" in words
    exec(_code(panel), {"result": result, "objects": objects.copy(), "fovs": fovs})  # noqa: S102
    setup, labelled = seen["setup"], seen["objects"][group]
    assert setup.rare_label == group and labelled.dtype == bool
    # The share of cells the rule labels, not the cluster's.
    assert setup.rare.object_prevalence == pytest.approx(labelled.mean(), abs=1e-4)
    # The share of fields holding it in the condition where it is rarest.
    key = ["condition", "biological_repeat", "technical_repeat", "image_file"]
    held = seen["objects"].groupby(key)[group].any().groupby(level="condition").mean()
    assert setup.rare.fov_incidence == pytest.approx(held[held > 0].min(), abs=1e-4)


def test_live_on_a_group_stops_on_the_group_alone(plain, monkeypatch):
    objects, fovs, result = plain
    monkeypatch.setattr(workflows, "LiveFolder", _Folder)
    choices = _choices(plan_card(result, objects, fovs, str), "plan-live")
    name, panel = next(iter(choices.items()))
    assert "has been seen in 3 fields" in name
    # max_risk is a posterior risk, not the false-stop rate at 3 occurrences (#345, in #343).
    assert "the posterior risk of stopping too early is 5%" in re.sub(r"\s+", " ", panel)
    exec(_code(panel), {"result": result, "objects": objects, "fovs": fovs})  # noqa: S102
    session = _Folder.seen["session"]
    assert _Folder.seen["watch"]["until_stop"] == "latest_group"
    assert session.objective.objective.value == "rare_discovery"
    [rule] = session.strategies
    assert isinstance(rule, RareIncidenceStrategy)


def test_live_on_a_mean_clears_the_other_declared_targets(declared, monkeypatch):
    objects, fovs, result = declared
    monkeypatch.setattr(workflows, "LiveFolder", _Folder)
    choices = _choices(plan_card(result, objects, fovs, str), "plan-live")
    panel = choices[f"…the mean of {result.setup.readout} is known"]
    exec(_code(panel), {"result": result, "objects": objects, "fovs": fovs})  # noqa: S102
    # The fingerprint and the rare target would wait for rules this stop does not run.
    assert _Folder.seen["session"].objective.objective.value == "population"


def test_the_declared_label_and_fingerprint_are_offered(declared, monkeypatch):
    objects, fovs, result = declared
    seen = {}
    monkeypatch.setattr(workflows, "optimise", _fake_optimise(seen))
    monkeypatch.setattr(workflows, "LiveFolder", _Folder)
    markup = plan_card(result, objects, fovs, str)
    plans, live = _choices(markup, "plan-optimise"), _choices(markup, "plan-live")
    assert "mitotic seen in at least 3 fields of every well" in plans
    assert "The fingerprint in every well" in plans
    # The declared label's group is not offered again as a new group.
    assert not any(o.startswith("G") for o in plans)
    panel = live["…mitotic has been seen in 3 fields"]
    exec(_code(panel), {"result": result, "objects": objects, "fovs": fovs})  # noqa: S102
    target = _Folder.seen["session"].objective.rare
    assert target.fov_incidence > 0 and target.min_fov_occurrences == 3


def test_a_difference_is_planned_in_repeats_for_the_conditions_not_yet_significant(plain):
    objects, fovs, result = plain
    choices = _choices(plan_card(result, objects, fovs, str), "plan-optimise")
    panel = choices["A difference from the control"]
    assert "<pre>" not in panel and "biological repeats" in panel
    confirmed = result.discovery.tests.query("confirmed")
    rows = re.findall(r"<tr><td>(.*?)</td><td>(.*?)</td>", panel)
    assert not {(f, c) for f, c in rows} & set(zip(confirmed["feature"], confirmed["condition"]))


def test_with_one_repeat_the_difference_says_how_many_to_plan(plain):
    objects, fovs, _ = plain
    one = fovs["biological_repeat"] == fovs["biological_repeat"].iloc[0]
    kept = objects["biological_repeat"].isin(set(fovs.loc[one, "biological_repeat"]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = report(
            objects[kept], fovs[one], _setup(fingerprint=(), rare=None, rare_label=None)
        )
    choices = _choices(plan_card(result, objects[kept], fovs[one], str), "plan-optimise")
    assert "With 1 biological repeat" in choices["A difference from the control"]


def test_live_session_takes_a_narrower_objective():
    setup = _setup(fingerprint=(), rare=None, rare_label=None)
    with pytest.raises(TypeError, match="ObjectiveConfig"):
        live_session(setup, objective="rare_discovery")
    target = _setup().rare
    target = type(target)(
        fov_incidence=0.2, detection_probability=0.95, min_fov_occurrences=3, phenotype="m"
    )
    session = live_session(
        setup,
        rules=(RareIncidenceStrategy("mitotic", 1.0, 1.0, 0.05),),
        objective=ObjectiveConfig("rare_discovery", rare=target),
    )
    assert session.objective.objective.value == "rare_discovery"


def test_the_page_ends_with_the_plan_its_code_left_as_it_is(plain, tmp_path):
    from morphocoverage.workflows import html_report

    objects, fovs, result = plain
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        page = html_report(result, objects, fovs, tmp_path).read_text(encoding="utf-8")
    order = [page.index(f'id="{key}"') for key in ("limits", "plan", "files")]
    assert order == sorted(order) and 'href="#plan"' in page
    plan = page[order[1] : order[2]]
    # Neither the code nor a list's options carry the glossary's links.
    for block in re.findall(r"<pre>.*?</pre>|<select.*?</select>", plan, re.DOTALL):
        assert "<a " not in block and 'class="tip"' not in block
    # The cards no longer give optimise's code: the plan does.
    before = page[: order[1]]
    assert "optimise(objects" not in html.unescape(before)


def test_with_one_feature_the_readouts_own_verdicts_say_what_is_significant(plain):
    from types import SimpleNamespace

    from morphocoverage.sampling_unit.reading_text import CONFIRMED, condition_readings
    from morphocoverage.workflows._html_plan import _difference_goal

    _, _, result = plain
    contrast = result.results["contrast"]
    one = SimpleNamespace(
        setup=SimpleNamespace(readout=result.setup.readout, columns=(result.setup.readout,)),
        discovery=None,
        results={"contrast": contrast},
    )
    readings = condition_readings(contrast)
    confirmed = set(readings.loc[readings["verdict"] == CONFIRMED, "condition"])
    _, panel = _difference_goal(one)
    listed = {c for _, c in re.findall(r"<tr><td>(.*?)</td><td>(.*?)</td>", panel)}
    assert not listed & confirmed


def test_a_pilot_condition_is_named_and_the_others_still_planned(plain):
    """A condition with 2 repeats beside conditions with 3: the pilot sentence names it, and
    the others keep their plan (stress test, 2026-10-02)."""
    objects, fovs, _ = plain
    last = max(set(fovs["biological_repeat"]))

    def drop(table):
        return table[~((table["condition"] == "narrow") & (table["biological_repeat"] == last))]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = report(
            drop(objects), drop(fovs), _setup(fingerprint=(), rare=None, rare_label=None)
        )
    panel = _choices(plan_card(result, drop(objects), drop(fovs), str), "plan-optimise")[
        "A difference from the control"
    ]
    words = html.unescape(re.sub(r"<[^>]+>", " ", panel))
    assert "With 2 biological repeats for narrow" in words
    listed = {c for _, c in re.findall(r"<tr><td>(.*?)</td><td>(.*?)</td>", panel)}
    assert "narrow" not in listed
    assert listed or "Every condition with 3 or more repeats is significant" in words


def test_the_incidence_counts_the_fields_without_observations():
    # Regression (#343): fields with no observation had no object row and left the
    # denominator, so the share of fields holding the label, and the plan, came out higher.
    from types import SimpleNamespace

    import pandas as pd

    from morphocoverage.workflows._html_plan import _incidence

    objects, fovs = _screen()
    empty = (
        fovs.groupby("condition", sort=False)
        .head(12)
        .assign(image_file=lambda t: t["image_file"].str.replace(".tif", "_empty.tif"))
    )
    every = pd.concat([fovs, empty], ignore_index=True)
    result = SimpleNamespace(setup=_setup(fingerprint=(), rare=None, rare_label=None))
    incidence, where = _incidence(result, objects, every, objects["mitotic"])
    held = objects.groupby("image_file")["mitotic"].any()
    shares = (
        every.assign(held=every["image_file"].map(held).fillna(False).astype(bool))
        .groupby("condition")["held"]
        .mean()
    )
    assert where == shares.idxmin() and incidence == pytest.approx(shares.min())
    assert incidence < _incidence(result, objects, fovs, objects["mitotic"])[0]


def test_a_condition_without_a_repeat_plan_is_said_not_assessable_in_the_table():
    # #90: a plan with no estimable answer was dropped silently; it stays, with its reason.
    from morphocoverage.sampling_unit.repeat_plan import DEGENERATE
    from morphocoverage.workflows._html_plan import _difference_goal

    discovery = SimpleNamespace(
        tests=pd.DataFrame(
            {
                "feature": ["area"] * 3,
                "condition": ["a", "b", "c"],
                "repeats": [3, 3, 3],
                "confirmed": [True, False, False],
            }
        ),
        ranking=pd.DataFrame({"feature": ["area"]}),
    )
    plan = pd.DataFrame(
        {
            "condition": ["a", "b", "c"],
            "repeats_needed": [4.0, np.nan, 5.0],
            "repeats_needed_twice_fields": [3.0, np.nan, np.inf],
            "planned_on": ["log ratio", "difference", "log ratio"],
            "change": [0.2, np.nan, -0.1],
            "status": ["ok", DEGENERATE, "ok"],
        }
    )
    contrast = SimpleNamespace(plan=plan, table=pd.DataFrame(), settings={})
    result = SimpleNamespace(
        discovery=discovery,
        setup=SimpleNamespace(readout="area", columns=("area",)),
        results={"contrast": contrast},
    )
    option, panel = _difference_goal(result)
    assert option == "A difference from the control"
    assert panel == (
        "<p>Optimise plans each well&#x27;s own measurements, not a comparison: a difference "
        "from the control is planned in biological repeats. For each condition not yet "
        "significant across repeats, the repeats a paired t-test on the repeat means needs for "
        "80% power, if the change and the variation seen here recur, each repeat with its own "
        "control. Significant across repeats already: area. Not assessable: these data give "
        "no plan, for the reason in its row.</p>"
        '<div class="table-wrap"><table class="sortable"><thead><tr><th>feature</th>'
        "<th>condition</th><th>change seen</th><th>repeats needed</th>"
        "<th>with twice the fields per repeat</th></tr></thead><tbody><tr><td>area</td>"
        f"<td>b</td><td>–</td><td>not assessable: {html.escape(DEGENERATE)}</td><td>–</td>"
        "</tr><tr><td>area</td>"
        "<td>c</td><td>×-0.10</td><td>5</td><td>more than 30</td></tr></tbody></table></div>"
    )

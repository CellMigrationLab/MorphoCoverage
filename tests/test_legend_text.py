"""Tests of ``sampling_unit.legend_text``: what a figure legend says for each plotted feature."""

from __future__ import annotations

import pandas as pd


def _legend_none():
    from morphocoverage.sampling_unit.legend_text import figure_legend

    return figure_legend(None, pd.DataFrame(), "cells")


EXPECTED_LEGEND_NONE = None


def test_legend_none():
    assert _legend_none() == EXPECTED_LEGEND_NONE

"""Tests of ``glossary``: the report's glossary."""

from __future__ import annotations

from morphocoverage.glossary import GLOSSARY


def test_the_glossary_says_unusual_apart_from_nothing_new_a24():
    terms = {term.name: term for term in GLOSSARY}
    assert "definition of new" not in terms["Unusual cells"].meaning
    assert "definition of new" in terms["'Nothing new'"].meaning


def test_the_sign_flip_entry_says_magnitudes_and_exact_or_monte_carlo_578():
    # The test flips the signs of the per-repeat differences and keeps their magnitudes
    # (not a test on the signs alone); it enumerates up to 16 repeats, Monte Carlo beyond.
    meaning = {term.name: term for term in GLOSSARY}["Sign-flip test"].meaning
    assert "test on the signs" not in meaning
    assert "magnitude" in meaning and "mean" in meaning
    assert "exact up to 16" in meaning and "Monte Carlo" in meaning

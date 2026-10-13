"""Tests of ``glossary``: the report's glossary."""

from __future__ import annotations

from morphocoverage.glossary import GLOSSARY


def test_the_glossary_says_unusual_apart_from_nothing_new_a24():
    terms = {term.name: term for term in GLOSSARY}
    assert "definition of new" not in terms["Unusual cells"].meaning
    assert "definition of new" in terms["'Nothing new'"].meaning

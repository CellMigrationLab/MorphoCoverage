"""The glossary of the report: every test, rule and term it uses, in plain words, with the
papers they come from.

``GLOSSARY`` is a tuple of ``Term`` (``name``, ``meaning``, ``where`` in the report, the
``references``), grouped by ``section``. The HTML report prints it as its last section, and
``glossary_markdown()`` is ``docs/glossary.md`` (a test keeps the two the same). A reference
links to its DOI only where the DOI is established; methods of the lab without a paper are
named without a link.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Reference:
    """A paper or tool: how to cite it, and its link (``None`` when there is none to give)."""

    text: str
    url: str | None = None


@dataclass(frozen=True)
class Term:
    """One glossary entry."""

    section: str
    name: str
    meaning: str
    where: str
    references: tuple[Reference, ...] = ()


ZWEIFACH = Reference(
    "Zweifach A. Determining how many cells to average for statistical testing of microscopy "
    "experiments. J Cell Biol. 2024.",
    "https://doi.org/10.1083/jcb.202401074",
)
MARCOTTI = Reference("Marcotti et al. J Cell Sci. 2026.", "https://doi.org/10.1242/jcs.264367")
SUPERPLOTS = Reference(
    "Lord SJ, Velle KB, Mullins RD, Fritz-Laylin LK. SuperPlots: communicating reproducibility "
    "and variability in cell biology. J Cell Biol. 2020;219(6):e202001064.",
    "https://doi.org/10.1083/jcb.202001064",
)
PLOTSOFDATA = Reference(
    "Postma M, Goedhart J. PlotsOfData: a web app for visualizing data together with their "
    "summaries. PLoS Biol. 2019;17(3):e3000202.",
    "https://doi.org/10.1371/journal.pbio.3000202",
)
BH = Reference(
    "Benjamini Y, Hochberg Y. Controlling the false discovery rate: a practical and powerful "
    "approach to multiple testing. J R Stat Soc B. 1995;57(1):289-300.",
    "https://doi.org/10.1111/j.2517-6161.1995.tb02031.x",
)
UMAP = Reference(
    "McInnes L, Healy J, Melville J. UMAP: Uniform Manifold Approximation and Projection for "
    "dimension reduction. arXiv:1802.03426. 2018.",
    "https://arxiv.org/abs/1802.03426",
)
REN = Reference("Ren et al. Nonparametric bootstrapping for hierarchical data. J Appl Stat. 2010.")
COHEN = Reference("Cohen J. Statistical Power Analysis for the Behavioral Sciences. 2nd ed. 1988.")
PLOT_AND_STATS = Reference("Plot&Stats, the Cell Migration Lab's analysis notebook.")
LAB_PAPER = Reference(
    "CCT8 associates with the MYO10 motor domain and regulates filopodia and breast cancer cell "
    "invasion. J Cell Sci. 2026 (the lab datasets).",
    "https://doi.org/10.1242/jcs.265278",
)

GLOSSARY: tuple[Term, ...] = (
    Term(
        "Reading the report",
        "Status markers",
        "✅ answered; ⚠️ answered, with a caution about these data; ❌ check the data first; ➖ "
        "not applicable or not assessed. A marker says how far to trust an answer, not what it "
        "found: ✅ can mean 'no difference'.",
        "At a glance, every answer",
    ),
    Term(
        "Reading the report",
        "Assumptions",
        "What an answer rests on that these data cannot check (cells as independent draws, "
        "fields as a random sample of the well). Stated once, after the answer, in grey.",
        "Every answer",
    ),
    Term(
        "The design",
        "Field (FOV)",
        "One imaged position of the stage: the unit a microscopist acquires, and the unit most "
        "answers count, because cells of one field are not independent of each other.",
        "Everywhere",
    ),
    Term(
        "The design",
        "Biological repeat",
        "An independent experiment (often its date). The replicate that a conclusion about the "
        "conditions must hold across.",
        "Question 9, the design scan",
    ),
    Term(
        "The design",
        "Technical repeat",
        "A well, dish or coverslip within a biological repeat; never a biological repeat. "
        "'single' when there is one.",
        "Question 9, the plots (colours), question 7",
    ),
    Term(
        "The design",
        "Pilot",
        "A dataset of one or two biological repeats. It gets every answer its design supports "
        "and advice for the next experiment, not a verdict across repeats.",
        "Question 9, the plan",
    ),
    Term(
        "The design",
        "Normalised values",
        "Each value divided by the mean of the control of its own biological repeat, so a "
        "scale shared by a repeat (a staining or imaging batch) is removed. Needs a positive "
        "control mean.",
        "Question 9, the plots, the PCA and UMAP",
    ),
    Term(
        "Did I image enough?",
        "Mean-precision rule",
        "Stops once the mean of the readout is precise enough: its averaging CV (one standard "
        "error as a share of the mean) at or below the target, e.g. 5%. 'Per cell' treats "
        "cells as independent; 'per field' uses field means. A CV of 5% is one standard error "
        "of 5%, not a ±5% bound.",
        "Question 2, the stopping rules",
        (ZWEIFACH,),
    ),
    Term(
        "Did I image enough?",
        "Mean-precision count check",
        "Whether the number of cells the mean-precision rule asks for still gives that "
        "precision when the cells come as whole fields, as they are acquired.",
        "Foundation",
        (ZWEIFACH,),
    ),
    Term(
        "Did I image enough?",
        "JCS spread and plateau rules",
        "Stop once the mean, median, SD and IQR of a feature stabilise as cells are added "
        "(within a tolerance you choose, in the feature's units).",
        "Question 2, the stopping rules",
        (MARCOTTI,),
    ),
    Term(
        "Did I image enough?",
        "Held-back fields",
        "Fields set aside before a rule is replayed, so whether the rule stopped too early is "
        "judged on fields it never saw.",
        "Questions 4, 5 and the rules' reliability (optimise)",
    ),
    Term(
        "Did I image enough?",
        "Resolved",
        "The 95% interval of the difference from the control, from the fields resampled as "
        "they were acquired, excludes zero.",
        "Question 2",
    ),
    Term(
        "Do the conditions differ?",
        "Cohen's d",
        "The difference of two means in units of their pooled standard deviation: 0.2 small, "
        "0.5 medium, 0.8 large by Cohen's convention. Sign: negative is lower than the "
        "control.",
        "Question 9, the feature ranking",
        (COHEN,),
    ),
    Term(
        "Do the conditions differ?",
        "Cell-level randomization test",
        "Shuffles the condition labels over every cell of a condition and its control, as the "
        "lab's Plot&Stats notebook does. It asks whether these wells differ, not whether the "
        "effect repeats: when the difference varies between repeats it gives p ≤ 0.05 more "
        "often than 5% without an effect, and the report prints that chance beside it.",
        "Question 9, the discovery gate",
        (PLOT_AND_STATS,),
    ),
    Term(
        "Do the conditions differ?",
        "Balanced test",
        "The cell-level test when the repeats hold different numbers of cells: each draw takes "
        "the same number from every repeat, so none weighs more; the median d and p over 100 "
        "draws.",
        "Question 9",
    ),
    Term(
        "Do the conditions differ?",
        "Sparse repeat",
        "A repeat with under a quarter of the median repeat's cells, or fewer than 5: left out "
        "of the balanced test only, and named so it can be checked.",
        "Question 9, the design scan",
    ),
    Term(
        "Do the conditions differ?",
        "Paired t-test on the repeat means",
        "Each biological repeat's condition mean paired with its own control mean, the repeat "
        "as the replicate (as in SuperPlots). Keeps false positives at 5%, but with 3 repeats "
        "it confirms only large, consistent effects. Given raw and on log ratios to the "
        "control (normalised).",
        "Question 9: the only test that confirms a difference",
        (SUPERPLOTS,),
    ),
    Term(
        "Do the conditions differ?",
        "Sign-flip test",
        "The exact test on the signs of the per-repeat differences; given only from 6 repeats, "
        "where its smallest p can reach 0.05 after the correction.",
        "Question 9, full detail",
    ),
    Term(
        "Do the conditions differ?",
        "Bonferroni correction",
        "Each p-value multiplied by the number of conditions tested, so testing several does "
        "not raise the chance of one false positive.",
        "Question 9",
    ),
    Term(
        "Do the conditions differ?",
        "Benjamini-Hochberg q-value",
        "A p-value corrected for the many features tested together: among the features with "
        "q ≤ 0.05, about 5% are expected to be false discoveries.",
        "Question 9, the feature ranking",
        (BH,),
    ),
    Term(
        "Do the conditions differ?",
        "Discovery gate and confirmed across repeats",
        "A feature passes the discovery gate when its cell-level test has q ≤ 0.05; it is "
        "confirmed across repeats when the paired test on the repeat means passes too. The "
        "first finds candidates; the second says the effect repeats.",
        "Question 9, the feature ranking",
    ),
    Term(
        "Do the conditions differ?",
        "Reagents of one target",
        "Two siRNAs (or other reagents) of one gene, named in the 'target' column: when both "
        "go the same way in every repeat, an off-target effect of one of them is less likely. "
        "Described, not tested.",
        "Question 9",
        (LAB_PAPER,),
    ),
    Term(
        "Do the conditions differ?",
        "JCS effect size",
        "The treated mean minus the control mean over the control SD, from cells drawn as "
        "random cells and as whole fields.",
        "Question 9, full detail",
        (MARCOTTI,),
    ),
    Term(
        "Figures",
        "PlotsOfData-style plot",
        "Every cell a point coloured by technical repeat, the biological repeats side by side, "
        "each repeat's median with its 95% interval from resampling whole technical repeats "
        "and their fields as acquired (never single cells).",
        "Question 9's figures",
        (PLOTSOFDATA, SUPERPLOTS, REN),
    ),
    Term(
        "Figures",
        "PCA and loadings",
        "The two directions along which the standardised features vary most, each labelled "
        "with its share of the variance; an arrow shows how much a best-ranked feature weighs "
        "on each direction. Describes, does not test.",
        "Question 9's figures",
    ),
    Term(
        "Figures",
        "UMAP",
        "A map that keeps each cell's nearest neighbours close, so it can show groups that PCA "
        "mixes. Its axes have no units and distances between groups are not to scale. "
        "Describes, does not test; on raw values a separation can be a batch between repeats.",
        "Question 9's figures",
        (UMAP,),
    ),
    Term(
        "The data",
        "Integrity checks",
        "Exact checks for accidents of image analysis: a field exported twice, rows pasted into "
        "another field, a change of units, a summary written twice. A finding is structure to "
        "inspect, not a judgement.",
        "The data, question 6, the design scan",
    ),
    Term(
        "The data",
        "Unusual cells and 'nothing new'",
        "Cells far from the control's, and whether later fields still showed states earlier "
        "fields did not (the definition of new is declared in the setup).",
        "Questions 2 and 3",
    ),
    Term(
        "The data",
        "Positive control (representation)",
        "A condition known to differ, used to check that the features can see the phenotype "
        "at all before 'nothing new' is trusted.",
        "Question 1",
    ),
    Term(
        "The data",
        "Between-field share",
        "The part of a feature's variance that lies between fields rather than between cells "
        "of one field: when it is large, more fields help more than more cells per field.",
        "Foundation, question 3",
    ),
)


def _reference_markdown(reference: Reference) -> str:
    return f"[{reference.text}]({reference.url})" if reference.url else reference.text


def glossary_markdown() -> str:
    """``docs/glossary.md``: the glossary by section."""
    lines = [
        "# Glossary",
        "",
        (
            "Every test, rule and term the report uses, in plain words, with where it appears "
            "and the papers it comes from. Generated from `morphocoverage.glossary` (the HTML "
            "report prints the same entries); edit them there."
        ),
    ]
    section = None
    for term in GLOSSARY:
        if term.section != section:
            section = term.section
            lines += ["", f"## {section}"]
        lines += ["", f"**{term.name}.** {term.meaning}", "", f"*Where:* {term.where}."]
        if term.references:
            refs = "; ".join(_reference_markdown(r) for r in term.references)
            lines.append(f"*See:* {refs}")
    return "\n".join(lines) + "\n"


__all__ = ["GLOSSARY", "Reference", "Term", "glossary_markdown"]

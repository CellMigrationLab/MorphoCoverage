"""The benchmark behind MorphoCoverage's rules (docs/common/validation.md).

A common design skeleton (``design``), separate outcome generators (``scalar``,
``embedding``, ``populations``, ``sequential``), one runner (``runner``, ``summary``), one
script per module (``p1_repeat_inference`` ... ``p5_multiple_testing``) and the candidate
methods that the package does not hold yet (``candidates``, never wired into the reports).
Run from the repository root with ``PYTHONPATH=src:data/simulations python -m
decisions.<script>`` (README.md). This is validation code, not part of the package.
"""

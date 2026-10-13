"""The guided session's scripted answers and its cells, shared by the guided tests."""

import numpy as np
import pandas as pd


def _cells(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for condition, shift in (("siCTRL", 0), ("siA#1", 8), ("siA#2", 9)):
        for repeat in ("2026_01_01", "2026_02_01", "2026_03_01"):
            for field in range(4):
                for label in range(6):
                    rows.append(
                        {
                            "condition": condition,
                            "biological_repeat": repeat,
                            "image_file": f"{condition}_{repeat}_{field}.tif",
                            "label_id": label + 1,
                            "area": rng.normal(100 + shift, 10),
                            "intensity": rng.normal(5, 1),
                            "aspect": rng.normal(2, 0.2),
                        }
                    )
    return pd.DataFrame(rows)


# The scale question of plan v2 (owner, 2026-10-06), asked for features named like an area or
# an intensity: answered with its default, Enter, unless a test scripts it (``scales=False``).
SCALE_PROMPT = "Test the changes of "


# Whether sets of conditions are separate experiments (audit 10, owner, 2026-10-06): answered
# with its default, No, unless a test scripts it (``experiments=False``).
EXPERIMENT_PROMPTS = ("Are the conditions of each control a separate", "Are they one experiment")


class Script:
    """Answers the prompts in order and records what was said."""

    def __init__(self, answers, scales=True, experiments=True):
        self.answers, self.said, self.prompts = list(answers), [], []
        self.scales, self.experiments = scales, experiments

    def ask(self, prompt):
        self.prompts.append(prompt)
        if self.scales and prompt.startswith(SCALE_PROMPT):
            return ""
        if self.experiments and prompt.startswith(EXPERIMENT_PROMPTS):
            return ""
        if not self.answers:
            raise AssertionError(f"no scripted answer left for {prompt!r}")
        return self.answers.pop(0)

    def say(self, text):
        self.said.append(text)

    @property
    def text(self):
        return "\n".join(self.said)

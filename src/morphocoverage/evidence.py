"""Evidence: what an estimator says about the visible acquisition, without deciding.

An estimator returns named ``Quantity`` values. Each one states what it is
evidence about (its objective ``component``), which direction is better, and
its units, so that decision policies and benchmarks never have to guess what a
bare number means. A quantity that cannot be computed yet, or that is undefined
for these data, says so explicitly instead of being a ``None`` score.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# Effort is framework-owned (AGENTS.md section 20); an estimator reporting its
# own count could disagree with the acquisition record, e.g. by dropping
# zero-object FOVs.
RESERVED_QUANTITY_NAMES = frozenset({"n_fovs", "n_objects"})


class Component(str, Enum):
    """The part of a sampling objective that a quantity is evidence about."""

    POPULATION = "population"
    RARE = "rare"


class Direction(str, Enum):
    """Which way a quantity moves as the acquisition becomes more adequate."""

    LOWER_IS_BETTER = "lower_is_better"
    HIGHER_IS_BETTER = "higher_is_better"


class Availability(str, Enum):
    """Whether a quantity has a value on the visible data.

    ``INSUFFICIENT_DATA``: more acquisition may make it computable.
    ``NOT_APPLICABLE``: it is undefined for these data (for example a CV of a
    feature with negative values) and more acquisition will not fix that.
    """

    AVAILABLE = "available"
    INSUFFICIENT_DATA = "insufficient_data"
    NOT_APPLICABLE = "not_applicable"


def _coerce(enum: type[Enum], value: object, name: str) -> Enum:
    try:
        return enum(value)
    except ValueError:
        allowed = [member.value for member in enum]
        raise ValueError(f"unknown {name} {value!r}; expected one of {allowed}") from None


@dataclass(frozen=True)
class Quantity:
    """One named piece of evidence, with its meaning stated.

    An available quantity has a numeric value (``inf`` is allowed, ``NaN`` is
    not: an undefined value must be reported as unavailable, with a reason).
    An unavailable quantity has ``value=None`` and a ``reason``.
    """

    value: float | None
    direction: Direction
    units: str
    component: Component
    availability: Availability = Availability.AVAILABLE
    reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "direction", _coerce(Direction, self.direction, "direction"))
        object.__setattr__(self, "component", _coerce(Component, self.component, "component"))
        object.__setattr__(
            self, "availability", _coerce(Availability, self.availability, "availability")
        )
        if not isinstance(self.units, str) or not self.units:
            raise ValueError("units must be a non-empty description")
        if self.availability is Availability.AVAILABLE:
            if isinstance(self.value, bool) or not isinstance(self.value, numbers.Real):
                raise TypeError("an available quantity needs a numeric value")
            if math.isnan(self.value):
                raise ValueError(
                    "NaN is not a value; report the quantity as insufficient_data or "
                    "not_applicable with a reason"
                )
            object.__setattr__(self, "value", float(self.value))
        else:
            if self.value is not None:
                raise ValueError("an unavailable quantity must have value=None")
            if not self.reason:
                raise ValueError("an unavailable quantity needs a reason")

    @classmethod
    def available(
        cls, value: float, *, direction: Direction, units: str, component: Component
    ) -> Quantity:
        """A computed value."""
        return cls(value, direction, units, component)

    @classmethod
    def insufficient(
        cls, reason: str, *, direction: Direction, units: str, component: Component
    ) -> Quantity:
        """Not computable yet; acquiring more may make it available."""
        return cls(None, direction, units, component, Availability.INSUFFICIENT_DATA, reason)

    @classmethod
    def not_applicable(
        cls, reason: str, *, direction: Direction, units: str, component: Component
    ) -> Quantity:
        """Undefined for these data; acquiring more will not make it available."""
        return cls(None, direction, units, component, Availability.NOT_APPLICABLE, reason)


@dataclass(frozen=True)
class Evidence:
    """Named quantities and free-form diagnostics from one estimator call."""

    quantities: Mapping[str, Quantity] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, quantity in self.quantities.items():
            if not isinstance(name, str) or not name:
                raise ValueError("quantity names must be non-empty strings")
            if name in RESERVED_QUANTITY_NAMES:
                raise ValueError(
                    f"{name!r} is framework-owned effort; estimators must not report it"
                )
            if not isinstance(quantity, Quantity):
                raise TypeError(f"quantity {name!r} must be a Quantity")

    def value(self, name: str) -> float | None:
        """The value of one quantity, ``None`` when it is unavailable."""
        return self.quantities[name].value

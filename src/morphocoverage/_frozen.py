"""Owned, read-only containers for frozen result snapshots (private; issue 108).

A dataclass advertised as a frozen scientific result must own what it holds: mutating the
dictionary a caller passed in, or the one the result exposes, must not change it. ``freeze``
copies the containers it is given (dicts, lists, sets, tuples, nested to any depth) into read-only
ones that still compare equal to, serialise as, pickle and deep-copy like the plain ones:

- a dict becomes a ``FrozenDict``, a list a ``FrozenList``, a set a ``frozenset``, a tuple a tuple
  of frozen items;
- anything else (numbers, text, enums, numpy arrays, pandas tables, objects) is kept as it is:
  the snapshot owns its containers, not the objects inside them, so a pandas table must be
  copied by its owner (a documented limit, not a silent one).

``thaw`` gives back plain, independent dicts and lists for code that needs to edit a copy.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, NoReturn

_READ_ONLY = "this snapshot is read-only: copy it (thaw(...)) to change a copy"


def _refuse(*_: Any, **__: Any) -> NoReturn:
    raise TypeError(_READ_ONLY)


class FrozenDict(dict[Any, Any]):
    """A dict that cannot be changed after construction (equal to, and serialised as, a dict)."""

    __setitem__ = __delitem__ = __ior__ = _refuse
    clear = pop = popitem = setdefault = update = _refuse

    def __reduce__(self) -> tuple[Any, ...]:
        return (FrozenDict, (dict(self),))

    def __deepcopy__(self, memo: dict[int, Any]) -> FrozenDict:
        return self

    def __copy__(self) -> FrozenDict:
        return self


class FrozenList(list[Any]):
    """A list that cannot be changed after construction (equal to, and serialised as, a list)."""

    __setitem__ = __delitem__ = __iadd__ = __imul__ = _refuse
    append = extend = insert = remove = pop = clear = sort = reverse = _refuse

    def __reduce__(self) -> tuple[Any, ...]:
        return (FrozenList, (list(self),))

    def __deepcopy__(self, memo: dict[int, Any]) -> FrozenList:
        return self

    def __copy__(self) -> FrozenList:
        return self


def freeze(value: Any) -> Any:
    """``value`` with every container it holds copied into a read-only one (see the module)."""
    if isinstance(value, FrozenDict | FrozenList):
        return value
    if isinstance(value, Mapping):
        return FrozenDict({k: freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return FrozenList(freeze(v) for v in value)
    if isinstance(value, tuple):
        return (
            type(value)(*map(freeze, value))
            if hasattr(value, "_fields")
            else tuple(freeze(v) for v in value)
        )
    if isinstance(value, set | frozenset):
        return frozenset(freeze(v) for v in value)
    return value


def thaw(value: Any) -> Any:
    """Plain, independent dicts, lists and tuples with the same contents as ``value``."""
    if isinstance(value, Mapping):
        return {k: thaw(v) for k, v in value.items()}
    if isinstance(value, list):
        return [thaw(v) for v in value]
    if isinstance(value, tuple):
        return tuple(thaw(v) for v in value)
    if isinstance(value, set | frozenset):
        return {thaw(v) for v in value}
    return value


def own(mapping: Mapping[str, Any] | Iterable[tuple[str, Any]]) -> FrozenDict:
    """A frozen copy of a mapping (``freeze`` for the top level a dataclass field holds)."""
    return freeze(dict(mapping))


__all__ = ["FrozenDict", "FrozenList", "freeze", "own", "thaw"]

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

import functools
from collections.abc import Callable, Iterable, Mapping
from typing import Any, NoReturn, TypeVar

import pandas as pd

_T = TypeVar("_T")

_READ_ONLY = "this snapshot is read-only: copy it (thaw(...)) to change a copy"


def _refuse(*_: Any, **__: Any) -> NoReturn:
    raise TypeError(_READ_ONLY)


class FrozenDict(dict[Any, Any]):
    """A dict that cannot be changed after construction (equal to, and serialised as, a dict).
    Built directly, it freezes what it is given, so ``freeze`` may trust any one (#581)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        dict.__init__(self, {k: freeze(v) for k, v in dict(*args, **kwargs).items()})

    __setitem__ = __delitem__ = __ior__ = _refuse
    clear = pop = popitem = setdefault = update = _refuse

    def __reduce__(self) -> tuple[Any, ...]:
        return (FrozenDict, (dict(self),))

    def __deepcopy__(self, memo: dict[int, Any]) -> FrozenDict:
        return self

    def __copy__(self) -> FrozenDict:
        return self


class FrozenList(list[Any]):
    """A list that cannot be changed after construction (equal to, and serialised as, a list).
    Built directly, it freezes what it is given, so ``freeze`` may trust any one (#581)."""

    def __init__(self, items: Iterable[Any] = ()) -> None:
        list.__init__(self, (freeze(v) for v in items))

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
        return FrozenDict(value.items())
    if isinstance(value, list):
        return FrozenList(value)
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


def own_value(value: Any) -> Any:
    """What a snapshot keeps of one field: containers frozen (``freeze``), a pandas table or
    column copied (an owned copy; a table cannot be made deeply read-only, so the limit is
    that the exposed table itself stays editable by whoever holds the result)."""
    if isinstance(value, pd.DataFrame | pd.Series):
        return value.copy(deep=True)
    return freeze(value)


def owned(*names: str) -> Callable[[type[_T]], type[_T]]:
    """Class decorator (above ``@dataclass(frozen=True)``): once the generated ``__init__`` has
    run, the fields ``names`` are replaced by owned copies (``own_value``), so mutating what the
    caller passed in cannot change the result (issue 108). It wraps ``__init__``, which also
    serves ``dataclasses.replace`` and runs after the class's own ``__post_init__``."""

    def wrap(cls: type[_T]) -> type[_T]:
        original = cls.__init__

        @functools.wraps(original)
        def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
            original(self, *args, **kwargs)
            for name in names:
                object.__setattr__(self, name, own_value(getattr(self, name)))

        cls.__init__ = __init__
        return cls

    return wrap


__all__ = ["FrozenDict", "FrozenList", "freeze", "own", "own_value", "owned", "thaw"]

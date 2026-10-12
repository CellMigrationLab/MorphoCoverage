"""Connected groups of items joined by pairs (union-find), shared by the analyses that
join features (``retrospective._feature_relations``) and conditions
(``sampling_unit._experiments``)."""

from __future__ import annotations

from collections.abc import Hashable, Iterable


def connected_groups(
    items: Iterable[Hashable], pairs: Iterable[tuple[Hashable, Hashable]]
) -> dict[Hashable, Hashable]:
    """Each item's group, named by the group's first item in ``items``: the items linked,
    directly or through others, by ``pairs``."""
    order: dict[Hashable, int] = {}
    for item in items:
        order.setdefault(item, len(order))
    listed = list(order)
    parent = list(range(len(listed)))

    def root(index: int) -> int:
        """The group's first item, the path halved on the way."""
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for first, second in pairs:
        a, b = root(order[first]), root(order[second])
        parent[max(a, b)] = min(a, b)
    return {item: listed[root(position)] for item, position in order.items()}


__all__ = ["connected_groups"]

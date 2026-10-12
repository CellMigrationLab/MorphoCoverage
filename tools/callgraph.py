"""Who calls what in the package, from the source tree alone (``ast``; imports nothing).

Each module's names are resolved through its import table (relative and absolute
``morphocoverage`` imports, re-exports followed, aliases kept), ``self.``/``cls.``/``super().``
through the class and its package bases, and enclosing functions' nested definitions. A row's
``calls`` lists the package definitions its own code calls or refers to (a callback passed
along counts: ``key=_order``), as ``module:qualified name``, and the outside names it calls:
``numpy.mean`` through an import, ``len`` for a builtin, ``?.name`` when the target cannot be
known from the source (a method of a value, a parameter). ``called_by`` is the reverse, for
package definitions only. Module and class rows hold what their own statements (constants,
decorators, bases, class attributes) refer to.
"""

from __future__ import annotations

import ast
import builtins
from collections.abc import Iterator

import function_index as fi

PACKAGE = "morphocoverage"
Binding = tuple[str, str]  # (what, target): def, class, module, from, external, builtin
_BUILTINS = set(dir(builtins))


class Package:
    """The definitions and import tables of every module, and name resolution over them."""

    def __init__(self, trees: list[tuple[str, ast.Module]]) -> None:
        self.modules = {name for name, _ in trees}
        self.packages = {"__init__"} | {
            m for m in self.modules if any(o.startswith(f"{m}.") for o in self.modules)
        }
        self.kinds: dict[str, str] = {}
        self.classes: dict[str, ast.ClassDef] = {}
        self.tables: dict[str, dict[str, Binding]] = {}
        for module, tree in trees:
            table = self._imports(module, tree)
            for qualified, node, kind in fi.walk_definitions(tree):
                key = f"{module}:{qualified}"
                self.kinds[key] = kind
                if isinstance(node, ast.ClassDef):
                    self.classes[key] = node
                if "." not in qualified and "#" not in qualified:
                    table[qualified] = ("def", key)  # the first definition of the name
            self.tables[module] = table

    def _module(self, dotted: str) -> str | None:
        name = dotted or "__init__"
        return name if name in self.modules else None

    def _imports(self, module: str, tree: ast.Module) -> dict[str, Binding]:
        table: dict[str, Binding] = {}
        if module in self.packages:
            package = "" if module == "__init__" else module
        else:
            package = module.rpartition(".")[0]
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    table.update(self._import(alias))
            elif isinstance(node, ast.ImportFrom):
                target = self._source(node, package)
                for alias in node.names:
                    if alias.name != "*":
                        table[alias.asname or alias.name] = self._from(node, target, alias)
        return table

    def _import(self, alias: ast.alias) -> dict[str, Binding]:
        name = alias.name
        inside = name == PACKAGE or name.startswith(f"{PACKAGE}.")
        if alias.asname is None:
            head = name.split(".")[0]
            return {head: ("module", "__init__") if head == PACKAGE else ("external", head)}
        if inside:
            module = self._module(name[len(PACKAGE) :].lstrip("."))
            return {alias.asname: ("module", module)} if module else {}
        return {alias.asname: ("external", name)}

    def _source(self, node: ast.ImportFrom, package: str) -> str | None:
        """The package-relative dotted name an ``ImportFrom`` reads, ``None`` outside."""
        if node.level:
            parts = package.split(".") if package else []
            parts = parts[: len(parts) - (node.level - 1)]
            return ".".join([*parts, *([node.module] if node.module else [])])
        module = node.module or ""
        if module == PACKAGE or module.startswith(f"{PACKAGE}."):
            return module[len(PACKAGE) :].lstrip(".")
        return None

    def _from(self, node: ast.ImportFrom, target: str | None, alias: ast.alias) -> Binding:
        if target is None:
            return ("external", f"{node.module}.{alias.name}")
        submodule = self._module(f"{target}.{alias.name}" if target else alias.name)
        if submodule:
            return ("module", submodule)
        return ("from", f"{self._module(target)}:{alias.name}")

    def symbol(self, module: str | None, name: str, depth: int = 0) -> Binding | None:
        """What ``name`` is in ``module``: re-exports followed to the definition."""
        if module is None or depth > 20:
            return None
        found = self.tables.get(module, {}).get(name)
        if found is None:
            submodule = self._module(name if module == "__init__" else f"{module}.{name}")
            return ("module", submodule) if submodule else None
        if found[0] == "from":
            source, _, inner = found[1].partition(":")
            return self.symbol(None if source == "None" else source, inner, depth + 1)
        return found

    def member(self, class_key: str, name: str, depth: int = 0) -> Binding | None:
        """The method or nested class ``name`` of a class, looked up through its package
        bases in order."""
        if f"{class_key}.{name}" in self.kinds:
            return ("def", f"{class_key}.{name}")
        if depth > 20 or class_key not in self.classes:
            return None
        module = class_key.partition(":")[0]
        for base in self.classes[class_key].bases:
            found = self.expression(base, module, "")
            if found and found[0] == "def" and found[1] in self.classes:
                inherited = self.member(found[1], name, depth + 1)
                if inherited:
                    return inherited
        return None

    def _scopes(self, module: str, qualified: str) -> tuple[list[str], str | None]:
        """The enclosing function scopes (innermost first) and the enclosing class."""
        parts = qualified.split(".") if qualified else []
        prefixes = [".".join(parts[: i + 1]) for i in range(len(parts))][::-1]
        functions, owner = [], None
        for prefix in prefixes:
            kind = self.kinds.get(f"{module}:{prefix}")
            if kind == "class":
                owner = owner or f"{module}:{prefix}"
            elif owner is None:
                functions.append(prefix)
        return functions, owner

    def expression(self, node: ast.expr, module: str, qualified: str) -> Binding | None:
        """What a ``Name`` or ``Attribute`` expression refers to, ``None`` when unknown."""
        if isinstance(node, ast.Name):
            functions, owner = self._scopes(module, qualified)
            for scope in functions:
                if f"{module}:{scope}.{node.id}" in self.kinds:
                    return ("def", f"{module}:{scope}.{node.id}")
            if node.id in ("self", "cls") and owner:
                return ("instance", owner)
            found = self.symbol(module, node.id)
            if found is None and node.id in _BUILTINS:
                return ("builtin", node.id)
            return found
        if isinstance(node, ast.Attribute):
            base = self._base(node.value, module, qualified)
            return self._attribute(base, node.attr) if base else None
        return None

    def _base(self, node: ast.expr, module: str, qualified: str) -> Binding | None:
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "super"
        ):
            owner = self._scopes(module, qualified)[1]
            return ("super", owner) if owner else None
        return self.expression(node, module, qualified)

    def _attribute(self, base: Binding, name: str) -> Binding | None:
        what, target = base
        if what == "module":
            return self.symbol(target, name)
        if what == "external":
            return ("external", f"{target}.{name}")
        if what in ("instance",) or (what == "def" and target in self.classes):
            return self.member(target, name)
        if what == "super":
            module = target.partition(":")[0]
            for parent in self.classes[target].bases:
                found = self.expression(parent, module, "")
                if found and found[0] == "def" and found[1] in self.classes:
                    inherited = self.member(found[1], name)
                    if inherited:
                        return inherited
        return None


def _references(nodes: Iterator[ast.AST]) -> Iterator[ast.expr]:
    """The outermost ``Name``/``Attribute`` loads (``a.b.c``, not ``a.b`` or ``a``) and the
    callee of every call, annotations left out."""
    inner: set[int] = set()
    skipped: set[int] = set()
    for node in nodes:
        if isinstance(node, ast.AnnAssign):
            skipped.update(id(n) for n in ast.walk(node.annotation))
        if id(node) in skipped:
            continue
        if isinstance(node, ast.Attribute):
            inner.add(id(node.value))
        loaded = isinstance(node, ast.Name | ast.Attribute) and isinstance(node.ctx, ast.Load)
        if loaded and id(node) not in inner:
            yield node


def _called(nodes: list[ast.AST]) -> set[int]:
    return {id(n.func) for n in nodes if isinstance(n, ast.Call)}


def references(package: Package, module: str, qualified: str, nodes: list[ast.AST]) -> list[str]:
    """``calls`` of one row: package definitions referred to, outside names called."""
    called = _called(nodes)
    found: set[str] = set()
    for node in _references(iter(nodes)):
        binding = package.expression(node, module, qualified)
        is_call = id(node) in called
        if binding and binding[0] == "def":
            found.add(binding[1])
        elif not is_call:
            continue
        elif binding and binding[0] in ("external", "builtin"):
            found.add(binding[1])
        elif binding is None or binding[0] not in ("module", "instance", "super"):
            found.add(f"?.{node.attr if isinstance(node, ast.Attribute) else node.id}")
    return sorted(found)


def _all_below(nodes: list[ast.AST]) -> list[ast.AST]:
    out: list[ast.AST] = []
    for node in nodes:
        out.append(node)
        out += [n for n in ast.walk(node) if n is not node]
    return out


def _own(node: ast.AST) -> list[ast.AST]:
    """A definition's own nodes for ``calls``: a function's body (lambdas included) and
    decorators and defaults; a class's bases, decorators and non-method statements; a
    module's statements outside its definitions."""
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
        args = node.args
        outer = [*node.decorator_list, *args.defaults, *(d for d in args.kw_defaults if d)]
        return [*fi.own_nodes(node, lambdas=True), *_all_below(outer)]
    statements = [
        n
        for n in node.body  # type: ignore[attr-defined]
        if not isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
    ]
    if isinstance(node, ast.ClassDef):
        statements = [*node.bases, *node.keywords, *node.decorator_list, *statements]
    return _all_below(statements)


def fill_calls(rows: list[dict[str, object]], trees: list[tuple[str, ast.Module]]) -> None:
    """Fill ``calls`` of every row and ``called_by`` of the classes and functions."""
    package = Package(trees)
    by_key = {fi.key(r): r for r in rows}
    for module, tree in trees:
        module_row = by_key[f"{module}:{fi.MODULE_ROW}"]
        module_row["calls"] = references(package, module, "", _own(tree))
        for qualified, node, _ in fi.walk_definitions(tree):
            row = by_key[f"{module}:{qualified}"]
            row["calls"] = references(package, module, qualified, _own(node))
    callers: dict[str, set[str]] = {}
    for row in rows:
        for target in row["calls"]:  # type: ignore[attr-defined]
            if target in by_key and target != fi.key(row):
                callers.setdefault(target, set()).add(fi.key(row))
    for target, found in callers.items():
        if "called_by" in by_key[target]:
            by_key[target]["called_by"] = sorted(found)

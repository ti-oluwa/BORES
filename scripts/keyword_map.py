"""
Builds `docs/deck-keyword-map.md`: every deck keyword the parser knows, where it is defined, which
modules read it, and which of its fields no module reads.

Run from the repository root with `python scripts/keyword_map.py`.

Usage is found by static analysis of `src/bores` (everything except `deck/keywords`, which only
defines the keywords). String literals can coincide with unrelated words (`OIL`, `GAS`, `FIELD`),
so a short name's users need a glance before being trusted:

- A keyword is *used* by a module that has its name as a string literal (`deck_file.get("PORO")`)
  or imports its constant from `bores.deck.keywords` (`deck_file.get(START)`). Docstrings and
  comments do not count.
- A field is *read* if its name occurs as a string literal anywhere outside `deck/keywords`
  (`record["drainage_radius"]`). Names shared by several keywords (`i`, `well`, ...) cannot be
  attributed to one of them from a literal alone, so they are listed as *shared name* rather than
  unused; every other field with no literal at all is *unread*.
"""

import ast
import collections
import importlib
import pathlib
import pkgutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "bores"
OUTPUT = ROOT / "docs" / "deck-keyword-map.md"
SECTION_OF_MODULE = {
    "runspec": "RUNSPEC",
    "grid": "GRID",
    "props": "PROPS",
    "regions": "REGIONS",
    "solution": "SOLUTION",
    "summary": "SUMMARY",
    "schedule": "SCHEDULE",
}


GENERIC_CONSUMERS = {
    "TSTEP": "`DeckFile.report_times` and the schedule machinery, by keyword kind",
    "DATES": "`DeckFile.report_times` and the schedule machinery, by keyword kind",
    "SPECGRID": "`DeckFile.dimensions`",
    "DIMENS": "`DeckFile.dimensions`",
    "METRIC": "`DeckFile.unit_system`",
    "FIELD": "`DeckFile.unit_system`",
    "LAB": "`DeckFile.unit_system`",
    "PVT-M": "`DeckFile.unit_system`",
}
"""Keywords read by the deck machinery through their kind rather than by name, so no module names them."""


def field_names(keyword) -> list[str]:
    """Every field, column or record name a keyword's records carry."""
    names: list[str] = []
    for attribute in ("fields", "columns", "header_fields", "detail_fields"):
        for item in getattr(keyword, attribute, None) or ():
            names.append(getattr(item, "name", item) if not isinstance(item, str) else item)
    for record in getattr(keyword, "records", None) or ():
        for item in getattr(record, "fields", None) or ():
            names.append(item.name)
    return list(dict.fromkeys(str(n) for n in names))


def docstring_nodes(tree: ast.AST) -> set[int]:
    """Ids of the string-constant nodes that are docstrings."""
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                ids.add(id(body[0].value))
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            ids.add(id(node.value))
    return ids


def registry_node_ids(tree: ast.AST) -> set[int]:
    """Ids of every node inside the `DEFAULT_KEYWORDS = frozenset({...})` registry or an `__all__` list."""
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in ("DEFAULT_KEYWORDS", "__all__")
            for target in node.targets
        ):
            ids.update(id(child) for child in ast.walk(node))
    return ids


def scan_source() -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """
    Return `(string literal -> files, keyword constant -> files)` for the modules that consume
    keywords. A constant counts only if the module imports it from the keyword package and then
    actually uses it (importing it for re-export or for the registry does not count).
    """
    literals: dict[str, set[str]] = collections.defaultdict(set)
    imports: dict[str, set[str]] = collections.defaultdict(set)
    for path in sorted(SRC.rglob("*.py")):
        relative = path.relative_to(SRC).as_posix()
        if (
            relative.startswith(("deck/keywords/", "simulation.old/"))
            or relative == "deck/__init__.py"
        ):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        skip = docstring_nodes(tree) | registry_node_ids(tree)
        imported: dict[str, str] = {}
        loaded: set[str] = set()
        for node in ast.walk(tree):
            if id(node) in skip:
                continue
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                literals[node.value].add(relative)
            elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "bores.deck.keywords"
            ):
                for alias in node.names:
                    imported[alias.asname or alias.name] = alias.name
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                loaded.add(node.id)
        for alias, name in imported.items():
            if alias in loaded:
                imports[name].add(relative)
    return literals, imports


def main() -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from bores.deck.file import DEFAULT_KEYWORDS
    from bores.deck.keywords.base import Keyword

    package = importlib.import_module("bores.deck.keywords")
    defined_in: dict[str, str] = {}
    for module_info in pkgutil.iter_modules(package.__path__):
        module = importlib.import_module(f"bores.deck.keywords.{module_info.name}")
        for value in vars(module).values():
            if isinstance(value, Keyword) and value in DEFAULT_KEYWORDS:
                defined_in.setdefault(value.name, module_info.name)

    literals, imports = scan_source()
    keywords = sorted(
        DEFAULT_KEYWORDS, key=lambda keyword: (defined_in.get(keyword.name, ""), keyword.name)
    )
    field_owner_count: collections.Counter[str] = collections.Counter()
    for keyword in keywords:
        field_owner_count.update(field_names(keyword))

    lines = [
        "# Deck keyword map",
        "",
        "Generated by `scripts/keyword_map.py`; regenerate it instead of editing it by hand.",
        "",
        "`used by` lists the modules that name the keyword. `unread fields` are fields whose name "
        "appears nowhere outside the keyword definitions, so no loader reads them. `shared name` "
        "fields have a name other keywords also have, so a loader reading that name may or may not "
        "be reading this keyword's field.",
        "",
    ]
    unread_total = 0
    for module_name in sorted({defined_in.get(k.name, "?") for k in keywords}):
        section = SECTION_OF_MODULE.get(module_name, module_name)
        group = [k for k in keywords if defined_in.get(k.name, "?") == module_name]
        lines += [f"## {section} ({module_name}.py, {len(group)} keywords)", ""]
        lines += [
            "| keyword | kind | used by | unread fields | shared name |",
            "|---|---|---|---|---|",
        ]
        for keyword in group:
            users = sorted(literals.get(keyword.name, set()) | imports.get(keyword.name, set()))
            note = GENERIC_CONSUMERS.get(keyword.name)
            names = field_names(keyword)
            unread = [n for n in names if n not in literals and field_owner_count[n] == 1]
            shared = [n for n in names if n not in literals and field_owner_count[n] > 1]
            unread_total += len(unread)
            lines.append(
                f"| `{keyword.name}` | {type(keyword).__name__} | "
                f"{', '.join(f'`{u}`' for u in users) if users else (note or '**none**')} | "
                f"{', '.join(f'`{n}`' for n in unread) or ''} | {', '.join(f'`{n}`' for n in shared) or ''} |"
            )
        lines.append("")
    lines += [f"Fields no module reads: {unread_total}.", ""]
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(
        f"wrote {OUTPUT.relative_to(ROOT)}: {len(keywords)} keywords, {unread_total} unread fields"
    )


if __name__ == "__main__":
    main()

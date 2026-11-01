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


NOT_BLACK_OIL = "not black-oil"
NOT_BUILT = "feature not built"
CANDIDATE = "candidate"
NO_LOADER = "no loader yet"
NO_HOME = "no model home"

TRIAGE: dict[str, tuple[str, str]] = {
    "GRIDUNIT": (NO_HOME, "the grid unit type is informational"),
    "NNC": (
        NOT_BUILT,
        "per-connection saturation and pressure table numbers need NNC-level region overrides",
    ),
    "PINCH": (
        NOT_BUILT,
        "pinch-out gap and connection options change transmissibility across pinched-out layers",
    ),
    "SPECGRID": (
        NOT_BUILT,
        "`numres` counts local grid refinements and `gflag` marks a radial grid; neither is supported",
    ),
    "ENDSCALE": (
        NOT_BUILT,
        "`combination_mode` selects how scaled end points are combined, which nothing consumes",
    ),
    "EQLDIMS": (NOT_BLACK_OIL, "tracer concentration table capacities"),
    "REGDIMS": (NOT_BLACK_OIL, "capacities for tracking, coal, operate regions and polymer"),
    "TABDIMS": (NOT_BLACK_OIL, "capacities for EOS, thermal and compaction tables"),
    "VFPPDIMS": (NO_HOME, "all of its items are now read into `ModelDimensions`"),
    "WELLDIMS": (
        NOT_BLACK_OIL,
        "capacities for separators, surface streams, mixtures and multi-segment well detail",
    ),
    "COMPDAT": (
        CANDIDATE,
        "`kh` and `pressure_equivalent_radius` override the connection factor's permeability-thickness and Peaceman radius; the well index calculation has no override to receive them yet",
    ),
    "COMPSEGS": (NOT_BUILT, "multi-segment wells are not modelled"),
    "WELSEGS": (NOT_BUILT, "multi-segment wells are not modelled"),
    "GCONINJE": (
        NOT_BUILT,
        "reinjection and voidage replacement targets and group responses are not implemented",
    ),
    "GCONPROD": (
        NOT_BUILT,
        "exceed actions, subordinate application and reservoir fluid targets are not implemented",
    ),
    "GECON": (NOT_BUILT, "group economic limits are not implemented"),
    "WCONINJE": (NOT_BLACK_OIL, "steam ratios are thermal"),
    "WECON": (
        NOT_BUILT,
        "secondary limits, follow-on wells and the limited quantity are not implemented",
    ),
    "WELSPECS": (
        CANDIDATE,
        "`drainage_radius` has no field on `Well` since the index calculation derives it; the rest are FrontSim, polymer and well model options",
    ),
    "WGRUPCON": (
        NOT_BUILT,
        "guide rate phase, scaling and group availability are not implemented",
    ),
    "AQUCT": (NOT_BLACK_OIL, "salt concentration"),
    "AQUFLUX": (NO_HOME, "the aquifer pressure of a constant flux aquifer is for reporting"),
    "EQUIL": (
        NOT_BLACK_OIL,
        "compositional initialization and the black-oil hydrocarbon-in-gas initialization option",
    ),
    "GRIDFILE": (NO_HOME, "controls Eclipse grid file output"),
    "MESSAGES": (NO_HOME, "Eclipse message print limits"),
    "NSTACK": (NO_HOME, "size of Eclipse's linear solver stack"),
    "TRACER": (NOT_BUILT, "tracers are not modelled"),
    "DRSDT": (
        NO_LOADER,
        "a scheduled limit on how fast dissolved gas can rise, with no matching limit in the model",
    ),
    "DRVDT": (
        NO_LOADER,
        "a scheduled limit on how fast vaporized oil can rise, with no matching limit in the model",
    ),
    "WCONHIST": (NO_LOADER, "history controls are not loaded"),
    "WCONINJH": (NO_LOADER, "history controls are not loaded"),
    "WDFAC": (NO_LOADER, "non-Darcy flow is not applied"),
    "WELPI": (NO_LOADER, "productivity index rescaling is not applied"),
    "WPAVE": (NO_LOADER, "well pressure averaging is not modelled"),
    "WTEST": (NO_LOADER, "well testing is not modelled"),
    "INIT": (NO_HOME, "asks Eclipse to write an initial file"),
    "FMTIN": (NO_HOME, "Eclipse input file format"),
    "FMTOUT": (NO_HOME, "Eclipse output file format"),
    "UNIFIN": (NO_HOME, "Eclipse unified input files"),
    "UNIFOUT": (NO_HOME, "Eclipse unified output files"),
    "NOECHO": (NO_HOME, "stops Eclipse echoing the input"),
    "RUNSUM": (NO_HOME, "asks Eclipse for a run summary file"),
    "ZIPPY2": (NO_HOME, "an OPM Flow settings keyword"),
    "NEWTRAN": (
        NOT_BUILT,
        "chooses between two Eclipse transmissibility calculations; there is only one here",
    ),
    "OLDTRAN": (
        NOT_BUILT,
        "chooses between two Eclipse transmissibility calculations; there is only one here",
    ),
    "SGL": (CANDIDATE, "connate gas end point array; `Rock` has no field for it"),
    "SGU": (CANDIDATE, "maximum gas saturation end point array; `Rock` has no field for it"),
    "SWU": (CANDIDATE, "maximum water saturation end point array; `Rock` has no field for it"),
    "SGLPC": (
        CANDIDATE,
        "end point used only to scale capillary pressure; `Rock` has no separate capillary end points",
    ),
    "SWLPC": (
        CANDIDATE,
        "end point used only to scale capillary pressure; `Rock` has no separate capillary end points",
    ),
    "SOIL": (
        CANDIDATE,
        "initial oil saturation; the initializer derives oil as 1 - Sw - Sg, so it needs a rule for when it is given as well",
    ),
}
KIND_TRIAGE: dict[str, tuple[str, str]] = {
    "MnemonicReportKeyword": (NO_HOME, "controls which Eclipse report files are written"),
    "SummaryVectorKeyword": (
        NO_HOME,
        "selects summary vectors, which are built from the model's own output",
    ),
}
"""
What an unlisted keyword of a given kind is, by its class name, when nothing names it in `TRIAGE`.
"""

SECTION_MODULES = ("sections",)

"""
Why each keyword's unread fields stay unread. Every keyword with unread fields needs an entry, so a
new one shows as untriaged until someone decides.
"""


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
            if unread or not users:
                category, reason = TRIAGE.get(
                    keyword.name,
                    ("section marker", "")
                    if module_name in SECTION_MODULES
                    else KIND_TRIAGE.get(type(keyword).__name__, ("untriaged", "")),
                )
            else:
                category, reason = "", ""
            lines.append(
                f"| `{keyword.name}` | {type(keyword).__name__} | "
                f"{', '.join(f'`{u}`' for u in users) if users else (note or '**none**')} | "
                f"{', '.join(f'`{n}`' for n in unread) or ''} | {', '.join(f'`{n}`' for n in shared) or ''} | "
                f"{(category + ': ' + reason) if reason else category} |"
            )
        lines.append("")
    untriaged = [
        keyword.name
        for keyword in keywords
        if [n for n in field_names(keyword) if n not in literals and field_owner_count[n] == 1]
        and keyword.name not in TRIAGE
    ]
    untriaged += [
        keyword.name
        for keyword in keywords
        if not (literals.get(keyword.name, set()) | imports.get(keyword.name, set()))
        and not GENERIC_CONSUMERS.get(keyword.name)
        and keyword.name not in TRIAGE
        and type(keyword).__name__ not in KIND_TRIAGE
        and defined_in.get(keyword.name) not in SECTION_MODULES
    ]
    lines += [f"Fields no module reads: {unread_total}.", ""]
    if untriaged:
        lines += [
            f"Keywords with no consumer or with unread fields, and no triage entry: {', '.join(untriaged)}.",
            "",
        ]
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(
        f"wrote {OUTPUT.relative_to(ROOT)}: {len(keywords)} keywords, {unread_total} unread fields"
    )


if __name__ == "__main__":
    main()

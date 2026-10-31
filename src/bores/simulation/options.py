"""
Switches and sizes a deck sets for the model as a whole.

These come from the `RUNSPEC` keywords that turn optional features on (`SATOPTS`, `EQLOPTS`), the
ones that declare how large the model is (`TABDIMS`, `EQLDIMS`, `REGDIMS`, `WELLDIMS`, `VFPPDIMS`,
`VFPIDIMS`) and `MINPV` from the `GRID` section. A deck sets them once for the whole run, so they
are held on the `RunSpec` instead of on the data they describe.
"""

import typing

import attrs

from bores.constants import c
from bores.deck.file import DeckFile
from bores.types import Integer, Number, UnitSystem

__all__ = [
    "EquilibrationOptions",
    "ModelDimensions",
    "SaturationOptions",
    "load_equilibration_options",
    "load_minimum_pore_volume",
    "load_model_dimensions",
    "load_saturation_options",
]


@attrs.frozen(kw_only=True, slots=True)
class SaturationOptions:
    """
    The optional saturation function features a deck switches on with `SATOPTS`.

    `HYSTER` is not here: it is `RunSpec.hysteresis_enabled`.
    """

    directional_relative_permeability: bool = False
    """Whether relative permeabilities depend on the flow direction (`DIRECT`)."""

    irreversible_directional_relative_permeability: bool = False
    """Whether the directional relative permeabilities are irreversible (`IRREVERS`)."""

    surface_tension_dependence: bool = False
    """Whether relative permeability and capillary pressure depend on surface tension (`SURFTENS`)."""

    unrecognized: frozenset[str] = frozenset()
    """Any other option the deck listed, upper-cased, since the options Eclipse accepts depend on its version."""


@attrs.frozen(kw_only=True, slots=True)
class EquilibrationOptions:
    """The optional equilibration features a deck switches on with `EQLOPTS`."""

    threshold_pressures: bool = False
    """Whether the deck defines pressure differences that flow between regions must exceed (`THPRES`)."""

    irreversible_threshold_pressures: bool = False
    """Whether a threshold pressure applies in one direction only (`IRREVERS`)."""

    quiescent_start: bool = False
    """Whether the initial state is adjusted so that nothing flows at the start (`QUIESC`)."""

    mobile_fluid_endpoint_correction: bool = False
    """Whether saturation end points are corrected so the initial fluids are mobile (`MOBILE`)."""

    unrecognized: frozenset[str] = frozenset()
    """Any other option the deck listed, upper-cased."""


@attrs.frozen(kw_only=True, slots=True)
class ModelDimensions:
    """
    The sizes a deck declares for its own tables, regions, wells and VFP tables.

    Every field is `None` when the deck does not write the keyword it comes from. They are
    capacities Eclipse reserves rather than anything a calculation needs, which is why they are
    checked against the deck (`bores.deck.dimensions`) and otherwise just carried.
    """

    saturation_tables: Integer | None = None
    """Number of saturation function regions (`TABDIMS`)."""

    pvt_tables: Integer | None = None
    """Number of PVT regions (`TABDIMS`)."""

    saturation_table_nodes: Integer | None = None
    """Most rows in a saturation function table (`TABDIMS`)."""

    pvt_pressure_nodes: Integer | None = None
    """Most pressure rows in a PVT table (`TABDIMS`)."""

    dissolved_gas_ratio_nodes: Integer | None = None
    """Most solution gas-oil ratio rows in a live oil table (`TABDIMS`)."""

    vaporized_oil_ratio_nodes: Integer | None = None
    """Most vaporized oil-gas ratio rows in a wet gas table (`TABDIMS`)."""

    equilibration_regions: Integer | None = None
    """Number of equilibration regions (`EQLDIMS`)."""

    pressure_depth_nodes: Integer | None = None
    """Most depth nodes in an equilibrium pressure table (`EQLDIMS`)."""

    ratio_depth_nodes: Integer | None = None
    """Most depth rows in a table of solution gas-oil or vaporized oil-gas ratio against depth (`EQLDIMS`)."""

    fluid_in_place_regions: Integer | None = None
    """Number of fluid-in-place regions in a region set (`REGDIMS`)."""

    fluid_in_place_region_sets: Integer | None = None
    """Number of fluid-in-place region sets, that is `FIPNUM` and the other `FIP<name>` arrays (`REGDIMS`)."""

    reservoir_regions: Integer | None = None
    """Number of independent reservoir regions (`REGDIMS`)."""

    flux_regions: Integer | None = None
    """Number of flux regions, `FLUXNUM` (`REGDIMS`)."""

    wells: Integer | None = None
    """Most wells in the model (`WELLDIMS`)."""

    connections_per_well: Integer | None = None
    """Most connections in one well (`WELLDIMS`)."""

    groups: Integer | None = None
    """Most well groups in the model (`WELLDIMS`)."""

    wells_per_group: Integer | None = None
    """Most wells or groups directly under one group (`WELLDIMS`)."""

    production_vfp_tables: Integer | None = None
    """Number of production VFP tables (`VFPPDIMS`)."""

    injection_vfp_tables: Integer | None = None
    """Number of injection VFP tables (`VFPIDIMS`)."""


DIMENSION_SOURCES: tuple[tuple[str, str, str], ...] = (
    ("TABDIMS", "num_saturation_tables", "saturation_tables"),
    ("TABDIMS", "num_pvt_tables", "pvt_tables"),
    ("TABDIMS", "max_saturation_table_nodes", "saturation_table_nodes"),
    ("TABDIMS", "max_pvt_pressure_nodes", "pvt_pressure_nodes"),
    ("TABDIMS", "max_dissolved_gas_ratio_nodes", "dissolved_gas_ratio_nodes"),
    ("TABDIMS", "max_vaporized_oil_ratio_nodes", "vaporized_oil_ratio_nodes"),
    ("EQLDIMS", "max_equilibration_regions", "equilibration_regions"),
    ("EQLDIMS", "max_pressure_depth_nodes", "pressure_depth_nodes"),
    ("EQLDIMS", "max_rs_rv_depth_nodes", "ratio_depth_nodes"),
    ("REGDIMS", "max_fluid_in_place_regions", "fluid_in_place_regions"),
    ("REGDIMS", "max_fluid_in_place_region_sets", "fluid_in_place_region_sets"),
    ("REGDIMS", "max_reservoir_regions", "reservoir_regions"),
    ("REGDIMS", "max_flux_regions", "flux_regions"),
    ("WELLDIMS", "max_wells", "wells"),
    ("WELLDIMS", "max_connections_per_well", "connections_per_well"),
    ("WELLDIMS", "max_groups", "groups"),
    ("WELLDIMS", "max_wells_per_group", "wells_per_group"),
    ("VFPPDIMS", "max_production_vfp_tables", "production_vfp_tables"),
    ("VFPIDIMS", "max_injection_vfp_tables", "injection_vfp_tables"),
)
"""Where each `ModelDimensions` field comes from: the keyword, the item in its record, and the field."""


def load_saturation_options(deck_file: DeckFile) -> SaturationOptions:
    """
    Read `SATOPTS`.

    :param deck_file: The deck to read.
    :returns: The options, all off if the deck has no `SATOPTS`.
    """
    listed = frozenset(str(option).upper() for option in deck_file.get("SATOPTS") or ())
    return SaturationOptions(
        directional_relative_permeability="DIRECT" in listed,
        irreversible_directional_relative_permeability="IRREVERS" in listed,
        surface_tension_dependence="SURFTENS" in listed,
        unrecognized=listed - {"HYSTER", "DIRECT", "IRREVERS", "SURFTENS"},
    )


def load_equilibration_options(deck_file: DeckFile) -> EquilibrationOptions:
    """
    Read `EQLOPTS`.

    :param deck_file: The deck to read.
    :returns: The options, all off if the deck has no `EQLOPTS`.
    """
    listed = frozenset(str(option).upper() for option in deck_file.get("EQLOPTS") or ())
    return EquilibrationOptions(
        threshold_pressures="THPRES" in listed,
        irreversible_threshold_pressures="IRREVERS" in listed,
        quiescent_start="QUIESC" in listed,
        mobile_fluid_endpoint_correction="MOBILE" in listed,
        unrecognized=listed - {"THPRES", "IRREVERS", "QUIESC", "MOBILE"},
    )


def load_model_dimensions(deck_file: DeckFile) -> ModelDimensions | None:
    """
    Read the deck's dimension keywords.

    :param deck_file: The deck to read.
    :returns: The dimensions, or `None` if the deck has none of the dimension keywords.
    """
    keywords = {keyword for keyword, _, _ in DIMENSION_SOURCES}
    if not deck_file.has_any(*keywords):
        return None
    records = {keyword: deck_file.get_shared(keyword) for keyword in keywords}
    values: dict[str, typing.Any] = {}
    for keyword, item, field in DIMENSION_SOURCES:
        record = records[keyword]
        if record is not None and record.get(item) is not None:
            values[field] = int(record[item])
    return ModelDimensions(**values)


def load_minimum_pore_volume(deck_file: DeckFile) -> Number | None:
    """
    Read `MINPV`, the pore volume below which a cell takes no part in the flow.

    The deck gives it in reservoir volume: cubic metres (`METRIC`), reservoir barrels (`FIELD`) or
    cubic centimetres (`LAB`). It is returned in the volume unit pore volumes have here (m³, ft³
    or cm³), so the two can be compared directly.

    :param deck_file: The deck to read.
    :returns: The minimum pore volume, or `None` if the deck has no `MINPV` (the simulator's own
        default then applies).
    """
    record = deck_file.get("MINPV")
    if record is None:
        return None
    minimum = float(record["minpv"])
    if deck_file.unit_system == UnitSystem.FIELD:
        minimum *= c.BARRELS_TO_CUBIC_FEET
    return minimum

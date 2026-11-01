"""
Switches and sizes a deck sets for the model as a whole.

These come from the `RUNSPEC` keywords that turn optional features on (`SATOPTS`, `EQLOPTS`), the
ones that declare how large the model is (`TABDIMS`, `EQLDIMS`, `REGDIMS`, `WELLDIMS`, `FAULTDIM`, `PIMTDIMS`,
`VFPPDIMS`, `VFPIDIMS`) and `MINPV` from the `GRID` section. A deck sets them once for the whole run, so they
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
    The saturation function features of a run, which a deck switches on with `SATOPTS`.

    Held on `RunSpec.saturation_options`.
    """

    hysteresis_enabled: bool = False
    """
    Enable saturation hysteresis tracking for residual saturations.

    When True, residual oil and gas saturations are updated each time step
    based on the saturation reversal history of each cell (drainage vs.
    imbibition regime). This is required when using hysteresis-aware
    relative permeability or capillary pressure models (e.g. Killough,
    Land trapping) where the scanning curves depend on the maximum
    historical saturation reached in each cell.

    When False (default), residual saturations are treated as fixed rock
    properties and the hysteresis state is not updated during the
    simulation. This is appropriate for standard black-oil simulations
    without hysteresis models, and avoids the overhead of per-step
    history tracking.

    This flag should only be enabled when the configured relative
    permeability and/or capillary pressure tables implement hysteresis-
    aware scanning curves. Enabling it with non-hysteretic tables has no
    physical effect and incurs unnecessary computation.
    """

    hysteresis_saturation_tolerance: float | None = attrs.field(
        default=None,
        validator=attrs.validators.optional(attrs.validators.ge(0)),  # type: ignore[arg-type]
    )
    """
    Smallest saturation change between time steps that counts as a phase moving up or down
    when tracking relative permeability hysteresis. `None` uses the
    `HYSTERESIS_SATURATION_TOLERANCE` constant of `constants`.
    """

    endpoint_scaling_enabled: bool = False
    """Whether the deck scales the saturation function end points cell by cell (`ENDSCALE`)."""

    directional_endpoint_scaling: bool = False
    """Whether end point scaling depends on the flow direction (`ENDSCALE` with `DIRECT`)."""

    irreversible_endpoint_scaling: bool = False
    """Whether end point scaling is irreversible, so one set applies to flow in each direction (`ENDSCALE` with `IRREVERS`)."""

    scale_critical_saturations: bool = False
    """Whether the critical saturations are scaled consistently with the other end points (`SCALECRS YES`)."""

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

    endpoint_depth_tables: Integer | None = None
    """Number of tables of end points against depth (`ENDSCALE`)."""

    endpoint_depth_nodes: Integer | None = None
    """Most depth nodes in a table of end points against depth (`ENDSCALE`)."""

    fault_segments: Integer | None = None
    """Most fault segments (`FAULTDIM`)."""

    multiplier_tables: Integer | None = None
    """Number of productivity index multiplier tables (`PIMTDIMS`)."""

    multiplier_table_rows: Integer | None = None
    """Most rows in a productivity index multiplier table (`PIMTDIMS`)."""

    vfp_flow_rate_values: Integer | None = None
    """Most flow rate values on a production VFP table's axis (`VFPPDIMS`)."""

    vfp_tubing_head_pressure_values: Integer | None = None
    """Most tubing head pressure values on a production VFP table's axis (`VFPPDIMS`)."""

    vfp_water_fraction_values: Integer | None = None
    """Most water fraction values on a production VFP table's axis (`VFPPDIMS`)."""

    vfp_gas_fraction_values: Integer | None = None
    """Most gas fraction values on a production VFP table's axis (`VFPPDIMS`)."""

    vfp_artificial_lift_values: Integer | None = None
    """Most artificial lift values on a production VFP table's axis (`VFPPDIMS`)."""

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
    ("ENDSCALE", "num_endpoint_depth_tables", "endpoint_depth_tables"),
    ("ENDSCALE", "num_endpoint_depth_nodes", "endpoint_depth_nodes"),
    ("FAULTDIM", "max_fault_segments", "fault_segments"),
    ("PIMTDIMS", "max_multiplier_tables", "multiplier_tables"),
    ("PIMTDIMS", "max_multiplier_table_rows", "multiplier_table_rows"),
    ("VFPPDIMS", "max_flow_rate_values", "vfp_flow_rate_values"),
    ("VFPPDIMS", "max_tubing_head_pressure_values", "vfp_tubing_head_pressure_values"),
    ("VFPPDIMS", "max_water_fraction_values", "vfp_water_fraction_values"),
    ("VFPPDIMS", "max_gas_fraction_values", "vfp_gas_fraction_values"),
    ("VFPPDIMS", "max_artificial_lift_values", "vfp_artificial_lift_values"),
    ("VFPPDIMS", "max_production_vfp_tables", "production_vfp_tables"),
    ("VFPIDIMS", "max_injection_vfp_tables", "injection_vfp_tables"),
)
"""Where each `ModelDimensions` field comes from: the keyword, the item in its record, and the field."""


def load_saturation_options(deck_file: DeckFile) -> SaturationOptions:
    """
    Read `SATOPTS`, `ENDSCALE` and `SCALECRS`.

    :param deck_file: The deck to read.
    :returns: The options, all off for whichever of the keywords the deck does not have.
    """
    listed = frozenset(str(option).upper() for option in deck_file.get("SATOPTS") or ())
    endpoint_scaling = deck_file.get_shared("ENDSCALE")
    critical_scaling = deck_file.get_shared("SCALECRS")
    return SaturationOptions(
        endpoint_scaling_enabled=endpoint_scaling is not None,
        directional_endpoint_scaling=bool(endpoint_scaling)
        and endpoint_scaling["directional"] == "DIRECT",
        irreversible_endpoint_scaling=bool(endpoint_scaling)
        and endpoint_scaling["reversibility"] == "IRREVERS",
        scale_critical_saturations=bool(critical_scaling) and bool(critical_scaling["enabled"]),
        hysteresis_enabled="HYSTER" in listed,
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

import logging
import typing

import attrs
from typing_extensions import Self

from bores.blackoil.satfunc.capillary_pressure.hysteresis import CurvatureCapillaryPressureTable
from bores.blackoil.satfunc.capillary_pressure.tables import (
    CapillaryPressureTable,
    ThreePhaseCapillaryPressureTable,
)
from bores.blackoil.satfunc.relperm.hysteresis import (
    CarlsonRelPermTable,
    EclipseKilloughRelPermTable,
    KilloughLandRelPermTable,
    compute_land_trapping_coefficient,
)
from bores.blackoil.satfunc.relperm.tables import (
    RelativePermeabilityTable,
    ThreePhaseRelPermTable,
    resolve_residual_oil_saturation_gas,
)
from bores.blackoil.satfunc.tables import SatFuncTables
from bores.constants import c
from bores.deck.file import DeckFile
from bores.errors import NotSupportedError, ValidationError
from bores.types import Boolean, FluidPhase, Number, RelPermHysteresisModel

logger = logging.getLogger(__name__)

__all__ = [
    "HYSTERESIS_WETTING_PHASE_MAP",
    "LIMITING_CURVES_MAP",
    "RELPERM_HYSTERESIS_MODEL_MAP",
    "HysteresisOptions",
    "LandParameters",
    "as_hysteresis_capillary_pressure_table",
    "as_hysteresis_relperm_table",
    "as_hysteresis_tables",
    "derive_land_parameters",
    "get_hysteresis_roles",
]

RELPERM_HYSTERESIS_MODEL_MAP: typing.Mapping[int, tuple[RelPermHysteresisModel, bool]] = {
    -1: (RelPermHysteresisModel.NONE, False),
    0: (RelPermHysteresisModel.CARLSON, False),
    1: (RelPermHysteresisModel.CARLSON, True),
    2: (RelPermHysteresisModel.KILLOUGH, False),
    3: (RelPermHysteresisModel.KILLOUGH, True),
}
"""
Maps `EHYSTR` item 2 to the relative permeability hysteresis model and to whether the wetting
phase takes its imbibition curve rather than its drainage curve. This is the one place the
numeric codes are translated.
"""

LIMITING_CURVES_MAP: typing.Mapping[str, tuple[bool, bool]] = {
    "BOTH": (True, True),
    "KR": (True, False),
    "PC": (False, True),
}
"""
Maps `EHYSTR` item 5 to `(apply_to_relperm, apply_to_capillary_pressure)`. This is the one place
the deck's abbreviations are translated.
"""

HYSTERESIS_WETTING_PHASE_MAP: typing.Mapping[str, FluidPhase] = {
    "OIL": FluidPhase.OIL,
    "GAS": FluidPhase.GAS,
    "WATER": FluidPhase.WATER,
}
"""Maps `EHYSTR` item 8 to the wetting phase."""


@attrs.frozen(kw_only=True, slots=True)
class HysteresisOptions:
    """
    How saturation function hysteresis is modelled, as set by `SATOPTS` and `EHYSTR`.

    Every field has the Eclipse default, so `HysteresisOptions()` is what a deck gets from
    `SATOPTS HYSTER` with no `EHYSTR`. The deck's abbreviations are translated once, when the
    options are loaded, so nothing here holds an Eclipse mnemonic.
    """

    relperm_model: RelPermHysteresisModel = RelPermHysteresisModel.CARLSON
    """
    Construction used for the relative permeability scanning curves. `BLENDED_SCANNING` keeps
    BORES's own weighted blend available next to Eclipse's two models.
    """

    wetting_phase_uses_imbibition: Boolean = False
    """Whether the wetting phase takes its imbibition curve instead of its drainage curve."""

    capillary_pressure_curvature: Number = attrs.field(
        factory=lambda: typing.cast(float, c.HYSTERESIS_CAPILLARY_PRESSURE_CURVATURE)
    )
    """Curvature parameter of the capillary pressure scanning curves (`EHYSTR` item 1)."""

    killough_wetting_curvature: Number = attrs.field(
        factory=lambda: typing.cast(float, c.HYSTERESIS_KILLOUGH_WETTING_CURVATURE)
    )
    """Curvature parameter of the Killough wetting phase scanning curves (`EHYSTR` item 3)."""

    killough_trapped_modifier: Number = attrs.field(
        factory=lambda: typing.cast(float, c.HYSTERESIS_KILLOUGH_TRAPPED_MODIFIER)
    )
    """Modification parameter of the Killough trapped non-wetting saturation (`EHYSTR` item 4)."""

    apply_to_relperm: Boolean = True
    """Whether relative permeability follows hysteresis. Set from `EHYSTR` item 5."""

    apply_to_capillary_pressure: Boolean = True
    """Whether capillary pressure follows hysteresis. Set from `EHYSTR` item 5."""

    retrace_capillary_pressure: Boolean = True
    """Whether capillary pressure scanning curves are retraced rather than new (`EHYSTR` item 6)."""

    starts_on_imbibition_curve: Boolean = False
    """Whether the mobility starts on the imbibition curve (`EHYSTR` item 7)."""

    wetting_phase: FluidPhase = FluidPhase.OIL
    """Wetting phase (`EHYSTR` item 8). Water and gas are the non-wetting phases when it is oil."""

    baker_interpolation_oil: Boolean = False
    """Whether Baker's interpolation is used for oil relative permeability (`EHYSTR` item 9)."""

    baker_interpolation_gas: Boolean = False
    """Whether Baker's interpolation is used for gas relative permeability (`EHYSTR` item 10)."""

    baker_interpolation_water: Boolean = False
    """Whether Baker's interpolation is used for water relative permeability (`EHYSTR` item 11)."""

    threshold_saturation: Number = 0.0
    """Saturation change below which hysteresis is not triggered (`EHYSTR` item 12)."""

    fix_killough_wetting_phase: Boolean = False
    """Whether the Killough wetting phase correction is applied (`EHYSTR` item 13)."""

    scale_capillary_pressure: Boolean = False
    """Whether capillary pressure is scaled (`EHYSTR` item 14)."""

    def __attrs_post_init__(self) -> None:
        if self.wetting_phase not in HYSTERESIS_WETTING_PHASE_MAP.values():
            raise ValidationError(
                f"`wetting_phase` must be oil, gas or water; got {self.wetting_phase}."
            )
        if self.capillary_pressure_curvature < 0.0:
            raise ValidationError("`capillary_pressure_curvature` must not be negative.")
        if self.killough_wetting_curvature <= 0.0:
            raise ValidationError("`killough_wetting_curvature` must be positive.")
        if self.killough_trapped_modifier < 0.0:
            raise ValidationError("`killough_trapped_modifier` must not be negative.")
        if not 0.0 <= self.threshold_saturation < 1.0:
            raise ValidationError("`threshold_saturation` must lie in [0, 1).")

    @property
    def relperm_hysteresis_active(self) -> Boolean:
        """Whether relative permeability follows hysteresis at all."""
        return self.apply_to_relperm and self.relperm_model != RelPermHysteresisModel.NONE

    @property
    def water_imbibition_when_increasing(self) -> Boolean:
        """
        Whether water imbibes as its saturation rises, which it does when water is the wetting
        phase. This is the argument the saturation history tracker takes for water.
        """
        return self.wetting_phase == FluidPhase.WATER

    @property
    def gas_imbibition_when_increasing(self) -> Boolean:
        """
        Whether gas imbibes as its saturation rises, which it does only when gas is the wetting
        phase. This is the argument the saturation history tracker takes for gas.
        """
        return self.wetting_phase == FluidPhase.GAS

    @classmethod
    def from_deck(
        cls,
        deck_file: DeckFile,
        *,
        relperm_model: RelPermHysteresisModel | None = None,
    ) -> Self | None:
        """
        Read the hysteresis options from a deck.

        Hysteresis is on when `SATOPTS` lists `HYSTER`. `EHYSTR` then sets the options, and
        any item it leaves out, or all of them if it is absent, keeps its Eclipse default.

        :param deck_file: The deck to read.
        :param relperm_model: Use this relative permeability model instead of the one `EHYSTR`
            item 2 selects. This is how `BLENDED_SCANNING`, which no `EHYSTR` code selects, is
            chosen for a deck. The wetting phase still follows item 2.
        :returns: The options, or `None` when the deck does not turn hysteresis on.
        """
        if "HYSTER" not in (deck_file.get("SATOPTS") or ()):
            return None

        record = deck_file.get("EHYSTR")
        if record is None:
            return cls() if relperm_model is None else cls(relperm_model=relperm_model)

        code = int(record["relative_permeability_model"])
        if code not in RELPERM_HYSTERESIS_MODEL_MAP:
            raise ValidationError(
                f"`EHYSTR` item 2 must be one of {sorted(RELPERM_HYSTERESIS_MODEL_MAP)}; got {code}."
            )

        deck_model, wetting_phase_uses_imbibition = RELPERM_HYSTERESIS_MODEL_MAP[code]
        apply_to_relperm, apply_to_capillary_pressure = LIMITING_CURVES_MAP[
            record["limiting_curves"]
        ]
        return cls(
            relperm_model=deck_model if relperm_model is None else relperm_model,
            wetting_phase_uses_imbibition=wetting_phase_uses_imbibition,
            capillary_pressure_curvature=float(record["capillary_pressure_curvature"]),
            killough_wetting_curvature=float(record["killough_wetting_curvature"]),
            killough_trapped_modifier=float(record["killough_trapped_modifier"]),
            apply_to_relperm=apply_to_relperm,
            apply_to_capillary_pressure=apply_to_capillary_pressure,
            retrace_capillary_pressure=record["capillary_pressure_shape"] == "RETR",
            starts_on_imbibition_curve=record["initial_mobility_curve"] == "IMBIB",
            wetting_phase=HYSTERESIS_WETTING_PHASE_MAP[record["wetting_phase"]],
            baker_interpolation_oil=record["baker_oil"] == "YES",
            baker_interpolation_gas=record["baker_gas"] == "YES",
            baker_interpolation_water=record["baker_water"] == "YES",
            threshold_saturation=float(record["threshold_saturation"]),
            fix_killough_wetting_phase=bool(record["fix_killough_wetting_phase"]),
            scale_capillary_pressure=bool(record["scale_capillary_pressure"]),
        )


class LandParameters(typing.NamedTuple):
    """Land's trapping parameters of one drainage and imbibition pair of tables."""

    land_trapping_coefficient_water: Number
    """Land coefficient of the oil-water system, whose non-wetting phase is oil."""

    land_trapping_coefficient_gas: Number
    """Land coefficient of the gas-oil system, whose non-wetting phase is gas."""

    maximum_residual_oil_saturation_water: Number | None
    """Most oil the imbibition table traps in the oil-water system, or `None` for none."""

    maximum_residual_oil_saturation_gas: Number | None
    """Most oil the imbibition table traps in the gas-oil system, or `None` for none."""

    maximum_residual_gas_saturation: Number | None
    """Most gas the imbibition table traps, or `None` for none."""


def derive_land_parameters(
    *,
    drainage: RelativePermeabilityTable,
    imbibition: RelativePermeabilityTable,
) -> LandParameters:
    """
    Derive Land's trapping parameters from a drainage and an imbibition table.

    The most that is trapped is the imbibition table's critical saturation. It is trapped when
    imbibition starts from the largest saturation the non-wetting phase reaches on the drainage
    table, which is one minus connate water, and minus the residual oil to gas flood for gas. The
    coefficient follows from those two, so nothing about Land is configured by hand. A phase the
    imbibition table traps none of gets `None`, which switches Land off for it.

    :param drainage: The drainage relative permeability table of the pair.
    :param imbibition: The imbibition relative permeability table of the pair.
    :returns: Land's parameters.
    """
    connate_water = drainage.get_connate_water_saturation()
    largest_gas = (
        1.0 - connate_water - resolve_residual_oil_saturation_gas(drainage, connate_water)
    )
    largest_oil = 1.0 - connate_water

    trapped_gas = imbibition.get_residual_gas_saturation()
    trapped_oil_water = imbibition.get_residual_oil_saturation_water()
    trapped_oil_gas = resolve_residual_oil_saturation_gas(
        imbibition, imbibition.get_connate_water_saturation()
    )
    return LandParameters(
        land_trapping_coefficient_water=compute_land_trapping_coefficient(
            maximum_residual_saturation=trapped_oil_water,
            maximum_non_wetting_saturation=largest_oil,
        ),
        land_trapping_coefficient_gas=compute_land_trapping_coefficient(
            maximum_residual_saturation=trapped_gas,
            maximum_non_wetting_saturation=largest_gas,
        ),
        maximum_residual_oil_saturation_water=trapped_oil_water if trapped_oil_water > 0 else None,
        maximum_residual_oil_saturation_gas=trapped_oil_gas if trapped_oil_gas > 0 else None,
        maximum_residual_gas_saturation=trapped_gas if trapped_gas > 0 else None,
    )


def get_hysteresis_roles(
    relative_permeability: RelativePermeabilityTable,
) -> tuple[Boolean, Boolean]:
    """
    The direction each phase imbibes in, as the saturation history tracker needs it.

    Read from the table's own wetting phases, since the wetting phase decides which way a phase
    moves while it imbibes and the tables already know it.

    :param relative_permeability: Any relative permeability table of the run.
    :returns: `(water_imbibition_when_increasing, gas_imbibition_when_increasing)`: water imbibes
        as it rises when it is the wetting phase of the oil-water system, and gas only when it is
        the wetting phase of the gas-oil system.
    """
    return (
        relative_permeability.get_oil_water_wetting_phase() == FluidPhase.WATER,
        relative_permeability.get_gas_oil_wetting_phase() == FluidPhase.GAS,
    )


def as_hysteresis_relperm_table(
    *,
    drainage: RelativePermeabilityTable,
    imbibition: RelativePermeabilityTable,
    options: HysteresisOptions,
) -> RelativePermeabilityTable:
    """
    Pair a drainage and an imbibition relative permeability table into a hysteresis table.

    The model comes from `options.relperm_model`. Relative permeability stays on the drainage table
    when the model is `NONE` or `options.apply_to_relperm` is off.

    :param drainage: The drainage table, which a cell is on until it first reverses.
    :param imbibition: The imbibition table.
    :param options: How hysteresis is modelled.
    :returns: The hysteresis table, or `drainage` when relative permeability does not follow
        hysteresis.
    :raises NotSupportedError: For a two-phase table.
    """
    if not options.relperm_hysteresis_active:
        return drainage

    if not isinstance(drainage, ThreePhaseRelPermTable) or not isinstance(
        imbibition, ThreePhaseRelPermTable
    ):
        raise NotSupportedError(
            "Relative permeability hysteresis needs three-phase tables, with both the oil-water "
            "and the gas-oil system. Two-phase decks are not supported yet."
        )

    model = options.relperm_model
    if model == RelPermHysteresisModel.CARLSON:
        return CarlsonRelPermTable(
            oil_water_drainage_table=drainage,
            gas_oil_drainage_table=drainage,
            oil_water_imbibition_table=imbibition,
            gas_oil_imbibition_table=imbibition,
            wetting_phase_uses_imbibition=options.wetting_phase_uses_imbibition,
        )

    if model == RelPermHysteresisModel.KILLOUGH:
        return EclipseKilloughRelPermTable(
            oil_water_drainage_table=drainage,
            gas_oil_drainage_table=drainage,
            oil_water_imbibition_table=imbibition,
            gas_oil_imbibition_table=imbibition,
            wetting_phase_hysteresis=options.wetting_phase_uses_imbibition,
            wetting_phase_curvature=options.killough_wetting_curvature,
        )

    if model == RelPermHysteresisModel.BLENDED_SCANNING:
        land = derive_land_parameters(drainage=drainage, imbibition=imbibition)
        return KilloughLandRelPermTable(
            oil_water_drainage_table=drainage,
            gas_oil_drainage_table=drainage,
            oil_water_imbibition_table=imbibition,
            gas_oil_imbibition_table=imbibition,
            land_trapping_coefficient_water=land.land_trapping_coefficient_water,
            land_trapping_coefficient_gas=land.land_trapping_coefficient_gas,
            maximum_residual_oil_saturation_water=land.maximum_residual_oil_saturation_water,
            maximum_residual_oil_saturation_gas=land.maximum_residual_oil_saturation_gas,
            maximum_residual_gas_saturation=land.maximum_residual_gas_saturation,
        )
    raise NotSupportedError(f"Relative permeability hysteresis model `{model}` is not built.")


def as_hysteresis_capillary_pressure_table(
    *,
    drainage: CapillaryPressureTable | None,
    imbibition: CapillaryPressureTable | None,
    options: HysteresisOptions,
) -> CapillaryPressureTable | None:
    """
    Pair a drainage and an imbibition capillary pressure table into a hysteresis table.

    Capillary pressure stays on the drainage table when `options.apply_to_capillary_pressure` is
    off, or when there is no imbibition table to scan towards.

    :param drainage: The drainage table, or `None` when the region has no capillary pressure.
    :param imbibition: The imbibition table, or `None`.
    :param options: How hysteresis is modelled.
    :returns: The hysteresis table, or `drainage` when capillary pressure does not follow
        hysteresis.
    :raises NotSupportedError: For a two-phase table.
    """
    if drainage is None or imbibition is None or not options.apply_to_capillary_pressure:
        return drainage

    if not isinstance(drainage, ThreePhaseCapillaryPressureTable) or not isinstance(
        imbibition, ThreePhaseCapillaryPressureTable
    ):
        raise NotSupportedError(
            "Capillary pressure hysteresis needs three-phase tables, with both the oil-water and "
            "the gas-oil system. Two-phase decks are not supported yet."
        )

    return CurvatureCapillaryPressureTable(
        oil_water_drainage_table=drainage.oil_water_table,
        gas_oil_drainage_table=drainage.gas_oil_table,
        oil_water_imbibition_table=imbibition.oil_water_table,
        gas_oil_imbibition_table=imbibition.gas_oil_table,
        scanning_curvature=options.capillary_pressure_curvature,
        retrace_scanning_curves=options.retrace_capillary_pressure,
    )


def as_hysteresis_tables(
    *,
    drainage: SatFuncTables,
    imbibition: SatFuncTables,
    options: HysteresisOptions,
) -> SatFuncTables:
    """
    Pair the drainage and imbibition tables of a cell into the tables it is evaluated with.

    :param drainage: The tables `SATNUM` selects.
    :param imbibition: The tables `IMBNUM` selects.
    :param options: How hysteresis is modelled.
    :returns: Relative permeability and capillary pressure that follow hysteresis as `options`
        say, and drainage tables where they do not.
    """
    return SatFuncTables(
        relative_permeability=as_hysteresis_relperm_table(
            drainage=drainage.relative_permeability,
            imbibition=imbibition.relative_permeability,
            options=options,
        ),
        capillary_pressure=as_hysteresis_capillary_pressure_table(
            drainage=drainage.capillary_pressure,
            imbibition=imbibition.capillary_pressure,
            options=options,
        ),
    )

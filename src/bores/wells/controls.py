"""Well control mechanisms for reservoir simulation."""

import logging
import threading
import typing

import attrs
import numba

from bores.constants import c
from bores.errors import ComputationError, ValidationError
from bores.serialization import Serializable, make_serializable_type_registrar
from bores.stores import StoreSerializable
from bores.tables.pvt import PVTTables
from bores.types import FluidPhase
from bores.wells.core import (
    WellFluid,
    compute_average_compressibility_factor,
    compute_gas_well_rate,
    compute_oil_well_rate,
    compute_required_bhp_for_gas_rate,
    compute_required_bhp_for_oil_rate,
    get_pseudo_pressure_table,
)

logger = logging.getLogger(__name__)

__all__ = [
    "AdaptiveRateControl",
    "BHPControl",
    "ControlResult",
    "CoupledRateControl",
    "InjectionClamp",
    "MultiPhaseControl",
    "ProductionClamp",
    "RateClamp",
    "RateControl",
    "WellControl",
    "rate_clamp",
    "well_control",
]


WellFluidTcon = typing.TypeVar("WellFluidTcon", bound=WellFluid, contravariant=True)


@attrs.frozen
class ControlResult:
    """
    Combined result of a single well control evaluation.

    Bundles the flow rate and effective bottom-hole pressure computed in one
    unified pass through the control logic. The motivation is efficiency and
    consistency: the pressure solver needs the BHP for the implicit coupling
    term, and the saturation solver needs the corresponding flow rate. If those
    two quantities were obtained from separate `get_flow_rate` /
    `get_bottom_hole_pressure` calls they would (a) duplicate expensive
    intermediate work (pseudo-pressure table lookups, Z-factor averages,
    `_compute_required_bhp` solves) and (b) risk subtle inconsistencies if the
    reservoir state changes between calls.

    :param rate: Flow rate (bbl/day or ft³/day). Positive = injection,
        negative = production. Zero when the well is inactive, when flow is
        disallowed by a phase-mobility check, or when a BHP constraint cannot be
        satisfied.
    :param bhp: Effective bottom-hole pressure (psi). Equal to the reservoir
        pressure (no drawdown) when the well is inactive or flow is disallowed.
        Otherwise reflects the actual operating BHP after all constraints and
        clamps have been applied.
    """

    rate: float
    """Flow rate (bbl/day or ft³/day). Positive for injection, negative for production."""
    bhp: float
    """Effective bottom-hole pressure (psi)."""

    def __iter__(self) -> typing.Iterator[float]:
        yield self.rate
        yield self.bhp


def _disallow_flow(
    fluid: typing.Optional[WellFluid],
    is_active: bool,
    phase_mobility: typing.Optional[float] = None,
    minimum_mobility: float = 1e-18,
) -> bool:
    """
    Check if well should not allow flow and just return zero flow rate
    or same reservoir pressure (wtih zero drawdown).

    :param fluid: Well fluid object (`None` means no fluid).
    :param is_active: Whether well is active/open.
    :param phase_mobility: Phase mobility (cP⁻¹).
    :param minimum_mobility: Minimum mobility threshold below which phase is considered immobile (cP⁻¹).
        Default 1e-12 cP⁻¹ corresponds to k_r ≈ 0.00001 (essentially zero).
    :return: True if no flow should happen, False otherwise.
    """
    return (
        fluid is None
        or (phase_mobility is not None and phase_mobility < minimum_mobility)
        or not is_active
    )


def _apply_clamp(
    pressure: float,
    control_type: str,
    rate: typing.Optional[float] = None,
    bhp: typing.Optional[float] = None,
    clamp: typing.Optional["RateClamp"] = None,
) -> typing.Optional[float]:
    """
    Apply clamping condition if provided.

    :return: Clamped rate/bhp if clamp condition is met, None if not clamped (caller should return original rate)
    """
    if clamp is not None:
        if rate is not None:
            clamped_rate = clamp.clamp_rate(rate, pressure)
            if clamped_rate is not None:
                logger.debug(
                    f"Clamping rate {rate:.6f} to {clamped_rate:.6f} "
                    f"({control_type}, pressure={pressure:.3f} psi)"
                )
                return clamped_rate
        elif bhp is not None:
            clamped_bhp = clamp.clamp_bhp(bhp, pressure)
            if clamped_bhp is not None:
                logger.debug(
                    f"Clamping BHP {bhp:.6f} to {clamped_bhp:.6f} "
                    f"({control_type}, pressure={pressure:.3f} psi)"
                )
                return clamped_bhp
    return None


def _compute_required_bhp(
    target_rate: float,
    fluid: WellFluid,
    well_index: float,
    pressure: float,
    temperature: float,
    phase_mobility: float,
    use_pseudo_pressure: bool,
    formation_volume_factor: float,
    fluid_compressibility: typing.Optional[float],
    incompressibility_threshold: float = 1e-6,
    pvt_tables: typing.Optional[PVTTables] = None,
) -> float:
    """
    Compute required BHP to achieve target rate.

    :return: Required bottom hole pressure (psi)
    :raises ValidationError: If computation is not possible (e.g., zero mobility)
    :raises ZeroDivisionError: If rate equation has numerical issues
    """
    if fluid.phase == FluidPhase.GAS:
        # Setup pseudo-pressure if needed
        use_pp, pp_table = get_pseudo_pressure_table(
            fluid=fluid,
            pressure=pressure,
            temperature=temperature,
            use_pseudo_pressure=use_pseudo_pressure,
            pvt_tables=pvt_tables,
        )

        # Compute Z-factor using reservoir pressure as initial estimate
        specific_gravity = typing.cast(
            float,
            fluid.get_specific_gravity(pressure=pressure, temperature=temperature),
        )
        if specific_gravity is None:
            raise ValidationError(
                "Well fluid has no specific gravity define. Specify a value or provide a PVT table for the fluid."
            )
        avg_z_factor = compute_average_compressibility_factor(
            pressure=pressure,
            temperature=temperature,
            gas_gravity=specific_gravity,
        )
        return compute_required_bhp_for_gas_rate(
            target_rate=target_rate,
            well_index=well_index,
            pressure=pressure,
            temperature=temperature,
            phase_mobility=phase_mobility,
            average_compressibility_factor=avg_z_factor,
            use_pseudo_pressure=use_pp,
            pseudo_pressure_table=pp_table,
            formation_volume_factor=formation_volume_factor,
        )
    # For oil/water
    return compute_required_bhp_for_oil_rate(
        target_rate=target_rate,
        well_index=well_index,
        pressure=pressure,
        phase_mobility=phase_mobility,
        fluid_compressibility=fluid_compressibility,
        incompressibility_threshold=incompressibility_threshold,
    )


class RateClamp(Serializable):
    """
    Base class for a well rate clamp.

    Determines when a computed flow rate or BHP should be clamped
    to prevent unphysical scenarios (e.g., production during injection).
    """

    __abstract_serializable__ = True

    def clamp_rate(
        self, rate: float, pressure: float, **kwargs
    ) -> typing.Optional[float]:
        """
        Determine if the flow rate should be clamped to zero.

        :param rate: The computed flow rate (bbl/day or ft³/day).
        :param pressure: The reservoir pressure at the well location (psi).
        :param kwargs: Additional context for clamping decision.
        :return: The clamped flow rate if clamping condition is met, else None.
        """
        raise NotImplementedError

    def clamp_bhp(
        self, bottom_hole_pressure: float, pressure: float, **kwargs
    ) -> typing.Optional[float]:
        """
        Determine if the bottom-hole pressure should be clamped.

        :param bottom_hole_pressure: The computed bottom-hole pressure (psi).
        :param pressure: The reservoir pressure at the well location (psi).
        :param kwargs: Additional context for clamping decision.
        :return: The clamped bottom-hole pressure if clamping condition is met, else None.
        """
        raise NotImplementedError


_CLAMP_TYPES: typing.Dict[str, typing.Type[RateClamp]] = {}
"""Registry for rate clamp types."""
rate_clamp = make_serializable_type_registrar(
    base_cls=RateClamp,
    registry=_CLAMP_TYPES,
    lock=threading.Lock(),
    key_attr="__type__",
    override=False,
    auto_register_serializer=True,
    auto_register_deserializer=True,
)
"""Decorator to register a new rate clamp type."""


@rate_clamp
@attrs.frozen
class ProductionClamp(RateClamp):
    """Clamp condition for production wells."""

    __type__ = "production_clamp"

    value: float = 0.0
    """Clamp value to return when condition is met."""

    def clamp_rate(
        self, rate: float, pressure: float, **kwargs
    ) -> typing.Optional[float]:
        """Clamp if rate is positive (injection during production)."""
        if rate > 0.0:
            return self.value
        return None

    def clamp_bhp(
        self, bottom_hole_pressure: float, pressure: float, **kwargs
    ) -> typing.Optional[float]:
        if bottom_hole_pressure > pressure:
            return pressure
        return None


@rate_clamp
@attrs.frozen
class InjectionClamp(RateClamp):
    """Clamp condition for injection wells."""

    __type__ = "injection_clamp"

    value: float = 0.0
    """Clamp value to return when condition is met."""

    def clamp_rate(
        self, rate: float, pressure: float, **kwargs
    ) -> typing.Optional[float]:
        """Clamp if rate is negative (production during injection)."""
        if rate < 0.0:
            return self.value
        return None

    def clamp_bhp(
        self, bottom_hole_pressure: float, pressure: float, **kwargs
    ) -> typing.Optional[float]:
        if bottom_hole_pressure < pressure:
            return pressure
        return None


WellControlType = typing.Literal["rate", "bhp", "custom"]


class WellControl(StoreSerializable, typing.Generic[WellFluidTcon]):
    """
    Base class for well control implementations.

    Interface for computing flow rates and bottom-hole pressures
    under different control strategies.
    """

    __abstract_serializable__ = True

    def is_bhp_control(self) -> bool:
        return self.get_type() == "bhp"

    def is_rate_control(self) -> bool:
        return self.get_type() == "rate"

    def is_custom_control(self) -> bool:
        return self.get_type() == "custom"

    def get_type(self) -> WellControlType:
        """Returns type of the well control. Either 'rate' or 'bhp'"""
        raise NotImplementedError

    def get_flow_rate(
        self,
        pressure: float,
        temperature: float,
        well_index: float,
        fluid: WellFluidTcon,
        formation_volume_factor: float,
        phase_mobility: typing.Optional[float] = None,
        allocation_fraction: float = 1.0,
        is_active: bool = True,
        use_pseudo_pressure: bool = False,
        fluid_compressibility: typing.Optional[float] = None,
        pvt_tables: typing.Optional[PVTTables] = None,
        **kwargs: typing.Any,
    ) -> float:
        """
        Compute the flow rate based on the control method.

        :param pressure: The reservoir pressure at the well location (psi).
        :param temperature: The reservoir temperature at the well location (°F).
        :param phase_mobility: The relative mobility of the fluid phase.
            Not so relevant for rate controlled injection wells
        :param well_index: The well index (md*ft).
        :param fluid: The fluid being produced or injected.
        :param formation_volume_factor: Formation volume factor (bbl/STB or ft³/SCF).
        :param allocation_fraction: Fraction of target rate to allocate to this cell (for multi-cell wells).
            For rate controls, cell_rate = target_rate x allocation_fraction.
            Typically allocation_fraction = cell_WI / total_well_WI.
            Default is 1.0 (single-cell well or no allocation needed).
        :param is_active: Whether the well is active/open.
        :param use_pseudo_pressure: Whether to use pseudo-pressure for gas wells.
        :param fluid_compressibility: Compressibility of the fluid (psi⁻¹).
        :param pvt_tables: `PVTTables` object for fluid property lookups
        :return: The flow rate in (bbl/day or ft³/day). Positive for injection, negative for production.
        """
        raise NotImplementedError

    def get_bottom_hole_pressure(
        self,
        pressure: float,
        temperature: float,
        well_index: float,
        fluid: WellFluid,
        formation_volume_factor: float,
        phase_mobility: typing.Optional[float] = None,
        allocation_fraction: float = 1.0,
        is_active: bool = True,
        use_pseudo_pressure: bool = False,
        fluid_compressibility: typing.Optional[float] = None,
        pvt_tables: typing.Optional[PVTTables] = None,
        **kwargs: typing.Any,
    ) -> float:
        """
        Compute the effective bottom-hole pressure for this control at current conditions.

        This is used for semi-implicit treatment in the pressure equation.

        For BHP control: returns the specified BHP
        For rate control: returns the BHP required to achieve target rate
        For adaptive control: returns BHP based on current operating mode

        :param pressure: Reservoir pressure at well location (psi)
        :param temperature: Reservoir temperature (°F)
        :param phase_mobility: Phase mobility (1/cP). Not so relevant for rate controlled injection wells
        :param well_index: Well index (md*ft)
        :param fluid: Fluid being produced/injected
        :param formation_volume_factor: Formation volume factor
        :param allocation_fraction: Fraction of target rate to allocate to this cell (for multi-cell wells).
            For rate controls, cell_rate = target_rate x allocation_fraction.
            Default is 1.0 (single-cell well or no allocation needed).
        :param is_active: Whether well is active
        :param use_pseudo_pressure: Whether to use pseudo-pressure for gas
        :param fluid_compressibility: Fluid compressibility (1/psi)
        :param pvt_tables: `PVTTables` object for fluid property lookups
        :return: Effective bottom-hole pressure (psi)
        """
        raise NotImplementedError

    def get_control(
        self,
        pressure: float,
        temperature: float,
        well_index: float,
        fluid: WellFluidTcon,
        formation_volume_factor: float,
        phase_mobility: typing.Optional[float] = None,
        allocation_fraction: float = 1.0,
        is_active: bool = True,
        use_pseudo_pressure: bool = False,
        fluid_compressibility: typing.Optional[float] = None,
        pvt_tables: typing.Optional[PVTTables] = None,
        **kwargs: typing.Any,
    ) -> ControlResult:
        """
        Compute both the flow rate and effective bottom-hole pressure in a single pass.

        Rather than calling `get_flow_rate` and `get_bottom_hole_pressure` separately
        (which would duplicate expensive intermediate work such as pseudo-pressure
        table lookups and Z-factor averages), callers should use this method and
        cache the returned `ControlResult`. The pressure solver can then
        use `result.bhp` for the implicit coupling term while the saturation
        solver uses `result.rate`, guaranteeing consistency between the two.

        The base implementation falls back to calling `get_flow_rate` and
        `get_bottom_hole_pressure` sequentially. Concrete subclasses override
        this method to share intermediate computations and avoid duplication.

        :param pressure: Reservoir pressure at the well location (psi).
        :param temperature: Reservoir temperature at the well location (°F).
        :param phase_mobility: Relative mobility of the fluid phase (cP⁻¹).
        :param well_index: Well index (md·ft).
        :param fluid: Fluid being produced or injected.
        :param formation_volume_factor: Formation volume factor (bbl/STB or ft³/SCF).
        :param allocation_fraction: Fraction of total well rate allocated to this
            cell (= cell_WI / total_WI for multi-cell wells).  Default 1.0.
        :param is_active: Whether the well is currently open.
        :param use_pseudo_pressure: Whether to use pseudo-pressure for gas wells.
        :param fluid_compressibility: Fluid compressibility (psi⁻¹).
        :param pvt_tables: PVT look-up tables for fluid properties.
        :param kwargs: Additional control-specific arguments (e.g. primary-phase
            context for `CoupledRateControl`).
        :return: `ControlResult` containing the flow rate (bbl/day or
            ft³/day) and effective BHP (psi).
        """
        rate = self.get_flow_rate(
            pressure=pressure,
            temperature=temperature,
            well_index=well_index,
            fluid=fluid,
            formation_volume_factor=formation_volume_factor,
            phase_mobility=phase_mobility,
            allocation_fraction=allocation_fraction,
            is_active=is_active,
            use_pseudo_pressure=use_pseudo_pressure,
            fluid_compressibility=fluid_compressibility,
            pvt_tables=pvt_tables,
            **kwargs,
        )
        bhp = self.get_bottom_hole_pressure(
            pressure=pressure,
            temperature=temperature,
            well_index=well_index,
            fluid=fluid,
            formation_volume_factor=formation_volume_factor,
            phase_mobility=phase_mobility,
            allocation_fraction=allocation_fraction,
            is_active=is_active,
            use_pseudo_pressure=use_pseudo_pressure,
            fluid_compressibility=fluid_compressibility,
            pvt_tables=pvt_tables,
            **kwargs,
        )
        return ControlResult(rate=rate, bhp=bhp)


_WELL_CONTROLS: typing.Dict[str, typing.Type[WellControl]] = {}
"""Registry for well control types."""
well_control = make_serializable_type_registrar(
    base_cls=WellControl,
    registry=_WELL_CONTROLS,
    lock=threading.Lock(),
    key_attr="__type__",
    override=False,
    auto_register_serializer=True,
    auto_register_deserializer=True,
)
"""Decorator to register a new well control type."""


@well_control
@attrs.frozen
class BHPControl(WellControl[WellFluidTcon]):
    """
    Bottom Hole Pressure (BHP) control.

    Computes flow rate based on pressure differential between reservoir and
    wellbore using Darcy's law. This is the traditional well control method.
    """

    __type__ = "bhp_control"

    bhp: float = attrs.field(validator=attrs.validators.gt(0))
    """Well bottom-hole flowing pressure in psi."""
    target_phase: typing.Optional[typing.Union[str, FluidPhase]] = None
    """Target fluid phase for the control."""
    clamp: typing.Optional[RateClamp] = None
    """Condition for clamping flow rates to zero. None means no clamping."""

    def __attrs_post_init__(self) -> None:
        """Validate control parameters."""
        if self.bhp <= 0.0:
            raise ValidationError("Well bottom hole pressure must be positive.")

        if self.target_phase is not None:
            object.__setattr__(self, "target_phase", FluidPhase(self.target_phase))

    def get_type(self) -> WellControlType:
        return "bhp"

    def get_flow_rate(
        self,
        pressure: float,
        temperature: float,
        well_index: float,
        fluid: WellFluidTcon,
        formation_volume_factor: float,
        phase_mobility: typing.Optional[float] = None,
        allocation_fraction: float = 1.0,
        is_active: bool = True,
        use_pseudo_pressure: bool = False,
        fluid_compressibility: typing.Optional[float] = None,
        pvt_tables: typing.Optional[PVTTables] = None,
        **kwargs: typing.Any,
    ) -> float:
        """
        Compute flow rate using BHP control (Darcy's law).

        Flow rate is proportional to (P_reservoir - P_wellbore).

        :param pressure: Reservoir pressure at the well location (psi).
        :param temperature: Reservoir temperature at the well location (°F).
        :param phase_mobility: Relative mobility of the fluid phase. Required for BHP control.
        :param well_index: Well index (md*ft).
        :param fluid: Fluid being produced or injected.
        :param formation_volume_factor: Formation volume factor (bbl/STB or ft³/SCF).
        :param allocation_fraction: Ignored for BHP control (rate naturally allocates proportionally to WI).
        :param is_active: Whether the well is active/open.
        :param use_pseudo_pressure: Whether to use pseudo-pressure for gas wells.
        :param fluid_compressibility: Compressibility of the fluid (psi⁻¹).
        :param pvt_tables: `PVTTables` object for fluid property lookups
        :return: Flow rate in (bbl/day or ft³/day).
        """
        if phase_mobility is None:
            raise ValidationError(
                "Phase mobility is required for bottom hole pressure (BHP) control"
            )

        if _disallow_flow(fluid=fluid, is_active=is_active) or (
            self.target_phase is not None and fluid.phase != self.target_phase
        ):
            return 0.0

        bhp = self.bhp

        # Compute rate based on fluid phase
        if fluid.phase == FluidPhase.GAS:
            # Setup pseudo-pressure if needed
            use_pp, pp_table = get_pseudo_pressure_table(
                fluid=fluid,
                pressure=pressure,
                temperature=temperature,
                use_pseudo_pressure=use_pseudo_pressure,
                pvt_tables=pvt_tables,
            )
            # Compute Z-factor using reservoir pressure as initial estimate
            specific_gravity = typing.cast(
                float,
                fluid.get_specific_gravity(pressure=pressure, temperature=temperature),
            )
            if specific_gravity is None:
                raise ValidationError(
                    "Well fluid has no specific gravity define. Specify a value or provide a PVT table for the fluid."
                )
            avg_z_factor = compute_average_compressibility_factor(
                pressure=pressure,
                temperature=temperature,
                gas_gravity=specific_gravity,
                bottom_hole_pressure=bhp,
            )
            rate = compute_gas_well_rate(
                well_index=well_index,
                pressure=pressure,
                temperature=temperature,
                bottom_hole_pressure=bhp,
                phase_mobility=phase_mobility,
                use_pseudo_pressure=use_pp,
                pseudo_pressure_table=pp_table,
                average_compressibility_factor=avg_z_factor,
                formation_volume_factor=formation_volume_factor,
            )
        else:
            # For water and oil wells
            rate = compute_oil_well_rate(
                well_index=well_index,
                pressure=pressure,
                bottom_hole_pressure=bhp,
                phase_mobility=phase_mobility,
                fluid_compressibility=fluid_compressibility,
                incompressibility_threshold=c.FLUID_INCOMPRESSIBILITY_THRESHOLD,
            )

        # Apply clamp condition if any
        clamped = _apply_clamp(
            rate=rate,
            clamp=self.clamp,
            pressure=pressure,
            control_type="BHP control",
        )
        return clamped if clamped is not None else rate

    def get_bottom_hole_pressure(
        self,
        pressure: float,
        temperature: float,
        well_index: float,
        fluid: WellFluid,
        formation_volume_factor: float,
        phase_mobility: typing.Optional[float] = None,
        allocation_fraction: float = 1.0,
        is_active: bool = True,
        use_pseudo_pressure: bool = False,
        fluid_compressibility: typing.Optional[float] = None,
        pvt_tables: typing.Optional[PVTTables] = None,
        **kwargs: typing.Any,
    ) -> float:
        """
        Return the specified bottom-hole pressure for BHP control.

        :param pressure: Reservoir pressure at well location (psi)
        :param temperature: Reservoir temperature (°F)
        :param phase_mobility: Phase mobility (1/cP). Required for BHP control.
        :param well_index: Well index (md*ft)
        :param fluid: Fluid being produced/injected
        :param formation_volume_factor: Formation volume factor
        :param allocation_fraction: Ignored for BHP control (BHP is same for all cells).
        :param is_active: Whether well is active
        :param use_pseudo_pressure: Whether to use pseudo-pressure for gas
        :param fluid_compressibility: Fluid compressibility (1/psi)
        :param pvt_tables: `PVTTables` object for fluid property lookups
        :return: Effective bottom-hole pressure (psi)
        """
        if phase_mobility is None:
            raise ValidationError(
                "Phase mobility is required for bottom hole pressure (BHP) control"
            )

        if _disallow_flow(fluid=fluid, is_active=is_active) or (
            self.target_phase is not None and fluid.phase != self.target_phase
        ):
            # Return reservoir pressure (no driving force)
            return pressure

        bhp = self.bhp
        return (
            _apply_clamp(
                pressure=pressure,
                control_type="BHP control",
                clamp=self.clamp,
                bhp=bhp,
            )
            or bhp
        )

    def get_control(
        self,
        pressure: float,
        temperature: float,
        well_index: float,
        fluid: WellFluidTcon,
        formation_volume_factor: float,
        phase_mobility: typing.Optional[float] = None,
        allocation_fraction: float = 1.0,
        is_active: bool = True,
        use_pseudo_pressure: bool = False,
        fluid_compressibility: typing.Optional[float] = None,
        pvt_tables: typing.Optional[PVTTables] = None,
        **kwargs: typing.Any,
    ) -> ControlResult:
        """
        Compute rate and BHP simultaneously for BHP control.

        BHP control fixes the wellbore pressure; both the effective BHP and the
        resulting Darcy rate share the same intermediate quantities (pseudo-pressure
        table and Z-factor average for gas, straightforward Darcy for liquid).
        Those quantities are therefore computed only once here.

        When flow is disallowed (inactive well, wrong phase, or phase-mobility
        below threshold), `rate=0` and `bhp=pressure` are returned immediately
        without any further computation.

        When a `clamp` is set the same clamped BHP/rate values that the
        individual methods would produce are returned, but derived from a single
        shared evaluation.

        :param pressure: Reservoir pressure at the well location (psi).
        :param temperature: Reservoir temperature at the well location (°F).
        :param phase_mobility: Relative mobility of the fluid phase (cP⁻¹).
            Required - raises `ValidationError` if `None`.
        :param well_index: Well index (md·ft).
        :param fluid: Fluid being produced or injected.
        :param formation_volume_factor: Formation volume factor (bbl/STB or ft³/SCF).
        :param allocation_fraction: Ignored for BHP control (Darcy allocation is
            implicit in the well index).
        :param is_active: Whether the well is currently open.
        :param use_pseudo_pressure: Whether to use pseudo-pressure for gas wells.
        :param fluid_compressibility: Fluid compressibility (psi⁻¹).
        :param pvt_tables: PVT look-up tables for fluid properties.
        :return: `ControlResult` with the Darcy rate and effective BHP.
        :raises ValidationError: If `phase_mobility` is `None`.
        """
        if phase_mobility is None:
            raise ValidationError(
                "Phase mobility is required for bottom hole pressure (BHP) control"
            )

        if _disallow_flow(fluid=fluid, is_active=is_active) or (
            self.target_phase is not None and fluid.phase != self.target_phase
        ):
            return ControlResult(rate=0.0, bhp=pressure)

        bhp = self.bhp

        # Apply BHP clamp before computing rate so the two are consistent.
        clamped_bhp = _apply_clamp(
            pressure=pressure,
            control_type="BHP control",
            clamp=self.clamp,
            bhp=bhp,
        )
        effective_bhp = clamped_bhp if clamped_bhp is not None else bhp

        # Compute rate from `effective_bhp` (shared intermediates)
        if fluid.phase == FluidPhase.GAS:
            use_pp, pp_table = get_pseudo_pressure_table(
                fluid=fluid,
                pressure=pressure,
                temperature=temperature,
                use_pseudo_pressure=use_pseudo_pressure,
                pvt_tables=pvt_tables,
            )
            specific_gravity = typing.cast(
                float,
                fluid.get_specific_gravity(pressure=pressure, temperature=temperature),
            )
            if specific_gravity is None:
                raise ValidationError(
                    "Well fluid has no specific gravity defined. "
                    "Specify a value or provide a PVT table for the fluid."
                )
            avg_z_factor = compute_average_compressibility_factor(
                pressure=pressure,
                temperature=temperature,
                gas_gravity=specific_gravity,
                bottom_hole_pressure=effective_bhp,
            )
            rate = compute_gas_well_rate(
                well_index=well_index,
                pressure=pressure,
                temperature=temperature,
                bottom_hole_pressure=effective_bhp,
                phase_mobility=phase_mobility,
                use_pseudo_pressure=use_pp,
                pseudo_pressure_table=pp_table,
                average_compressibility_factor=avg_z_factor,
                formation_volume_factor=formation_volume_factor,
            )
        else:
            rate = compute_oil_well_rate(
                well_index=well_index,
                pressure=pressure,
                bottom_hole_pressure=effective_bhp,
                phase_mobility=phase_mobility,
                fluid_compressibility=fluid_compressibility,
                incompressibility_threshold=c.FLUID_INCOMPRESSIBILITY_THRESHOLD,
            )

        # Apply rate clamp (BHP has already been clamped above).
        clamped_rate = _apply_clamp(
            rate=rate,
            clamp=self.clamp,
            pressure=pressure,
            control_type="BHP control",
        )
        final_rate = clamped_rate if clamped_rate is not None else rate
        return ControlResult(rate=final_rate, bhp=effective_bhp)

    def __str__(self) -> str:
        """String representation."""
        return f"BHP Control: BHP={self.bhp:.3e} psi"


@well_control
@attrs.frozen
class RateControl(WellControl[WellFluidTcon]):
    """
    Constant rate control.

    Maintains a target flow rate regardless of reservoir pressure,
    as long as the pressure constraint is satisfied.

    **IMPORTANT:** For injection wells, it is **highly recommended** to set `bhp_limit`
    to prevent unrealistic injection pressures, especially when injecting into low-mobility
    zones (e.g., water injection at connate saturation).
    """

    __type__ = "rate_control"

    target_rate: float
    """Target flow rate (STB/day or SCF/day). Positive for injection, negative for production."""
    bhp_limit: typing.Optional[float] = None
    """
    Minimum allowable BHP for production wells, and maximum allowable BHP for injection wells.

    BHP constraint for rate control:
    - For production: Minimum allowable BHP (well won't flow if pressure drops below this).
    - For injection: Maximum allowable BHP (prevents fracturing/unrealistic pressures).
    
    **Strongly recommended for injection wells** to avoid numerical issues when injecting
    into low-mobility zones.
    
    If not specified, no BHP constraint is applied (rate is always achieved regardless of
    required pressure. Use with caution!).
    """
    target_phase: typing.Optional[typing.Union[str, FluidPhase]] = None
    """Target fluid phase for the control."""
    clamp: typing.Optional[RateClamp] = None
    """Condition for clamping flow rates to zero. None means no clamping."""

    def __attrs_post_init__(self) -> None:
        """Validate control parameters."""
        if self.target_rate == 0.0:
            raise ValidationError(
                "Target rate cannot be zero. Use `well.shut_in` instead."
            )
        if self.bhp_limit is not None and self.bhp_limit <= 0.0:
            raise ValidationError("Minimum bottom hole pressure must be positive.")

        if self.target_phase is not None:
            object.__setattr__(self, "target_phase", FluidPhase(self.target_phase))

    def get_type(self) -> WellControlType:
        return "rate"

    def get_flow_rate(
        self,
        pressure: float,
        temperature: float,
        well_index: float,
        fluid: WellFluidTcon,
        formation_volume_factor: float,
        phase_mobility: typing.Optional[float] = None,
        allocation_fraction: float = 1.0,
        is_active: bool = True,
        use_pseudo_pressure: bool = False,
        fluid_compressibility: typing.Optional[float] = None,
        pvt_tables: typing.Optional[PVTTables] = None,
        **kwargs: typing.Any,
    ) -> float:
        """
        Return constant target rate, subject to BHP constraint.

        :param pressure: Reservoir pressure at the well location (psi).
        :param temperature: Reservoir temperature at the well location (°F).
        :param phase_mobility: Relative mobility of the fluid phase. Leave as `None` to operate in strict rate mode.
        :param well_index: Well index (md*ft).
        :param fluid: Fluid being produced or injected.
        :param formation_volume_factor: Formation volume factor (bbl/STB or ft³/SCF).
        :param allocation_fraction: Fraction of target rate to allocate to this cell (for multi-cell wells).
            This parameter allocates the well's total target rate proportionally across perforated cells.
            Typically allocation_fraction = cell_WI / total_well_WI.
        :param is_active: Whether the well is active/open.
        :param use_pseudo_pressure: Whether to use pseudo-pressure for gas wells.
        :param fluid_compressibility: Compressibility of the fluid (psi⁻¹).
        :param pvt_tables: `PVTTables` object for fluid property lookups
        :return: Target flow rate if the required bottom hole pressure to produce/inject
            is above or equal to the minimum bottom hole pressure constraint (if any). Otherwise returns 0.0.
        """
        if _disallow_flow(fluid=fluid, is_active=is_active) or (
            self.target_phase is not None and fluid.phase != self.target_phase
        ):
            return 0.0

        # Apply allocation to target rate
        target_rate = (
            self.target_rate * allocation_fraction * formation_volume_factor
        )  # Convert to reservoir rate and allocate to cell

        # If phase mobility is None, then the rate control is strict rate mode so no BHP checks. Else,
        # Check if achieving target rate would violate minimum bottom hole pressure constraint
        if phase_mobility is not None and self.bhp_limit is not None:
            bhp_limit = self.bhp_limit
            is_production = target_rate < 0.0  # Negative rate indicates production
            try:
                required_bhp = _compute_required_bhp(
                    target_rate=target_rate,
                    fluid=fluid,
                    well_index=well_index,
                    pressure=pressure,
                    temperature=temperature,
                    phase_mobility=phase_mobility,
                    use_pseudo_pressure=use_pseudo_pressure,
                    fluid_compressibility=fluid_compressibility,
                    incompressibility_threshold=c.FLUID_INCOMPRESSIBILITY_THRESHOLD,
                    formation_volume_factor=formation_volume_factor,
                    pvt_tables=pvt_tables,
                )

            except (ValueError, ZeroDivisionError, ComputationError) as exc:
                logger.warning(
                    f"Failed to compute required BHP for target rate {target_rate:.6f}: {exc}. "
                    "Returning 0."
                )
                return 0.0
            else:
                logger.debug(
                    f"Required BHP: {required_bhp:.6f} psi, Reservoir pressure: {pressure:.6f} psi, Fluid phase: {fluid.phase}"
                )
                if is_production:
                    can_achieve_rate = required_bhp >= bhp_limit
                else:
                    can_achieve_rate = required_bhp <= bhp_limit

                if can_achieve_rate is False:
                    logger.debug(
                        f"Cannot achieve target rate {target_rate:.6f} "
                        f"without violating bottom hole pressure limit {bhp_limit:.3f} psi "
                        f"(required BHP: {required_bhp:.3f} psi, pressure: {pressure:.3f} psi)"
                    )
                    return 0.0

        clamped = _apply_clamp(
            rate=target_rate,
            clamp=self.clamp,
            pressure=pressure,
            control_type="constant rate control",
        )
        return clamped if clamped is not None else target_rate

    def get_bottom_hole_pressure(
        self,
        pressure: float,
        temperature: float,
        well_index: float,
        fluid: WellFluid,
        formation_volume_factor: float,
        phase_mobility: typing.Optional[float] = None,
        allocation_fraction: float = 1.0,
        is_active: bool = True,
        use_pseudo_pressure: bool = False,
        fluid_compressibility: typing.Optional[float] = None,
        pvt_tables: typing.Optional[PVTTables] = None,
        **kwargs: typing.Any,
    ) -> float:
        """
        Compute BHP required to achieve target rate.

        :param pressure: Reservoir pressure at well location (psi)
        :param temperature: Reservoir temperature (°F)
        :param phase_mobility: Phase mobility (1/cP). This is required to get the effective BHP for specified rate.
        :param well_index: Well index (md*ft)
        :param fluid: Fluid being produced/injected
        :param formation_volume_factor: Formation volume factor
        :param allocation_fraction: Fraction of target rate to allocate to this cell.
        :param is_active: Whether well is active
        :param use_pseudo_pressure: Whether to use pseudo-pressure for gas
        :param fluid_compressibility: Fluid compressibility (1/psi)
        :param pvt_tables: `PVTTables` object for fluid property lookups
        :return: Effective bottom-hole pressure (psi)
        """
        if phase_mobility is None:
            raise ValidationError(
                "Phase mobility is required to get the effective bottom hole pressure (BHP) for the rate control."
            )

        if _disallow_flow(fluid=fluid, is_active=is_active) or (
            self.target_phase is not None and fluid.phase != self.target_phase
        ):
            return pressure

        # Apply allocation to target rate and convert to reservoir rate
        target_rate_reservoir = (
            self.target_rate * allocation_fraction * formation_volume_factor
        )
        # Compute required BHP for target rate
        try:
            required_bhp = _compute_required_bhp(
                target_rate=target_rate_reservoir,
                fluid=fluid,
                well_index=well_index,
                pressure=pressure,
                temperature=temperature,
                phase_mobility=phase_mobility,
                use_pseudo_pressure=use_pseudo_pressure,
                formation_volume_factor=formation_volume_factor,
                fluid_compressibility=fluid_compressibility,
                incompressibility_threshold=c.FLUID_INCOMPRESSIBILITY_THRESHOLD,
                pvt_tables=pvt_tables,
            )
        except (ValueError, ZeroDivisionError, ComputationError) as exc:
            logger.warning(
                f"Cannot compute required BHP: {exc}. Using reservoir pressure."
            )
            return (
                _apply_clamp(
                    pressure=pressure,
                    control_type="constant rate control",
                    clamp=self.clamp,
                    bhp=pressure,
                )
                or pressure
            )

        # Check BHP constraint
        bhp = required_bhp
        bhp_limit = self.bhp_limit
        if bhp_limit is not None:
            is_production = target_rate_reservoir < 0.0

            if is_production:
                # Production: BHP must be >= bhp_limit
                if required_bhp < bhp_limit:
                    logger.debug(
                        f"Required BHP {required_bhp:.4f} < min {bhp_limit:.4f}. "
                        f"Using constraint BHP."
                    )
                    bhp = bhp_limit
            else:
                # Injection: BHP must be <= max_bhp (bhp_limit is actually max here)
                if required_bhp > bhp_limit:
                    logger.debug(
                        f"Required BHP {required_bhp:.4f} > max {bhp_limit:.4f}. "
                        f"Using constraint BHP."
                    )
                    bhp = bhp_limit

        return (
            _apply_clamp(
                pressure=pressure,
                control_type="constant rate control",
                clamp=self.clamp,
                bhp=bhp,
            )
            or bhp
        )

    def get_control(
        self,
        pressure: float,
        temperature: float,
        well_index: float,
        fluid: WellFluidTcon,
        formation_volume_factor: float,
        phase_mobility: typing.Optional[float] = None,
        allocation_fraction: float = 1.0,
        is_active: bool = True,
        use_pseudo_pressure: bool = False,
        fluid_compressibility: typing.Optional[float] = None,
        pvt_tables: typing.Optional[PVTTables] = None,
        **kwargs: typing.Any,
    ) -> ControlResult:
        """
        Compute rate and effective BHP in a single pass for constant rate control.

        The key expense shared between `get_flow_rate` and
        `get_bottom_hole_pressure` is the call to `_compute_required_bhp`
        (which may involve a pseudo-pressure table look-up and a Z-factor solve
        for gas). This method performs that solve exactly once and derives both
        outputs from the result.

        **Operating modes:**

        * *Strict rate mode* (`phase_mobility is None` or `bhp_limit is None`):
          The allocated target rate is returned directly without a feasibility
          check. BHP is back-computed from that rate; if the back-computation
          fails the reservoir pressure is used as a fallback.

        * *Constrained rate mode* (both `phase_mobility` and `bhp_limit` are
          provided): `_compute_required_bhp` is called once.  If the required
          BHP satisfies the constraint the target rate and required BHP are
          returned. If the constraint would be violated `rate=0` and
          `bhp=pressure` are returned (the well effectively shuts in at this
          timestep).

        In all cases the same `clamp` logic that the individual methods
        apply is honoured, and the returned rate and BHP are mutually consistent.

        :param pressure: Reservoir pressure at the well location (psi).
        :param temperature: Reservoir temperature at the well location (°F).
        :param phase_mobility: Relative mobility of the fluid phase (cP⁻¹).
            Pass `None` for strict rate mode (no BHP feasibility check).
        :param well_index: Well index (md·ft).
        :param fluid: Fluid being produced or injected.
        :param formation_volume_factor: Formation volume factor (bbl/STB or ft³/SCF).
        :param allocation_fraction: Fraction of total well rate allocated to this
            cell (= cell_WI / total_WI for multi-cell wells). Default 1.0.
        :param is_active: Whether the well is currently open.
        :param use_pseudo_pressure: Whether to use pseudo-pressure for gas wells.
        :param fluid_compressibility: Fluid compressibility (psi⁻¹).
        :param pvt_tables: PVT look-up tables for fluid properties.
        :return: `ControlResult` with the flow rate and effective BHP.
            When the BHP constraint cannot be satisfied both fields reflect the
            shut-in state: `rate=0`, `bhp=pressure`.
        """
        if _disallow_flow(fluid=fluid, is_active=is_active) or (
            self.target_phase is not None and fluid.phase != self.target_phase
        ):
            return ControlResult(rate=0.0, bhp=pressure)

        # Allocated reservoir rate
        target_rate = self.target_rate * allocation_fraction * formation_volume_factor

        # Strict rate mode: no mobility provided hence skip BHP feasibility check.
        # We still need to report a BHP; try to back-compute it from the rate,
        # falling back to reservoir pressure if the solve fails.
        if phase_mobility is None:
            clamped_rate = _apply_clamp(
                rate=target_rate,
                clamp=self.clamp,
                pressure=pressure,
                control_type="constant rate control",
            )
            final_rate = clamped_rate if clamped_rate is not None else target_rate
            # BHP is indeterminate in strict mode - return reservoir pressure as
            # a conservative sentinel (zero drawdown).
            return ControlResult(rate=final_rate, bhp=pressure)

        # Constrained mode: solve for required BHP once.
        bhp_limit = self.bhp_limit
        is_production = target_rate < 0.0

        try:
            required_bhp = _compute_required_bhp(
                target_rate=target_rate,
                fluid=fluid,
                well_index=well_index,
                pressure=pressure,
                temperature=temperature,
                phase_mobility=phase_mobility,
                use_pseudo_pressure=use_pseudo_pressure,
                fluid_compressibility=fluid_compressibility,
                incompressibility_threshold=c.FLUID_INCOMPRESSIBILITY_THRESHOLD,
                formation_volume_factor=formation_volume_factor,
                pvt_tables=pvt_tables,
            )
        except (ValueError, ZeroDivisionError, ComputationError) as exc:
            logger.warning(
                f"Failed to compute required BHP for target rate {target_rate:.6f}: {exc}. "
                "Returning shut-in state (rate=0, bhp=reservoir pressure)."
            )
            return ControlResult(rate=0.0, bhp=pressure)

        logger.debug(
            f"Required BHP: {required_bhp:.6f} psi, Reservoir pressure: {pressure:.6f} psi, "
            f"Fluid phase: {fluid.phase}"
        )

        if bhp_limit is not None:
            if is_production:
                can_achieve_rate = required_bhp >= bhp_limit
            else:
                can_achieve_rate = required_bhp <= bhp_limit

            if not can_achieve_rate:
                logger.debug(
                    f"Cannot achieve target rate {target_rate:.6f} without violating "
                    f"BHP limit {bhp_limit:.3f} psi (required BHP: {required_bhp:.3f} psi, "
                    f"reservoir pressure: {pressure:.3f} psi). Returning shut-in state."
                )
                return ControlResult(rate=0.0, bhp=pressure)

        clamped_rate = _apply_clamp(
            rate=target_rate,
            clamp=self.clamp,
            pressure=pressure,
            control_type="constant rate control",
        )
        final_rate = clamped_rate if clamped_rate is not None else target_rate

        # Clamp the BHP that corresponds to the (possibly clamped) rate. If
        # the rate clamp fired it means the well is effectively shut; reflect
        # that in BHP too.
        if clamped_rate is not None and clamped_rate == 0.0:
            final_bhp = pressure
        else:
            clamped_bhp = _apply_clamp(
                pressure=pressure,
                control_type="constant rate control",
                clamp=self.clamp,
                bhp=required_bhp,
            )
            final_bhp = clamped_bhp if clamped_bhp is not None else required_bhp

        return ControlResult(rate=final_rate, bhp=final_bhp)

    def update(
        self,
        target_rate: typing.Optional[float] = None,
        bhp_limit: typing.Optional[float] = None,
        clamp: typing.Optional[RateClamp] = None,
    ) -> "RateControl[WellFluidTcon]":
        """
        Create a new `RateControl` with updated parameters.

        :param target_rate: New target flow rate. If None, retains existing.
        :param bhp_limit: New minimum BHP. If None, retains existing.
        :param clamp: New clamp condition. If None, retains existing.
        :return: New `RateControl` instance with updated parameters.
        """
        return type(self)(
            target_rate=target_rate or self.target_rate,
            target_phase=self.target_phase,
            bhp_limit=(bhp_limit or self.bhp_limit),
            clamp=clamp or self.clamp,
        )

    def __str__(self) -> str:
        """String representation."""
        if self.bhp_limit:
            if self.target_rate < 0:
                bhp_str = f"\nMin BHP={self.bhp_limit:.3e}psi"
            else:
                bhp_str = f"\nMax BHP={self.bhp_limit:.3e}psi"
        else:
            bhp_str = ""
        return f"Constant Rate Control: Rate={self.target_rate:.3e}{bhp_str}"


@well_control
@attrs.frozen
class AdaptiveRateControl(WellControl[WellFluidTcon]):
    """
    Adaptive control that switches between rate and BHP control.

    Operates at constant rate until BHP limit is reached, then switches
    to BHP control. This prevents excessive pressure drawdown while maintaining
    target production/injection when feasible.
    """

    __type__ = "adaptive_bhp_rate_control"

    target_rate: float
    """
    Target flow rate (STB/day or SCF/day). Positive for injection, negative for production.
    """
    bhp_limit: float
    """
    Minimum allowable BHP for production wells, and maximum allowable BHP for injection wells.

    Control switches from rate to BHP control when this limit is reached.
    """
    target_phase: typing.Optional[typing.Union[str, FluidPhase]] = None
    """Target fluid phase for the control."""
    clamp: typing.Optional[RateClamp] = None
    """Condition for clamping flow rates to zero. None means no clamping."""

    def __attrs_post_init__(self) -> None:
        """Validate control parameters."""
        if self.target_rate == 0.0:
            raise ValidationError(
                "Target rate cannot be zero. Use `well.shut_in` instead."
            )
        if self.bhp_limit <= 0.0:
            raise ValidationError("Minimum bottom hole pressure must be positive.")

        if self.target_phase is not None:
            object.__setattr__(self, "target_phase", FluidPhase(self.target_phase))

    def get_type(self) -> WellControlType:
        # This is a rate first control although it uses bhp when `bhp_limit` kicks in.
        # Hence the name adaptive "rate" control
        return "rate"

    def get_flow_rate(
        self,
        pressure: float,
        temperature: float,
        well_index: float,
        fluid: WellFluidTcon,
        formation_volume_factor: float,
        phase_mobility: typing.Optional[float] = None,
        allocation_fraction: float = 1.0,
        is_active: bool = True,
        use_pseudo_pressure: bool = False,
        fluid_compressibility: typing.Optional[float] = None,
        pvt_tables: typing.Optional[PVTTables] = None,
        **kwargs: typing.Any,
    ) -> float:
        """
        Compute flow rate adaptively.

        Uses rate control if achievable within BHP constraint,
        otherwise switches to BHP control at `bhp_limit`.

        :param pressure: Reservoir pressure at the well location (psi).
        :param temperature: Reservoir temperature at the well location (°F).
        :param phase_mobility: Relative mobility of the fluid phase.
        :param well_index: Well index (md*ft).
        :param fluid: Fluid being produced or injected.
        :param formation_volume_factor: Formation volume factor (bbl/STB or ft³/SCF).
        :param allocation_fraction: Fraction of target rate to allocate (applies in rate mode only).
        :param is_active: Whether the well is active/open.
        :param use_pseudo_pressure: Whether to use pseudo-pressure for gas wells.
        :param fluid_compressibility: Compressibility of the fluid (psi⁻¹).
        :param pvt_tables: `PVTTables` object for fluid property lookups
        :return: Flow rate in (bbl/day or ft³/day).
        """
        if _disallow_flow(fluid=fluid, is_active=is_active) or (
            self.target_phase is not None and fluid.phase != self.target_phase
        ):
            return 0.0

        # Apply allocation to target rate (for rate mode) and convert to reservoir rate
        target_rate = self.target_rate * allocation_fraction * formation_volume_factor
        is_production = target_rate < 0.0  # Negative rate indicates production
        bhp_limit = self.bhp_limit
        incompressibility_threshold = c.FLUID_INCOMPRESSIBILITY_THRESHOLD

        # If no mobility provided, skip BHP feasibility check, return rate directly.
        if phase_mobility is None:
            clamped = _apply_clamp(
                rate=target_rate,
                clamp=self.clamp,
                pressure=pressure,
                control_type="adaptive control - strict rate mode",
            )
            return clamped if clamped is not None else target_rate

        # Compute required BHP to achieve target rate
        try:
            required_bhp = _compute_required_bhp(
                target_rate=target_rate,
                fluid=fluid,
                well_index=well_index,
                pressure=pressure,
                temperature=temperature,
                phase_mobility=phase_mobility,
                use_pseudo_pressure=use_pseudo_pressure,
                fluid_compressibility=fluid_compressibility,
                formation_volume_factor=formation_volume_factor,
                incompressibility_threshold=incompressibility_threshold,
                pvt_tables=pvt_tables,
            )
        except (ValueError, ZeroDivisionError, ComputationError) as exc:
            logger.warning(
                f"Failed to compute required BHP for adaptive control: {exc}. "
                "Switching to BHP mode.",
                exc_info=True,
            )
        else:
            logger.debug(
                f"Required BHP: {required_bhp:.6f} psi, Reservoir pressure: {pressure:.6f} psi, Fluid phase: {fluid.phase}"
            )
            if is_production:
                can_achieve_rate = required_bhp >= bhp_limit
            else:
                can_achieve_rate = required_bhp <= bhp_limit

            if can_achieve_rate:
                # Can achieve target rate without violating minimum bottom hole pressure
                clamped = _apply_clamp(
                    rate=target_rate,
                    clamp=self.clamp,
                    pressure=pressure,
                    control_type="adaptive control - rate mode",
                )
                if clamped is not None:
                    return clamped

                logger.debug(
                    f"Using rate control at {target_rate:.6f} "
                    f"(required BHP: {required_bhp:.3f} psi > minimum: {bhp_limit:.3f} psi)"
                )
                return target_rate

        # Target rate would violate minimum bottom hole pressure, switch to BHP control
        logger.debug(
            f"Switching to BHP control at {bhp_limit:.3f} psi "
            f"(target rate not achievable within pressure constraints)"
        )

        # Compute rate at minimum bottom hole pressure using same logic as BHP control
        if fluid.phase == FluidPhase.GAS:
            use_pp, pp_table = get_pseudo_pressure_table(
                fluid=fluid,
                pressure=pressure,
                temperature=temperature,
                use_pseudo_pressure=use_pseudo_pressure,
                pvt_tables=pvt_tables,
            )
            # Compute Z-factor using reservoir pressure as initial estimate
            specific_gravity = typing.cast(
                float,
                fluid.get_specific_gravity(pressure=pressure, temperature=temperature),
            )
            if specific_gravity is None:
                raise ValidationError(
                    "Well fluid has no specific gravity define. Specify a value or provide a PVT table for the fluid."
                )
            avg_z_factor = compute_average_compressibility_factor(
                pressure=pressure,
                temperature=temperature,
                gas_gravity=specific_gravity,
                bottom_hole_pressure=bhp_limit,
            )
            rate = compute_gas_well_rate(
                well_index=well_index,
                pressure=pressure,
                temperature=temperature,
                bottom_hole_pressure=bhp_limit,
                phase_mobility=phase_mobility,
                use_pseudo_pressure=use_pp,
                pseudo_pressure_table=pp_table,
                average_compressibility_factor=avg_z_factor,
                formation_volume_factor=formation_volume_factor,
            )
        else:
            # For water and oil wells
            rate = compute_oil_well_rate(
                well_index=well_index,
                pressure=pressure,
                bottom_hole_pressure=bhp_limit,
                phase_mobility=phase_mobility,
                fluid_compressibility=fluid_compressibility,
                incompressibility_threshold=incompressibility_threshold,
            )

        clamped = _apply_clamp(
            rate=rate,
            clamp=self.clamp,
            pressure=pressure,
            control_type="adaptive control - BHP mode",
        )
        return clamped if clamped is not None else rate

    def get_bottom_hole_pressure(
        self,
        pressure: float,
        temperature: float,
        well_index: float,
        fluid: WellFluid,
        formation_volume_factor: float,
        phase_mobility: typing.Optional[float] = None,
        allocation_fraction: float = 1.0,
        is_active: bool = True,
        use_pseudo_pressure: bool = False,
        fluid_compressibility: typing.Optional[float] = None,
        pvt_tables: typing.Optional[PVTTables] = None,
        **kwargs: typing.Any,
    ) -> float:
        """
        Compute BHP based on current operating mode.

        :param pressure: Reservoir pressure at well location (psi)
        :param temperature: Reservoir temperature (°F)
        :param phase_mobility: Phase mobility (1/cP)
        :param well_index: Well index (md*ft)
        :param fluid: Fluid being produced/injected
        :param formation_volume_factor: Formation volume factor
        :param allocation_fraction: Fraction of target rate to allocate (applies in rate mode only).
        :param is_active: Whether well is active
        :param use_pseudo_pressure: Whether to use pseudo-pressure for gas
        :param fluid_compressibility: Fluid compressibility (1/psi)
        :param pvt_tables: `PVTTables` object for fluid property lookups
        :return: Effective bottom-hole pressure (psi)
        """
        if phase_mobility is None:
            raise ValidationError(
                "Phase mobility is required to get the effective bottom hole pressure (BHP) for the control."
            )

        if _disallow_flow(fluid=fluid, is_active=is_active) or (
            self.target_phase is not None and fluid.phase != self.target_phase
        ):
            return pressure

        # Apply allocation to target rate (for rate mode)
        target_rate_reservoir = (
            self.target_rate * allocation_fraction * formation_volume_factor
        )
        bhp_limit = self.bhp_limit

        # Try to compute required BHP for target rate
        try:
            required_bhp = _compute_required_bhp(
                target_rate=target_rate_reservoir,
                fluid=fluid,
                well_index=well_index,
                pressure=pressure,
                temperature=temperature,
                phase_mobility=phase_mobility,
                use_pseudo_pressure=use_pseudo_pressure,
                formation_volume_factor=formation_volume_factor,
                fluid_compressibility=fluid_compressibility,
                incompressibility_threshold=c.FLUID_INCOMPRESSIBILITY_THRESHOLD,
                pvt_tables=pvt_tables,
            )
        except (ValueError, ZeroDivisionError, ComputationError) as exc:
            logger.debug(f"Cannot achieve rate mode: {exc}. Using BHP mode.")
            return (
                _apply_clamp(
                    pressure=pressure,
                    control_type="adaptive control - BHP mode",
                    clamp=self.clamp,
                    bhp=bhp_limit,
                )
                or bhp_limit
            )

        # Check if rate is achievable within BHP constraint
        is_production = target_rate_reservoir < 0.0
        if is_production:
            can_achieve = required_bhp >= bhp_limit
        else:
            can_achieve = required_bhp <= bhp_limit

        if can_achieve:
            logger.debug(f"Adaptive control: rate mode (BHP={required_bhp:.4f})")
            bhp = required_bhp
        else:
            logger.debug(f"Adaptive control: BHP mode (BHP={bhp_limit:.4f})")
            bhp = bhp_limit
        return (
            _apply_clamp(
                pressure=pressure,
                control_type="adaptive control",
                clamp=self.clamp,
                bhp=bhp,
            )
            or bhp
        )

    def get_control(
        self,
        pressure: float,
        temperature: float,
        well_index: float,
        fluid: WellFluidTcon,
        formation_volume_factor: float,
        phase_mobility: typing.Optional[float] = None,
        allocation_fraction: float = 1.0,
        is_active: bool = True,
        use_pseudo_pressure: bool = False,
        fluid_compressibility: typing.Optional[float] = None,
        pvt_tables: typing.Optional[PVTTables] = None,
        **kwargs: typing.Any,
    ) -> ControlResult:
        """
        Compute rate and effective BHP in a single pass for adaptive rate control.

        `AdaptiveRateControl` has two operating modes that are selected at
        runtime based on whether the target rate is feasible within the BHP
        constraint. Determining the operating mode requires calling
        `_compute_required_bhp`, the same solve that `get_flow_rate` and
        `get_bottom_hole_pressure` each perform independently. This method
        calls that solve exactly once and derives both outputs from the result,
        removing the duplication.

        **Mode selection (with `phase_mobility` provided):**

        1. *Rate mode* - `_compute_required_bhp` succeeds and the required BHP
           satisfies `bhp_limit`: the allocated target rate is returned together
           with the required BHP (both subject to `clamp`).

        2. *BHP mode* - the required BHP violates `bhp_limit`, or the BHP solve
           throws: the well falls back to operating at `bhp_limit`. The rate
           is then computed via Darcy's law at that BHP (gas or liquid path,
           mirroring `get_flow_rate`'s BHP-mode branch). Pseudo-pressure table
           and Z-factor are built once and shared between rate and BHP.

        **Strict rate mode** (`phase_mobility is None`): the allocated target
        rate is returned directly (no feasibility check). BHP is returned as
        reservoir pressure (zero drawdown sentinel) because no mobility is
        available for a back-solve.

        In all cases the `clamp` is applied consistently to both rate and
        BHP before returning, so the two are always mutually consistent.

        :param pressure: Reservoir pressure at the well location (psi).
        :param temperature: Reservoir temperature at the well location (°F).
        :param phase_mobility: Relative mobility of the fluid phase (cP⁻¹).
            Pass `None` for strict rate mode (no BHP feasibility check).
        :param well_index: Well index (md·ft).
        :param fluid: Fluid being produced or injected.
        :param formation_volume_factor: Formation volume factor (bbl/STB or ft³/SCF).
        :param allocation_fraction: Fraction of total well rate allocated to this
            cell (= cell_WI / total_WI for multi-cell wells). Default 1.0.
        :param is_active: Whether the well is currently open.
        :param use_pseudo_pressure: Whether to use pseudo-pressure for gas wells.
        :param fluid_compressibility: Fluid compressibility (psi⁻¹).
        :param pvt_tables: PVT look-up tables for fluid properties.
        :return: `ControlResult` with the flow rate and effective BHP.
            BHP equals `bhp_limit` when operating in BHP mode, or the
            required BHP when in rate mode. Reservoir pressure is used as the
            BHP sentinel in strict rate mode.
        """
        if _disallow_flow(fluid=fluid, is_active=is_active) or (
            self.target_phase is not None and fluid.phase != self.target_phase
        ):
            return ControlResult(rate=0.0, bhp=pressure)

        target_rate = self.target_rate * allocation_fraction * formation_volume_factor
        is_production = target_rate < 0.0
        bhp_limit = self.bhp_limit
        incompressibility_threshold = c.FLUID_INCOMPRESSIBILITY_THRESHOLD

        # Strict rate mode - no mobility, skip feasibility check.
        if phase_mobility is None:
            clamped = _apply_clamp(
                rate=target_rate,
                clamp=self.clamp,
                pressure=pressure,
                control_type="adaptive control - strict rate mode",
            )
            final_rate = clamped if clamped is not None else target_rate
            # No mobility → cannot back-solve for BHP; use reservoir pressure.
            return ControlResult(rate=final_rate, bhp=pressure)

        # Attempt to solve for the BHP required to deliver the target rate.
        bhp_solve_failed = False
        # Default. Will be overwritten or used as fallback
        required_bhp: float = bhp_limit

        try:
            required_bhp = _compute_required_bhp(
                target_rate=target_rate,
                fluid=fluid,
                well_index=well_index,
                pressure=pressure,
                temperature=temperature,
                phase_mobility=phase_mobility,
                use_pseudo_pressure=use_pseudo_pressure,
                fluid_compressibility=fluid_compressibility,
                formation_volume_factor=formation_volume_factor,
                incompressibility_threshold=incompressibility_threshold,
                pvt_tables=pvt_tables,
            )
        except (ValueError, ZeroDivisionError, ComputationError) as exc:
            logger.warning(
                f"Failed to compute required BHP for adaptive control: {exc}. "
                "Switching to BHP mode.",
                exc_info=True,
            )
            bhp_solve_failed = True

        # Determine operating mode from the solve result.
        in_rate_mode: bool
        if bhp_solve_failed:
            in_rate_mode = False
        else:
            logger.debug(
                f"Required BHP: {required_bhp:.6f} psi, Reservoir pressure: {pressure:.6f} psi, "
                f"Fluid phase: {fluid.phase}"
            )
            if is_production:
                in_rate_mode = required_bhp >= bhp_limit
            else:
                in_rate_mode = required_bhp <= bhp_limit

        # Rate mode: target rate is achievable.
        if in_rate_mode:
            clamped_rate = _apply_clamp(
                rate=target_rate,
                clamp=self.clamp,
                pressure=pressure,
                control_type="adaptive control - rate mode",
            )
            final_rate = clamped_rate if clamped_rate is not None else target_rate

            # If clamp zeroed the rate the effective BHP collapses to reservoir pressure.
            if clamped_rate is not None and clamped_rate == 0.0:
                final_bhp = pressure
            else:
                clamped_bhp = _apply_clamp(
                    pressure=pressure,
                    control_type="adaptive control - rate mode",
                    clamp=self.clamp,
                    bhp=required_bhp,
                )
                final_bhp = clamped_bhp if clamped_bhp is not None else required_bhp

            logger.debug(
                f"Adaptive control - rate mode: rate={final_rate:.6f}, BHP={final_bhp:.4f} psi"
            )
            return ControlResult(rate=final_rate, bhp=final_bhp)

        # BHP mode: operate at bhp_limit and compute the resulting Darcy rate.
        logger.debug(
            f"Adaptive control - BHP mode at {bhp_limit:.3f} psi "
            f"(target rate not achievable within pressure constraints)"
        )

        if fluid.phase == FluidPhase.GAS:
            use_pp, pp_table = get_pseudo_pressure_table(
                fluid=fluid,
                pressure=pressure,
                temperature=temperature,
                use_pseudo_pressure=use_pseudo_pressure,
                pvt_tables=pvt_tables,
            )
            specific_gravity = typing.cast(
                float,
                fluid.get_specific_gravity(pressure=pressure, temperature=temperature),
            )
            if specific_gravity is None:
                raise ValidationError(
                    "Well fluid has no specific gravity defined. "
                    "Specify a value or provide a PVT table for the fluid."
                )
            # Z-factor averaged between reservoir pressure and bhp_limit - single compute.
            avg_z_factor = compute_average_compressibility_factor(
                pressure=pressure,
                temperature=temperature,
                gas_gravity=specific_gravity,
                bottom_hole_pressure=bhp_limit,
            )
            rate = compute_gas_well_rate(
                well_index=well_index,
                pressure=pressure,
                temperature=temperature,
                bottom_hole_pressure=bhp_limit,
                phase_mobility=phase_mobility,
                use_pseudo_pressure=use_pp,
                pseudo_pressure_table=pp_table,
                average_compressibility_factor=avg_z_factor,
                formation_volume_factor=formation_volume_factor,
            )
        else:
            rate = compute_oil_well_rate(
                well_index=well_index,
                pressure=pressure,
                bottom_hole_pressure=bhp_limit,
                phase_mobility=phase_mobility,
                fluid_compressibility=fluid_compressibility,
                incompressibility_threshold=incompressibility_threshold,
            )

        clamped_rate = _apply_clamp(
            rate=rate,
            clamp=self.clamp,
            pressure=pressure,
            control_type="adaptive control - BHP mode",
        )
        final_rate = clamped_rate if clamped_rate is not None else rate

        if clamped_rate is not None and clamped_rate == 0.0:
            final_bhp = pressure
        else:
            clamped_bhp = _apply_clamp(
                pressure=pressure,
                control_type="adaptive control - BHP mode",
                clamp=self.clamp,
                bhp=bhp_limit,
            )
            final_bhp = clamped_bhp if clamped_bhp is not None else bhp_limit

        return ControlResult(rate=final_rate, bhp=final_bhp)

    def update(
        self,
        target_rate: typing.Optional[float] = None,
        bhp_limit: typing.Optional[float] = None,
        clamp: typing.Optional[RateClamp] = None,
    ) -> "AdaptiveRateControl[WellFluidTcon]":
        """
        Create a new `AdaptiveRateControl` with updated parameters.

        :param target_rate: New target flow rate. If None, retains existing.
        :param bhp_limit: New minimum BHP. If None, retains existing.
        :param clamp: New clamp condition. If None, retains existing.
        :return: New `AdaptiveRateControl` instance with updated parameters.
        """
        return type(self)(
            target_rate=target_rate or self.target_rate,
            target_phase=self.target_phase,
            bhp_limit=(bhp_limit or self.bhp_limit),
            clamp=clamp or self.clamp,
        )

    def __str__(self) -> str:
        if self.bhp_limit:
            if self.target_rate < 0:
                bhp_str = f"\nMin BHP={self.bhp_limit:.3e}psi"
            else:
                bhp_str = f"\nMax BHP={self.bhp_limit:.3e}psi"
        else:
            bhp_str = ""
        return f"Adaptive Rate Control:\nRate={self.target_rate:.3e}{bhp_str}"


@well_control
@attrs.frozen
class CoupledRateControl(WellControl[WellFluidTcon]):
    """
    Well control that fixes one phase's rate and lets other phases flow at the resulting BHP.

    Standard approach in reservoir simulation for production wells: specify an oil (or gas/water)
    target rate, and the simulator determines the BHP required to deliver that rate. Water and gas
    then produce at whatever their natural Darcy rates are at that BHP.

    Phases are coupled through a shared BHP

    NOTE: This rate control is to be used for **production wells only**.

    Example:
    ```python
    control = CoupledRateControl(
        primary_phase=FluidPhase.OIL,
        primary_control=AdaptiveRateControl(
            target_rate=-500, target_phase="oil", bhp_limit=1500,
        ),
        secondary_clamp=ProductionClamp(),
    )
    ```

    :param primary_phase: The phase whose rate is fixed (determines BHP).
    :param primary_control: Rate or adaptive control applied to the primary phase.
    :param secondary_clamp: Optional clamp on secondary phase rates (e.g. prevent backflow).
    """

    __type__ = "primary_phase_rate_control"

    primary_phase: typing.Union[FluidPhase, str] = attrs.field(converter=FluidPhase)
    """Phase whose rate is fixed (determines BHP)."""

    primary_control: typing.Union[RateControl, AdaptiveRateControl]
    """Rate control applied to the primary phase."""

    secondary_clamp: typing.Optional[RateClamp] = None
    """Optional clamp on secondary (non-primary) phase rates."""

    def __attrs_post_init__(self) -> None:
        object.__setattr__(self, "primary_phase", FluidPhase(self.primary_phase))

    def get_type(self) -> WellControlType:
        return "rate"

    def _compute_primary_bhp(
        self,
        pressure: float,
        temperature: float,
        primary_phase_mobility: typing.Optional[float],
        well_index: float,
        primary_fluid: WellFluid,
        primary_formation_volume_factor: float,
        allocation_fraction: float,
        use_pseudo_pressure: bool,
        primary_fluid_compressibility: typing.Optional[float],
        pvt_tables: typing.Optional[PVTTables],
    ) -> float:
        """Compute the BHP established by the primary phase's rate control."""
        return self.primary_control.get_bottom_hole_pressure(
            pressure=pressure,
            temperature=temperature,
            phase_mobility=primary_phase_mobility,
            well_index=well_index,
            fluid=primary_fluid,
            formation_volume_factor=primary_formation_volume_factor,
            allocation_fraction=allocation_fraction,
            is_active=True,
            use_pseudo_pressure=use_pseudo_pressure,
            fluid_compressibility=primary_fluid_compressibility,
            pvt_tables=pvt_tables,
        )

    def get_bottom_hole_pressure(
        self,
        pressure: float,
        temperature: float,
        well_index: float,
        fluid: WellFluid,
        formation_volume_factor: float,
        phase_mobility: typing.Optional[float] = None,
        allocation_fraction: float = 1.0,
        is_active: bool = True,
        use_pseudo_pressure: bool = False,
        fluid_compressibility: typing.Optional[float] = None,
        pvt_tables: typing.Optional[PVTTables] = None,
        primary_phase_mobility: typing.Optional[float] = None,
        primary_fluid: typing.Optional[WellFluid] = None,
        primary_formation_volume_factor: typing.Optional[float] = None,
        primary_fluid_compressibility: typing.Optional[float] = None,
        **kwargs: typing.Any,
    ) -> float:
        """
        Compute BHP for semi-implicit pressure equation coupling.

        For the primary phase, BHP is derived from the rate control using its own properties.
        For secondary phases, BHP is derived using the primary phase's properties so that all
        phases share a consistent drawdown.

        :param primary_phase_mobility: Mobility of primary phase (required for secondary phases).
        :param primary_fluid: Primary phase fluid object (required for secondary phases).
        :param primary_formation_volume_factor: FVF of primary phase (required for secondary phases).
        :param primary_fluid_compressibility: Compressibility of primary phase (required for secondary phases).
        """
        if not is_active:
            return pressure

        if fluid.phase == self.primary_phase:
            return self._compute_primary_bhp(
                pressure=pressure,
                temperature=temperature,
                primary_phase_mobility=phase_mobility,
                well_index=well_index,
                primary_fluid=fluid,
                primary_formation_volume_factor=formation_volume_factor,
                allocation_fraction=allocation_fraction,
                use_pseudo_pressure=use_pseudo_pressure,
                primary_fluid_compressibility=fluid_compressibility,
                pvt_tables=pvt_tables,
            )

        if (
            primary_phase_mobility is None
            or primary_fluid is None
            or primary_formation_volume_factor is None
        ):
            logger.warning(
                f"Cannot compute BHP for secondary phase {fluid.phase!s} - "
                f"primary phase properties not provided. Using cell pressure."
            )
            return pressure

        return self._compute_primary_bhp(
            pressure=pressure,
            temperature=temperature,
            primary_phase_mobility=primary_phase_mobility,
            well_index=well_index,
            primary_fluid=primary_fluid,
            primary_formation_volume_factor=primary_formation_volume_factor,
            allocation_fraction=allocation_fraction,
            use_pseudo_pressure=use_pseudo_pressure,
            primary_fluid_compressibility=primary_fluid_compressibility,
            pvt_tables=pvt_tables,
        )

    def get_flow_rate(
        self,
        pressure: float,
        temperature: float,
        well_index: float,
        fluid: WellFluidTcon,
        formation_volume_factor: float,
        phase_mobility: typing.Optional[float] = None,
        allocation_fraction: float = 1.0,
        is_active: bool = True,
        use_pseudo_pressure: bool = False,
        fluid_compressibility: typing.Optional[float] = None,
        pvt_tables: typing.Optional[PVTTables] = None,
        primary_phase_mobility: typing.Optional[float] = None,
        primary_fluid: typing.Optional[WellFluid] = None,
        primary_formation_volume_factor: typing.Optional[float] = None,
        primary_fluid_compressibility: typing.Optional[float] = None,
        **kwargs: typing.Any,
    ) -> float:
        """
        Compute flow rate for a given phase.

        The primary phase rate comes directly from the rate control. Secondary phase rates
        are computed via Darcy's law at the BHP established by the primary phase, using
        each secondary phase's own mobility and FVF.

        :param primary_phase_mobility: Mobility of primary phase (required for secondary phases).
        :param primary_fluid: Primary phase fluid object (required for secondary phases).
        :param primary_formation_volume_factor: FVF of primary phase (required for secondary phases).
        :param primary_fluid_compressibility: Compressibility of primary phase (required for secondary phases).
        """
        if fluid.phase == self.primary_phase:
            return self.primary_control.get_flow_rate(
                pressure=pressure,
                temperature=temperature,
                phase_mobility=phase_mobility,
                well_index=well_index,
                fluid=fluid,
                formation_volume_factor=formation_volume_factor,
                allocation_fraction=allocation_fraction,
                is_active=is_active,
                use_pseudo_pressure=use_pseudo_pressure,
                fluid_compressibility=fluid_compressibility,
                pvt_tables=pvt_tables,
            )

        if (
            primary_phase_mobility is None
            or primary_fluid is None
            or primary_formation_volume_factor is None
        ):
            logger.warning(
                f"Cannot compute flow rate for secondary phase {fluid.phase!s} - "
                f"primary phase properties not provided. Returning 0."
            )
            return 0.0

        if phase_mobility is None:
            raise ValidationError(
                "Phase mobility is required to get the effective bottom hole pressure (BHP) for the secondary phases."
            )

        if _disallow_flow(
            fluid=fluid,
            phase_mobility=phase_mobility,
            is_active=is_active,
        ):
            return 0.0

        bhp = self._compute_primary_bhp(
            pressure=pressure,
            temperature=temperature,
            primary_phase_mobility=primary_phase_mobility,
            well_index=well_index,
            primary_fluid=primary_fluid,
            primary_formation_volume_factor=primary_formation_volume_factor,
            allocation_fraction=allocation_fraction,
            use_pseudo_pressure=use_pseudo_pressure,
            primary_fluid_compressibility=primary_fluid_compressibility,
            pvt_tables=pvt_tables,
        )

        # Compute secondary phase rate at the primary-phase-derived BHP
        if fluid.phase == FluidPhase.GAS:
            use_pp, pp_table = get_pseudo_pressure_table(
                fluid=fluid,
                pressure=pressure,
                temperature=temperature,
                use_pseudo_pressure=use_pseudo_pressure,
                pvt_tables=pvt_tables,
            )
            # Compute Z-factor using reservoir pressure as initial estimate
            specific_gravity = typing.cast(
                float,
                fluid.get_specific_gravity(pressure=pressure, temperature=temperature),
            )
            if specific_gravity is None:
                raise ValidationError(
                    "Well fluid has no specific gravity define. Specify a value or provide a PVT table for the fluid."
                )
            avg_z_factor = compute_average_compressibility_factor(
                pressure=pressure,
                temperature=temperature,
                gas_gravity=specific_gravity,
                bottom_hole_pressure=bhp,
            )
            rate = compute_gas_well_rate(
                well_index=well_index,
                pressure=pressure,
                temperature=temperature,
                bottom_hole_pressure=bhp,
                phase_mobility=phase_mobility,
                use_pseudo_pressure=use_pp,
                pseudo_pressure_table=pp_table,
                average_compressibility_factor=avg_z_factor,
                formation_volume_factor=formation_volume_factor,
            )
        else:
            rate = compute_oil_well_rate(
                well_index=well_index,
                pressure=pressure,
                bottom_hole_pressure=bhp,
                phase_mobility=phase_mobility,
                fluid_compressibility=fluid_compressibility,
                incompressibility_threshold=c.FLUID_INCOMPRESSIBILITY_THRESHOLD,
            )

        clamped = _apply_clamp(
            rate=rate,
            clamp=self.secondary_clamp,
            pressure=pressure,
            control_type="primary phase rate control (secondary)",
        )
        return clamped if clamped is not None else rate

    def get_control(
        self,
        pressure: float,
        temperature: float,
        well_index: float,
        fluid: WellFluidTcon,
        formation_volume_factor: float,
        phase_mobility: typing.Optional[float] = None,
        allocation_fraction: float = 1.0,
        is_active: bool = True,
        use_pseudo_pressure: bool = False,
        fluid_compressibility: typing.Optional[float] = None,
        pvt_tables: typing.Optional[PVTTables] = None,
        primary_phase_mobility: typing.Optional[float] = None,
        primary_fluid: typing.Optional[WellFluid] = None,
        primary_formation_volume_factor: typing.Optional[float] = None,
        primary_fluid_compressibility: typing.Optional[float] = None,
        **kwargs: typing.Any,
    ) -> ControlResult:
        """
        Compute rate and effective BHP simultaneously for coupled rate control.

        `CoupledRateControl` derives a shared BHP from the primary phase's
        rate control and then computes each secondary phase's Darcy rate at that
        BHP. Both `get_flow_rate` and `get_bottom_hole_pressure` ultimately
        call `_compute_primary_bhp`, which in turn calls
        `primary_control.get_bottom_hole_pressure` (itself containing a
        `_compute_required_bhp` solve for rate controls). This method makes
        that call once and reuses the result for both outputs.

        **Primary phase:** `get_control` is delegated to
        `primary_control.get_control`, which computes the primary rate and BHP
        in one pass. The BHP from that result becomes the shared coupling BHP.

        **Secondary phases:** the shared BHP (derived from the primary phase) is
        used directly. The secondary phase's Darcy rate is computed at that BHP
        using the secondary phase's own mobility and FVF (gas or liquid path).
        Pseudo-pressure table and Z-factor (for gas secondaries) are built once
        and used for both the rate and the returned BHP.

        **Missing primary context:** if the caller does not supply
        `primary_phase_mobility`, `primary_fluid`, or
        `primary_formation_volume_factor` for a secondary phase, both rate and
        BHP fall back gracefully (`rate=0`, `bhp=pressure`) with a warning,
        matching the behaviour of the individual methods.

        :param pressure: Reservoir pressure at the well location (psi).
        :param temperature: Reservoir temperature at the well location (°F).
        :param phase_mobility: Mobility of the phase being evaluated (cP⁻¹).
            Required for secondary phases.
        :param well_index: Well index (md·ft).
        :param fluid: Fluid being produced or injected.
        :param formation_volume_factor: FVF of `fluid` (bbl/STB or ft³/SCF).
        :param allocation_fraction: Fraction of total well rate allocated to this
            cell (applies to primary phase rate control). Default 1.0.
        :param is_active: Whether the well is currently open.
        :param use_pseudo_pressure: Whether to use pseudo-pressure for gas wells.
        :param fluid_compressibility: Compressibility of `fluid` (psi⁻¹).
        :param pvt_tables: PVT look-up tables for fluid properties.
        :param primary_phase_mobility: Mobility of the primary phase (cP⁻¹).
            Required when evaluating a secondary phase.
        :param primary_fluid: Fluid object for the primary phase.
            Required when evaluating a secondary phase.
        :param primary_formation_volume_factor: FVF of the primary phase.
            Required when evaluating a secondary phase.
        :param primary_fluid_compressibility: Compressibility of the primary phase
            (psi⁻¹). Used when the primary is a liquid phase.
        :return: `ControlResult` with the flow rate and effective BHP for
            `fluid`. For the primary phase the BHP is what its rate control
            requires; for secondary phases it is the same shared BHP.
        """
        if not is_active:
            return ControlResult(rate=0.0, bhp=pressure)

        # Primary phase: delegate entirely to primary_control.get_control so
        # that the BHP solve and rate computation share intermediates there.
        if fluid.phase == self.primary_phase:
            return self.primary_control.get_control(
                pressure=pressure,
                temperature=temperature,
                well_index=well_index,
                fluid=fluid,
                formation_volume_factor=formation_volume_factor,
                phase_mobility=phase_mobility,
                allocation_fraction=allocation_fraction,
                is_active=is_active,
                use_pseudo_pressure=use_pseudo_pressure,
                fluid_compressibility=fluid_compressibility,
                pvt_tables=pvt_tables,
            )

        # Secondary phase needs primary context to derive shared BHP.
        if (
            primary_phase_mobility is None
            or primary_fluid is None
            or primary_formation_volume_factor is None
        ):
            logger.warning(
                f"Cannot compute control result for secondary phase {fluid.phase!s} - "
                f"primary phase properties not provided. Returning zero rate / cell pressure."
            )
            return ControlResult(rate=0.0, bhp=pressure)

        if phase_mobility is None:
            raise ValidationError(
                "Phase mobility is required for secondary phase flow rate computation "
                "in CoupledRateControl."
            )

        if _disallow_flow(
            fluid=fluid, phase_mobility=phase_mobility, is_active=is_active
        ):
            return ControlResult(rate=0.0, bhp=pressure)

        # Compute the shared BHP from the primary phase
        shared_bhp = self._compute_primary_bhp(
            pressure=pressure,
            temperature=temperature,
            primary_phase_mobility=primary_phase_mobility,
            well_index=well_index,
            primary_fluid=primary_fluid,
            primary_formation_volume_factor=primary_formation_volume_factor,
            allocation_fraction=allocation_fraction,
            use_pseudo_pressure=use_pseudo_pressure,
            primary_fluid_compressibility=primary_fluid_compressibility,
            pvt_tables=pvt_tables,
        )

        # Compute secondary Darcy rate at shared_bhp
        if fluid.phase == FluidPhase.GAS:
            use_pp, pp_table = get_pseudo_pressure_table(
                fluid=fluid,
                pressure=pressure,
                temperature=temperature,
                use_pseudo_pressure=use_pseudo_pressure,
                pvt_tables=pvt_tables,
            )
            specific_gravity = typing.cast(
                float,
                fluid.get_specific_gravity(pressure=pressure, temperature=temperature),
            )
            if specific_gravity is None:
                raise ValidationError(
                    "Well fluid has no specific gravity defined. "
                    "Specify a value or provide a PVT table for the fluid."
                )
            avg_z_factor = compute_average_compressibility_factor(
                pressure=pressure,
                temperature=temperature,
                gas_gravity=specific_gravity,
                bottom_hole_pressure=shared_bhp,
            )
            rate = compute_gas_well_rate(
                well_index=well_index,
                pressure=pressure,
                temperature=temperature,
                bottom_hole_pressure=shared_bhp,
                phase_mobility=phase_mobility,
                use_pseudo_pressure=use_pp,
                pseudo_pressure_table=pp_table,
                average_compressibility_factor=avg_z_factor,
                formation_volume_factor=formation_volume_factor,
            )
        else:
            rate = compute_oil_well_rate(
                well_index=well_index,
                pressure=pressure,
                bottom_hole_pressure=shared_bhp,
                phase_mobility=phase_mobility,
                fluid_compressibility=fluid_compressibility,
                incompressibility_threshold=c.FLUID_INCOMPRESSIBILITY_THRESHOLD,
            )

        clamped_rate = _apply_clamp(
            rate=rate,
            clamp=self.secondary_clamp,
            pressure=pressure,
            control_type="coupled rate control (secondary)",
        )
        final_rate = clamped_rate if clamped_rate is not None else rate

        return ControlResult(rate=final_rate, bhp=shared_bhp)

    def build_primary_phase_context(
        self,
        produced_fluids: typing.Sequence[WellFluid],
        oil_mobility: float,
        water_mobility: float,
        gas_mobility: float,
        oil_fvf: float,
        water_fvf: float,
        gas_fvf: float,
        oil_compressibility: float,
        water_compressibility: float,
        gas_compressibility: float,
    ) -> dict[str, typing.Any]:
        """
        Build kwargs for the primary phase cell properties for passing to
        `get_flow_rate(...)` / `get_bottom_hole_pressure(...)`.

        Call once per cell before iterating over produced fluids. The returned dictionary
        can be unpacked as `**kwargs`.
        """
        primary_fluid = None
        for fluid in produced_fluids:
            if fluid.phase == self.primary_phase:
                primary_fluid = fluid
                break

        if primary_fluid is None:
            return {}

        phase_props = {
            FluidPhase.OIL: (oil_mobility, oil_fvf, oil_compressibility),
            FluidPhase.GAS: (gas_mobility, gas_fvf, gas_compressibility),
            FluidPhase.WATER: (water_mobility, water_fvf, water_compressibility),
        }
        mobility, fvf, compressibility = phase_props[FluidPhase(self.primary_phase)]
        return {
            "primary_phase_mobility": mobility,
            "primary_fluid": primary_fluid,
            "primary_formation_volume_factor": fvf,
            "primary_fluid_compressibility": compressibility,
        }

    def __str__(self) -> str:
        return f"Coupled Rate Control:\nPrimary phase: {self.primary_phase!s}\nControl:\n\t{self.primary_control!s}"


@well_control
@attrs.frozen
class MultiPhaseControl(WellControl):
    """
    Multi-phase well control for wells.

    Defines separate well controls for oil, gas, and water phases.
    """

    __type__ = "multi_phase_rate_control"

    oil_control: typing.Optional[WellControl] = None
    """Oil phase well control. Ensure that this is intended for oil phase."""
    gas_control: typing.Optional[WellControl] = None
    """Gas phase well control. Ensure that this is intended for gas phase."""
    water_control: typing.Optional[WellControl] = None
    """Water phase well control. Ensure that this is intended for water phase."""

    def get_type(self) -> WellControlType:
        return "custom"

    def get_phase_control_type(
        self, phase: FluidPhase
    ) -> typing.Optional[WellControlType]:
        """Return the control type for a specific phase, or None if no control for that phase."""
        if phase == FluidPhase.OIL and self.oil_control is not None:
            return self.oil_control.get_type()
        elif phase == FluidPhase.GAS and self.gas_control is not None:
            return self.gas_control.get_type()
        elif phase == FluidPhase.WATER and self.water_control is not None:
            return self.water_control.get_type()
        return None

    def get_flow_rate(
        self,
        pressure: float,
        temperature: float,
        well_index: float,
        fluid: WellFluid,
        formation_volume_factor: float,
        phase_mobility: typing.Optional[float] = None,
        allocation_fraction: float = 1.0,
        is_active: bool = True,
        use_pseudo_pressure: bool = False,
        fluid_compressibility: typing.Optional[float] = None,
        pvt_tables: typing.Optional[PVTTables] = None,
        **kwargs: typing.Any,
    ) -> float:
        """
        Compute flow rate based on fluid phase.

        :param pressure: Reservoir pressure at the well location (psi).
        :param temperature: Reservoir temperature at the well location (°F).
        :param phase_mobility: Relative mobility of the fluid phase.
        :param well_index: Well index (md*ft).
        :param fluid: Fluid being produced or injected.
        :param is_active: Whether the well is active/open.
        :param use_pseudo_pressure: Whether to use pseudo-pressure for gas wells.
        :param fluid_compressibility: Compressibility of the fluid (psi⁻¹).
        :param formation_volume_factor: Formation volume factor (bbl/STB or ft³/SCF).
        :param pvt_tables: `PVTTables` object for fluid property lookups
        :return: Flow rate in (bbl/day or ft³/day).
        """
        if fluid.phase == FluidPhase.OIL and self.oil_control is not None:
            return self.oil_control.get_flow_rate(
                pressure=pressure,
                temperature=temperature,
                phase_mobility=phase_mobility,
                well_index=well_index,
                fluid=fluid,
                formation_volume_factor=formation_volume_factor,
                allocation_fraction=allocation_fraction,
                is_active=is_active,
                use_pseudo_pressure=use_pseudo_pressure,
                fluid_compressibility=fluid_compressibility,
                pvt_tables=pvt_tables,
                **kwargs,
            )
        elif fluid.phase == FluidPhase.GAS and self.gas_control is not None:
            return self.gas_control.get_flow_rate(
                pressure=pressure,
                temperature=temperature,
                phase_mobility=phase_mobility,
                well_index=well_index,
                fluid=fluid,
                formation_volume_factor=formation_volume_factor,
                allocation_fraction=allocation_fraction,
                is_active=is_active,
                use_pseudo_pressure=use_pseudo_pressure,
                fluid_compressibility=fluid_compressibility,
                pvt_tables=pvt_tables,
                **kwargs,
            )
        elif fluid.phase == FluidPhase.WATER and self.water_control is not None:
            return self.water_control.get_flow_rate(
                pressure=pressure,
                temperature=temperature,
                phase_mobility=phase_mobility,
                well_index=well_index,
                fluid=fluid,
                formation_volume_factor=formation_volume_factor,
                allocation_fraction=allocation_fraction,
                is_active=is_active,
                use_pseudo_pressure=use_pseudo_pressure,
                fluid_compressibility=fluid_compressibility,
                pvt_tables=pvt_tables,
                **kwargs,
            )
        return 0.0

    def get_bottom_hole_pressure(
        self,
        pressure: float,
        temperature: float,
        well_index: float,
        fluid: WellFluid,
        formation_volume_factor: float,
        phase_mobility: typing.Optional[float] = None,
        allocation_fraction: float = 1.0,
        is_active: bool = True,
        use_pseudo_pressure: bool = False,
        fluid_compressibility: typing.Optional[float] = None,
        pvt_tables: typing.Optional[PVTTables] = None,
        **kwargs: typing.Any,
    ) -> float:
        """Delegate to appropriate phase control."""
        if fluid.phase == FluidPhase.OIL and self.oil_control is not None:
            return self.oil_control.get_bottom_hole_pressure(
                pressure=pressure,
                temperature=temperature,
                phase_mobility=phase_mobility,
                well_index=well_index,
                fluid=fluid,
                formation_volume_factor=formation_volume_factor,
                allocation_fraction=allocation_fraction,
                is_active=is_active,
                use_pseudo_pressure=use_pseudo_pressure,
                fluid_compressibility=fluid_compressibility,
                pvt_tables=pvt_tables,
                **kwargs,
            )
        elif fluid.phase == FluidPhase.GAS and self.gas_control is not None:
            return self.gas_control.get_bottom_hole_pressure(
                pressure=pressure,
                temperature=temperature,
                phase_mobility=phase_mobility,
                well_index=well_index,
                fluid=fluid,
                formation_volume_factor=formation_volume_factor,
                allocation_fraction=allocation_fraction,
                is_active=is_active,
                use_pseudo_pressure=use_pseudo_pressure,
                fluid_compressibility=fluid_compressibility,
                pvt_tables=pvt_tables,
                **kwargs,
            )
        elif fluid.phase == FluidPhase.WATER and self.water_control is not None:
            return self.water_control.get_bottom_hole_pressure(
                pressure=pressure,
                temperature=temperature,
                phase_mobility=phase_mobility,
                well_index=well_index,
                fluid=fluid,
                formation_volume_factor=formation_volume_factor,
                allocation_fraction=allocation_fraction,
                is_active=is_active,
                use_pseudo_pressure=use_pseudo_pressure,
                fluid_compressibility=fluid_compressibility,
                pvt_tables=pvt_tables,
                **kwargs,
            )
        return pressure

    def get_control(
        self,
        pressure: float,
        temperature: float,
        well_index: float,
        fluid: WellFluid,
        formation_volume_factor: float,
        phase_mobility: typing.Optional[float] = None,
        allocation_fraction: float = 1.0,
        is_active: bool = True,
        use_pseudo_pressure: bool = False,
        fluid_compressibility: typing.Optional[float] = None,
        pvt_tables: typing.Optional[PVTTables] = None,
        **kwargs: typing.Any,
    ) -> ControlResult:
        """
        Compute rate and effective BHP for the appropriate per-phase sub-control.

        If no sub-control is registered for `fluid.phase` the result is the
        no-flow / no-drawdown sentinel: `rate=0`, `bhp=pressure`.

        All `**kwargs` are forwarded to the sub-control's `get_control`
        unchanged, so phase-specific extras (e.g. primary-phase context for a
        `CoupledRateControl` sub-control) are transparently supported.

        :param pressure: Reservoir pressure at the well location (psi).
        :param temperature: Reservoir temperature at the well location (°F).
        :param phase_mobility: Relative mobility of the fluid phase (cP⁻¹).
        :param well_index: Well index (md·ft).
        :param fluid: Fluid being produced or injected.
        :param formation_volume_factor: FVF of `fluid` (bbl/STB or ft³/SCF).
        :param allocation_fraction: Fraction of total well rate allocated to this
            cell. Default 1.0.
        :param is_active: Whether the well is currently open.
        :param use_pseudo_pressure: Whether to use pseudo-pressure for gas wells.
        :param fluid_compressibility: Fluid compressibility (psi⁻¹).
        :param pvt_tables: PVT look-up tables for fluid properties.
        :param kwargs: Additional arguments forwarded to the sub-control's
            `get_control` method.
        :return: `ControlResult` from the matching sub-control, or
            `ControlResult(rate=0.0, bhp=pressure)` if no sub-control exists
            for this phase.
        """
        kwds = dict(  # noqa
            pressure=pressure,
            temperature=temperature,
            well_index=well_index,
            fluid=fluid,
            formation_volume_factor=formation_volume_factor,
            phase_mobility=phase_mobility,
            allocation_fraction=allocation_fraction,
            is_active=is_active,
            use_pseudo_pressure=use_pseudo_pressure,
            fluid_compressibility=fluid_compressibility,
            pvt_tables=pvt_tables,
            **kwargs,
        )

        if fluid.phase == FluidPhase.OIL and self.oil_control is not None:
            return self.oil_control.get_control(**kwds)  # type: ignore[arg-type]
        elif fluid.phase == FluidPhase.GAS and self.gas_control is not None:
            return self.gas_control.get_control(**kwds)  # type: ignore[arg-type]
        elif fluid.phase == FluidPhase.WATER and self.water_control is not None:
            return self.water_control.get_control(**kwds)  # type: ignore[arg-type]

        # No sub-control for this phase, hence no flow, no drawdown.
        return ControlResult(rate=0.0, bhp=pressure)

    def update(
        self,
        oil_control: typing.Optional[WellControl] = None,
        gas_control: typing.Optional[WellControl] = None,
        water_control: typing.Optional[WellControl] = None,
    ) -> "MultiPhaseControl":
        """
        Create a new `MultiPhaseControl` with updated controls.

        :param oil_control: New oil phase control. If None, retains existing.
        :param gas_control: New gas phase control. If None, retains existing.
        :param water_control: New water phase control. If None, retains existing.
        :return: New `MultiPhaseControl` instance with updated controls.
        """
        return type(self)(
            oil_control=oil_control or self.oil_control,
            gas_control=gas_control or self.gas_control,
            water_control=water_control or self.water_control,
        )

    def __str__(self) -> str:
        return f"Multi-Phase Rate Control:\nOil Control: {self.oil_control!s}\nGas Control: {self.gas_control!s}\nWater Control: {self.water_control!s}"

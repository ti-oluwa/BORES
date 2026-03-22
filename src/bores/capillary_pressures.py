"""Capillary pressure models and tables for multi-phase flow simulations."""

import threading
import typing

import attrs
import numpy as np
import numpy.typing as npt

from bores.errors import ValidationError
from bores.grids.base import array as bores_array
from bores.serialization import Serializable, make_serializable_type_registrar
from bores.stores import StoreSerializable
from bores.types import (
    CapillaryPressureDerivatives,
    CapillaryPressures,
    FloatOrArray,
    FluidPhase,
    Wettability,
)
from bores.utils import piecewise_linear_slope

__all__ = [
    "BrooksCoreyCapillaryPressureModel",
    "LeverettJCapillaryPressureModel",
    "ThreePhaseCapillaryPressureTable",
    "TwoPhaseCapillaryPressureTable",
    "VanGenuchtenCapillaryPressureModel",
    "capillary_pressure_table",
]


class CapillaryPressureTable(StoreSerializable):
    """
    Protocol for a capillary pressure table that computes
    capillary pressures based on fluid saturations.
    """

    __abstract_serializable__ = True

    def get_capillary_pressures(
        self,
        water_saturation: FloatOrArray,
        oil_saturation: FloatOrArray,
        gas_saturation: FloatOrArray,
        **kwargs: typing.Any,
    ) -> CapillaryPressures:
        """
        Compute capillary pressures for three-phase system.

        Supports both scalar and array inputs.

        :param water_saturation: Water saturation (fraction, 0-1) - scalar or array.
        :param oil_saturation: Oil saturation (fraction, 0-1) - scalar or array.
        :param gas_saturation: Gas saturation (fraction, 0-1) - scalar or array.
        :return: `CapillaryPressures` dictionary.
        """
        raise NotImplementedError

    def get_capillary_pressure_derivatives(
        self,
        water_saturation: FloatOrArray,
        oil_saturation: FloatOrArray,
        gas_saturation: FloatOrArray,
        **kwargs: typing.Any,
    ) -> CapillaryPressureDerivatives:
        """
        Compute capillary pressure derivatives for three-phase system.

        Supports both scalar and array inputs.

        :param water_saturation: Water saturation (fraction, 0-1) - scalar or array.
        :param oil_saturation: Oil saturation (fraction, 0-1) - scalar or array.
        :param gas_saturation: Gas saturation (fraction, 0-1) - scalar or array.
        :return: `CapillaryPressureDerivatives` dictionary containing the partial derivatives as described above.
        """
        raise NotImplementedError

    def __call__(
        self,
        water_saturation: FloatOrArray,
        oil_saturation: FloatOrArray,
        gas_saturation: FloatOrArray,
        **kwargs: typing.Any,
    ) -> CapillaryPressures:
        """
        Computes capillary pressures based on fluid saturations.

        :param water_saturation: Water saturation (fraction, 0-1) - scalar or array.
        :param oil_saturation: Oil saturation (fraction, 0-1) - scalar or array.
        :param gas_saturation: Gas saturation (fraction, 0-1) - scalar or array.
        :return: `CapillaryPressures` dictionary.
        """
        return self.get_capillary_pressures(
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            **kwargs,
        )

    def derivatives(
        self,
        water_saturation: FloatOrArray,
        oil_saturation: FloatOrArray,
        gas_saturation: FloatOrArray,
        **kwargs: typing.Any,
    ) -> CapillaryPressureDerivatives:
        """
        Compute capillary pressure derivatives for three-phase system.

        Supports both scalar and array inputs.

        :param water_saturation: Water saturation (fraction, 0-1) - scalar or array.
        :param oil_saturation: Oil saturation (fraction, 0-1) - scalar or array.
        :param gas_saturation: Gas saturation (fraction, 0-1) - scalar or array.
        :return: `CapillaryPressureDerivatives` dictionary containing the partial derivatives as described above.
        """
        return self.get_capillary_pressure_derivatives(
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            **kwargs,
        )


_CAPILLARY_PRESSURE_TABLES: typing.Dict[str, typing.Type[CapillaryPressureTable]] = {}
"""Registry for capillary pressure table types."""
_capillary_pressure_table_lock = threading.Lock()
capillary_pressure_table = make_serializable_type_registrar(
    base_cls=CapillaryPressureTable,
    registry=_CAPILLARY_PRESSURE_TABLES,
    key_attr="__type__",
    lock=_capillary_pressure_table_lock,
    override=False,
    auto_register_serializer=True,
    auto_register_deserializer=True,
)


def list_capillary_pressure_tables() -> typing.List[str]:
    """
    List all registered capillary pressure table types.

    :return: List of capillary pressure table type names.
    """
    with _capillary_pressure_table_lock:
        return list(_CAPILLARY_PRESSURE_TABLES.keys())


def get_capillary_pressure_table(name: str) -> typing.Type[CapillaryPressureTable]:
    """
    Get a registered capillary pressure table type by name.

    :param name: Name of the capillary pressure table type.
    :return: Capillary pressure table class.
    :raises KeyError: If the type name is not registered.
    """
    with _capillary_pressure_table_lock:
        if name not in _CAPILLARY_PRESSURE_TABLES:
            raise ValidationError(
                f"Capillary pressure table type '{name}' is not registered. "
                f"Use `@capillary_pressure_table` to register it. "
                f"Available types: {list(_CAPILLARY_PRESSURE_TABLES.keys())}"
            )
        return _CAPILLARY_PRESSURE_TABLES[name]


@attrs.frozen
class TwoPhaseCapillaryPressureTable(Serializable):
    """
    Two-phase capillary pressure lookup table.

    Interpolates capillary pressure for two fluid phases based on saturation values.
    Uses `np.interp` for fast vectorized interpolation.

    Supports both scalar and array inputs up to 3D.
    """

    wetting_phase: typing.Union[FluidPhase, str] = attrs.field(converter=FluidPhase)
    """The first fluid phase (typically the wetting phase), e.g., 'water' or 'oil'."""
    non_wetting_phase: typing.Union[FluidPhase, str] = attrs.field(converter=FluidPhase)
    """The second fluid phase (typically the non-wetting phase), e.g., 'oil' or 'gas'."""
    wetting_phase_saturation: npt.NDArray = attrs.field(converter=bores_array)
    """The saturation values for the wetting phase (phase1), ranging from 0 to 1."""
    capillary_pressure: npt.NDArray = attrs.field(converter=bores_array)
    """Capillary pressure values (Pc = P_non-wetting - P_wetting) corresponding to saturations."""

    def __attrs_post_init__(self) -> None:
        """Validate table data."""
        if len(self.wetting_phase_saturation) != len(self.capillary_pressure):
            raise ValidationError(
                f"Saturation and pressure arrays must have same length. "
                f"Got {len(self.wetting_phase_saturation)} vs {len(self.capillary_pressure)}"
            )
        if len(self.wetting_phase_saturation) < 2:
            raise ValidationError("At least 2 points required for interpolation")

        # Ensure arrays are sorted by saturation (required for np.interp)
        if not np.all(np.diff(self.wetting_phase_saturation) >= 0):
            raise ValidationError(
                "Wetting phase saturation must be monotonically increasing"
            )

    def get_capillary_pressure(
        self, wetting_phase_saturation: FloatOrArray
    ) -> FloatOrArray:
        """
        Get capillary pressure at given wetting phase saturation(s).

        Uses `np.interp` for fast linear interpolation. Supports both scalar
        and array inputs up to 3D. For out-of-bounds values, uses constant
        extrapolation (returns edge values).

        :param wetting_phase_saturation: Saturation of the wetting phase (scalar or array).
        :return: Capillary pressure value(s) - type matches input type.

        Examples:
        ```python
        # Scalar input
        pc = table.get_capillary_pressure(0.5)  # Returns float

        # Array input
        sw_grid = np.array([[0.2, 0.3], [0.4, 0.5]])
        pc_grid = table.get_capillary_pressure(sw_grid)  # Returns same shape array
        ```
        """
        # Handle scalar input
        is_scalar = np.isscalar(wetting_phase_saturation)
        saturation = np.atleast_1d(wetting_phase_saturation)

        # Store original shape for multi-dimensional arrays
        original_shape = saturation.shape

        # Flatten to 1D for `np.interp` (required)
        saturation_flat = saturation.ravel(order="C")

        # Fast linear interpolation using np.interp
        # left/right specify extrapolation values for out-of-bounds
        capillary_pressure_flat = np.interp(
            x=saturation_flat,
            xp=self.wetting_phase_saturation,  # type: ignore[arg-type]
            fp=self.capillary_pressure,  # type: ignore[arg-type]
            left=self.capillary_pressure[0],  # Extrapolate low saturations
            right=self.capillary_pressure[-1],  # Extrapolate high saturations
        )

        # Reshape back to original dimensions
        capillary_pressure = capillary_pressure_flat.reshape(original_shape)
        return capillary_pressure[0] if is_scalar else capillary_pressure

    def __call__(
        self, wetting_phase_saturation: FloatOrArray, **kwargs: typing.Any
    ) -> FloatOrArray:
        """
        Get capillary pressure at given wetting phase saturation(s).

        :param wetting_phase_saturation: Saturation of the wetting phase (scalar or array).
        :return: Capillary pressure value(s).
        """
        return self.get_capillary_pressure(wetting_phase_saturation)

    def get_capillary_pressure_derivative(
        self,
        wetting_phase_saturation: FloatOrArray,
    ) -> FloatOrArray:
        """
        Derivative of capillary pressure with respect to wetting-phase saturation:
        `dPc / d(wetting-phase saturation)`.

        Uses the exact piecewise-linear slope of the tabulated capillary pressure
        curve, the same interpolant used by `get_capillary_pressure`.  Zero
        outside the tabulated saturation range (constant extrapolation).

        :param wetting_phase_saturation: Wetting-phase saturation (scalar or array).
        :return: Derivative value(s) with the same shape as the input.
        """
        return piecewise_linear_slope(
            query=wetting_phase_saturation,
            table_x=self.wetting_phase_saturation,
            table_y=self.capillary_pressure,
        )


@capillary_pressure_table
@attrs.frozen
class ThreePhaseCapillaryPressureTable(
    CapillaryPressureTable,
    load_exclude={"supports_arrays"},
    dump_exclude={"supports_arrays"},
):
    """
    Three-phase capillary pressure lookup table.

    Uses two two-phase tables (oil-water and gas-oil) to compute capillary pressures
    in a three-phase system (water, oil, gas).

    Pcow = Po - Pw (oil-water capillary pressure)
    Pcgo = Pg - Po (gas-oil capillary pressure)
    """

    __type__ = "three_phase_capillary_pressure_table"

    oil_water_table: TwoPhaseCapillaryPressureTable
    """
    Capillary pressure table for oil-water system (wetting phase = water or oil).

    A table of Pcow against wetting phase saturation (water saturation if water is wetting phase,
    oil saturation if oil is wetting phase).
    """
    gas_oil_table: TwoPhaseCapillaryPressureTable
    """
    Capillary pressure table for gas-oil system (wetting phase = oil).

    A table of Pcgo against oil saturation.
    """

    supports_arrays: bool = attrs.field(init=False, repr=False, default=True)
    """Flag indicating support for array inputs."""

    def __attrs_post_init__(self) -> None:
        """Validate that the tables are set up correctly for three-phase flow."""
        if {
            self.oil_water_table.wetting_phase,
            self.oil_water_table.non_wetting_phase,
        } != {FluidPhase.WATER, FluidPhase.OIL}:
            raise ValidationError(
                "`oil_water_table` must be between water and oil phases."
            )
        if {self.gas_oil_table.wetting_phase, self.gas_oil_table.non_wetting_phase} != {
            FluidPhase.OIL,
            FluidPhase.GAS,
        }:
            raise ValidationError("`gas_oil_table` must be between oil and gas phases.")

        if self.oil_water_table.wetting_phase == self.gas_oil_table.non_wetting_phase:
            raise ValidationError(
                "Wetting phase of `oil_water_table` cannot be the same as non-wetting phase of `gas_oil_table`."
            )

    def get_capillary_pressures(
        self,
        water_saturation: FloatOrArray,
        oil_saturation: FloatOrArray,
        gas_saturation: FloatOrArray,
        **kwargs: typing.Any,
    ) -> CapillaryPressures:
        """
        Compute capillary pressures for three-phase system.

        Supports both scalar and array inputs.

        :param water_saturation: Water saturation (fraction, 0-1) - scalar or array.
        :param oil_saturation: Oil saturation (fraction, 0-1) - scalar or array.
        :param gas_saturation: Gas saturation (fraction, 0-1) - scalar or array.
        :return: `CapillaryPressures` dictionary (matching input type).
        """
        # Oil-water capillary pressure (based on wetting phase saturation)
        if self.oil_water_table.wetting_phase == FluidPhase.WATER:
            pcow = self.oil_water_table(water_saturation)
        else:
            pcow = self.oil_water_table(oil_saturation)

        # Gas-oil capillary pressure (based on oil saturation - wetting phase)
        pcgo = self.gas_oil_table(oil_saturation)
        return CapillaryPressures(oil_water=pcow, gas_oil=pcgo)  # type: ignore[typeddict-item]

    def get_capillary_pressures_derivatives(
        self,
        water_saturation: FloatOrArray,
        oil_saturation: FloatOrArray,
        gas_saturation: FloatOrArray,
        **kwargs: typing.Any,
    ) -> CapillaryPressureDerivatives:
        """
        Compute the partial derivatives of the oil-water and gas-oil capillary
        pressures with respect to saturation.

        Returns a dictionary containing:

        ```
        (dPcow/dSw, dPcow/dSo, dPcgo/dSo)
        ```

        - `dPcow/dSw`: non-zero when the oil-water table uses water saturation
        as its reference axis (water-wet convention).
        - `dPcow/dSo`: non-zero when the oil-water table uses oil saturation
        as its reference axis (oil-wet convention).
        - `dPcgo/dSo`: the gas-oil table always uses oil saturation as its
        wetting-phase reference axis, so this is always the active gas-oil
        derivative.

        Note: `dPcgo/dSg` is not returned because this table type indexes the
        gas-oil curve by oil saturation. There is no direct gas saturation axis.
        Callers that need `dPcgo/dSg` should apply the saturation constraint
        `So = 1 - Sw - Sg` with the chain rule.

        All derivatives are exact piecewise-linear slopes from the underlying
        two-phase tables.

        :param water_saturation: Water saturation (scalar or array).
        :param oil_saturation: Oil saturation (scalar or array).
        :param gas_saturation: Gas saturation (not used directly by this table
            type; included for a consistent three-phase calling convention).
        :return: `CapillaryPressureDerivatives` dictionary containing the partial derivatives as described above.
        """
        oil_water_table = self.oil_water_table
        gas_oil_table = self.gas_oil_table

        is_scalar = np.isscalar(water_saturation)
        zero_scalar = 0.0
        zero_array = np.zeros_like(
            np.atleast_1d(np.asarray(water_saturation, dtype=np.float64))
        )

        if oil_water_table.wetting_phase == FluidPhase.WATER:
            d_pcow_d_sw = oil_water_table.get_capillary_pressure_derivative(
                wetting_phase_saturation=water_saturation,
            )
            d_pcow_d_so = zero_scalar if is_scalar else zero_array.copy()
        else:
            d_pcow_d_sw = zero_scalar if is_scalar else zero_array.copy()
            d_pcow_d_so = oil_water_table.get_capillary_pressure_derivative(
                wetting_phase_saturation=oil_saturation,
            )

        d_pcgo_d_so = gas_oil_table.get_capillary_pressure_derivative(
            wetting_phase_saturation=oil_saturation,
        )
        return CapillaryPressureDerivatives(
            dPcow_dSw=d_pcow_d_sw,
            dPcow_dSo=d_pcow_d_so,
            dPcgo_dSg=d_pcgo_d_so,
        )


def compute_brooks_corey_capillary_pressures(
    water_saturation: FloatOrArray,
    oil_saturation: FloatOrArray,
    gas_saturation: FloatOrArray,
    irreducible_water_saturation: FloatOrArray,
    residual_oil_saturation_water: FloatOrArray,
    residual_oil_saturation_gas: FloatOrArray,
    residual_gas_saturation: FloatOrArray,
    wettability: Wettability,
    oil_water_entry_pressure_water_wet: float,
    oil_water_entry_pressure_oil_wet: float,
    oil_water_pore_size_distribution_index_water_wet: float,
    oil_water_pore_size_distribution_index_oil_wet: float,
    gas_oil_entry_pressure: float,
    gas_oil_pore_size_distribution_index: float,
    mixed_wet_water_fraction: float = 0.5,
) -> typing.Tuple[FloatOrArray, FloatOrArray]:
    """
    Computes capillary pressures (Pcow, Pcgo) using Brooks-Corey model.

    Supports both scalar and array inputs (up to 3D).

    Pcow is defined as Po - Pw.
    Pcgo is defined as Pg - Po.

    Brooks-Corey model: Pc = Pd * (Se)^(-1/λ)
    where:
    - Pd is the displacement/entry pressure
    - Se is the effective saturation
    - λ is the pore size distribution index

    Wettability behavior:
    - WATER_WET: Pcow > 0, Pcgo > 0 (water preferentially wets rock)
    - OIL_WET:   Pcow < 0, Pcgo > 0 (oil preferentially wets rock)
    - MIXED_WET: Pcow varies with saturation (weighted combination)

    :param water_saturation: Current water saturation (fraction, 0-1) - scalar or array.
    :param oil_saturation: Current oil saturation (fraction, 0-1) - scalar or array.
    :param gas_saturation: Current gas saturation (fraction, 0-1) - scalar or array.
    :param irreducible_water_saturation: Irreducible water saturation (Swc) - scalar or array.
    :param residual_oil_saturation_water: Residual oil saturation during water flooding (Sorw) - scalar or array.
    :param residual_oil_saturation_gas: Residual oil saturation during gas flooding (Sorg) - scalar or array.
    :param residual_gas_saturation: Residual gas saturation (Sgr) - scalar or array.
    :param wettability: Wettability type (WATER_WET, OIL_WET, or MIXED_WET).
    :param oil_water_entry_pressure_water_wet: Entry pressure for oil-water in water-wet system (psi).
    :param oil_water_entry_pressure_oil_wet: Entry pressure for oil-water in oil-wet system (psi).
    :param oil_water_pore_size_distribution_index_water_wet: Pore size distribution index (λ) for oil-water in water-wet.
    :param oil_water_pore_size_distribution_index_oil_wet: Pore size distribution index (λ) for oil-water in oil-wet.
    :param gas_oil_entry_pressure: Entry pressure for gas-oil (psi).
    :param gas_oil_pore_size_distribution_index: Pore size distribution index (λ) for gas-oil.
    :param mixed_wet_water_fraction: Fraction of pore space that is water-wet (0-1, default 0.5).
    :return: Tuple of (oil_water_capillary_pressure, gas_oil_capillary_pressure) in psi.
    """
    # Convert to arrays for vectorized operations
    sw = np.atleast_1d(water_saturation)
    so = np.atleast_1d(oil_saturation)
    sg = np.atleast_1d(gas_saturation)
    Swc = np.atleast_1d(irreducible_water_saturation)
    Sorw = np.atleast_1d(residual_oil_saturation_water)
    Sorg = np.atleast_1d(residual_oil_saturation_gas)
    Sgr = np.atleast_1d(residual_gas_saturation)

    # Broadcast all arrays to same shape
    sw, so, sg, Swc, Sorw, Sorg, Sgr = np.broadcast_arrays(
        sw, so, sg, Swc, Sorw, Sorg, Sgr
    )

    # Validate saturations
    if np.any((sw < 0) | (sw > 1) | (so < 0) | (so > 1) | (sg < 0) | (sg > 1)):
        raise ValidationError("Saturations must be between 0 and 1.")

    # Normalize saturations if they do not sum to 1
    total_saturation = sw + so + sg
    needs_norm = (np.abs(total_saturation - 1.0) > 1e-6) & (total_saturation > 0.0)
    if np.any(needs_norm):
        sw = np.where(needs_norm, sw / total_saturation, sw)
        so = np.where(needs_norm, so / total_saturation, so)
        sg = np.where(needs_norm, sg / total_saturation, sg)

    # Effective pore spaces
    total_mobile_pore_space_water = 1.0 - Swc - Sorw
    total_mobile_pore_space_gas = 1.0 - Swc - Sorg - Sgr

    # Pcow (Po - Pw)
    oil_water_capillary_pressure = np.zeros_like(sw)

    # Mask for valid mobile pore space
    valid_water = total_mobile_pore_space_water > 1e-9

    if np.any(valid_water):
        effective_water_saturation = np.where(
            valid_water, (sw - Swc) / total_mobile_pore_space_water, 0.0
        )
        effective_water_saturation = np.clip(effective_water_saturation, 1e-6, 1.0)
        # Mask for undersaturated conditions
        undersaturated = valid_water & (effective_water_saturation < 1.0 - 1e-6)

        if np.any(undersaturated):
            if wettability == Wettability.WATER_WET:
                # Pure water-wet: Pcow > 0
                pcow = oil_water_entry_pressure_water_wet * (
                    effective_water_saturation
                    ** (-1.0 / oil_water_pore_size_distribution_index_water_wet)
                )
                oil_water_capillary_pressure = np.where(undersaturated, pcow, 0.0)

            elif wettability == Wettability.OIL_WET:
                # Pure oil-wet: Pcow < 0
                pcow = -(
                    oil_water_entry_pressure_oil_wet
                    * effective_water_saturation
                    ** (-1.0 / oil_water_pore_size_distribution_index_oil_wet)
                )
                oil_water_capillary_pressure = np.where(undersaturated, pcow, 0.0)

            elif wettability == Wettability.MIXED_WET:
                # Mixed-wet: Weighted average
                pcoil_water_contact_anglewater_wet = (
                    oil_water_entry_pressure_water_wet
                    * (
                        effective_water_saturation
                        ** (-1.0 / oil_water_pore_size_distribution_index_water_wet)
                    )
                )
                pcoil_water_contact_angleoil_wet = -(
                    oil_water_entry_pressure_oil_wet
                    * effective_water_saturation
                    ** (-1.0 / oil_water_pore_size_distribution_index_oil_wet)
                )
                pcow = (
                    mixed_wet_water_fraction * pcoil_water_contact_anglewater_wet
                    + (1.0 - mixed_wet_water_fraction)
                    * pcoil_water_contact_angleoil_wet
                )
                oil_water_capillary_pressure = np.where(undersaturated, pcow, 0.0)

    # Pcgo (Pg - Po)
    gas_oil_capillary_pressure = np.zeros_like(sg)
    valid_gas = total_mobile_pore_space_gas > 1e-9

    if np.any(valid_gas):
        effective_gas_saturation = np.where(
            valid_gas, (sg - Sgr) / total_mobile_pore_space_gas, 0.0
        )
        effective_gas_saturation = np.clip(effective_gas_saturation, 1e-6, 1.0)

        undersaturated_gas = valid_gas & (effective_gas_saturation < 1.0 - 1e-6)

        if np.any(undersaturated_gas):
            pcgo = gas_oil_entry_pressure * (
                effective_gas_saturation
                ** (-1.0 / gas_oil_pore_size_distribution_index)
            )
            gas_oil_capillary_pressure = np.where(undersaturated_gas, pcgo, 0.0)

    # Return scalars if inputs were scalars
    is_scalar = np.isscalar(water_saturation)
    if is_scalar:
        return float(oil_water_capillary_pressure), float(gas_oil_capillary_pressure)
    return oil_water_capillary_pressure, gas_oil_capillary_pressure


@capillary_pressure_table
@attrs.frozen
class BrooksCoreyCapillaryPressureModel(
    CapillaryPressureTable,
    load_exclude={"supports_arrays"},
    dump_exclude={"supports_arrays"},
):
    """
    Brooks-Corey capillary pressure model for three-phase systems.

    Implements the Brooks-Corey model: Pc = Pd * (Se)^(-1/λ)

    Supports water-wet, oil-wet, and mixed-wet systems.
    """

    __type__ = "brooks_corey_capillary_pressure_model"

    irreducible_water_saturation: typing.Optional[float] = None
    """Default irreducible water saturation (Swc). Can be overridden per call."""
    residual_oil_saturation_water: typing.Optional[float] = None
    """Default residual oil saturation after water flood (Sorw). Can be overridden per call."""
    residual_oil_saturation_gas: typing.Optional[float] = None
    """Default residual oil saturation after gas flood (Sorg). Can be overridden per call."""
    residual_gas_saturation: typing.Optional[float] = None
    """Default residual gas saturation (Sgr). Can be overridden per call."""
    oil_water_entry_pressure_water_wet: float = 5.0
    """Entry pressure for oil-water in water-wet system (psi)."""
    oil_water_entry_pressure_oil_wet: float = 5.0
    """Entry pressure for oil-water in oil-wet system (psi)."""
    oil_water_pore_size_distribution_index_water_wet: float = 2.0
    """Pore size distribution index (λ) for oil-water in water-wet system."""
    oil_water_pore_size_distribution_index_oil_wet: float = 2.0
    """Pore size distribution index (λ) for oil-water in oil-wet system."""
    gas_oil_entry_pressure: float = 1.0
    """Entry pressure for gas-oil (psi)."""
    gas_oil_pore_size_distribution_index: float = 2.0
    """Pore size distribution index (λ) for gas-oil."""
    wettability: Wettability = Wettability.WATER_WET
    """Wettability type (WATER_WET, OIL_WET, or MIXED_WET)."""
    mixed_wet_water_fraction: float = 0.5
    """Fraction of pore space that is water-wet in mixed-wet systems (0-1)."""

    supports_arrays: bool = attrs.field(init=False, repr=False, default=True)
    """Flag indicating support for array inputs."""

    def get_capillary_pressures(  # type: ignore[override]
        self,
        water_saturation: FloatOrArray,
        oil_saturation: FloatOrArray,
        gas_saturation: FloatOrArray,
        irreducible_water_saturation: typing.Optional[FloatOrArray] = None,
        residual_oil_saturation_water: typing.Optional[FloatOrArray] = None,
        residual_oil_saturation_gas: typing.Optional[FloatOrArray] = None,
        residual_gas_saturation: typing.Optional[FloatOrArray] = None,
    ) -> CapillaryPressures:
        """
        Compute capillary pressures using Brooks-Corey model.

        Supports both scalar and array inputs.

        :param water_saturation: Water saturation (fraction, 0-1) - scalar or array.
        :param oil_saturation: Oil saturation (fraction, 0-1) - scalar or array.
        :param gas_saturation: Gas saturation (fraction, 0-1) - scalar or array.
        :param irreducible_water_saturation: Optional override for Swc - scalar or array.
        :param residual_oil_saturation_water: Optional override for Sorw - scalar or array.
        :param residual_oil_saturation_gas: Optional override for Sorg - scalar or array.
        :param residual_gas_saturation: Optional override for Sgr - scalar or array.
        :return: `CapillaryPressures` dictionary.
        """
        Swc = (
            irreducible_water_saturation
            if irreducible_water_saturation is not None
            else self.irreducible_water_saturation
        )
        Sorw = (
            residual_oil_saturation_water
            if residual_oil_saturation_water is not None
            else self.residual_oil_saturation_water
        )
        Sorg = (
            residual_oil_saturation_gas
            if residual_oil_saturation_gas is not None
            else self.residual_oil_saturation_gas
        )
        Sgr = (
            residual_gas_saturation
            if residual_gas_saturation is not None
            else self.residual_gas_saturation
        )

        params_missing = []
        if Swc is None:
            params_missing.append("Swc")
        if Sorw is None:
            params_missing.append("Sorw")
        if Sorg is None:
            params_missing.append("Sorg")
        if Sgr is None:
            params_missing.append("Sgr")
        if params_missing:
            raise ValidationError(
                f"Residual saturations must be provided either as model defaults or in the call. "
                f"Missing: {', '.join(params_missing)}"
            )

        pcow, pcgo = compute_brooks_corey_capillary_pressures(
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            irreducible_water_saturation=Swc,  # type: ignore[arg-type]
            residual_oil_saturation_water=Sorw,  # type: ignore[arg-type]
            residual_oil_saturation_gas=Sorg,  # type: ignore[arg-type]
            residual_gas_saturation=Sgr,  # type: ignore[arg-type]
            wettability=self.wettability,
            oil_water_entry_pressure_water_wet=self.oil_water_entry_pressure_water_wet,
            oil_water_entry_pressure_oil_wet=self.oil_water_entry_pressure_oil_wet,
            oil_water_pore_size_distribution_index_water_wet=self.oil_water_pore_size_distribution_index_water_wet,
            oil_water_pore_size_distribution_index_oil_wet=self.oil_water_pore_size_distribution_index_oil_wet,
            gas_oil_entry_pressure=self.gas_oil_entry_pressure,
            gas_oil_pore_size_distribution_index=self.gas_oil_pore_size_distribution_index,
            mixed_wet_water_fraction=self.mixed_wet_water_fraction,
        )
        return CapillaryPressures(oil_water=pcow, gas_oil=pcgo)  # type: ignore[typeddict-item]

    def get_capillary_pressures_derivatives(
        self,
        water_saturation: FloatOrArray,
        oil_saturation: FloatOrArray,
        gas_saturation: FloatOrArray,
        irreducible_water_saturation: typing.Optional[FloatOrArray] = None,
        residual_oil_saturation_water: typing.Optional[FloatOrArray] = None,
        residual_oil_saturation_gas: typing.Optional[FloatOrArray] = None,
        residual_gas_saturation: typing.Optional[FloatOrArray] = None,
    ) -> CapillaryPressureDerivatives:
        """
        Compute the partial derivatives of the Brooks-Corey oil-water and gas-oil
        capillary pressures with respect to saturation.

        Returns a dictionary contianing:
        ```
        (dPcow/dSw, dPcow/dSo, dPcgo/dSg)
        ```

        - `dPcow/dSw`: analytically derived from the Brooks-Corey power law via
        the chain rule through effective water saturation.
        - `dPcow/dSo`: zero for this model - the oil-water capillary pressure is
        a function of water saturation only, regardless of wettability.
        - `dPcgo/dSg`: analytically derived via the chain rule through effective
        gas saturation.

        The Brooks-Corey capillary pressure formulae are:

        ```
        Pcow = Pd_ow * Se_w^(-1/lambda_ow)          (water-wet)
        Pcow = -Pd_ow * Se_w^(-1/lambda_ow)         (oil-wet)
        Pcgo = Pd_go * Se_g^(-1/lambda_go)
        ```

        where effective water saturation:
        ```
        Se_w = (Sw - irreducible water saturation)
            / (1 - irreducible water saturation
                    - residual oil saturation to water flooding)
        ```

        and effective gas saturation:

        ```
        Se_g = (Sg - residual gas saturation)
            / (1 - irreducible water saturation
                    - residual gas saturation
                    - residual oil saturation to gas flooding)
        ```

        `Pd` is the displacement entry pressure and `lambda` is the pore
        size distribution index.

        :param water_saturation: Water saturation (scalar or array).
        :param oil_saturation: Oil saturation (scalar or array, not used by
            this model but included for API consistency).
        :param gas_saturation: Gas saturation (scalar or array).
        :param irreducible_water_saturation: Optional override for the
            irreducible (connate) water saturation.
        :param residual_oil_saturation_water: Optional override for the residual
            oil saturation to water flooding.
        :param residual_oil_saturation_gas: Optional override for the residual
            oil saturation to gas flooding.
        :param residual_gas_saturation: Optional override for the residual gas
            saturation.
        :return: `CapillaryPressureDerivatives` dictionary containing the partial derivatives as described above.
        """
        Swc = (
            irreducible_water_saturation
            if irreducible_water_saturation is not None
            else self.irreducible_water_saturation
        )
        Sorw = (
            residual_oil_saturation_water
            if residual_oil_saturation_water is not None
            else self.residual_oil_saturation_water
        )
        Sorg = (
            residual_oil_saturation_gas
            if residual_oil_saturation_gas is not None
            else self.residual_oil_saturation_gas
        )
        Sgr = (
            residual_gas_saturation
            if residual_gas_saturation is not None
            else self.residual_gas_saturation
        )

        params_missing = []
        if Swc is None:
            params_missing.append("Swc")
        if Sorw is None:
            params_missing.append("Sorw")
        if Sorg is None:
            params_missing.append("Sorg")
        if Sgr is None:
            params_missing.append("Sgr")
        if params_missing:
            raise ValidationError(
                f"Residual saturations must be provided either as model defaults or in the call. "
                f"Missing: {', '.join(params_missing)}"
            )

        wettability = self.wettability
        is_scalar = np.isscalar(water_saturation)
        sw = np.atleast_1d(np.asarray(water_saturation, dtype=np.float64))
        sg = np.atleast_1d(np.asarray(gas_saturation, dtype=np.float64))

        # dPcow/dSw via chain rule:  dPcow/dSw = (dPcow/dSe_w) * (1 / mobile_water_range)
        mobile_water_range = 1.0 - Swc - Sorw  # type: ignore
        if mobile_water_range > 1e-9:
            se_w = np.clip(
                (sw - Swc) / mobile_water_range,  # type: ignore
                1e-6,
                1.0,
            )
            if wettability == Wettability.MIXED_WET:
                water_wet_fraction = self.mixed_wet_water_fraction
                exp_ww = -1.0 / self.oil_water_pore_size_distribution_index_water_wet
                exp_ow = -1.0 / self.oil_water_pore_size_distribution_index_oil_wet
                d_se_w_ww = (
                    self.oil_water_entry_pressure_water_wet
                    * exp_ww
                    * (se_w ** (exp_ww - 1.0))
                )
                d_se_w_ow = -(
                    self.oil_water_entry_pressure_oil_wet
                    * exp_ow
                    * (se_w ** (exp_ow - 1.0))
                )
                d_pcow_d_se_w = (
                    water_wet_fraction * d_se_w_ww
                    + (1.0 - water_wet_fraction) * d_se_w_ow
                )
            else:
                if wettability == Wettability.WATER_WET:
                    pore_distribution_index = (
                        self.oil_water_pore_size_distribution_index_water_wet
                    )
                    entry_pressure = self.oil_water_entry_pressure_water_wet
                    sign = 1.0
                else:  # OIL_WET
                    pore_distribution_index = (
                        self.oil_water_pore_size_distribution_index_oil_wet
                    )
                    entry_pressure = self.oil_water_entry_pressure_oil_wet
                    sign = -1.0
                exp = -1.0 / pore_distribution_index
                d_pcow_d_se_w = sign * entry_pressure * exp * (se_w ** (exp - 1.0))

            d_pcow_d_sw = d_pcow_d_se_w / mobile_water_range
        else:
            d_pcow_d_sw = np.zeros_like(sw)

        d_pcow_d_so = np.zeros_like(sw)  # Pcow has no So dependence in this model

        # dPcgo/dSg via chain rule:  dPcgo/dSg = (dPcgo/dSe_g) * (1 / mobile_gas_range)
        mobile_gas_range = 1.0 - Swc - Sorg - Sgr  # type: ignore
        if mobile_gas_range > 1e-9:
            se_g = np.clip(
                (sg - Sgr) / mobile_gas_range,  # type: ignore
                1e-6,
                1.0,
            )
            exp_go = -1.0 / self.gas_oil_pore_size_distribution_index
            d_pcgo_d_se_g = (
                self.gas_oil_entry_pressure * exp_go * (se_g ** (exp_go - 1.0))
            )
            d_pcgo_d_sg = d_pcgo_d_se_g / mobile_gas_range
        else:
            d_pcgo_d_sg = np.zeros_like(sg)

        if is_scalar:
            return CapillaryPressureDerivatives(
                dPcow_dSw=float(np.ravel(d_pcow_d_sw)[0]),
                dPcow_dSo=float(np.ravel(d_pcow_d_so)[0]),
                dPcgo_dSg=float(np.ravel(d_pcgo_d_sg)[0]),
            )
        return CapillaryPressureDerivatives(
            dPcow_dSw=d_pcow_d_sw,
            dPcow_dSo=d_pcow_d_so,
            dPcgo_dSg=d_pcgo_d_sg,
        )

    def __call__(
        self,
        water_saturation: FloatOrArray,
        oil_saturation: FloatOrArray,
        gas_saturation: FloatOrArray,
        **kwargs: typing.Any,
    ) -> CapillaryPressures:
        """
        Compute capillary pressures using Brooks-Corey model.

        :param water_saturation: Water saturation (fraction, 0-1) - scalar or array.
        :param oil_saturation: Oil saturation (fraction, 0-1) - scalar or array.
        :param gas_saturation: Gas saturation (fraction, 0-1) - scalar or array.
        :kwarg irreducible_water_saturation: Optional override for Swc.
        :kwarg residual_oil_saturation_water: Optional override for Sorw.
        :kwarg residual_oil_saturation_gas: Optional override for Sorg.
        :kwarg residual_gas_saturation: Optional override for Sgr.
        :return: `CapillaryPressures` dictionary.
        """
        return self.get_capillary_pressures(
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            irreducible_water_saturation=kwargs.get("irreducible_water_saturation"),
            residual_oil_saturation_water=kwargs.get("residual_oil_saturation_water"),
            residual_oil_saturation_gas=kwargs.get("residual_oil_saturation_gas"),
            residual_gas_saturation=kwargs.get("residual_gas_saturation"),
        )

    def derivatives(
        self,
        water_saturation: FloatOrArray,
        oil_saturation: FloatOrArray,
        gas_saturation: FloatOrArray,
        **kwargs: typing.Any,
    ) -> CapillaryPressureDerivatives:
        """
        Compute capillary pressure derivatives using Brooks-Corey model.

        :param water_saturation: Water saturation (fraction, 0-1) - scalar or array.
        :param oil_saturation: Oil saturation (fraction, 0-1) - scalar or array.
        :param gas_saturation: Gas saturation (fraction, 0-1) - scalar or array.
        :kwarg irreducible_water_saturation: Optional override for Swc.
        :kwarg residual_oil_saturation_water: Optional override for Sorw.
        :kwarg residual_oil_saturation_gas: Optional override for Sorg.
        :kwarg residual_gas_saturation: Optional override for Sgr.
        :return: `CapillaryPressureDerivatives` dictionary containing the partial derivatives as described above.
        """
        return self.get_capillary_pressure_derivatives(
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            irreducible_water_saturation=kwargs.get("irreducible_water_saturation"),
            residual_oil_saturation_water=kwargs.get("residual_oil_saturation_water"),
            residual_oil_saturation_gas=kwargs.get("residual_oil_saturation_gas"),
            residual_gas_saturation=kwargs.get("residual_gas_saturation"),
        )


def compute_van_genuchten_capillary_pressures(
    water_saturation: FloatOrArray,
    oil_saturation: FloatOrArray,
    gas_saturation: FloatOrArray,
    irreducible_water_saturation: FloatOrArray,
    residual_oil_saturation_water: FloatOrArray,
    residual_oil_saturation_gas: FloatOrArray,
    residual_gas_saturation: FloatOrArray,
    wettability: Wettability,
    oil_water_alpha_water_wet: float,
    oil_water_alpha_oil_wet: float,
    oil_water_n_water_wet: float,
    oil_water_n_oil_wet: float,
    gas_oil_alpha: float,
    gas_oil_n: float,
    mixed_wet_water_fraction: float = 0.5,
) -> typing.Tuple[FloatOrArray, FloatOrArray]:
    """
    Computes capillary pressures using van Genuchten model.

    Supports both scalar and array inputs (up to 3D).

    van Genuchten model: Pc = (1/α) * [(Se^(-1/m) - 1)^(1/n)]
    where m = 1 - 1/n

    This model is widely used in unsaturated soil mechanics and petroleum engineering.
    Provides smoother transitions near residual saturations compared to Brooks-Corey.

    :param water_saturation: Current water saturation (fraction, 0-1) - scalar or array.
    :param oil_saturation: Current oil saturation (fraction, 0-1) - scalar or array.
    :param gas_saturation: Current gas saturation (fraction, 0-1) - scalar or array.
    :param irreducible_water_saturation: Irreducible water saturation (Swc) - scalar or array.
    :param residual_oil_saturation_water: Residual oil saturation during water flooding (Sorw) - scalar or array.
    :param residual_oil_saturation_gas: Residual oil saturation during gas flooding (Sorg) - scalar or array.
    :param residual_gas_saturation: Residual gas saturation (Sgr) - scalar or array.
    :param wettability: Wettability type (WATER_WET, OIL_WET, or MIXED_WET).
    :param oil_water_alpha_water_wet: van Genuchten α parameter for oil-water (water-wet) [1/psi].
    :param oil_water_alpha_oil_wet: van Genuchten α parameter for oil-water (oil-wet) [1/psi].
    :param oil_water_n_water_wet: van Genuchten n parameter for oil-water (water-wet).
    :param oil_water_n_oil_wet: van Genuchten n parameter for oil-water (oil-wet).
    :param gas_oil_alpha: van Genuchten α parameter for gas-oil [1/psi].
    :param gas_oil_n: van Genuchten n parameter for gas-oil.
    :param mixed_wet_water_fraction: Fraction of pore space that is water-wet (0-1, default 0.5).
    :return: Tuple of (oil_water_capillary_pressure, gas_oil_capillary_pressure) in psi.
    """
    # Parameter validation
    if oil_water_alpha_water_wet <= 0.0 or oil_water_alpha_oil_wet <= 0.0:
        raise ValidationError("Oil-water alpha parameters must be positive.")
    if gas_oil_alpha <= 0.0:
        raise ValidationError("Gas-oil alpha parameter must be positive.")
    if oil_water_n_water_wet <= 1.0 or oil_water_n_oil_wet <= 1.0:
        raise ValidationError("Oil-water n parameters must be greater than 1.")
    if gas_oil_n <= 1.0:
        raise ValidationError("Gas-oil n parameter must be greater than 1.")

    # Convert to arrays for vectorized operations
    sw = np.atleast_1d(water_saturation)
    so = np.atleast_1d(oil_saturation)
    sg = np.atleast_1d(gas_saturation)
    Swc = np.atleast_1d(irreducible_water_saturation)
    Sorw = np.atleast_1d(residual_oil_saturation_water)
    Sorg = np.atleast_1d(residual_oil_saturation_gas)
    Sgr = np.atleast_1d(residual_gas_saturation)

    # Broadcast all arrays to same shape
    sw, so, sg, Swc, Sorw, Sorg, Sgr = np.broadcast_arrays(
        sw, so, sg, Swc, Sorw, Sorg, Sgr
    )

    # Validate saturations
    if np.any((sw < 0) | (sw > 1) | (so < 0) | (so > 1) | (sg < 0) | (sg > 1)):
        raise ValidationError("Saturations must be between 0 and 1.")

    # Normalize saturations if they do not sum to 1
    total_saturation = sw + so + sg
    needs_norm = (np.abs(total_saturation - 1.0) > 1e-6) & (total_saturation > 0.0)
    if np.any(needs_norm):
        sw = np.where(needs_norm, sw / total_saturation, sw)
        so = np.where(needs_norm, so / total_saturation, so)
        sg = np.where(needs_norm, sg / total_saturation, sg)

    # Effective pore spaces
    total_mobile_pore_space_water = 1.0 - Swc - Sorw
    total_mobile_pore_space_gas = 1.0 - Swc - Sorg - Sgr

    #  Pcow (Po - Pw)
    oil_water_capillary_pressure = np.zeros_like(sw)
    # Mask for valid mobile pore space
    valid_water = total_mobile_pore_space_water > 1e-9

    if np.any(valid_water):
        effective_water_saturation = np.where(
            valid_water, (sw - Swc) / total_mobile_pore_space_water, 0.0
        )
        effective_water_saturation = np.clip(
            effective_water_saturation, 1e-6, 1.0 - 1e-6
        )

        if wettability == Wettability.WATER_WET:
            m_ww = 1.0 - 1.0 / oil_water_n_water_wet
            term = (effective_water_saturation ** (-1.0 / m_ww) - 1.0) ** (
                1.0 / oil_water_n_water_wet
            )
            pcow = (1.0 / oil_water_alpha_water_wet) * term
            oil_water_capillary_pressure = np.where(valid_water, pcow, 0.0)

        elif wettability == Wettability.OIL_WET:
            m_ow = 1.0 - 1.0 / oil_water_n_oil_wet
            term = (effective_water_saturation ** (-1.0 / m_ow) - 1.0) ** (
                1.0 / oil_water_n_oil_wet
            )
            pcow = -(1.0 / oil_water_alpha_oil_wet) * term
            oil_water_capillary_pressure = np.where(valid_water, pcow, 0.0)

        elif wettability == Wettability.MIXED_WET:
            # Water-wet contribution
            m_ww = 1.0 - 1.0 / oil_water_n_water_wet
            term_ww = (effective_water_saturation ** (-1.0 / m_ww) - 1.0) ** (
                1.0 / oil_water_n_water_wet
            )
            pcoil_water_contact_anglewater_wet = (
                1.0 / oil_water_alpha_water_wet
            ) * term_ww

            # Oil-wet contribution
            m_ow = 1.0 - 1.0 / oil_water_n_oil_wet
            term_ow = (effective_water_saturation ** (-1.0 / m_ow) - 1.0) ** (
                1.0 / oil_water_n_oil_wet
            )
            pcoil_water_contact_angleoil_wet = (
                -(1.0 / oil_water_alpha_oil_wet) * term_ow
            )

            pcow = (
                mixed_wet_water_fraction * pcoil_water_contact_anglewater_wet
                + (1.0 - mixed_wet_water_fraction) * pcoil_water_contact_angleoil_wet
            )
            oil_water_capillary_pressure = np.where(valid_water, pcow, 0.0)

    #  Pcgo (Pg - Po)
    gas_oil_capillary_pressure = np.zeros_like(sg)
    valid_gas = total_mobile_pore_space_gas > 1e-9

    if np.any(valid_gas):
        effective_gas_saturation = np.where(
            valid_gas, (sg - Sgr) / total_mobile_pore_space_gas, 0.0
        )
        effective_gas_saturation = np.clip(effective_gas_saturation, 1e-6, 1.0 - 1e-6)

        m_go = 1.0 - 1.0 / gas_oil_n
        term = (effective_gas_saturation ** (-1.0 / m_go) - 1.0) ** (1.0 / gas_oil_n)
        pcgo = (1.0 / gas_oil_alpha) * term
        gas_oil_capillary_pressure = np.where(valid_gas, pcgo, 0.0)

    # Return scalars if inputs were scalars
    is_scalar = np.isscalar(water_saturation)
    if is_scalar:
        return float(oil_water_capillary_pressure), float(gas_oil_capillary_pressure)
    return oil_water_capillary_pressure, gas_oil_capillary_pressure


def _van_genuchten_pc_slope_wrt_effective_saturation(
    effective_saturation: npt.NDArray,
    alpha: float,
    n: float,
    sign: float,
) -> npt.NDArray:
    """
    Analytical derivative of the van Genuchten capillary pressure with
    respect to effective (normalised) saturation.

    The van Genuchten model is:

    ```
    Pc = sign * (1/alpha) * (Se^(-1/m) - 1)^(1/n)   where m = 1 - 1/n
    ```

    Let u = Se^(-1/m) - 1.  By the chain rule::

    ```
    dPc/dSe = (dPc/du) * (du/dSe)
            = sign*(1/alpha)*(1/n)*u^(1/n - 1) * (-1/m)*Se^(-1/m - 1)
    ```

    :param effective_saturation: Normalised saturation, clamped to
        (1e-6, 1-1e-6) internally.
    :param alpha: van Genuchten alpha parameter (1/pressure, positive).
    :param n: van Genuchten n parameter (greater than 1).
    :param sign: +1 for water-wet, -1 for oil-wet oil-water capillary pressure.
    :return: Derivative array with the same shape as `effective_saturation`.
    """
    m = 1.0 - 1.0 / n
    se = np.clip(effective_saturation, 1e-6, 1.0 - 1e-6)
    u = se ** (-1.0 / m) - 1.0
    u_safe = np.where(u > 1e-30, u, 1e-30)
    d_pc_d_u = sign * (1.0 / alpha) * (1.0 / n) * (u_safe ** (1.0 / n - 1.0))
    d_u_d_se = (-1.0 / m) * (se ** (-1.0 / m - 1.0))
    return d_pc_d_u * d_u_d_se


@capillary_pressure_table
@attrs.frozen
class VanGenuchtenCapillaryPressureModel(
    CapillaryPressureTable,
    load_exclude={"supports_arrays"},
    dump_exclude={"supports_arrays"},
):
    """
    van Genuchten capillary pressure model for three-phase systems.

    Implements: Pc = (1/α) * [(Se^(-1/m) - 1)^(1/n)] where m = 1 - 1/n

    Provides smoother transitions than Brooks-Corey model.
    """

    __type__ = "van_genuchten_capillary_pressure_model"

    irreducible_water_saturation: typing.Optional[float] = None
    """Default irreducible water saturation (Swc). Can be overridden per call."""
    residual_oil_saturation_water: typing.Optional[float] = None
    """Default residual oil saturation after water flood (Sorw). Can be overridden per call."""
    residual_oil_saturation_gas: typing.Optional[float] = None
    """Default residual oil saturation after gas flood (Sorg). Can be overridden per call."""
    residual_gas_saturation: typing.Optional[float] = None
    """Default residual gas saturation (Sgr). Can be overridden per call."""
    oil_water_alpha_water_wet: float = 0.01
    """van Genuchten α parameter for oil-water (water-wet) [1/psi]."""
    oil_water_alpha_oil_wet: float = 0.01
    """van Genuchten α parameter for oil-water (oil-wet) [1/psi]."""
    oil_water_n_water_wet: float = 2.0
    """van Genuchten n parameter for oil-water (water-wet)."""
    oil_water_n_oil_wet: float = 2.0
    """van Genuchten n parameter for oil-water (oil-wet)."""
    gas_oil_alpha: float = 0.01
    """van Genuchten α parameter for gas-oil [1/psi]."""
    gas_oil_n: float = 2.0
    """van Genuchten n parameter for gas-oil."""
    wettability: Wettability = Wettability.WATER_WET
    """Wettability type (WATER_WET, OIL_WET, or MIXED_WET)."""
    mixed_wet_water_fraction: float = 0.5
    """Fraction of pore space that is water-wet in mixed-wet systems (0-1)."""
    supports_arrays: bool = attrs.field(init=False, repr=False, default=True)
    """Flag indicating support for array inputs."""

    def get_capillary_pressures(  # type: ignore[override]
        self,
        water_saturation: FloatOrArray,
        oil_saturation: FloatOrArray,
        gas_saturation: FloatOrArray,
        irreducible_water_saturation: typing.Optional[FloatOrArray] = None,
        residual_oil_saturation_water: typing.Optional[FloatOrArray] = None,
        residual_oil_saturation_gas: typing.Optional[FloatOrArray] = None,
        residual_gas_saturation: typing.Optional[FloatOrArray] = None,
    ) -> CapillaryPressures:
        """
        Compute capillary pressures using van Genuchten model.

        Supports both scalar and array inputs.

        :param water_saturation: Water saturation (fraction, 0-1) - scalar or array.
        :param oil_saturation: Oil saturation (fraction, 0-1) - scalar or array.
        :param gas_saturation: Gas saturation (fraction, 0-1) - scalar or array.
        :param irreducible_water_saturation: Optional override for Swc - scalar or array.
        :param residual_oil_saturation_water: Optional override for Sorw - scalar or array.
        :param residual_oil_saturation_gas: Optional override for Sorg - scalar or array.
        :param residual_gas_saturation: Optional override for Sgr - scalar or array.
        :return: `CapillaryPressures` dictionary.
        """
        Swc = (
            irreducible_water_saturation
            if irreducible_water_saturation is not None
            else self.irreducible_water_saturation
        )
        Sorw = (
            residual_oil_saturation_water
            if residual_oil_saturation_water is not None
            else self.residual_oil_saturation_water
        )
        Sorg = (
            residual_oil_saturation_gas
            if residual_oil_saturation_gas is not None
            else self.residual_oil_saturation_gas
        )
        Sgr = (
            residual_gas_saturation
            if residual_gas_saturation is not None
            else self.residual_gas_saturation
        )

        params_missing = []
        if Swc is None:
            params_missing.append("Swc")
        if Sorw is None:
            params_missing.append("Sorw")
        if Sorg is None:
            params_missing.append("Sorg")
        if Sgr is None:
            params_missing.append("Sgr")
        if params_missing:
            raise ValidationError(
                f"Residual saturations must be provided either as model defaults or in the call. "
                f"Missing: {', '.join(params_missing)}"
            )

        pcow, pcgo = compute_van_genuchten_capillary_pressures(
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            irreducible_water_saturation=Swc,  # type: ignore[arg-type]
            residual_oil_saturation_water=Sorw,  # type: ignore[arg-type]
            residual_oil_saturation_gas=Sorg,  # type: ignore[arg-type]
            residual_gas_saturation=Sgr,  # type: ignore[arg-type]
            wettability=self.wettability,
            oil_water_alpha_water_wet=self.oil_water_alpha_water_wet,
            oil_water_alpha_oil_wet=self.oil_water_alpha_oil_wet,
            oil_water_n_water_wet=self.oil_water_n_water_wet,
            oil_water_n_oil_wet=self.oil_water_n_oil_wet,
            gas_oil_alpha=self.gas_oil_alpha,
            gas_oil_n=self.gas_oil_n,
            mixed_wet_water_fraction=self.mixed_wet_water_fraction,
        )
        return CapillaryPressures(oil_water=pcow, gas_oil=pcgo)  # type: ignore[typeddict-item]

    def get_capillary_pressures_derivatives(
        self,
        water_saturation: FloatOrArray,
        oil_saturation: FloatOrArray,
        gas_saturation: FloatOrArray,
        irreducible_water_saturation: typing.Optional[FloatOrArray] = None,
        residual_oil_saturation_water: typing.Optional[FloatOrArray] = None,
        residual_oil_saturation_gas: typing.Optional[FloatOrArray] = None,
        residual_gas_saturation: typing.Optional[FloatOrArray] = None,
    ) -> CapillaryPressureDerivatives:
        """
        Compute the partial derivatives of the van Genuchten oil-water and
        gas-oil capillary pressures with respect to saturation.

        Returns a dictionary containing:

        ```
        (dPcow/dSw, dPcow/dSo, dPcgo/dSg)
        ```

        - `dPcow/dSw`: analytically derived via the chain rule through
        effective water saturation (see
        `_van_genuchten_pc_slope_wrt_effective_saturation`).
        - `dPcow/dSo`: zero - the oil-water capillary pressure depends only
        on water saturation in this model.
        - `dPcgo/dSg`: analytically derived via the chain rule through
        effective gas saturation.

        The van Genuchten model is:
        ```
        Pc = (1/alpha) * [(Se^(-1/m) - 1)^(1/n)]    where m = 1 - 1/n
        ```

        :param water_saturation: Water saturation (scalar or array).
        :param oil_saturation: Oil saturation (scalar or array).
        :param gas_saturation: Gas saturation (scalar or array).
        :param irreducible_water_saturation: Optional override for the
            irreducible (connate) water saturation.
        :param residual_oil_saturation_water: Optional override for the residual
            oil saturation to water flooding.
        :param residual_oil_saturation_gas: Optional override for the residual
            oil saturation to gas flooding.
        :param residual_gas_saturation: Optional override for the residual gas
            saturation.
        :return: `CapillaryPressureDerivatives` dictionary containing the partial derivatives as described above.
        """
        Swc = (
            irreducible_water_saturation
            if irreducible_water_saturation is not None
            else self.irreducible_water_saturation
        )
        Sorw = (
            residual_oil_saturation_water
            if residual_oil_saturation_water is not None
            else self.residual_oil_saturation_water
        )
        Sorg = (
            residual_oil_saturation_gas
            if residual_oil_saturation_gas is not None
            else self.residual_oil_saturation_gas
        )
        Sgr = (
            residual_gas_saturation
            if residual_gas_saturation is not None
            else self.residual_gas_saturation
        )

        params_missing = []
        if Swc is None:
            params_missing.append("Swc")
        if Sorw is None:
            params_missing.append("Sorw")
        if Sorg is None:
            params_missing.append("Sorg")
        if Sgr is None:
            params_missing.append("Sgr")
        if params_missing:
            raise ValidationError(
                f"Residual saturations must be provided either as model defaults or in the call. "
                f"Missing: {', '.join(params_missing)}"
            )

        wettability = self.wettability
        is_scalar = np.isscalar(water_saturation)
        sw = np.atleast_1d(np.asarray(water_saturation, dtype=np.float64))
        sg = np.atleast_1d(np.asarray(gas_saturation, dtype=np.float64))

        mobile_water_range = 1.0 - Swc - Sorw  # type: ignore
        if mobile_water_range > 1e-9:
            se_w = np.clip(
                (sw - Swc) / mobile_water_range,  # type: ignore
                1e-6,
                1.0 - 1e-6,
            )
            if wettability == Wettability.WATER_WET:
                d_pcow_d_se_w = _van_genuchten_pc_slope_wrt_effective_saturation(
                    se_w,
                    self.oil_water_alpha_water_wet,
                    self.oil_water_n_water_wet,
                    sign=+1.0,
                )
            elif wettability == Wettability.OIL_WET:
                d_pcow_d_se_w = _van_genuchten_pc_slope_wrt_effective_saturation(
                    se_w,
                    self.oil_water_alpha_oil_wet,
                    self.oil_water_n_oil_wet,
                    sign=-1.0,
                )
            else:  # MIXED_WET
                water_wet_fraction = self.mixed_wet_water_fraction
                d_ww = _van_genuchten_pc_slope_wrt_effective_saturation(
                    se_w,
                    self.oil_water_alpha_water_wet,
                    self.oil_water_n_water_wet,
                    sign=+1.0,
                )
                d_ow = _van_genuchten_pc_slope_wrt_effective_saturation(
                    se_w,
                    self.oil_water_alpha_oil_wet,
                    self.oil_water_n_oil_wet,
                    sign=-1.0,
                )
                d_pcow_d_se_w = (
                    water_wet_fraction * d_ww + (1.0 - water_wet_fraction) * d_ow
                )

            d_pcow_d_sw = d_pcow_d_se_w / mobile_water_range
        else:
            d_pcow_d_sw = np.zeros_like(sw)

        d_pcow_d_so = np.zeros_like(sw)

        mobile_gas_range = 1.0 - Swc - Sorg - Sgr  # type: ignore
        if mobile_gas_range > 1e-9:
            se_g = np.clip(
                (sg - Sgr) / mobile_gas_range,  # type: ignore
                1e-6,
                1.0 - 1e-6,
            )
            d_pcgo_d_se_g = _van_genuchten_pc_slope_wrt_effective_saturation(
                se_g,
                self.gas_oil_alpha,
                self.gas_oil_n,
                sign=+1.0,
            )
            d_pcgo_d_sg = d_pcgo_d_se_g / mobile_gas_range
        else:
            d_pcgo_d_sg = np.zeros_like(sg)

        if is_scalar:
            return CapillaryPressureDerivatives(
                dPcow_dSw=float(np.ravel(d_pcow_d_sw)[0]),
                dPcow_dSo=float(np.ravel(d_pcow_d_so)[0]),
                dPcgo_dSg=float(np.ravel(d_pcgo_d_sg)[0]),
            )
        return CapillaryPressureDerivatives(
            dPcow_dSw=d_pcow_d_sw,
            dPcow_dSo=d_pcow_d_so,
            dPcgo_dSg=d_pcgo_d_sg,
        )

    def __call__(
        self,
        water_saturation: FloatOrArray,
        oil_saturation: FloatOrArray,
        gas_saturation: FloatOrArray,
        **kwargs: typing.Any,
    ) -> CapillaryPressures:
        """
        Compute capillary pressures using van Genuchten model.

        :param water_saturation: Water saturation (fraction, 0-1) - scalar or array.
        :param oil_saturation: Oil saturation (fraction, 0-1) - scalar or array.
        :param gas_saturation: Gas saturation (fraction, 0-1) - scalar or array.
        :kwarg irreducible_water_saturation: Optional override for Swc.
        :kwarg residual_oil_saturation_water: Optional override for Sorw.
        :kwarg residual_oil_saturation_gas: Optional override for Sorg.
        :kwarg residual_gas_saturation: Optional override for Sgr.
        :return: `CapillaryPressures` dictionary.
        """
        return self.get_capillary_pressures(
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            irreducible_water_saturation=kwargs.get("irreducible_water_saturation"),
            residual_oil_saturation_water=kwargs.get("residual_oil_saturation_water"),
            residual_oil_saturation_gas=kwargs.get("residual_oil_saturation_gas"),
            residual_gas_saturation=kwargs.get("residual_gas_saturation"),
        )

    def derivatives(
        self,
        water_saturation: FloatOrArray,
        oil_saturation: FloatOrArray,
        gas_saturation: FloatOrArray,
        **kwargs: typing.Any,
    ) -> CapillaryPressureDerivatives:
        """
        Compute capillary pressure derivatives using van Genuchten model.

        :param water_saturation: Water saturation (fraction, 0-1) - scalar or array.
        :param oil_saturation: Oil saturation (fraction, 0-1) - scalar or array.
        :param gas_saturation: Gas saturation (fraction, 0-1) - scalar or array.
        :kwarg irreducible_water_saturation: Optional override for Swc.
        :kwarg residual_oil_saturation_water: Optional override for Sorw.
        :kwarg residual_oil_saturation_gas: Optional override for Sorg.
        :kwarg residual_gas_saturation: Optional override for Sgr.
        :return: `CapillaryPressureDerivatives` dictionary containing the partial derivatives as described above.
        """
        return self.get_capillary_pressure_derivatives(
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            irreducible_water_saturation=kwargs.get("irreducible_water_saturation"),
            residual_oil_saturation_water=kwargs.get("residual_oil_saturation_water"),
            residual_oil_saturation_gas=kwargs.get("residual_oil_saturation_gas"),
            residual_gas_saturation=kwargs.get("residual_gas_saturation"),
        )


def compute_leverett_j_capillary_pressures(
    water_saturation: FloatOrArray,
    oil_saturation: FloatOrArray,
    gas_saturation: FloatOrArray,
    irreducible_water_saturation: FloatOrArray,
    residual_oil_saturation_water: FloatOrArray,
    residual_oil_saturation_gas: FloatOrArray,
    residual_gas_saturation: FloatOrArray,
    permeability: FloatOrArray,
    porosity: FloatOrArray,
    oil_water_interfacial_tension: float,
    gas_oil_interfacial_tension: float,
    oil_water_contact_angle: float = 0.0,
    gas_oil_contact_angle: float = 0.0,
    j_function_coefficient: float = 0.5,
    j_function_exponent: float = 0.5,
    mixed_wet_water_fraction: float = 0.5,
    wettability: Wettability = Wettability.WATER_WET,
) -> typing.Tuple[FloatOrArray, FloatOrArray]:
    """
    Computes capillary pressures using Leverett J-function approach.

    Supports both scalar and array inputs (up to 3D).

    The Leverett J-function is a dimensionless correlation that relates capillary pressure
    to rock properties (porosity, permeability) and fluid properties (IFT, contact angle).

    Pc = σ * cos(θ) * sqrt(φ/k) * J(Se)

    where:
    - σ is interfacial tension
    - θ is contact angle
    - φ is porosity
    - k is permeability
    - J(Se) is the dimensionless Leverett J-function (typically fit to data)

    For this implementation, we use a simplified power-law form:
    J(Se) = a * Se^(-b)

    :param water_saturation: Current water saturation (fraction, 0-1) - scalar or array.
    :param oil_saturation: Current oil saturation (fraction, 0-1) - scalar or array.
    :param gas_saturation: Current gas saturation (fraction, 0-1) - scalar or array.
    :param irreducible_water_saturation: Irreducible water saturation (Swc) - scalar or array.
    :param residual_oil_saturation_water: Residual oil saturation during water flooding (Sorw) - scalar or array.
    :param residual_oil_saturation_gas: Residual oil saturation during gas flooding (Sorg) - scalar or array.
    :param residual_gas_saturation: Residual gas saturation (Sgr) - scalar or array.
    :param permeability: Absolute permeability (mD) - scalar or array.
    :param porosity: Porosity (fraction, 0-1) - scalar or array.
    :param oil_water_interfacial_tension: Oil-water interfacial tension (dyne/cm).
    :param gas_oil_interfacial_tension: Gas-oil interfacial tension (dyne/cm).
    :param oil_water_contact_angle: Oil-water contact angle in degrees (0° = water-wet).
    :param gas_oil_contact_angle: Gas-oil contact angle in degrees (0° = oil-wet).
    :param j_function_coefficient: Empirical coefficient 'a' in J(Se) = a * Se^(-b). Fit to core data (can be tuned to match experimental data).
    :param j_function_exponent: Empirical exponent 'b' in J(Se) = a * Se^(-b). Fit to core data (can be tuned to match experimental data).
    :param mixed_wet_water_fraction: Fraction of pore space that is water-wet in mixed-wet systems (0-1).
    :param wettability: Wettability type (affects sign of capillary pressure).
    :return: Tuple of (oil_water_capillary_pressure, gas_oil_capillary_pressure) in psi.
    """
    # Convert to arrays for vectorized operations
    sw = np.atleast_1d(water_saturation)
    so = np.atleast_1d(oil_saturation)
    sg = np.atleast_1d(gas_saturation)
    Swc = np.atleast_1d(irreducible_water_saturation)
    Sorw = np.atleast_1d(residual_oil_saturation_water)
    Sorg = np.atleast_1d(residual_oil_saturation_gas)
    Sgr = np.atleast_1d(residual_gas_saturation)
    perm = np.atleast_1d(permeability)
    phi = np.atleast_1d(porosity)

    # Broadcast all arrays to same shape
    sw, so, sg, Swc, Sorw, Sorg, Sgr, perm, phi = np.broadcast_arrays(
        sw, so, sg, Swc, Sorw, Sorg, Sgr, perm, phi
    )

    # Validate saturations
    if np.any((sw < 0) | (sw > 1) | (so < 0) | (so > 1) | (sg < 0) | (sg > 1)):
        raise ValidationError("Saturations must be between 0 and 1.")
    if np.any(perm < 0.0):
        raise ValidationError("Permeability must be positive.")
    if np.any((phi < 0.0) | (phi > 1.0)):
        raise ValidationError("Porosity must be between 0 and 1.")

    # Normalize saturations if they do not sum to 1
    total_saturation = sw + so + sg
    needs_norm = (np.abs(total_saturation - 1.0) > 1e-6) & (total_saturation > 0.0)
    if np.any(needs_norm):
        sw = np.where(needs_norm, sw / total_saturation, sw)
        so = np.where(needs_norm, so / total_saturation, so)
        sg = np.where(needs_norm, sg / total_saturation, sg)

    # Effective pore spaces
    # Water mobile pore space: excludes connate water and residual oil (to water)
    # Gas residual doesn't affect water-oil mobile space
    total_mobile_pore_space_water = 1.0 - Swc - Sorw
    # Gas mobile pore space: excludes connate water, residual oil (to gas), and residual gas
    total_mobile_pore_space_gas = 1.0 - Swc - Sorg - Sgr

    # Convert contact angles to radians
    theta_oil_water_contact_anglerad = np.deg2rad(oil_water_contact_angle)
    theta_go_rad = np.deg2rad(gas_oil_contact_angle)

    # Leverett scaling factor: sqrt(φ/k)
    # Check for zero/invalid values
    valid_rock = (perm > 0) & (phi > 0)
    leverett_factor = np.where(valid_rock, np.sqrt(phi / perm), 0.0)

    # Pcow (Po - Pw)
    oil_water_capillary_pressure = np.zeros_like(sw)
    valid_water = (total_mobile_pore_space_water > 1e-9) & valid_rock

    if np.any(valid_water):
        effective_water_saturation = np.where(
            valid_water, (sw - Swc) / total_mobile_pore_space_water, 0.0
        )
        effective_water_saturation = np.clip(
            effective_water_saturation, 1e-6, 1.0 - 1e-6
        )

        # J-function value
        j_value_ow = j_function_coefficient * (
            effective_water_saturation ** (-j_function_exponent)
        )

        # Capillary pressure (converting dyne/cm to psi: 1 dyne/cm = 0.00145038 psi)
        # Pc = σ * cos(θ) * sqrt(φ/k) * J(Se)
        pc_ow = (
            oil_water_interfacial_tension
            * 4.725  # Convert to psi
            * np.cos(theta_oil_water_contact_anglerad)
            * leverett_factor
            * j_value_ow
        )

        # Apply wettability sign convention
        if wettability == Wettability.WATER_WET:
            oil_water_capillary_pressure = np.where(valid_water, pc_ow, 0.0)
        elif wettability == Wettability.OIL_WET:
            oil_water_capillary_pressure = np.where(valid_water, -pc_ow, 0.0)
        else:  # MIXED_WET
            # Weighted average: positive (water-wet) and negative (oil-wet) contributions
            mixed_pc_ow = mixed_wet_water_fraction * pc_ow + (
                1.0 - mixed_wet_water_fraction
            ) * (-pc_ow)
            oil_water_capillary_pressure = np.where(valid_water, mixed_pc_ow, 0.0)

    # Pcgo (Pg - Po)
    gas_oil_capillary_pressure = np.zeros_like(sg)
    valid_gas = (total_mobile_pore_space_gas > 1e-9) & valid_rock

    if np.any(valid_gas):
        effective_gas_saturation = np.where(
            valid_gas, (sg - Sgr) / total_mobile_pore_space_gas, 0.0
        )
        effective_gas_saturation = np.clip(effective_gas_saturation, 1e-6, 1.0 - 1e-6)

        # J-function value
        j_value_go = j_function_coefficient * (
            effective_gas_saturation ** (-j_function_exponent)
        )

        # Capillary pressure
        pcgo = (
            gas_oil_interfacial_tension
            * 4.725  # Convert to psi
            * np.cos(theta_go_rad)
            * leverett_factor
            * j_value_go
        )
        gas_oil_capillary_pressure = np.where(valid_gas, pcgo, 0.0)

    # Return scalars if inputs were scalars
    is_scalar = np.isscalar(water_saturation)
    if is_scalar:
        return float(oil_water_capillary_pressure), float(gas_oil_capillary_pressure)
    return oil_water_capillary_pressure, gas_oil_capillary_pressure


@capillary_pressure_table
@attrs.frozen
class LeverettJCapillaryPressureModel(
    CapillaryPressureTable,
    load_exclude={"supports_arrays"},
    dump_exclude={"supports_arrays"},
):
    """
    Leverett J-function capillary pressure model for three-phase systems.

    Uses dimensionless J-function correlation to relate capillary pressure
    to rock and fluid properties: Pc = σ * cos(θ) * sqrt(φ/k) * J(Se)

    Useful when capillary pressure data needs to be scaled across different
    rock types or fluid systems.
    """

    __type__ = "leverett_j_capillary_pressure_model"

    irreducible_water_saturation: typing.Optional[float] = None
    """Default irreducible water saturation (Swc). Can be overridden per call."""
    residual_oil_saturation_water: typing.Optional[float] = None
    """Default residual oil saturation after water flood (Sorw). Can be overridden per call."""
    residual_oil_saturation_gas: typing.Optional[float] = None
    """Default residual oil saturation after gas flood (Sorg). Can be overridden per call."""
    residual_gas_saturation: typing.Optional[float] = None
    """Default residual gas saturation (Sgr). Can be overridden per call."""
    permeability: float = 100.0
    """Absolute permeability (mD)."""
    porosity: float = 0.2
    """Porosity (fraction, 0-1)."""
    oil_water_interfacial_tension: float = 30.0
    """Oil-water interfacial tension (dyne/cm)."""
    gas_oil_interfacial_tension: float = 20.0
    """Gas-oil interfacial tension (dyne/cm)."""
    oil_water_contact_angle: float = 0.0
    """Oil-water contact angle in degrees (0° = water-wet, 180° = oil-wet)."""
    gas_oil_contact_angle: float = 0.0
    """Gas-oil contact angle in degrees (0° = oil-wet to gas)."""
    mixed_wet_water_fraction: float = 0.5
    """Fraction of pore space that is water-wet in mixed-wet systems (0-1)."""
    wettability: Wettability = Wettability.WATER_WET
    """Wettability type (affects sign convention)."""
    j_function_coefficient: float = 0.5
    """Empirical coefficient 'a' in J(Se) = a * Se^(-b). Fit to core data (can be tuned to match experimental data)."""
    j_function_exponent: float = 0.5
    """Empirical exponent 'b' in J(Se) = a * Se^(-b). Fit to core data (can be tuned to match experimental data)."""
    supports_arrays: bool = attrs.field(init=False, repr=False, default=True)
    """Flag indicating support for array inputs."""

    def get_capillary_pressures(  # type: ignore[override]
        self,
        water_saturation: FloatOrArray,
        oil_saturation: FloatOrArray,
        gas_saturation: FloatOrArray,
        irreducible_water_saturation: typing.Optional[FloatOrArray] = None,
        residual_oil_saturation_water: typing.Optional[FloatOrArray] = None,
        residual_oil_saturation_gas: typing.Optional[FloatOrArray] = None,
        residual_gas_saturation: typing.Optional[FloatOrArray] = None,
    ) -> CapillaryPressures:
        """
        Compute capillary pressures using Leverett J-function.

        Supports both scalar and array inputs.

        :param water_saturation: Water saturation (fraction, 0-1) - scalar or array.
        :param oil_saturation: Oil saturation (fraction, 0-1) - scalar or array.
        :param gas_saturation: Gas saturation (fraction, 0-1) - scalar or array.
        :param irreducible_water_saturation: Optional override for Swc - scalar or array.
        :param residual_oil_saturation_water: Optional override for Sorw - scalar or array.
        :param residual_oil_saturation_gas: Optional override for Sorg - scalar or array.
        :param residual_gas_saturation: Optional override for Sgr - scalar or array.
        :return: `CapillaryPressures` dictionary.
        """
        Swc = (
            irreducible_water_saturation
            if irreducible_water_saturation is not None
            else self.irreducible_water_saturation
        )
        Sorw = (
            residual_oil_saturation_water
            if residual_oil_saturation_water is not None
            else self.residual_oil_saturation_water
        )
        Sorg = (
            residual_oil_saturation_gas
            if residual_oil_saturation_gas is not None
            else self.residual_oil_saturation_gas
        )
        Sgr = (
            residual_gas_saturation
            if residual_gas_saturation is not None
            else self.residual_gas_saturation
        )

        params_missing = []
        if Swc is None:
            params_missing.append("Swc")
        if Sorw is None:
            params_missing.append("Sorw")
        if Sorg is None:
            params_missing.append("Sorg")
        if Sgr is None:
            params_missing.append("Sgr")
        if params_missing:
            raise ValidationError(
                f"Residual saturations must be provided either as model defaults or in the call. "
                f"Missing: {', '.join(params_missing)}"
            )

        pcow, pcgo = compute_leverett_j_capillary_pressures(
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            irreducible_water_saturation=Swc,  # type: ignore[arg-type]
            residual_oil_saturation_water=Sorw,  # type: ignore[arg-type]
            residual_oil_saturation_gas=Sorg,  # type: ignore[arg-type]
            residual_gas_saturation=Sgr,  # type: ignore[arg-type]
            permeability=self.permeability,
            porosity=self.porosity,
            oil_water_interfacial_tension=self.oil_water_interfacial_tension,
            gas_oil_interfacial_tension=self.gas_oil_interfacial_tension,
            oil_water_contact_angle=self.oil_water_contact_angle,
            gas_oil_contact_angle=self.gas_oil_contact_angle,
            j_function_coefficient=self.j_function_coefficient,
            j_function_exponent=self.j_function_exponent,
            mixed_wet_water_fraction=self.mixed_wet_water_fraction,
            wettability=self.wettability,
        )
        return CapillaryPressures(oil_water=pcow, gas_oil=pcgo)  # type: ignore[typeddict-item]

    def get_capillary_pressures_derivatives(
        self,
        water_saturation: FloatOrArray,
        oil_saturation: FloatOrArray,
        gas_saturation: FloatOrArray,
        irreducible_water_saturation: typing.Optional[FloatOrArray] = None,
        residual_oil_saturation_water: typing.Optional[FloatOrArray] = None,
        residual_oil_saturation_gas: typing.Optional[FloatOrArray] = None,
        residual_gas_saturation: typing.Optional[FloatOrArray] = None,
    ) -> CapillaryPressureDerivatives:
        """
        Compute the partial derivatives of the Leverett J-function oil-water and
        gas-oil capillary pressures with respect to saturation.

        Returns a dictionary containing:

        ```
        (dPcow/dSw, dPcow/dSo, dPcgo/dSg)
        ```

        - `dPcow/dSw`: analytically derived via the chain rule through
        effective water saturation.
        - `dPcow/dSo`: zero - the oil-water capillary pressure depends only on
        water saturation in this model.
        - `dPcgo/dSg`: analytically derived via the chain rule through
        effective gas saturation.

        The Leverett J-function model is:

        ```
        Pc = sigma * cos(theta) * sqrt(porosity / permeability) * J(Se)
        J(Se) = j_function_coefficient * Se^(-j_function_exponent)
        ```

        Analytical derivative:

        ```
        dPc/dSe = sigma * cos(theta) * sqrt(phi/k)
                * j_function_coefficient * (-j_function_exponent)
                * Se^(-j_function_exponent - 1)
        dPc/dSw = dPc/dSe / mobile_water_saturation_range
        ```

        Unit conversion from dyne/cm to psi uses the factor 4.725.

        :param water_saturation: Water saturation (scalar or array).
        :param oil_saturation: Oil saturation (scalar or array).
        :param gas_saturation: Gas saturation (scalar or array).
        :param irreducible_water_saturation: Optional override for the
            irreducible (connate) water saturation.
        :param residual_oil_saturation_water: Optional override for the residual
            oil saturation to water flooding.
        :param residual_oil_saturation_gas: Optional override for the residual
            oil saturation to gas flooding.
        :param residual_gas_saturation: Optional override for the residual gas
            saturation.
        :return: `CapillaryPressureDerivatives` dictionary containing the partial derivatives as described above.
        """
        Swc = (
            irreducible_water_saturation
            if irreducible_water_saturation is not None
            else self.irreducible_water_saturation
        )
        Sorw = (
            residual_oil_saturation_water
            if residual_oil_saturation_water is not None
            else self.residual_oil_saturation_water
        )
        Sorg = (
            residual_oil_saturation_gas
            if residual_oil_saturation_gas is not None
            else self.residual_oil_saturation_gas
        )
        Sgr = (
            residual_gas_saturation
            if residual_gas_saturation is not None
            else self.residual_gas_saturation
        )

        params_missing = []
        if Swc is None:
            params_missing.append("Swc")
        if Sorw is None:
            params_missing.append("Sorw")
        if Sorg is None:
            params_missing.append("Sorg")
        if Sgr is None:
            params_missing.append("Sgr")
        if params_missing:
            raise ValidationError(
                f"Residual saturations must be provided either as model defaults or in the call. "
                f"Missing: {', '.join(params_missing)}"
            )

        wettability = self.wettability
        absolute_permeability = self.permeability
        porosity = self.porosity
        oil_water_interfacial_tension = self.oil_water_interfacial_tension
        gas_oil_interfacial_tension = self.gas_oil_interfacial_tension
        oil_water_contact_angle_deg = self.oil_water_contact_angle
        gas_oil_contact_angle_deg = self.gas_oil_contact_angle
        j_function_coefficient = self.j_function_coefficient
        j_function_exponent = self.j_function_exponent
        mixed_wet_water_fraction = self.mixed_wet_water_fraction

        is_scalar = np.isscalar(water_saturation)
        sw = np.atleast_1d(np.asarray(water_saturation, dtype=np.float64))
        sg = np.atleast_1d(np.asarray(gas_saturation, dtype=np.float64))

        leverett_rock_factor = (
            np.sqrt(porosity / absolute_permeability)
            if absolute_permeability > 0.0 and porosity > 0.0
            else 0.0
        )
        dyne_per_cm_to_psi = 4.725  # matches the forward model

        mobile_water_range = 1.0 - Swc - Sorw  # type: ignore
        if mobile_water_range > 1e-9 and leverett_rock_factor > 0.0:
            se_w = np.clip(
                (sw - Swc) / mobile_water_range,  # type: ignore
                1e-6,
                1.0 - 1e-6,
            )
            d_j_d_se_w = (
                -j_function_coefficient
                * j_function_exponent
                * (se_w ** (-j_function_exponent - 1.0))
            )
            cos_ow = np.cos(np.deg2rad(oil_water_contact_angle_deg))
            ow_scale = (
                oil_water_interfacial_tension
                * dyne_per_cm_to_psi
                * cos_ow
                * leverett_rock_factor
            )
            if wettability == Wettability.WATER_WET:
                wettability_sign = 1.0
            elif wettability == Wettability.OIL_WET:
                wettability_sign = -1.0
            else:  # MIXED_WET
                wettability_sign = 2.0 * mixed_wet_water_fraction - 1.0
            d_pcow_d_sw = wettability_sign * ow_scale * d_j_d_se_w / mobile_water_range
        else:
            d_pcow_d_sw = np.zeros_like(sw)

        d_pcow_d_so = np.zeros_like(sw)

        mobile_gas_range = 1.0 - Swc - Sorg - Sgr  # type: ignore
        if mobile_gas_range > 1e-9 and leverett_rock_factor > 0.0:
            se_g = np.clip(
                (sg - Sgr) / mobile_gas_range,  # type: ignore
                1e-6,
                1.0 - 1e-6,
            )
            d_j_d_se_g = (
                -j_function_coefficient
                * j_function_exponent
                * (se_g ** (-j_function_exponent - 1.0))
            )
            cos_go = np.cos(np.deg2rad(gas_oil_contact_angle_deg))
            go_scale = (
                gas_oil_interfacial_tension
                * dyne_per_cm_to_psi
                * cos_go
                * leverett_rock_factor
            )
            d_pcgo_d_sg = go_scale * d_j_d_se_g / mobile_gas_range
        else:
            d_pcgo_d_sg = np.zeros_like(sg)

        if is_scalar:
            return CapillaryPressureDerivatives(
                dPcow_dSw=float(np.ravel(d_pcow_d_sw)[0]),
                dPcow_dSo=float(np.ravel(d_pcow_d_so)[0]),
                dPcgo_dSg=float(np.ravel(d_pcgo_d_sg)[0]),
            )
        return CapillaryPressureDerivatives(
            dPcow_dSw=d_pcow_d_sw,
            dPcow_dSo=d_pcow_d_so,
            dPcgo_dSg=d_pcgo_d_sg,
        )

    def __call__(
        self,
        water_saturation: FloatOrArray,
        oil_saturation: FloatOrArray,
        gas_saturation: FloatOrArray,
        **kwargs: typing.Any,
    ) -> CapillaryPressures:
        """
        Compute capillary pressures using Leverett J-function.

        :param water_saturation: Water saturation (fraction, 0-1) - scalar or array.
        :param oil_saturation: Oil saturation (fraction, 0-1) - scalar or array.
        :param gas_saturation: Gas saturation (fraction, 0-1) - scalar or array.
        :kwarg irreducible_water_saturation: Optional override for Swc.
        :kwarg residual_oil_saturation_water: Optional override for Sorw.
        :kwarg residual_oil_saturation_gas: Optional override for Sorg.
        :kwarg residual_gas_saturation: Optional override for Sgr.
        :return: `CapillaryPressures` dictionary.
        """
        return self.get_capillary_pressures(
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            irreducible_water_saturation=kwargs.get("irreducible_water_saturation"),
            residual_oil_saturation_water=kwargs.get("residual_oil_saturation_water"),
            residual_oil_saturation_gas=kwargs.get("residual_oil_saturation_gas"),
            residual_gas_saturation=kwargs.get("residual_gas_saturation"),
        )

    def derivatives(
        self,
        water_saturation: FloatOrArray,
        oil_saturation: FloatOrArray,
        gas_saturation: FloatOrArray,
        **kwargs: typing.Any,
    ) -> CapillaryPressureDerivatives:
        """
        Compute capillary pressure derivatives using Leverett J-function.

        :param water_saturation: Water saturation (fraction, 0-1) - scalar or array.
        :param oil_saturation: Oil saturation (fraction, 0-1) - scalar or array.
        :param gas_saturation: Gas saturation (fraction, 0-1) - scalar or array.
        :kwarg irreducible_water_saturation: Optional override for Swc.
        :kwarg residual_oil_saturation_water: Optional override for Sorw.
        :kwarg residual_oil_saturation_gas: Optional override for Sorg.
        :kwarg residual_gas_saturation: Optional override for Sgr.
        :return: `CapillaryPressureDerivatives` dictionary containing the partial derivatives as described above.
        """
        return self.get_capillary_pressure_derivatives(
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            irreducible_water_saturation=kwargs.get("irreducible_water_saturation"),
            residual_oil_saturation_water=kwargs.get("residual_oil_saturation_water"),
            residual_oil_saturation_gas=kwargs.get("residual_oil_saturation_gas"),
            residual_gas_saturation=kwargs.get("residual_gas_saturation"),
        )

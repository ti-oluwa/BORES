"""Core well calculations and utilities."""

import logging
import typing

import attrs
import numba
import numpy as np

from bores.constants import c
from bores.correlations.arrays import (
    compute_gas_compressibility as compute_gas_compressibility_vectorized,
)
from bores.correlations.arrays import (
    compute_gas_compressibility_factor as compute_gas_compressibility_factor_vectorized,
)
from bores.correlations.arrays import (
    compute_gas_density as compute_gas_density_vectorized,
)
from bores.correlations.arrays import (
    compute_gas_formation_volume_factor as compute_gas_formation_volume_factor_vectorized,
)
from bores.correlations.arrays import (
    compute_gas_free_water_formation_volume_factor as compute_gas_free_water_formation_volume_factor_vectorized,
)
from bores.correlations.arrays import (
    compute_gas_viscosity as compute_gas_viscosity_vectorized,
)
from bores.correlations.arrays import (
    compute_water_compressibility as compute_water_compressibility_vectorized,
)
from bores.correlations.arrays import (
    compute_water_density as compute_water_density_vectorized,
)
from bores.correlations.arrays import (
    compute_water_formation_volume_factor as compute_water_formation_volume_factor_vectorized,
)
from bores.correlations.arrays import (
    compute_water_viscosity as compute_water_viscosity_vectorized,
)
from bores.correlations.core import (
    compute_gas_compressibility,
    compute_gas_compressibility_factor,
    compute_gas_density,
    compute_gas_formation_volume_factor,
    compute_gas_free_water_formation_volume_factor,
    compute_gas_viscosity,
    compute_water_compressibility,
    compute_water_density,
    compute_water_formation_volume_factor,
    compute_water_viscosity,
    fahrenheit_to_rankine,
)
from bores.errors import ComputationError, ValidationError
from bores.fluids import Fluid
from bores.tables.pseudo_pressure import PseudoPressureTable
from bores.tables.pvt import PVTTables
from bores.types import (
    FloatOrArray,
    FluidPhase,
    Orientation,
    ThreeDimensions,
    TwoDimensions,
)

logger = logging.getLogger(__name__)

__all__ = [
    "InjectedFluid",
    "ProducedFluid",
    "WellFluid",
    "compute_2D_effective_drainage_radius",
    "compute_3D_effective_drainage_radius",
    "compute_average_compressibility_factor",
    "compute_gas_well_rate",
    "compute_oil_well_rate",
    "compute_required_bhp_for_gas_rate",
    "compute_required_bhp_for_oil_rate",
    "compute_well_index",
]


@numba.njit(cache=True)
def compute_well_index(
    permeability: float,
    interval_thickness: float,
    wellbore_radius: float,
    effective_drainage_radius: float,
    skin_factor: float = 0.0,
    net_to_gross: float = 1.0,
) -> float:
    """
    Compute the well index for a given well using the Peaceman equation.

    The well index is a measure of the productivity of a well, defined as the ratio of the
    well flow rate to the pressure drop across the well.

    The formula for the well index is:

    W = (k * h * net_to_gross) / (ln(re/rw) + s)

    where:
        - W is the well index (md*ft)
        - k is the absolute permeability of the reservoir rock (mD)
        - h is the thickness of the reservoir interval (ft)
        - re is the effective drainage radius (ft)
        - rw is the wellbore radius (ft)
        - s is the skin factor (dimensionless, default is 0)

    :param permeability: Absolute permeability of the reservoir rock (mD).
    :param interval_thickness: Thickness of the reservoir interval (ft).
    :param wellbore_radius: Radius of the wellbore (ft).
    :param effective_drainage_radius: Effective drainage radius (ft).
    :param skin_factor: Skin factor for the well (dimensionless, default is 0).
    :param net_to_gross: Net-to-gross ratio of the reservoir interval (default is 1.0).
        Reduces the effective thickness of the reservoir contributing to flow into the well.
    :return: The well index in (mD*ft).
    """
    return (permeability * interval_thickness * net_to_gross) / (
        np.log(effective_drainage_radius / wellbore_radius) + skin_factor
    )


@numba.njit(cache=True)
def compute_3D_effective_drainage_radius(
    interval_thickness: ThreeDimensions,
    permeability: ThreeDimensions,
    well_orientation: Orientation,
) -> float:
    """
    Compute the effective drainage radius for a well in a 3D reservoir model
    using Peaceman's (1983) anisotropic effective drainage radius formula.

    Peaceman's formula for a well aligned along a given axis uses the two
    grid dimensions and permeabilities *perpendicular* to the wellbore.
    For a well in the Z-direction (standard vertical well):

    ```
    r_o = 0.28 * sqrt[ (ky/kx)^0.5 * dx^2 + (kx/ky)^0.5 * dy^2 ]
                        -----------------------------------------
                                (ky/kx)^0.25 + (kx/ky)^0.25
    ```

    The same pattern applies by cyclic permutation for X- and Y-oriented
    (horizontal) wells, substituting the two directions perpendicular to
    the wellbore axis in each case.

    For the isotropic case (kx = ky = k) this reduces to:

    ```
    r_o = 0.28 * sqrt(dx^2 + dy^2) / 2^0.5
        = 0.198 * sqrt(dx^2 + dy^2)
    ```

    which matches the standard Peaceman result for uniform square grids
    (r_o = 0.2 * dx when dx = dy).

    If either perpendicular permeability is zero (e.g. a tight layer or shale
    streak), the well has no drainage capacity in that plane and 0.0 is
    returned. The caller should treat a zero return value as an indicator
    that no well index should be assigned to this interval.

    :param interval_thickness: Tuple of cell dimensions (dx, dy, dz) in ft.
    :param permeability: Tuple of permeabilities (kx, ky, kz) in mD.
    :param well_orientation: Wellbore axis. `Orientation.X` for a horizontal
        well along x, `Orientation.Y` for a horizontal well along y, or
        `Orientation.Z` for a standard vertical well.
    :return: Effective drainage (Peaceman) radius in ft, or 0.0 if either
        perpendicular permeability is zero.
    """
    dx, dy, dz = interval_thickness
    kx, ky, kz = permeability

    if well_orientation == Orientation.X:
        # Wellbore along x: perpendicular plane is y-z
        if ky <= 0.0 or kz <= 0.0:
            return 0.0
        r1, r2 = ky / kz, kz / ky
        numerator = np.sqrt(r1) * dy**2 + np.sqrt(r2) * dz**2
        denominator = r1**0.25 + r2**0.25
    elif well_orientation == Orientation.Y:
        # Wellbore along y: perpendicular plane is x-z
        if kx <= 0.0 or kz <= 0.0:
            return 0.0
        r1, r2 = kx / kz, kz / kx
        numerator = np.sqrt(r1) * dx**2 + np.sqrt(r2) * dz**2
        denominator = r1**0.25 + r2**0.25
    elif well_orientation == Orientation.Z:
        # Wellbore along z: perpendicular plane is x-y (standard vertical well)
        if kx <= 0.0 or ky <= 0.0:
            return 0.0
        r1, r2 = ky / kx, kx / ky
        numerator = np.sqrt(r1) * dx**2 + np.sqrt(r2) * dy**2
        denominator = r1**0.25 + r2**0.25
    else:
        raise ValidationError(f"Invalid well orientation {well_orientation}")

    return 0.28 * np.sqrt(numerator / denominator)


@numba.njit(cache=True)
def compute_2D_effective_drainage_radius(
    interval_thickness: TwoDimensions, permeability: TwoDimensions
) -> float:
    """
    Compute the effective drainage radius for a well in a 2D reservoir model
    using Peaceman's (1983) anisotropic formula.

    In a 2D x-y model the wellbore is always perpendicular to the grid plane
    (i.e. implicitly Z-oriented), so there is a single drainage radius
    expression that uses both in-plane grid dimensions and permeabilities:

    ```
    r_o = 0.28 * sqrt[ (ky/kx)^0.5 * dx^2 + (kx/ky)^0.5 * dy^2 ]
                        -----------------------------------------
                                (ky/kx)^0.25 + (kx/ky)^0.25
    ```

    This is identical to the Z-orientation case of `compute_3D_effective_drainage_radius`.

    For the isotropic case (kx = ky) this reduces to:

    ```
    r_o = 0.28 * sqrt(dx^2 + dy^2) / sqrt(2)
    ```

    which gives r_o = 0.2 * dx for a uniform square grid (dx = dy),
    matching the classic Peaceman result.

    If either permeability is zero or negative (e.g. a tight or shale cell),
    the well has no drainage capacity and 0.0 is returned.  The caller should
    treat a zero return value as an indicator that no well index should be
    assigned to this interval.

    :param interval_thickness: Tuple of cell dimensions (dx, dy) in ft.
    :param permeability: Tuple of permeabilities (kx, ky) in mD.
    :return: Effective drainage (Peaceman) radius in ft, or 0.0 if either
        permeability is zero or negative.
    """
    kx, ky = permeability
    if kx <= 0.0 or ky <= 0.0:
        return 0.0

    dx, dy = interval_thickness[0], interval_thickness[1]
    r1, r2 = ky / kx, kx / ky
    numerator = np.sqrt(r1) * dx**2 + np.sqrt(r2) * dy**2
    denominator = r1**0.25 + r2**0.25
    return 0.28 * np.sqrt(numerator / denominator)


@numba.njit(cache=True)
def compute_oil_well_rate(
    well_index: float,
    pressure: float,
    bottom_hole_pressure: float,
    phase_mobility: float,
    fluid_compressibility: typing.Optional[float] = None,
    incompressibility_threshold: float = 1e-6,
) -> float:
    """
    Compute the well rate at reservoir conditions using the Darcy well equation.

    Returns the rate at **reservoir conditions** (rb/day).  The caller is
    responsible for converting to stock-tank conditions by dividing by the
    formation volume factor (Bo, Bw, or Bg), consistent with the rest of the
    simulator.

    Because `phase_mobility = kr / mu` (no FVF term), and because Bo is
    applied *downstream*, the linear Darcy formula is the correct formulation
    for black-oil phases.  Bo already encodes fluid compressibility relative to
    surface conditions via `co = -(1/Bo) * (dBo/dP)`, so applying an
    additional exponential compressibility correction on top of a downstream Bo
    division would double-count the same physical effect.

    The slightly-compressible exponential correction is retained only for
    phases where no FVF is applied downstream (e.g. a standalone single-phase
    water or gas module that does not track Bw/Bg explicitly), and only when
    `c * dP` is large enough to produce a materially different result from
    the linear formula (> 1%) but still within the range where the exponential
    density model is physically valid (c * dP <= 0.7).

    Sign convention:
    - Negative rate indicates production (BHP < reservoir pressure).
    - Positive rate  indicates injection  (BHP > reservoir pressure).

    Formula (linear / incompressible):

        `Q = 7.08e-3 * W * M * (P_bhp - P)`

    Formula (slightly compressible, exponential):

        `Q = 7.08e-3 * W * M * [exp(c * dP) - 1] / c`

    where `dP = P_bhp - P`.

    :param well_index: Well index (mD*ft).
    :param pressure: Reservoir cell pressure at the perforation interval (psi).
    :param bottom_hole_pressure: Well bottom-hole pressure (psi).
    :param phase_mobility: Phase relative mobility kr/mu (md/cP).
        Must not include the FVF term; Bo/Bw/Bg is applied by the caller.
    :param fluid_compressibility: Fluid compressibility (psi^-1). When
        provided and above `incompressibility_threshold`, the exponential
        correction is evaluated. Pass `None` (default) to always use the
        linear formula, which is correct for black-oil phases.
    :param incompressibility_threshold: Minimum compressibility (psi^-1) below
        which the fluid is treated as incompressible and the linear formula is
        used regardless. Default `1e-6 psi^-1`.
    :return: Well rate at reservoir conditions (bbl/day). Negative for
        production, positive for injection.
    """
    if well_index < 0:
        raise ValidationError("Well index must be a positive value.")
    if phase_mobility < 0:
        raise ValidationError("Phase mobility must be a positive value.")

    if well_index == 0.0 or phase_mobility == 0.0:
        return 0.0

    pressure_difference = bottom_hole_pressure - pressure
    is_compressible = (
        fluid_compressibility is not None
        and fluid_compressibility >= incompressibility_threshold
    )
    if is_compressible:
        argument = fluid_compressibility * pressure_difference  # type: ignore
        # Apply exponential correction only when c*dP is large enough to
        # produce a >1% deviation from the linear result, but small enough
        # that the slightly-compressible exponential model remains valid.
        # For black-oil oil/water with Bo tracked downstream this branch is
        # essentially never entered (typical c*dP << 0.01); it is provided
        # for single-phase modules that do not apply a downstream FVF.
        #
        # c*dP < 1e-4 : exp(x)-1 ~ x, numerically identical to linear.
        # c*dP > 0.7  : model has broken down at this drawdown magnitude;
        #               fall back to linear; Bo correction handles the rest.
        if 1e-4 < abs(argument) <= 0.7:
            return (
                7.08e-3
                * well_index
                * phase_mobility
                * (np.exp(argument) - 1.0)
                / fluid_compressibility
            )

    return 7.08e-3 * well_index * phase_mobility * pressure_difference


@numba.njit(cache=True)
def compute_required_bhp_for_oil_rate(
    target_rate: float,
    well_index: float,
    pressure: float,
    phase_mobility: float,
    fluid_compressibility: typing.Optional[float] = None,
    incompressibility_threshold: float = 1e-6,
) -> float:
    """
    Compute the bottom-hole pressure required to achieve a target well rate.

    This is the exact algebraic inverse of `compute_oil_well_rate`.
    The same FVF and compressibility convention applies: `target_rate` must
    be at reservoir conditions (rb/day), and `phase_mobility = kr / mu`
    with no FVF term.

    Sign convention:
    - Negative `target_rate` indicates production (returned BHP < reservoir pressure).
    - Positive `target_rate`  indicates injection  (returned BHP > reservoir pressure).

    Inverse formula (linear / incompressible):

        `P_bhp = P + Q / (7.08e-3 * W * M)`

    Inverse formula (slightly compressible, exponential):

        `P_bhp = P + ln(Q * c / (7.08e-3 * W * M) + 1) / c`

    The logarithmic inverse is the algebraically exact inverse of the
    exponential forward formula, not an approximation.  The same c * dP
    validity window used in `compute_oil_well_rate` is enforced here
    by checking that the `ln` argument is positive and within a physically
    meaningful range before applying it, guaranteeing that the two functions
    are exact inverses of each other within the same operating regime.

    :param target_rate: Target well rate at reservoir conditions (bbl/day).
        Negative for production, positive for injection.
    :param well_index: Well index (mD*ft).
    :param pressure: Reservoir cell pressure at the perforation interval (psi).
    :param phase_mobility: Phase relative mobility kr/mu (md/cP).
        Must not include the FVF term.
    :param fluid_compressibility: Fluid compressibility (psi^-1).  When
        provided and above `incompressibility_threshold`, the logarithmic
        inverse is evaluated. Pass `None` to always use the linear inverse.
    :param incompressibility_threshold: Minimum compressibility (psi^-1) below
        which the fluid is treated as incompressible.  Default 1e-6 psi^-1.
    :return: Required bottom-hole pressure (psi).
    """
    if well_index < 0:
        raise ValidationError("Well index must be a positive value.")
    if phase_mobility < 0:
        raise ValidationError("Phase mobility must be a positive value.")

    if well_index == 0.0 or phase_mobility == 0.0:
        return float(pressure)

    denominator = 7.08e-3 * well_index * phase_mobility

    is_compressible = (
        fluid_compressibility is not None
        and fluid_compressibility >= incompressibility_threshold
    )
    if is_compressible:
        # Exact inverse of Q = (7.08e-3 * W * M / c) * [exp(c * dP) - 1]:
        #   c * dP = ln(Q * c / (7.08e-3 * W * M) + 1)
        #   dP     = ln(argument) / c
        argument = target_rate * fluid_compressibility / denominator + 1.0  # type: ignore

        # argument must be > 0 for ln() to be defined.
        # argument <= 0 means the target rate exceeds what the
        # compressibility-driven supply can deliver at this pressure;
        # fall through to the linear formula as a conservative limit.
        #
        # argument > 1e10 implies a c*dP outside the valid range of the
        # slightly-compressible model; fall through to linear.
        if 0.0 < argument <= 1e10:
            delta_p = np.log(argument) / fluid_compressibility
            # Enforce the same c*dP <= 0.7 validity window as the forward
            # function so the two are guaranteed to be exact inverses within
            # the same operating regime.
            if abs(fluid_compressibility * delta_p) <= 0.7:
                return float(pressure + delta_p)

    # Linear / incompressible inverse: dP = Q / (7.08e-3 * W * M)
    return float(pressure + target_rate / denominator)


@numba.njit(cache=True)
def compute_oil_productivity_index(
    well_index: float,
    phase_mobility: float,
    md_per_cp_to_ft2_per_psi_per_day: float = 0.001127,
) -> float:
    """
    Compute the well productivity index for oil/water phases.

    The PI relates flow rate to drawdown via Darcy's law:

        Q_res = PI * (P_bhp - P_reservoir)   [ft³/day]

    where Q_res is at reservoir conditions (rb/day before FVF conversion).

    This is the linear incompressible formulation, consistent with
    `compute_oil_well_rate`. The FVF is applied downstream by the caller.

    PI = WI * (kr/μ) * 0.001127 (or as set) [ft³/(psi·day)]

    :param well_index: Well index (mD·ft).
    :param phase_mobility: Phase relative mobility kr/μ (1/cP).
    :param md_per_cp_to_ft2_per_psi_per_day: Unit conversion factor (0.001127).
    :return: Productivity index (ft³/(psi·day)).
    """
    return well_index * phase_mobility * md_per_cp_to_ft2_per_psi_per_day


def compute_gas_well_rate(
    well_index: float,
    pressure: float,
    temperature: float,
    bottom_hole_pressure: float,
    phase_mobility: float,
    average_compressibility_factor: float = 1.0,
    use_pseudo_pressure: bool = True,
    pseudo_pressure_table: typing.Optional[PseudoPressureTable] = None,
    formation_volume_factor: typing.Optional[float] = None,
    gas_gravity: typing.Optional[float] = None,
) -> float:
    """
    Compute the gas well rate at reservoir conditions (ft³/day).

    The well equation is first evaluated at surface conditions (SCF/day) via
    the (Tsc/Psc) prefactor, then multiplied by Bg (ft³/SCF) to convert to
    reservoir conditions, consistent with the oil/water rate functions which
    return reservoir bbl/day. The caller divides by Bg downstream to obtain
    surface SCF/day, following the same pattern used throughout the simulator.

    Two formulations are supported:

    Pseudo-pressure (recommended, valid over the full pressure range):

    ```
    Q_scf = 1.9875e-2 * (Tsc/Psc) * (W * M / T) * (m(Pbhp) - m(P))
    Q_res = Q_scf * Bg
    ```

    Pressure-squared (valid when mu*Z is approximately constant, i.e.
    low-to-moderate pressures below ~2000 psi):

    ```
    Q_scf = 1.9875e-2 * (Tsc/Psc) * (W * M / T) * (Pbhp^2 - P^2) / Z
    Q_res = Q_scf * Bg
    ```

    Since phase_mobility = kr/mu (no Bg term), Bg must be supplied via
    `formation_volume_factor` or will be computed internally from `gas_gravity`.

    Sign convention:
    - Negative rate indicates production (BHP < reservoir pressure).
    - Positive rate indicates injection (BHP > reservoir pressure).

    :param well_index: Well index (mD*ft).
    :param pressure: Reservoir pressure at the perforation interval (psi).
    :param temperature: Reservoir temperature (deg F).
    :param bottom_hole_pressure: Well bottom-hole pressure (psi).
    :param phase_mobility: Phase relative mobility kr/mu (md/cP).
        Must not include the Bg term; surface-to-reservoir conversion is
        handled here via `formation_volume_factor` or `gas_gravity`.
    :param average_compressibility_factor: Average Z-factor over the pressure
        interval. Used only for the pressure-squared formulation. Default 1.0.
    :param use_pseudo_pressure: If True (default), use the pseudo-pressure
        formulation. If False, use the pressure-squared formulation.
    :param pseudo_pressure_table: Pre-computed pseudo-pressure lookup table.
        Required when `use_pseudo_pressure=True`.
    :param formation_volume_factor: Gas formation volume factor Bg (ft³/SCF)
        at reservoir conditions. If None, Bg is computed internally from
        `gas_gravity`, `pressure`, and `temperature`.
    :param gas_gravity: Gas specific gravity (air = 1.0). Required when
        `formation_volume_factor` is None.
    :return: Gas well rate at reservoir conditions (ft³/day). Negative for
        production, positive for injection.
    """
    if well_index < 0:
        raise ValidationError("Well index must be a positive value.")
    if phase_mobility < 0:
        raise ValidationError("Phase mobility must be a positive value.")

    if well_index == 0.0 or phase_mobility == 0.0:
        return 0.0

    Tsc = c.STANDARD_TEMPERATURE_RANKINE
    Psc = c.STANDARD_PRESSURE_IMPERIAL
    temperature_rankine = fahrenheit_to_rankine(temperature)

    if formation_volume_factor is not None:
        gas_fvf = formation_volume_factor
    else:
        if gas_gravity is None:
            raise ComputationError(
                "`gas_gravity` is required if `formation_volume_factor` is not provided."
            )
        Z = compute_gas_compressibility_factor(
            pressure=pressure,
            temperature=temperature,
            gas_gravity=gas_gravity,
            method="dak",
        )
        # Bg = 0.02827 * Z * T / P  (ft³/SCF)
        gas_fvf = 0.02827 * Z * temperature_rankine / pressure

    # Common prefactor: 1.9875e-2 * (Tsc/Psc) * (W * M / T)
    # Gives Q in SCF/day; multiply by Bg at the end for ft³/day.
    prefactor = (
        1.9875e-2 * (Tsc / Psc) * (well_index * phase_mobility / temperature_rankine)
    )
    if use_pseudo_pressure:
        if pseudo_pressure_table is None:
            raise ValidationError(
                "`pseudo_pressure_table` must be provided when `use_pseudo_pressure` is True."
            )
        pseudo_pressure_difference = pseudo_pressure_table(
            bottom_hole_pressure
        ) - pseudo_pressure_table(pressure)
        well_rate_scf = prefactor * pseudo_pressure_difference
    else:
        pressure_squared_difference = bottom_hole_pressure**2 - pressure**2
        well_rate_scf = (
            prefactor * pressure_squared_difference / average_compressibility_factor
        )

    # Convert SCF/day to reservoir ft³/day
    return well_rate_scf * gas_fvf


def compute_required_bhp_for_gas_rate(
    target_rate: float,
    well_index: float,
    pressure: float,
    temperature: float,
    phase_mobility: float,
    average_compressibility_factor: float = 1.0,
    use_pseudo_pressure: bool = True,
    pseudo_pressure_table: typing.Optional[PseudoPressureTable] = None,
    formation_volume_factor: typing.Optional[float] = None,
) -> float:
    """
    Compute the bottom-hole pressure required to achieve a target gas rate.

    This is the exact algebraic inverse of `compute_gas_well_rate`.
    `target_rate` must be at reservoir conditions (ft³/day), consistent
    with the return value of `compute_gas_well_rate`.

    Sign convention:
    - Negative `target_rate` indicates production (returned BHP < reservoir pressure).
    - Positive `target_rate` indicates injection (returned BHP > reservoir pressure).

    Inverse formula (pseudo-pressure):

    ```
    Q_scf   = target_rate / Bg
    m(Pbhp) = m(P) + Q_scf * T * Psc / (1.9875e-2 * Tsc * W * M)
    Pbhp    = m^-1( m(Pbhp) )  [inverse table interpolation]
    ```

    Inverse formula (pressure-squared):

    ```
    Q_scf  = target_rate / Bg
    Pbhp   = sqrt( P^2 + Q_scf * T * Psc * Z / (1.9875e-2 * Tsc * W * M) )
    ```

    :param target_rate: Target gas rate at reservoir conditions (ft³/day).
        Negative for production, positive for injection.
    :param well_index: Well index (mD*ft).
    :param pressure: Reservoir pressure at the perforation interval (psi).
    :param temperature: Reservoir temperature (deg F).
    :param phase_mobility: Phase relative mobility kr/mu (md/cP).
        Must not include the gas formation volume factor term.
    :param average_compressibility_factor: Average Z-factor over the pressure
        interval. Used only for the pressure-squared formulation. Default 1.0.
    :param use_pseudo_pressure: If True (default), use the pseudo-pressure
        inverse. If False, use the pressure-squared inverse.
    :param pseudo_pressure_table: Pre-computed pseudo-pressure lookup table
        supporting inverse interpolation. Required when `use_pseudo_pressure=True`.
    :param formation_volume_factor: Gas formation volume factor (ft³/SCF) at reservoir conditions.
        Required to convert the reservoir-condition target rate back to
        surface SCF/day before inverting the well equation.
    :return: Required bottom-hole pressure (psi).
    """
    if well_index < 0:
        raise ValidationError("Well index must be a positive value.")
    if phase_mobility < 0:
        raise ValidationError("Phase mobility must be a positive value.")
    if formation_volume_factor is None:
        raise ValidationError(
            "`formation_volume_factor` (Bg) must be provided to convert the "
            "reservoir-condition target rate to surface conditions."
        )
    if well_index == 0.0 or phase_mobility == 0.0:
        return float(pressure)

    Tsc = c.STANDARD_TEMPERATURE_RANKINE
    Psc = c.STANDARD_PRESSURE_IMPERIAL
    temperature_rankine = fahrenheit_to_rankine(temperature)

    # Convert reservoir ft³/day back to SCF/day for inversion
    target_rate_scf = target_rate / formation_volume_factor

    # Common denominator: 1.9875e-2 * Tsc * W * M / T
    denominator = 1.9875e-2 * Tsc * well_index * phase_mobility / temperature_rankine

    if use_pseudo_pressure:
        if pseudo_pressure_table is None:
            raise ValidationError(
                "`pseudo_pressure_table` must be provided when `use_pseudo_pressure` is True."
            )
        # m(Pbhp) = m(P) + Q_scf * Psc / denominator
        required_pseudo_pressure = (
            pseudo_pressure_table(pressure) + target_rate_scf * Psc / denominator
        )
        try:
            return float(
                pseudo_pressure_table.inverse(pseudo_pressure=required_pseudo_pressure)
            )
        except ValidationError as exc:
            min_pseudo_pressure = pseudo_pressure_table.pseudo_pressures[0]
            max_pseudo_pressure = pseudo_pressure_table.pseudo_pressures[-1]
            raise ComputationError(
                f"Cannot achieve target rate {target_rate:.2f} ft³/day — "
                f"required pseudo-pressure {required_pseudo_pressure:.2f} is outside "
                f"the valid table range [{min_pseudo_pressure:.2f}, {max_pseudo_pressure:.2f}]."
            ) from exc

    # Pressure-squared inverse.
    # Pbhp^2 = P^2 + Q_scf * Psc * Z / denominator
    required_bhp_squared = (
        pressure**2
        + target_rate_scf * Psc * average_compressibility_factor / denominator
    )
    if required_bhp_squared < 0:
        raise ComputationError(
            f"Cannot achieve target rate {target_rate:.2f} ft³/day. Rate results in negative pressure squared.  "
            "The requested rate exceeds reservoir deliverability."
        )
    return float(np.sqrt(required_bhp_squared))


def compute_gas_productivity_index(
    well_index: float,
    phase_mobility: float,
    pressure: float,
    temperature: float,
    bottom_hole_pressure: float,
    formation_volume_factor: float,
    use_pseudo_pressure: bool = False,
    pseudo_pressure_table: typing.Optional[PseudoPressureTable] = None,
    average_compressibility_factor: float = 1.0,
) -> float:
    drawdown = pressure - bottom_hole_pressure
    if abs(drawdown) < 1.0:
        # Near-zero drawdown: use tangent approximation to avoid division by zero
        # fall through to tangent formula below
        pass
    else:
        # Secant: PI = Q_res / drawdown, where Q_res = compute_gas_well_rate(...)
        q_res = compute_gas_well_rate(
            well_index=well_index,
            pressure=pressure,
            temperature=temperature,
            bottom_hole_pressure=bottom_hole_pressure,
            phase_mobility=phase_mobility,
            average_compressibility_factor=average_compressibility_factor,
            use_pseudo_pressure=use_pseudo_pressure,
            pseudo_pressure_table=pseudo_pressure_table,
            formation_volume_factor=formation_volume_factor,
        )
        return abs(q_res) / abs(drawdown)  # ft³/(psi·day), always positive

    # Tangent fallback for near-zero drawdown
    Tsc = c.STANDARD_TEMPERATURE_RANKINE
    Psc = c.STANDARD_PRESSURE_IMPERIAL
    T_rankine = fahrenheit_to_rankine(temperature)
    prefactor = 1.9875e-2 * (Tsc / Psc) * (well_index * phase_mobility / T_rankine)

    if use_pseudo_pressure and pseudo_pressure_table is not None:
        delta_p = max(1.0, pressure * 1e-4)
        p_hi = min(pressure + delta_p, pseudo_pressure_table.pressures[-1])
        p_lo = max(pressure - delta_p, pseudo_pressure_table.pressures[0])
        dm_dp = (pseudo_pressure_table(p_hi) - pseudo_pressure_table(p_lo)) / (
            p_hi - p_lo
        )
        return prefactor * dm_dp * formation_volume_factor

    return (
        prefactor
        * (2.0 * pressure / average_compressibility_factor)
        * formation_volume_factor
    )


@attrs.frozen
class WellFluid(Fluid):
    """Base class for fluid properties in wells."""

    specific_gravity: typing.Optional[float] = attrs.field(
        default=None, validator=attrs.validators.optional(attrs.validators.ge(0))
    )
    """
    Specific gravity of the fluid (dimensionless).
 
    For gas: relative to air (air = 1.0).
    For water: relative to fresh water (water = 1.0).
 
    Required when `pvt_table` is not set and correlation-based property
    calculations are needed (e.g. density, viscosity, pseudo-pressure table).
    When `pvt_table` is provided this field is optional.
    """

    molecular_weight: typing.Optional[float] = attrs.field(
        default=None, validator=attrs.validators.optional(attrs.validators.ge(0))
    )
    """
    Molecular weight of the fluid in (g/mol).
 
    Required for gas viscosity correlations (Lee-Kesler) when `pvt_table`
    is not set. When `pvt_table` is provided this field is optional.
    """

    def get_specific_gravity(
        self,
        pressure: FloatOrArray,
        temperature: FloatOrArray,
        salinity: typing.Optional[FloatOrArray] = None,
    ) -> typing.Optional[FloatOrArray]:
        """
        Get the specific gravity of the fluid at given pressure and temperature.

        Lookup priority:

        1. `self.pvt_table.specific_gravity(pressure, temperature)` is used when
           a PVT table is available and returns a non-`None` result.
        2. `self.specific_gravity` scalar field is returned directly if set.
        3. `None` is returned if neither source is available.

        :param pressure: The pressure at which to evaluate the specific gravity (psi).
        :param temperature: The temperature at which to evaluate the specific gravity (°F).
        :return: The specific gravity of the fluid (dimensionless), or `None` if
            neither a PVT table nor a scalar value is available.
        """
        if self.pvt_table is not None:
            result = self.pvt_table.specific_gravity(pressure, temperature, salinity)
            if result is not None:
                return result
        return self.specific_gravity

    def get_molecular_weight(
        self,
        pressure: FloatOrArray,
        temperature: FloatOrArray,
        salinity: typing.Optional[FloatOrArray] = None,
    ) -> typing.Optional[FloatOrArray]:
        """
        Get the molecular weight of the fluid at given pressure and temperature.

        Lookup priority:

        1. `self.pvt_table.molecular_weight(pressure, temperature)` is used when
           a PVT table is available and returns a non-`None` result.
        2. `self.molecular_weight` scalar field is returned directly if set.
        3. `None` is returned if neither source is available.

        :param pressure: The pressure at which to evaluate the molecular weight (psi).
        :param temperature: The temperature at which to evaluate the molecular weight (°F).
        :return: The molecular weight of the fluid (g/mol), or `None` if
            neither a PVT table nor a scalar value is available.
        """
        if self.pvt_table is not None:
            result = self.pvt_table.molecular_weight(pressure, temperature, salinity)
            if result is not None:
                return result
        return self.molecular_weight


@typing.final
@attrs.frozen
class InjectedFluid(WellFluid):
    """Properties of the fluid being injected into or produced by a well."""

    salinity: typing.Optional[float] = None
    """Salinity of the fluid (if water) in (ppm NaCl)."""

    is_miscible: bool = False
    """Whether this fluid is miscible with oil (e.g., CO2, N2)"""

    todd_longstaff_omega: float = attrs.field(
        validator=attrs.validators.and_(
            attrs.validators.ge(0.0), attrs.validators.le(1.0)
        ),
        default=0.67,
    )
    """Todd-Longstaff mixing parameter for miscible displacement (0 to 1)."""

    minimum_miscibility_pressure: typing.Optional[float] = None
    """Minimum miscibility pressure for this fluid-oil system (psi)"""

    miscibility_transition_width: float = attrs.field(  # type: ignore
        default=500.0, validator=attrs.validators.ge(0)
    )
    """Pressure range over which miscibility transitions from immiscible to miscible (psi)"""

    concentration: float = attrs.field(
        default=1.0,
        validator=attrs.validators.and_(
            attrs.validators.ge(0.0), attrs.validators.le(1.0)
        ),
    )
    """Concentration (preferably volume-based) of the fluid in the mixture (0 to 1). Relevant for miscible fluids."""

    density: typing.Optional[float] = None
    """
    Fluid density (lbm/ft³) at reservoir conditions.

    When provided, bypasses both `pvt_table` and correlation-based density
    calculations entirely. Useful for non-ideal gases such as CO2 where a
    measured or equation-of-state density is available.
    """

    viscosity: typing.Optional[float] = None
    """
    Fluid viscosity (cP) at reservoir conditions.

    When provided, bypasses both `pvt_table` and correlation-based viscosity
    calculations entirely. Useful for non-ideal gases such as CO2 where a
    measured or equation-of-state viscosity is available.
    """

    def __attrs_post_init__(self) -> None:
        """Validate the fluid properties."""
        if self.phase not in (FluidPhase.GAS, FluidPhase.WATER):
            raise ValidationError("Only gases and water are supported for injection.")

        if self.is_miscible:
            if self.phase != FluidPhase.GAS:
                raise ValidationError("Only gas phase can be miscible.")
            elif not self.minimum_miscibility_pressure or not self.todd_longstaff_omega:
                raise ValidationError(
                    "Miscible fluids must have both `minimum_miscibility_pressure` and `todd_longstaff_omega` defined."
                )

    def get_density(
        self, pressure: FloatOrArray, temperature: FloatOrArray, **kwargs: typing.Any
    ) -> FloatOrArray:
        """
        Get the density of the fluid at given pressure and temperature.

        Lookup priority:

        1. `self.density` scalar override. Returned directly if set.
        2. `self.pvt_table.density(pressure, temperature, salinity=...)` — used
           when a PVT table is available and returns a non-`None` result.
        3. Correlation fallback. Uses `specific_gravity` via the
           appropriate phase correlation (McCain for water, real-gas law for gas).

        :param pressure: The pressure at which to evaluate the density (psi).
        :param temperature: The temperature at which to evaluate the density (°F).
        :kwargs: Additional parameters for phase density calculations.

        For water:
            - `gas_free_water_formation_volume_factor`: pre-computed Bwf (bbl/STB).
                Computed internally if not provided.
            - `gas_solubility_in_water`: Rsw (SCF/STB). Defaults to 0.0
                (injection water assumed gas-free).
            - `gas_gravity`: gas specific gravity. Defaults to `self.specific_gravity`.

        For gas:
            - `gas_compressibility_factor`: Z-factor (dimensionless).
                Computed via DAK if not provided.

        :return: The density of the fluid (lbm/ft³).
        """
        if self.density is not None:
            if isinstance(pressure, np.ndarray):
                return np.full_like(pressure, self.density)
            return self.density

        if self.pvt_table is not None:
            salinity = self.salinity if self.phase == FluidPhase.WATER else None
            result = self.pvt_table.density(pressure, temperature, salinity=salinity)
            if result is not None:
                return result

        if self.specific_gravity is None:
            raise ValidationError(
                "`specific_gravity` is required if fluid has not `pvt_table`."
            )

        vectorize_pressure = isinstance(pressure, np.ndarray)
        vectorize_temperature = isinstance(temperature, np.ndarray)
        use_vectorization = vectorize_pressure or vectorize_temperature
        if use_vectorization and not vectorize_pressure:
            pressure = np.full_like(temperature, pressure)
        elif use_vectorization and not vectorize_temperature:
            temperature = np.full_like(pressure, temperature)

        if self.phase == FluidPhase.WATER:
            gas_free_water_fvf = kwargs.get(
                "gas_free_water_formation_volume_factor", None
            )
            if gas_free_water_fvf is None:
                if use_vectorization:
                    gas_free_water_fvf = (
                        compute_gas_free_water_formation_volume_factor_vectorized(
                            pressure=pressure,  # type: ignore
                            temperature=temperature,  # type: ignore
                        )
                    )
                else:
                    gas_free_water_fvf = compute_gas_free_water_formation_volume_factor(
                        pressure=pressure,  # type: ignore
                        temperature=temperature,  # type: ignore
                    )
                kwargs["gas_free_water_formation_volume_factor"] = gas_free_water_fvf

            # Assume no-gas in injection water if not explicitly specified
            if "gas_solubility_in_water" not in kwargs:
                kwargs["gas_solubility_in_water"] = 0.0
            if "gas_gravity" not in kwargs:
                kwargs["gas_gravity"] = self.specific_gravity

            if use_vectorization:
                return compute_water_density_vectorized(
                    pressure=pressure,  # type: ignore
                    temperature=temperature,  # type: ignore
                    salinity=self.salinity or 0.0,
                    **kwargs,
                )

            return compute_water_density(
                pressure=pressure,  # type: ignore
                temperature=temperature,  # type: ignore
                salinity=self.salinity or 0.0,
                **kwargs,
            )

        gas_z_factor = kwargs.get("gas_compressibility_factor", None)
        if gas_z_factor is None:
            if use_vectorization:
                gas_z_factor = compute_gas_compressibility_factor_vectorized(
                    pressure=pressure,  # type: ignore
                    temperature=temperature,  # type: ignore
                    gas_gravity=np.full_like(pressure, self.specific_gravity),
                    method="dak",
                )
            else:
                gas_z_factor = compute_gas_compressibility_factor(
                    pressure=pressure,  # type: ignore
                    temperature=temperature,  # type: ignore
                    gas_gravity=self.specific_gravity,
                    method="dak",
                )

        if use_vectorization:
            return compute_gas_density_vectorized(
                pressure=pressure,  # type: ignore
                temperature=temperature,  # type: ignore
                gas_gravity=np.full_like(pressure, self.specific_gravity),
                gas_compressibility_factor=gas_z_factor,
            )
        return compute_gas_density(
            pressure=pressure,  # type: ignore
            temperature=temperature,  # type: ignore
            gas_gravity=self.specific_gravity,
            gas_compressibility_factor=gas_z_factor,  # type: ignore
        )

    def get_viscosity(
        self, pressure: FloatOrArray, temperature: FloatOrArray, **kwargs: typing.Any
    ) -> FloatOrArray:
        """
        Get the viscosity of the fluid at given pressure and temperature.

        Lookup priority:

        1. `self.viscosity` scalar override. Returned directly if set.
        2. `self.pvt_table.viscosity(pressure, temperature, salinity=...)`. Used
           when a PVT table is available and returns a non-`None` result.
        3. Correlation fallback. Uses Lee-Kesler for gas (requires
           `specific_gravity` and `molecular_weight`), and the McCain
           correlation for water.

        :param pressure: The pressure at which to evaluate the viscosity (psi).
        :param temperature: The temperature at which to evaluate the viscosity (°F).
        :kwargs: Additional parameters for viscosity calculations.

        For gas:
            - `gas_density`: pre-computed gas density (lbm/ft³).
                Computed internally if not provided.
            - `gas_compressibility_factor`: Z-factor (dimensionless).
                Computed via DAK if neither `gas_density` nor this value
                is provided.

        :return: The viscosity of the fluid (cP).
        """
        if self.viscosity is not None:
            if isinstance(pressure, np.ndarray):
                return np.full_like(pressure, self.viscosity)
            return self.viscosity

        if self.pvt_table is not None:
            salinity = self.salinity if self.phase == FluidPhase.WATER else None
            result = self.pvt_table.viscosity(pressure, temperature, salinity=salinity)
            if result is not None:
                return result

        vectorize_pressure = isinstance(pressure, np.ndarray)
        vectorize_temperature = isinstance(temperature, np.ndarray)
        use_vectorization = vectorize_pressure or vectorize_temperature
        if use_vectorization and not vectorize_pressure:
            pressure = np.full_like(temperature, pressure)
        elif use_vectorization and not vectorize_temperature:
            temperature = np.full_like(pressure, temperature)

        if self.phase == FluidPhase.WATER:
            if use_vectorization:
                return compute_water_viscosity_vectorized(
                    pressure=pressure,  # type: ignore
                    temperature=temperature,  # type: ignore
                    salinity=self.salinity or 0.0,
                )
            return compute_water_viscosity(
                pressure=pressure,  # type: ignore
                temperature=temperature,  # type: ignore
                salinity=self.salinity or 0.0,
            )

        if self.specific_gravity is None:
            raise ValidationError(
                "`specific_gravity` is required if fluid has not `pvt_table`."
            )
        if self.molecular_weight is None:
            raise ValidationError(
                "`molecular_weight` is required if fluid has not `pvt_table`."
            )

        gas_density = kwargs.get("gas_density", None)
        if gas_density is None:
            gas_z_factor = kwargs.get("gas_compressibility_factor", None)
            if gas_z_factor is None:
                if use_vectorization:
                    gas_z_factor = compute_gas_compressibility_factor_vectorized(
                        pressure=pressure,  # type: ignore
                        temperature=temperature,  # type: ignore
                        gas_gravity=np.full_like(pressure, self.specific_gravity),
                        method="dak",
                    )
                else:
                    gas_z_factor = compute_gas_compressibility_factor(
                        pressure=pressure,  # type: ignore
                        temperature=temperature,  # type: ignore
                        gas_gravity=self.specific_gravity,
                        method="dak",
                    )

            if use_vectorization:
                gas_density = compute_gas_density_vectorized(
                    pressure=pressure,  # type: ignore
                    temperature=temperature,  # type: ignore
                    gas_gravity=np.full_like(pressure, self.specific_gravity),
                    gas_compressibility_factor=gas_z_factor,
                )
            else:
                gas_density = compute_gas_density(
                    pressure=pressure,  # type: ignore
                    temperature=temperature,  # type: ignore
                    gas_gravity=self.specific_gravity,
                    gas_compressibility_factor=gas_z_factor,  # type: ignore
                )

        if use_vectorization:
            return compute_gas_viscosity_vectorized(
                temperature=temperature,  # type: ignore
                gas_density=gas_density,  # type: ignore
                gas_molecular_weight=self.molecular_weight,
            )
        return compute_gas_viscosity(
            temperature=temperature,  # type: ignore
            gas_density=gas_density,  # type: ignore
            gas_molecular_weight=self.molecular_weight,
        )

    def get_compressibility(
        self, pressure: FloatOrArray, temperature: FloatOrArray, **kwargs: typing.Any
    ) -> FloatOrArray:
        """
        Get the compressibility of the fluid at given pressure and temperature.

        Lookup priority:

        1. `self.pvt_table.compressibility(pressure, temperature, salinity=...)`.
           Used when a PVT table is available and returns a non-`None` result.
        2. Correlation fallback. Uses the appropriate phase correlation.

        :param pressure: The pressure at which to evaluate the compressibility (psi).
        :param temperature: The temperature at which to evaluate the compressibility (°F).
        :kwargs: Additional parameters for compressibility calculations.

        For water:
            - `bubble_point_pressure`: water bubble point pressure (psi).
            - `gas_formation_volume_factor`: Bg (ft³/scf).
            - `gas_solubility_in_water`: Rsw (scf/STB).
            - `gas_free_water_formation_volume_factor`: Bwf (bbl/STB).
                Computed internally if not provided.

        For gas:
            - `gas_gravity`: gas specific gravity (dimensionless).
                Uses `self.specific_gravity` if not provided.
            - `gas_compressibility_factor`: Z-factor (dimensionless).

        :return: The compressibility of the fluid (psi⁻¹).
        """
        if self.pvt_table is not None:
            salinity = self.salinity if self.phase == FluidPhase.WATER else None
            result = self.pvt_table.compressibility(
                pressure, temperature, salinity=salinity
            )
            if result is not None:
                return result

        vectorize_pressure = isinstance(pressure, np.ndarray)
        vectorize_temperature = isinstance(temperature, np.ndarray)
        use_vectorization = vectorize_pressure or vectorize_temperature
        if use_vectorization and not vectorize_pressure:
            pressure = np.full_like(temperature, pressure)
        elif use_vectorization and not vectorize_temperature:
            temperature = np.full_like(pressure, temperature)

        if self.phase == FluidPhase.WATER:
            gas_free_water_fvf = kwargs.get(
                "gas_free_water_formation_volume_factor", None
            )
            if gas_free_water_fvf is None:
                if use_vectorization:
                    gas_free_water_fvf = (
                        compute_gas_free_water_formation_volume_factor_vectorized(
                            pressure=pressure,  # type: ignore
                            temperature=temperature,  # type: ignore
                        )
                    )
                else:
                    gas_free_water_fvf = compute_gas_free_water_formation_volume_factor(
                        pressure=pressure,  # type: ignore
                        temperature=temperature,  # type: ignore
                    )
                kwargs["gas_free_water_formation_volume_factor"] = gas_free_water_fvf

            if use_vectorization:
                return compute_water_compressibility_vectorized(
                    pressure=pressure,  # type: ignore
                    temperature=temperature,  # type: ignore
                    salinity=self.salinity or 0.0,
                    **kwargs,
                )

            return compute_water_compressibility(
                pressure=pressure,  # type: ignore
                temperature=temperature,  # type: ignore
                salinity=self.salinity or 0.0,
                **kwargs,
            )

        kwargs.setdefault("gas_gravity", self.specific_gravity)
        if use_vectorization:
            if "gas_gravity" not in kwargs:
                kwargs["gas_gravity"] = np.full_like(pressure, self.specific_gravity)
            return compute_gas_compressibility_vectorized(
                pressure=pressure,  # type: ignore
                temperature=temperature,  # type: ignore
                **kwargs,
            )
        return compute_gas_compressibility(
            pressure=pressure,  # type: ignore
            temperature=temperature,  # type: ignore
            **kwargs,
        )

    def get_formation_volume_factor(
        self, pressure: FloatOrArray, temperature: FloatOrArray, **kwargs: typing.Any
    ) -> FloatOrArray:
        """
        Get the formation volume factor of the fluid at given pressure and temperature.

        Lookup priority:

        1. `self.pvt_table.formation_volume_factor(pressure, temperature, salinity=...)`.
           Used when a PVT table is available and returns a non-`None` result.
        2. Correlation fallback. Uses the appropriate phase correlation.

        :param pressure: The pressure at which to evaluate the formation volume factor (psi).
        :param temperature: The temperature at which to evaluate the formation volume factor (°F).
        :kwargs: Additional parameters for formation volume factor calculations.

        For water:
            - `water_density`: pre-computed water density (lbm/ft³).
                Computed internally from correlations if not provided.

        For gas:
            - `gas_compressibility_factor`: Z-factor (dimensionless).
                Computed via DAK if not provided.

        :return: The formation volume factor of the fluid (bbl/STB for water, ft³/SCF for gas).
        """
        if self.pvt_table is not None:
            salinity = self.salinity if self.phase == FluidPhase.WATER else None
            result = self.pvt_table.formation_volume_factor(
                pressure, temperature, salinity=salinity
            )
            if result is not None:
                return result

        vectorize_pressure = isinstance(pressure, np.ndarray)
        vectorize_temperature = isinstance(temperature, np.ndarray)
        use_vectorization = vectorize_pressure or vectorize_temperature
        if use_vectorization and not vectorize_pressure:
            pressure = np.full_like(temperature, pressure)
        elif use_vectorization and not vectorize_temperature:
            temperature = np.full_like(pressure, temperature)

        if self.phase == FluidPhase.WATER:
            water_density = kwargs.get("water_density", None)
            if water_density is None:
                # No need for gas-free FVF or gas FVF here since injection water
                # is typically gas-free fresh water or degassed formation water.
                if use_vectorization:
                    water_density = compute_water_density_vectorized(
                        pressure=pressure,  # type: ignore
                        temperature=temperature,  # type: ignore
                        salinity=self.salinity or 0.0,
                    )
                else:
                    water_density = compute_water_density(
                        pressure=pressure,  # type: ignore
                        temperature=temperature,  # type: ignore
                        salinity=self.salinity or 0.0,
                    )

            if use_vectorization:
                return compute_water_formation_volume_factor_vectorized(
                    salinity=self.salinity or 0.0,
                    water_density=water_density,  # type: ignore
                )
            return compute_water_formation_volume_factor(
                salinity=self.salinity or 0.0,
                water_density=water_density,  # type: ignore
            )

        if self.specific_gravity is None:
            raise ValidationError(
                "`specific_gravity` is required if fluid has not `pvt_table`."
            )

        gas_z_factor = kwargs.get("gas_compressibility_factor", None)
        if gas_z_factor is None:
            if use_vectorization:
                gas_z_factor = compute_gas_compressibility_factor_vectorized(
                    pressure=pressure,  # type: ignore
                    temperature=temperature,  # type: ignore
                    gas_gravity=np.full_like(pressure, self.specific_gravity),
                    method="dak",
                )
            else:
                gas_z_factor = compute_gas_compressibility_factor(
                    pressure=pressure,  # type: ignore
                    temperature=temperature,  # type: ignore
                    gas_gravity=self.specific_gravity,
                    method="dak",
                )

        if use_vectorization:
            return compute_gas_formation_volume_factor_vectorized(
                pressure=pressure,  # type: ignore
                temperature=temperature,  # type: ignore
                gas_compressibility_factor=gas_z_factor,  # type: ignore
            )
        return compute_gas_formation_volume_factor(
            pressure=pressure,  # type: ignore
            temperature=temperature,  # type: ignore
            gas_compressibility_factor=gas_z_factor,  # type: ignore
        )


@typing.final
@attrs.frozen
class ProducedFluid(WellFluid):
    """Properties of the fluid being produced by a well."""

    pass


WellFluidT = typing.TypeVar("WellFluidT", bound=WellFluid)


@numba.njit(cache=True, inline="always")
def _geometric_mean(values: typing.Sequence[float]) -> float:
    prod = 1.0
    n = 0
    for v in values:
        prod *= max(v, 0.0)  # ensure non-negative
        n += 1

    if n == 0:
        raise ValidationError("No permeability values provided")
    return prod ** (1.0 / n)


@numba.njit(cache=True)
def compute_effective_permeability_for_well(
    permeability: typing.Sequence[float], orientation: Orientation
) -> float:
    """
    Compute `k_eff` for Peaceman WI using geometric mean of the two permeabilities
    perpendicular to the well axis. `permeability` is (kx, ky, kz).
    orientation is one of Orientation.X/Y/Z (or a string equivalent).
    """
    if len(permeability) != 3:
        # If 2D, fall back to geometric mean of available components
        return _geometric_mean(permeability)

    kx, ky, kz = permeability
    if orientation == Orientation.Z:  # vertical well: transverse are x,y
        return np.sqrt(max(kx, 0.0) * max(ky, 0.0))
    elif orientation == Orientation.X:  # well along x: transverse are y,z
        return np.sqrt(max(ky, 0.0) * max(kz, 0.0))
    elif orientation == Orientation.Y:  # well along y: transverse are x,z
        return np.sqrt(max(kx, 0.0) * max(kz, 0.0))
    # For Oblique/unknown orientation, use geometric mean of all three
    return _geometric_mean((kx, ky, kz))


def get_pseudo_pressure_table(
    fluid: Fluid,
    temperature: float,
    use_pseudo_pressure: bool,
    pvt_tables: typing.Optional[PVTTables] = None,
) -> typing.Tuple[bool, typing.Optional[PseudoPressureTable]]:
    """
    Get existing pseudo-pressure table or setup a new one for gas well fluid if needed.

    :return: Tuple of (use_pseudo_pressure, pseudo_pressure_table)
    """
    if not use_pseudo_pressure:
        return False, None

    pseudo_pressure_table = fluid.get_pseudo_pressure_table(
        temperature=temperature,
        points=c.GAS_PSEUDO_PRESSURE_POINTS,
        pvt_tables=pvt_tables,
    )
    return True, pseudo_pressure_table


@numba.njit(cache=True)
def compute_average_compressibility_factor(
    pressure: float,
    temperature: float,
    gas_gravity: float,
    bottom_hole_pressure: typing.Optional[float] = None,
) -> float:
    """
    Compute average gas compressibility factor.

    :param bottom_hole_pressure: If provided, uses average of reservoir and BHP.
        Otherwise uses reservoir pressure.
    """
    if bottom_hole_pressure is not None:
        avg_pressure = (pressure + bottom_hole_pressure) * 0.5
    else:
        avg_pressure = pressure
    return compute_gas_compressibility_factor(
        pressure=avg_pressure,
        temperature=temperature,
        gas_gravity=gas_gravity,
        method="dak",
    )

import logging
import typing

import attrs
import numba
import numpy as np
import numpy.typing as npt

from bores.config import Config
from bores.constants import c
from bores.datastructures import Rates
from bores.grids.base import CapillaryPressureGrids, RelativeMobilityGrids
from bores.models import FluidProperties, RockProperties
from bores.solvers.base import Solution
from bores.solvers.rates import WellRates
from bores.transmissibility import FaceTransmissibilities
from bores.types import OneDimensionalGrid, ThreeDimensionalGrid, ThreeDimensions
from bores.wells.indices import WellsIndices

__all__ = ["solve_transport"]

logger = logging.getLogger(__name__)


@attrs.frozen
class CFLMeta:
    cfl_threshold: float
    maximum_cfl_encountered: float
    cell: typing.Tuple[int, int, int]
    time_step: int
    violated: bool


@attrs.frozen
class FluxesMeta:
    total_water_inflow: float
    total_water_outflow: float
    total_gas_inflow: float
    total_gas_outflow: float
    total_inflow: float
    total_outflow: float


@attrs.frozen
class VolumesMeta:
    oil_volume: float
    water_volume: float
    gas_volume: float
    pore_volume: float


@attrs.frozen
class SaturationEvolutionMeta:
    cfl_info: CFLMeta
    fluxes: typing.Optional[FluxesMeta] = None
    volumes: typing.Optional[VolumesMeta] = None


@attrs.frozen
class ExplicitSaturationSolution:
    water_saturation_grid: ThreeDimensionalGrid
    oil_saturation_grid: ThreeDimensionalGrid
    gas_saturation_grid: ThreeDimensionalGrid
    maximum_cfl_encountered: float
    cfl_threshold: float
    maximum_oil_saturation_change: float
    maximum_water_saturation_change: float
    maximum_gas_saturation_change: float
    solvent_concentration_grid: typing.Optional[ThreeDimensionalGrid] = None


def solve_transport(
    cell_dimension: typing.Tuple[float, float],
    thickness_grid: ThreeDimensionalGrid,
    elevation_grid: ThreeDimensionalGrid,
    time_step: int,
    time_step_size: float,
    rock_properties: RockProperties[ThreeDimensions],
    fluid_properties: FluidProperties[ThreeDimensions],
    old_water_density_grid: ThreeDimensionalGrid,
    old_oil_density_grid: ThreeDimensionalGrid,
    old_gas_density_grid: ThreeDimensionalGrid,
    old_solution_gas_to_oil_ratio_grid: ThreeDimensionalGrid,
    old_gas_solubility_in_water_grid: ThreeDimensionalGrid,
    old_gas_formation_volume_factor_grid: ThreeDimensionalGrid,
    old_oil_formation_volume_factor_grid: ThreeDimensionalGrid,
    old_water_formation_volume_factor_grid: ThreeDimensionalGrid,
    relative_mobility_grids: RelativeMobilityGrids[ThreeDimensions],
    capillary_pressure_grids: CapillaryPressureGrids[ThreeDimensions],
    face_transmissibilities: FaceTransmissibilities,
    pressure_boundaries: ThreeDimensionalGrid,
    flux_boundaries: ThreeDimensionalGrid,
    config: Config,
    rates: typing.Optional[WellRates[ThreeDimensions]] = None,
    dtype: npt.DTypeLike = np.float64,
) -> Solution[ExplicitSaturationSolution, SaturationEvolutionMeta]:
    """
    Computes the new saturation distribution for water, oil, and gas across the
    reservoir grid using a mass-based explicit upwind finite difference method.

    The governing equations conserve fluid mass rather than reservoir-condition
    volume. For water:

        (phi*V/dt) * (current_water_density*Sw_new - old_water_density*Sw_old) = sum_faces(upwind_water_densitywind * F_w) + rho_w * q_w

    For gas (total: free + dissolved in oil + dissolved in water):

        (phi*V/dt) * [M_g_new - M_g_old] = sum_faces(upwind_gas_densitywind*F_g + upwind_oil_densitywind*alpha_Rs_upwind*F_o
                                            + upwind_water_densitywind*alpha_Rsw_upwind*F_w) + well_mass_terms

    where `alpha_Rs = Rs * Bg / Bo` (dimensionless reservoir-condition gas fraction
    dissolved in oil) and `alpha_Rsw = Rsw * Bg / Bw` (same for water). Oil
    saturation is derived from `So = 1 - Sw - Sg`.

    The PVT volume correction (`Delta_S_pvt`) is not applied because density
    changes between old and new conditions already capture fluid
    expansion/contraction within the mass accumulation term.

    The CFL stability criterion is unchanged - it is evaluated on volumetric
    outflows relative to pore volume, which is unaffected by density weighting.

    :param cell_dimension: Tuple (cell_size_x, cell_size_y) in feet.
    :param thickness_grid: Cell thickness grid (ft).
    :param elevation_grid: Cell elevation grid (ft).
    :param time_step: Current time step index (starting from 0).
    :param time_step_size: Time step duration in seconds.
    :param rock_properties: `RockProperties` containing rock physical properties.
    :param fluid_properties: `FluidProperties` containing fluid physical properties,
        including current pressure, saturation, density, Rs, and FVF grids at the
        new-pressure level (i.e. after `update_fluid_properties` has run).
    :param relative_mobility_grids: Three-phase relative mobility grids (water, oil, gas).
    :param capillary_pressure_grids: Capillary pressure grids (oil-water, gas-oil).
    :param face_transmissibilities: Precomputed geometric face transmissibilities.
    :param pressure_boundaries: Padded boundary pressure grid (nx+2, ny+2, nz+2).
        NaN indicates a Neumann face.
    :param flux_boundaries: Padded boundary flux grid (nx+2, ny+2, nz+2).
    :param config: Simulation config and parameters.
    :param wells_indices: Cache of well indices.
    :param injection_rates: Injection rates for each phase and cell (ft³/day).
    :param production_rates: Production rates for each phase and cell (ft³/day).
    :param dtype: Numpy dtype for output arrays.
    :return: `Solution` containing updated saturations.
    """
    time_step_in_days = time_step_size * c.DAYS_PER_SECOND
    porosity_grid = rock_properties.porosity_grid
    net_to_gross_grid = rock_properties.net_to_gross_grid

    current_water_density_grid = fluid_properties.water_density_grid
    current_oil_density_grid = fluid_properties.oil_effective_density_grid
    current_gas_density_grid = fluid_properties.gas_density_grid
    solution_gas_to_oil_ratio_grid = fluid_properties.solution_gas_to_oil_ratio_grid
    gas_solubility_in_water_grid = fluid_properties.gas_solubility_in_water_grid
    gas_formation_volume_factor_grid = fluid_properties.gas_formation_volume_factor_grid
    oil_formation_volume_factor_grid = fluid_properties.oil_formation_volume_factor_grid
    water_formation_volume_factor_grid = (
        fluid_properties.water_formation_volume_factor_grid
    )

    pressure_grid = fluid_properties.pressure_grid
    current_water_saturation_grid = fluid_properties.water_saturation_grid
    current_oil_saturation_grid = fluid_properties.oil_saturation_grid
    current_gas_saturation_grid = fluid_properties.gas_saturation_grid

    cell_count_x, cell_count_y, cell_count_z = pressure_grid.shape
    cell_size_x, cell_size_y = cell_dimension

    (
        water_relative_mobility_grid,
        oil_relative_mobility_grid,
        gas_relative_mobility_grid,
    ) = relative_mobility_grids
    oil_water_capillary_pressure_grid, gas_oil_capillary_pressure_grid = (
        capillary_pressure_grids
    )

    md_per_cp_to_ft2_per_psi_per_day = (
        c.MILLIDARCIES_PER_CENTIPOISE_TO_SQUARE_FEET_PER_PSI_PER_DAY
    )
    gravitational_constant = (
        c.ACCELERATION_DUE_TO_GRAVITY_FEET_PER_SECONDS_SQUARE
        / c.GRAVITATIONAL_CONSTANT_LBM_FT_PER_LBF_S2
    )

    (
        net_water_mass_flux_grid,
        net_gas_total_mass_flux_grid,
        net_volumetric_outflow_grid,
    ) = assemble_flux_contributions(
        pressure_grid=pressure_grid,
        cell_count_x=cell_count_x,
        cell_count_y=cell_count_y,
        cell_count_z=cell_count_z,
        pressure_boundaries=pressure_boundaries,
        flux_boundaries=flux_boundaries,
        water_relative_mobility_grid=water_relative_mobility_grid,
        oil_relative_mobility_grid=oil_relative_mobility_grid,
        gas_relative_mobility_grid=gas_relative_mobility_grid,
        face_transmissibilities_x=face_transmissibilities.x,
        face_transmissibilities_y=face_transmissibilities.y,
        face_transmissibilities_z=face_transmissibilities.z,
        oil_water_capillary_pressure_grid=oil_water_capillary_pressure_grid,
        gas_oil_capillary_pressure_grid=gas_oil_capillary_pressure_grid,
        water_density_grid=current_water_density_grid,
        oil_density_grid=current_oil_density_grid,
        gas_density_grid=current_gas_density_grid,
        solution_gas_to_oil_ratio_grid=solution_gas_to_oil_ratio_grid,
        gas_solubility_in_water_grid=gas_solubility_in_water_grid,
        gas_formation_volume_factor_grid=gas_formation_volume_factor_grid,
        oil_formation_volume_factor_grid=oil_formation_volume_factor_grid,
        water_formation_volume_factor_grid=water_formation_volume_factor_grid,
        elevation_grid=elevation_grid,
        gravitational_constant=gravitational_constant,
        md_per_cp_to_ft2_per_psi_per_day=md_per_cp_to_ft2_per_psi_per_day,
        dtype=dtype,
    )

    if rates is not None:
        net_water_well_rate_grid = rates.net_water_well_rate_grid
        net_oil_well_rate_grid = rates.net_oil_well_rate_grid
        net_gas_well_rate_grid = rates.net_gas_well_rate_grid
        net_water_well_mass_rate_grid = rates.net_water_well_mass_rate_grid
        net_oil_well_mass_rate_grid = rates.net_oil_well_mass_rate_grid
        net_gas_well_mass_rate_grid = rates.net_gas_well_mass_rate_grid
    else:
        zeros_grid = np.zeros_like(pressure_grid, dtype=dtype)
        net_water_well_rate_grid = zeros_grid
        net_oil_well_rate_grid = zeros_grid
        net_gas_well_rate_grid = zeros_grid
        net_water_well_mass_rate_grid = zeros_grid
        net_oil_well_mass_rate_grid = zeros_grid
        net_gas_well_mass_rate_grid = zeros_grid

    (
        updated_water_saturation_grid,
        updated_oil_saturation_grid,
        updated_gas_saturation_grid,
        cfl_violation_info,
    ) = apply_updates(
        updated_water_saturation_grid=current_water_saturation_grid.copy(),
        updated_oil_saturation_grid=current_oil_saturation_grid.copy(),
        updated_gas_saturation_grid=current_gas_saturation_grid.copy(),
        old_water_saturation_grid=current_water_saturation_grid,
        old_oil_saturation_grid=current_oil_saturation_grid,
        old_gas_saturation_grid=current_gas_saturation_grid,
        net_water_mass_flux_grid=net_water_mass_flux_grid,
        net_gas_total_mass_flux_grid=net_gas_total_mass_flux_grid,
        net_volumetric_outflow_grid=net_volumetric_outflow_grid,
        net_water_well_rate_grid=net_water_well_rate_grid,
        net_oil_well_rate_grid=net_oil_well_rate_grid,
        net_gas_well_rate_grid=net_gas_well_rate_grid,
        net_water_well_mass_rate_grid=net_water_well_mass_rate_grid,
        net_oil_well_mass_rate_grid=net_oil_well_mass_rate_grid,
        net_gas_well_mass_rate_grid=net_gas_well_mass_rate_grid,
        old_water_density_grid=old_water_density_grid,
        old_oil_density_grid=old_oil_density_grid,
        old_gas_density_grid=old_gas_density_grid,
        current_water_density_grid=current_water_density_grid,
        current_oil_density_grid=current_oil_density_grid,
        current_gas_density_grid=current_gas_density_grid,
        solution_gas_to_oil_ratio_grid=solution_gas_to_oil_ratio_grid,
        gas_solubility_in_water_grid=gas_solubility_in_water_grid,
        gas_formation_volume_factor_grid=gas_formation_volume_factor_grid,
        oil_formation_volume_factor_grid=oil_formation_volume_factor_grid,
        water_formation_volume_factor_grid=water_formation_volume_factor_grid,
        old_solution_gas_to_oil_ratio_grid=old_solution_gas_to_oil_ratio_grid,
        old_gas_solubility_in_water_grid=old_gas_solubility_in_water_grid,
        old_gas_formation_volume_factor_grid=old_gas_formation_volume_factor_grid,
        old_oil_formation_volume_factor_grid=old_oil_formation_volume_factor_grid,
        old_water_formation_volume_factor_grid=old_water_formation_volume_factor_grid,
        cell_count_x=cell_count_x,
        cell_count_y=cell_count_y,
        cell_count_z=cell_count_z,
        thickness_grid=thickness_grid,
        porosity_grid=porosity_grid,
        net_to_gross_grid=net_to_gross_grid,
        cell_size_x=cell_size_x,
        cell_size_y=cell_size_y,
        time_step_in_days=time_step_in_days,
        cfl_threshold=config.saturation_cfl_threshold,
        dtype=dtype,
    )

    maximum_oil_saturation_change = float(
        np.max(np.abs(updated_oil_saturation_grid - current_oil_saturation_grid))
    )
    maximum_water_saturation_change = float(
        np.max(np.abs(updated_water_saturation_grid - current_water_saturation_grid))
    )
    maximum_gas_saturation_change = float(
        np.max(np.abs(updated_gas_saturation_grid - current_gas_saturation_grid))
    )

    if cfl_violation_info[0] > 0.0:
        i, j, k = (
            int(cfl_violation_info[1]),
            int(cfl_violation_info[2]),
            int(cfl_violation_info[3]),
        )
        maximum_cfl_encountered = float(cfl_violation_info[4])
        cfl_threshold = float(cfl_violation_info[5])

        cell_thickness = float(thickness_grid[i, j, k])
        cell_total_volume = cell_size_x * cell_size_y * cell_thickness
        cell_porosity = float(porosity_grid[i, j, k])
        cell_pore_volume = cell_total_volume * cell_porosity

        total_outflow = float(net_volumetric_outflow_grid[i, j, k])
        well_water_outflow = abs(min(0.0, float(net_water_well_rate_grid[i, j, k])))
        well_oil_outflow = abs(min(0.0, float(net_oil_well_rate_grid[i, j, k])))
        well_gas_outflow = abs(min(0.0, float(net_gas_well_rate_grid[i, j, k])))
        well_water_inflow = max(0.0, float(net_water_well_rate_grid[i, j, k]))
        well_gas_inflow = max(0.0, float(net_gas_well_rate_grid[i, j, k]))

        cell_pressure = float(pressure_grid[i, j, k])
        cell_bubble_point = float(
            fluid_properties.oil_bubble_point_pressure_grid[i, j, k]
        )
        pressure_state = (
            "undersaturated" if cell_pressure > cell_bubble_point else "saturated"
        )
        average_reservoir_pressure = float(np.mean(pressure_grid))

        oil_saturation = float(current_oil_saturation_grid[i, j, k])
        water_saturation = float(current_water_saturation_grid[i, j, k])
        gas_saturation = float(current_gas_saturation_grid[i, j, k])
        oil_volume = cell_pore_volume * oil_saturation
        water_volume = cell_pore_volume * water_saturation
        gas_volume = cell_pore_volume * gas_saturation

        msg = f"""
        CFL condition violated at cell ({i}, {j}, {k}) at timestep {time_step}:

        Max CFL number {maximum_cfl_encountered:.4f} exceeds limit {cfl_threshold:.4f}.

        Pressure diagnostics:
        Cell pressure = {cell_pressure:.2f} psi, Bubble point = {cell_bubble_point:.2f} psi ({pressure_state})
        Avg reservoir pressure = {average_reservoir_pressure:.2f} psi

        Total volumetric outflow = {total_outflow:.12f} ft³/day,
        Water well outflow = {well_water_outflow:.12f} ft³/day,
        Gas well outflow = {well_gas_outflow:.12f} ft³/day,
        Oil well outflow = {well_oil_outflow:.12f} ft³/day,
        Water well inflow = {well_water_inflow:.12f} ft³/day,
        Gas well inflow = {well_gas_inflow:.12f} ft³/day,
        Oil volume = {oil_volume:.12f} ft³, Water volume = {water_volume:.12f} ft³,
        Gas volume = {gas_volume:.12f} ft³, Pore volume = {cell_pore_volume:.12f} ft³.

        Consider reducing time step size from {time_step_size} seconds.
        """
        return Solution(
            success=False,
            value=ExplicitSaturationSolution(
                water_saturation_grid=updated_water_saturation_grid.astype(
                    dtype, copy=False
                ),
                oil_saturation_grid=updated_oil_saturation_grid.astype(
                    dtype, copy=False
                ),
                gas_saturation_grid=updated_gas_saturation_grid.astype(
                    dtype, copy=False
                ),
                maximum_cfl_encountered=maximum_cfl_encountered,
                cfl_threshold=cfl_threshold,
                maximum_oil_saturation_change=maximum_oil_saturation_change,
                maximum_water_saturation_change=maximum_water_saturation_change,
                maximum_gas_saturation_change=maximum_gas_saturation_change,
            ),
            scheme="explicit",
            message=msg,
            metadata=SaturationEvolutionMeta(
                cfl_info=CFLMeta(
                    cfl_threshold=cfl_threshold,
                    maximum_cfl_encountered=maximum_cfl_encountered,
                    cell=(i, j, k),
                    time_step=time_step,
                    violated=True,
                ),
                fluxes=FluxesMeta(
                    total_water_inflow=well_water_inflow,
                    total_water_outflow=well_water_outflow,
                    total_gas_inflow=well_gas_inflow,
                    total_gas_outflow=well_gas_outflow,
                    total_inflow=well_water_inflow + well_gas_inflow,
                    total_outflow=total_outflow,
                ),
                volumes=VolumesMeta(
                    oil_volume=oil_volume,
                    water_volume=water_volume,
                    gas_volume=gas_volume,
                    pore_volume=cell_pore_volume,
                ),
            ),
        )

    cfl_threshold = float(cfl_violation_info[5])
    maximum_cfl_encountered = float(cfl_violation_info[4])
    cfl_i, cfl_j, cfl_k = (
        int(cfl_violation_info[1]),
        int(cfl_violation_info[2]),
        int(cfl_violation_info[3]),
    )
    return Solution(
        value=ExplicitSaturationSolution(
            water_saturation_grid=updated_water_saturation_grid.astype(
                dtype, copy=False
            ),
            oil_saturation_grid=updated_oil_saturation_grid.astype(dtype, copy=False),
            gas_saturation_grid=updated_gas_saturation_grid.astype(dtype, copy=False),
            maximum_cfl_encountered=maximum_cfl_encountered,
            cfl_threshold=cfl_threshold,
            maximum_oil_saturation_change=maximum_oil_saturation_change,
            maximum_water_saturation_change=maximum_water_saturation_change,
            maximum_gas_saturation_change=maximum_gas_saturation_change,
        ),
        scheme="explicit",
        success=True,
        metadata=SaturationEvolutionMeta(
            cfl_info=CFLMeta(
                cfl_threshold=cfl_threshold,
                maximum_cfl_encountered=maximum_cfl_encountered,
                cell=(cfl_i, cfl_j, cfl_k),
                time_step=time_step,
                violated=False,
            )
        ),
        message=f"Explicit saturation evolution time step {time_step} successful.",
    )


@numba.njit(cache=True, inline="always")
def compute_face_fluxes(
    cell_indices: ThreeDimensions,
    neighbour_indices: ThreeDimensions,
    pressure_grid: ThreeDimensionalGrid,
    face_transmissibility: float,
    water_relative_mobility_grid: ThreeDimensionalGrid,
    oil_relative_mobility_grid: ThreeDimensionalGrid,
    gas_relative_mobility_grid: ThreeDimensionalGrid,
    oil_water_capillary_pressure_grid: ThreeDimensionalGrid,
    gas_oil_capillary_pressure_grid: ThreeDimensionalGrid,
    oil_density_grid: ThreeDimensionalGrid,
    water_density_grid: ThreeDimensionalGrid,
    gas_density_grid: ThreeDimensionalGrid,
    elevation_grid: ThreeDimensionalGrid,
    gravitational_constant: float,
    md_per_cp_to_ft2_per_psi_per_day: float,
) -> typing.Tuple[float, float, float, float, float, float]:
    """
    Compute volumetric fluxes and upwind densities for all three phases between
    a cell and its interior neighbour.

    Returns both the volumetric face fluxes (ft³/day) and the upwind density
    (lb/ft³) selected for each phase, so the caller can form mass fluxes
    without re-evaluating the potential or upwind direction.

    The upwind density for each phase is selected based on the sign of that
    phase's potential difference (pressure + gravity + capillary):
    positive potential difference means the neighbour is higher-potential, so
    flow goes from neighbour into this cell and the neighbour density is used.

    :param cell_indices: (i, j, k) of the current cell.
    :param neighbour_indices: (i, j, k) of the neighbouring cell.
    :param pressure_grid: Oil pressure grid (psi).
    :param face_transmissibility: Geometric transmissibility at this face (mD·ft).
    :param water_relative_mobility_grid: Water relative mobility (ft²/psi·day).
    :param oil_relative_mobility_grid: Oil relative mobility (ft²/psi·day).
    :param gas_relative_mobility_grid: Gas relative mobility (ft²/psi·day).
    :param oil_water_capillary_pressure_grid: Oil-water capillary pressure (psi).
    :param gas_oil_capillary_pressure_grid: Gas-oil capillary pressure (psi).
    :param oil_density_grid: Oil density (lb/ft³).
    :param water_density_grid: Water density (lb/ft³).
    :param gas_density_grid: Gas density (lb/ft³).
    :param elevation_grid: Cell elevation (ft).
    :param gravitational_constant: g/gc conversion factor (lbf/lbm).
    :param md_per_cp_to_ft2_per_psi_per_day: Unit conversion factor.
    :return: Tuple of (water_flux, oil_flux, gas_flux, upwind_water_density,
        upwind_oil_density, upwind_gas_density) where fluxes are in ft³/day
        and densities are in lb/ft³. Positive flux means net inflow to `cell_indices`.
    """
    oil_pressure_difference = (
        pressure_grid[neighbour_indices] - pressure_grid[cell_indices]
    )
    oil_water_capillary_pressure_difference = (
        oil_water_capillary_pressure_grid[neighbour_indices]
        - oil_water_capillary_pressure_grid[cell_indices]
    )
    water_pressure_difference = (
        oil_pressure_difference - oil_water_capillary_pressure_difference
    )
    gas_oil_capillary_pressure_difference = (
        gas_oil_capillary_pressure_grid[neighbour_indices]
        - gas_oil_capillary_pressure_grid[cell_indices]
    )
    gas_pressure_difference = (
        oil_pressure_difference + gas_oil_capillary_pressure_difference
    )

    elevation_difference = (
        elevation_grid[neighbour_indices] - elevation_grid[cell_indices]
    )

    # Density upwinding: use neighbour density when neighbour has higher pressure
    upwind_water_density = (
        water_density_grid[neighbour_indices]
        if water_pressure_difference > 0.0
        else water_density_grid[cell_indices]
    )
    upwind_oil_density = (
        oil_density_grid[neighbour_indices]
        if oil_pressure_difference > 0.0
        else oil_density_grid[cell_indices]
    )
    upwind_gas_density = (
        gas_density_grid[neighbour_indices]
        if gas_pressure_difference > 0.0
        else gas_density_grid[cell_indices]
    )

    water_gravity_potential = (
        upwind_water_density * gravitational_constant * elevation_difference
    ) / 144.0
    oil_gravity_potential = (
        upwind_oil_density * gravitational_constant * elevation_difference
    ) / 144.0
    gas_gravity_potential = (
        upwind_gas_density * gravitational_constant * elevation_difference
    ) / 144.0

    water_potential_difference = water_pressure_difference + water_gravity_potential
    oil_potential_difference = oil_pressure_difference + oil_gravity_potential
    gas_potential_difference = gas_pressure_difference + gas_gravity_potential

    # Mobility upwinding: based on total phase potential
    upwind_water_mobility = (
        water_relative_mobility_grid[neighbour_indices]
        if water_potential_difference > 0.0
        else water_relative_mobility_grid[cell_indices]
    )
    upwind_oil_mobility = (
        oil_relative_mobility_grid[neighbour_indices]
        if oil_potential_difference > 0.0
        else oil_relative_mobility_grid[cell_indices]
    )
    upwind_gas_mobility = (
        gas_relative_mobility_grid[neighbour_indices]
        if gas_potential_difference > 0.0
        else gas_relative_mobility_grid[cell_indices]
    )

    water_flux = (
        upwind_water_mobility
        * water_potential_difference
        * face_transmissibility
        * md_per_cp_to_ft2_per_psi_per_day
    )
    oil_flux = (
        upwind_oil_mobility
        * oil_potential_difference
        * face_transmissibility
        * md_per_cp_to_ft2_per_psi_per_day
    )
    gas_flux = (
        upwind_gas_mobility
        * gas_potential_difference
        * face_transmissibility
        * md_per_cp_to_ft2_per_psi_per_day
    )

    # Return upwind density at the same upwinding direction as mobility
    selected_water_density = (
        water_density_grid[neighbour_indices]
        if water_potential_difference > 0.0
        else water_density_grid[cell_indices]
    )
    selected_oil_density = (
        oil_density_grid[neighbour_indices]
        if oil_potential_difference > 0.0
        else oil_density_grid[cell_indices]
    )
    selected_gas_density = (
        gas_density_grid[neighbour_indices]
        if gas_potential_difference > 0.0
        else gas_density_grid[cell_indices]
    )
    return (
        water_flux,
        oil_flux,
        gas_flux,
        selected_water_density,
        selected_oil_density,
        selected_gas_density,
    )


@numba.njit(parallel=True, cache=True)
def assemble_flux_contributions(
    pressure_grid: ThreeDimensionalGrid,
    cell_count_x: int,
    cell_count_y: int,
    cell_count_z: int,
    pressure_boundaries: ThreeDimensionalGrid,
    flux_boundaries: ThreeDimensionalGrid,
    water_relative_mobility_grid: ThreeDimensionalGrid,
    oil_relative_mobility_grid: ThreeDimensionalGrid,
    gas_relative_mobility_grid: ThreeDimensionalGrid,
    face_transmissibilities_x: ThreeDimensionalGrid,
    face_transmissibilities_y: ThreeDimensionalGrid,
    face_transmissibilities_z: ThreeDimensionalGrid,
    oil_water_capillary_pressure_grid: ThreeDimensionalGrid,
    gas_oil_capillary_pressure_grid: ThreeDimensionalGrid,
    water_density_grid: ThreeDimensionalGrid,
    oil_density_grid: ThreeDimensionalGrid,
    gas_density_grid: ThreeDimensionalGrid,
    solution_gas_to_oil_ratio_grid: ThreeDimensionalGrid,
    gas_solubility_in_water_grid: ThreeDimensionalGrid,
    gas_formation_volume_factor_grid: ThreeDimensionalGrid,
    oil_formation_volume_factor_grid: ThreeDimensionalGrid,
    water_formation_volume_factor_grid: ThreeDimensionalGrid,
    elevation_grid: ThreeDimensionalGrid,
    gravitational_constant: float,
    md_per_cp_to_ft2_per_psi_per_day: float,
    dtype: npt.DTypeLike,
) -> typing.Tuple[ThreeDimensionalGrid, ThreeDimensionalGrid, ThreeDimensionalGrid]:
    """
    Compute net mass fluxes into each cell from all six face neighbours, plus
    the total volumetric outflow per cell for the CFL check.

    For water, the net mass flux into cell (i,j,k) is:

        net_mass_water_flux[i,j,k] = sum_faces ( upwind_water_densitywind * F_w_face )

    where `upwind_water_densitywind` is the water density at the upwind cell determined by
    the water phase potential gradient, and `F_w_face` (ft³/day) is the
    volumetric Darcy flux for water at that face.

    For total gas (free + dissolved in oil + dissolved in water), the net mass
    flux is:

        net_mass_gas_total_flux[i,j,k] = sum_faces (
            upwind_gas_densitywind * F_g_face
            + upwind_oil_densitywind * alpha_Rs_upwind * F_o_face
            + upwind_water_densitywind * alpha_Rsw_upwind * F_w_face
        )

    where:
        `alpha_Rs  = Rs  * Bg / Bo`  (dimensionless, reservoir-condition gas
                                      volume fraction dissolved in oil phase)
        `alpha_Rsw = Rsw * Bg / Bw`  (same for water phase)
        `Rs` [SCF/STB], `Bg` [bbl/SCF], `Bo` [bbl/STB] - so Rs*Bg/Bo is
        dimensionless (reservoir-condition volume of dissolved gas per
        reservoir-condition volume of oil).

    The Rs and density values used in the face flux term are taken from the
    upwind cell for each phase, consistent with the upwinding direction of
    the phase mobility (i.e. the upwind Rs for oil is from the cell from
    which oil flows).

    Boundary faces follow the same convention as the volumetric solver:
    Dirichlet boundaries use the interior cell's mobility (one-sided flux)
    and the interior cell's density for the mass weighting. Neumann boundaries
    supply a known total volumetric flux that is split by mobility fraction and
    then density-weighted using the interior cell values.

    The `net_volumetric_outflow_grid` returned here contains the total
    volumetric outflow from each cell (ft³/day, always >= 0) for use in the
    CFL stability check. It is computed from the sum of
    `|min(0, F_w)| + |min(0, F_o)| + |min(0, F_g)|` across all faces plus
    well outflows (well outflows are added in `apply_updates`).

    :param pressure_grid: Current oil pressure grid (psi).
    :param cell_count_x: Number of cells in x-direction.
    :param cell_count_y: Number of cells in y-direction.
    :param cell_count_z: Number of cells in z-direction.
    :param pressure_boundaries: Padded boundary pressure grid (nx+2, ny+2, nz+2).
        NaN indicates a Neumann face.
    :param flux_boundaries: Padded boundary flux grid (nx+2, ny+2, nz+2).
    :param water_relative_mobility_grid: Water relative mobility (ft²/psi·day).
    :param oil_relative_mobility_grid: Oil relative mobility (ft²/psi·day).
    :param gas_relative_mobility_grid: Gas relative mobility (ft²/psi·day).
    :param face_transmissibilities_x: x-direction face transmissibilities (mD·ft).
    :param face_transmissibilities_y: y-direction face transmissibilities (mD·ft).
    :param face_transmissibilities_z: z-direction face transmissibilities (mD·ft).
    :param oil_water_capillary_pressure_grid: Oil-water capillary pressure (psi).
    :param gas_oil_capillary_pressure_grid: Gas-oil capillary pressure (psi).
    :param water_density_grid: Water density at new pressure (lb/ft³).
    :param oil_density_grid: Oil effective density at new pressure (lb/ft³).
    :param gas_density_grid: Gas density at new pressure (lb/ft³).
    :param solution_gas_to_oil_ratio_grid: Rs at new pressure (SCF/STB).
    :param gas_solubility_in_water_grid: Rsw at new pressure (SCF/STB).
    :param gas_formation_volume_factor_grid: Bg at new pressure (bbl/SCF).
    :param oil_formation_volume_factor_grid: Bo at new pressure (bbl/STB).
    :param water_formation_volume_factor_grid: Bw at new pressure (bbl/STB).
    :param elevation_grid: Cell elevation (ft).
    :param gravitational_constant: g/gc conversion factor (lbf/lbm).
    :param md_per_cp_to_ft2_per_psi_per_day: Unit conversion factor.
    :param dtype: Numpy dtype for output arrays.
    :return: Tuple of (`net_water_mass_flux_grid`, `net_gas_total_mass_flux_grid`,
        `net_volumetric_outflow_grid`). Mass flux units are lbm/day; volumetric
        outflow units are ft³/day (always >= 0).
    """
    net_water_mass_flux_grid = np.zeros(
        (cell_count_x, cell_count_y, cell_count_z), dtype=dtype
    )
    net_gas_total_mass_flux_grid = np.zeros(
        (cell_count_x, cell_count_y, cell_count_z), dtype=dtype
    )
    # Total volumetric outflow per cell used for CFL checks
    net_volumetric_outflow_grid = np.zeros(
        (cell_count_x, cell_count_y, cell_count_z), dtype=dtype
    )

    for i in numba.prange(cell_count_x):  # type: ignore
        for j in range(cell_count_y):
            for k in range(cell_count_z):
                cell_pressure = pressure_grid[i, j, k]
                cell_water_mobility = water_relative_mobility_grid[i, j, k]
                cell_oil_mobility = oil_relative_mobility_grid[i, j, k]
                cell_gas_mobility = gas_relative_mobility_grid[i, j, k]
                cell_total_mobility = (
                    cell_water_mobility + cell_oil_mobility + cell_gas_mobility
                )

                # Interior cell PVT values used for boundary mass weighting
                cell_water_density = water_density_grid[i, j, k]
                cell_oil_density = oil_density_grid[i, j, k]
                cell_gas_density = gas_density_grid[i, j, k]

                safe_oil_fvf = oil_formation_volume_factor_grid[i, j, k]
                safe_water_fvf = water_formation_volume_factor_grid[i, j, k]
                safe_gas_fvf = gas_formation_volume_factor_grid[i, j, k]
                if safe_oil_fvf < 1e-30:
                    safe_oil_fvf = 1e-30
                if safe_water_fvf < 1e-30:
                    safe_water_fvf = 1e-30
                if safe_gas_fvf < 1e-30:
                    safe_gas_fvf = 1e-30

                # alpha_Rs and alpha_Rsw for interior cell (used in boundary faces)
                cell_alpha_solution_gor = (
                    solution_gas_to_oil_ratio_grid[i, j, k]
                    * safe_gas_fvf
                    / safe_oil_fvf
                )
                cell_alpha_gas_solubility_in_water = (
                    gas_solubility_in_water_grid[i, j, k]
                    * safe_gas_fvf
                    / safe_water_fvf
                )

                net_mass_water_flux = 0.0
                net_mass_gas_total_flux = 0.0
                volumetric_outflow = 0.0

                # EAST (i+1, j, k)
                east_i = i + 1
                if east_i < cell_count_x:
                    (
                        water_flux,
                        oil_flux,
                        gas_flux,
                        upwind_water_density,
                        upwind_oil_density,
                        upwind_gas_density,
                    ) = compute_face_fluxes(
                        cell_indices=(i, j, k),
                        neighbour_indices=(east_i, j, k),
                        pressure_grid=pressure_grid,
                        face_transmissibility=face_transmissibilities_x[
                            i + 1, j + 1, k + 1
                        ],
                        water_relative_mobility_grid=water_relative_mobility_grid,
                        oil_relative_mobility_grid=oil_relative_mobility_grid,
                        gas_relative_mobility_grid=gas_relative_mobility_grid,
                        oil_water_capillary_pressure_grid=oil_water_capillary_pressure_grid,
                        gas_oil_capillary_pressure_grid=gas_oil_capillary_pressure_grid,
                        oil_density_grid=oil_density_grid,
                        water_density_grid=water_density_grid,
                        gas_density_grid=gas_density_grid,
                        elevation_grid=elevation_grid,
                        gravitational_constant=gravitational_constant,
                        md_per_cp_to_ft2_per_psi_per_day=md_per_cp_to_ft2_per_psi_per_day,
                    )
                    # Upwind Rs/Rsw for oil and water faces
                    if oil_flux > 0.0:
                        alpha_solution_gor_face = (
                            solution_gas_to_oil_ratio_grid[east_i, j, k]
                            * gas_formation_volume_factor_grid[east_i, j, k]
                            / max(oil_formation_volume_factor_grid[east_i, j, k], 1e-30)
                        )
                    else:
                        alpha_solution_gor_face = cell_alpha_solution_gor

                    if water_flux > 0.0:
                        alpha_gas_solubility_in_water_face = (
                            gas_solubility_in_water_grid[east_i, j, k]
                            * gas_formation_volume_factor_grid[east_i, j, k]
                            / max(
                                water_formation_volume_factor_grid[east_i, j, k], 1e-30
                            )
                        )
                    else:
                        alpha_gas_solubility_in_water_face = (
                            cell_alpha_gas_solubility_in_water
                        )

                    net_mass_water_flux += upwind_water_density * water_flux
                    net_mass_gas_total_flux += (
                        upwind_gas_density * gas_flux
                        + upwind_oil_density * alpha_solution_gor_face * oil_flux
                        + upwind_water_density
                        * alpha_gas_solubility_in_water_face
                        * water_flux
                    )
                    volumetric_outflow += abs(min(0.0, water_flux))
                    volumetric_outflow += abs(min(0.0, oil_flux))
                    volumetric_outflow += abs(min(0.0, gas_flux))
                else:
                    pei, pej, pek = east_i + 1, j + 1, k + 1
                    pressure_boundary = pressure_boundaries[pei, pej, pek]
                    T = (
                        face_transmissibilities_x[pei, pej, pek]
                        * md_per_cp_to_ft2_per_psi_per_day
                    )
                    if not np.isnan(pressure_boundary):
                        pressure_difference = pressure_boundary - cell_pressure
                        water_flux = cell_water_mobility * T * pressure_difference
                        oil_flux = cell_oil_mobility * T * pressure_difference
                        gas_flux = cell_gas_mobility * T * pressure_difference
                        net_mass_water_flux += cell_water_density * water_flux
                        net_mass_gas_total_flux += (
                            cell_gas_density * gas_flux
                            + cell_oil_density * cell_alpha_solution_gor * oil_flux
                            + cell_water_density
                            * cell_alpha_gas_solubility_in_water
                            * water_flux
                        )
                        volumetric_outflow += abs(min(0.0, water_flux))
                        volumetric_outflow += abs(min(0.0, oil_flux))
                        volumetric_outflow += abs(min(0.0, gas_flux))
                    else:
                        flux_boundary = flux_boundaries[pei, pej, pek]
                        if cell_total_mobility > 0.0:
                            water_fraction = cell_water_mobility / cell_total_mobility
                            oil_fraction = cell_oil_mobility / cell_total_mobility
                            gas_fraction = cell_gas_mobility / cell_total_mobility
                            water_flux = flux_boundary * water_fraction
                            oil_flux = flux_boundary * oil_fraction
                            gas_flux = flux_boundary * gas_fraction
                            net_mass_water_flux += cell_water_density * water_flux
                            net_mass_gas_total_flux += (
                                cell_gas_density * gas_flux
                                + cell_oil_density * cell_alpha_solution_gor * oil_flux
                                + cell_water_density
                                * cell_alpha_gas_solubility_in_water
                                * water_flux
                            )
                            volumetric_outflow += abs(min(0.0, water_flux))
                            volumetric_outflow += abs(min(0.0, oil_flux))
                            volumetric_outflow += abs(min(0.0, gas_flux))

                # WEST (i-1, j, k)
                west_i = i - 1
                pwi, pwj, pwk = west_i + 1, j + 1, k + 1
                if west_i >= 0:
                    (
                        water_flux,
                        oil_flux,
                        gas_flux,
                        upwind_water_density,
                        upwind_oil_density,
                        upwind_gas_density,
                    ) = compute_face_fluxes(
                        cell_indices=(i, j, k),
                        neighbour_indices=(west_i, j, k),
                        pressure_grid=pressure_grid,
                        face_transmissibility=face_transmissibilities_x[pwi, pwj, pwk],
                        water_relative_mobility_grid=water_relative_mobility_grid,
                        oil_relative_mobility_grid=oil_relative_mobility_grid,
                        gas_relative_mobility_grid=gas_relative_mobility_grid,
                        oil_water_capillary_pressure_grid=oil_water_capillary_pressure_grid,
                        gas_oil_capillary_pressure_grid=gas_oil_capillary_pressure_grid,
                        oil_density_grid=oil_density_grid,
                        water_density_grid=water_density_grid,
                        gas_density_grid=gas_density_grid,
                        elevation_grid=elevation_grid,
                        gravitational_constant=gravitational_constant,
                        md_per_cp_to_ft2_per_psi_per_day=md_per_cp_to_ft2_per_psi_per_day,
                    )
                    if oil_flux > 0.0:
                        alpha_solution_gor_face = (
                            solution_gas_to_oil_ratio_grid[west_i, j, k]
                            * gas_formation_volume_factor_grid[west_i, j, k]
                            / max(oil_formation_volume_factor_grid[west_i, j, k], 1e-30)
                        )
                    else:
                        alpha_solution_gor_face = cell_alpha_solution_gor

                    if water_flux > 0.0:
                        alpha_gas_solubility_in_water_face = (
                            gas_solubility_in_water_grid[west_i, j, k]
                            * gas_formation_volume_factor_grid[west_i, j, k]
                            / max(
                                water_formation_volume_factor_grid[west_i, j, k], 1e-30
                            )
                        )
                    else:
                        alpha_gas_solubility_in_water_face = (
                            cell_alpha_gas_solubility_in_water
                        )

                    net_mass_water_flux += upwind_water_density * water_flux
                    net_mass_gas_total_flux += (
                        upwind_gas_density * gas_flux
                        + upwind_oil_density * alpha_solution_gor_face * oil_flux
                        + upwind_water_density
                        * alpha_gas_solubility_in_water_face
                        * water_flux
                    )
                    volumetric_outflow += abs(min(0.0, water_flux))
                    volumetric_outflow += abs(min(0.0, oil_flux))
                    volumetric_outflow += abs(min(0.0, gas_flux))
                else:
                    pressure_boundary = pressure_boundaries[pwi, pwj, pwk]
                    T = (
                        face_transmissibilities_x[pwi, pwj, pwk]
                        * md_per_cp_to_ft2_per_psi_per_day
                    )
                    if not np.isnan(pressure_boundary):
                        pressure_difference = pressure_boundary - cell_pressure
                        water_flux = cell_water_mobility * T * pressure_difference
                        oil_flux = cell_oil_mobility * T * pressure_difference
                        gas_flux = cell_gas_mobility * T * pressure_difference
                        net_mass_water_flux += cell_water_density * water_flux
                        net_mass_gas_total_flux += (
                            cell_gas_density * gas_flux
                            + cell_oil_density * cell_alpha_solution_gor * oil_flux
                            + cell_water_density
                            * cell_alpha_gas_solubility_in_water
                            * water_flux
                        )
                        volumetric_outflow += abs(min(0.0, water_flux))
                        volumetric_outflow += abs(min(0.0, oil_flux))
                        volumetric_outflow += abs(min(0.0, gas_flux))
                    else:
                        flux_boundary = flux_boundaries[pwi, pwj, pwk]
                        if cell_total_mobility > 0.0:
                            water_fraction = cell_water_mobility / cell_total_mobility
                            oil_fraction = cell_oil_mobility / cell_total_mobility
                            gas_fraction = cell_gas_mobility / cell_total_mobility
                            water_flux = flux_boundary * water_fraction
                            oil_flux = flux_boundary * oil_fraction
                            gas_flux = flux_boundary * gas_fraction
                            net_mass_water_flux += cell_water_density * water_flux
                            net_mass_gas_total_flux += (
                                cell_gas_density * gas_flux
                                + cell_oil_density * cell_alpha_solution_gor * oil_flux
                                + cell_water_density
                                * cell_alpha_gas_solubility_in_water
                                * water_flux
                            )
                            volumetric_outflow += abs(min(0.0, water_flux))
                            volumetric_outflow += abs(min(0.0, oil_flux))
                            volumetric_outflow += abs(min(0.0, gas_flux))

                # SOUTH (i, j+1, k)
                south_j = j + 1
                if south_j < cell_count_y:
                    (
                        water_flux,
                        oil_flux,
                        gas_flux,
                        upwind_water_density,
                        upwind_oil_density,
                        upwind_gas_density,
                    ) = compute_face_fluxes(
                        cell_indices=(i, j, k),
                        neighbour_indices=(i, south_j, k),
                        pressure_grid=pressure_grid,
                        face_transmissibility=face_transmissibilities_y[
                            i + 1, j + 1, k + 1
                        ],
                        water_relative_mobility_grid=water_relative_mobility_grid,
                        oil_relative_mobility_grid=oil_relative_mobility_grid,
                        gas_relative_mobility_grid=gas_relative_mobility_grid,
                        oil_water_capillary_pressure_grid=oil_water_capillary_pressure_grid,
                        gas_oil_capillary_pressure_grid=gas_oil_capillary_pressure_grid,
                        oil_density_grid=oil_density_grid,
                        water_density_grid=water_density_grid,
                        gas_density_grid=gas_density_grid,
                        elevation_grid=elevation_grid,
                        gravitational_constant=gravitational_constant,
                        md_per_cp_to_ft2_per_psi_per_day=md_per_cp_to_ft2_per_psi_per_day,
                    )
                    if oil_flux > 0.0:
                        alpha_solution_gor_face = (
                            solution_gas_to_oil_ratio_grid[i, south_j, k]
                            * gas_formation_volume_factor_grid[i, south_j, k]
                            / max(
                                oil_formation_volume_factor_grid[i, south_j, k], 1e-30
                            )
                        )
                    else:
                        alpha_solution_gor_face = cell_alpha_solution_gor

                    if water_flux > 0.0:
                        alpha_gas_solubility_in_water_face = (
                            gas_solubility_in_water_grid[i, south_j, k]
                            * gas_formation_volume_factor_grid[i, south_j, k]
                            / max(
                                water_formation_volume_factor_grid[i, south_j, k], 1e-30
                            )
                        )
                    else:
                        alpha_gas_solubility_in_water_face = (
                            cell_alpha_gas_solubility_in_water
                        )

                    net_mass_water_flux += upwind_water_density * water_flux
                    net_mass_gas_total_flux += (
                        upwind_gas_density * gas_flux
                        + upwind_oil_density * alpha_solution_gor_face * oil_flux
                        + upwind_water_density
                        * alpha_gas_solubility_in_water_face
                        * water_flux
                    )
                    volumetric_outflow += abs(min(0.0, water_flux))
                    volumetric_outflow += abs(min(0.0, oil_flux))
                    volumetric_outflow += abs(min(0.0, gas_flux))
                else:
                    psi, psj, psk = i + 1, south_j + 1, k + 1
                    pressure_boundary = pressure_boundaries[psi, psj, psk]
                    T = (
                        face_transmissibilities_y[psi, psj, psk]
                        * md_per_cp_to_ft2_per_psi_per_day
                    )
                    if not np.isnan(pressure_boundary):
                        pressure_difference = pressure_boundary - cell_pressure
                        water_flux = cell_water_mobility * T * pressure_difference
                        oil_flux = cell_oil_mobility * T * pressure_difference
                        gas_flux = cell_gas_mobility * T * pressure_difference
                        net_mass_water_flux += cell_water_density * water_flux
                        net_mass_gas_total_flux += (
                            cell_gas_density * gas_flux
                            + cell_oil_density * cell_alpha_solution_gor * oil_flux
                            + cell_water_density
                            * cell_alpha_gas_solubility_in_water
                            * water_flux
                        )
                        volumetric_outflow += abs(min(0.0, water_flux))
                        volumetric_outflow += abs(min(0.0, oil_flux))
                        volumetric_outflow += abs(min(0.0, gas_flux))
                    else:
                        flux_boundary = flux_boundaries[psi, psj, psk]
                        if cell_total_mobility > 0.0:
                            water_fraction = cell_water_mobility / cell_total_mobility
                            oil_fraction = cell_oil_mobility / cell_total_mobility
                            gas_fraction = cell_gas_mobility / cell_total_mobility
                            water_flux = flux_boundary * water_fraction
                            oil_flux = flux_boundary * oil_fraction
                            gas_flux = flux_boundary * gas_fraction
                            net_mass_water_flux += cell_water_density * water_flux
                            net_mass_gas_total_flux += (
                                cell_gas_density * gas_flux
                                + cell_oil_density * cell_alpha_solution_gor * oil_flux
                                + cell_water_density
                                * cell_alpha_gas_solubility_in_water
                                * water_flux
                            )
                            volumetric_outflow += abs(min(0.0, water_flux))
                            volumetric_outflow += abs(min(0.0, oil_flux))
                            volumetric_outflow += abs(min(0.0, gas_flux))

                # NORTH (i, j-1, k)
                north_j = j - 1
                pni, pnj, pnk = i + 1, north_j + 1, k + 1
                if north_j >= 0:
                    (
                        water_flux,
                        oil_flux,
                        gas_flux,
                        upwind_water_density,
                        upwind_oil_density,
                        upwind_gas_density,
                    ) = compute_face_fluxes(
                        cell_indices=(i, j, k),
                        neighbour_indices=(i, north_j, k),
                        pressure_grid=pressure_grid,
                        face_transmissibility=face_transmissibilities_y[pni, pnj, pnk],
                        water_relative_mobility_grid=water_relative_mobility_grid,
                        oil_relative_mobility_grid=oil_relative_mobility_grid,
                        gas_relative_mobility_grid=gas_relative_mobility_grid,
                        oil_water_capillary_pressure_grid=oil_water_capillary_pressure_grid,
                        gas_oil_capillary_pressure_grid=gas_oil_capillary_pressure_grid,
                        oil_density_grid=oil_density_grid,
                        water_density_grid=water_density_grid,
                        gas_density_grid=gas_density_grid,
                        elevation_grid=elevation_grid,
                        gravitational_constant=gravitational_constant,
                        md_per_cp_to_ft2_per_psi_per_day=md_per_cp_to_ft2_per_psi_per_day,
                    )
                    if oil_flux > 0.0:
                        alpha_solution_gor_face = (
                            solution_gas_to_oil_ratio_grid[i, north_j, k]
                            * gas_formation_volume_factor_grid[i, north_j, k]
                            / max(
                                oil_formation_volume_factor_grid[i, north_j, k], 1e-30
                            )
                        )
                    else:
                        alpha_solution_gor_face = cell_alpha_solution_gor

                    if water_flux > 0.0:
                        alpha_gas_solubility_in_water_face = (
                            gas_solubility_in_water_grid[i, north_j, k]
                            * gas_formation_volume_factor_grid[i, north_j, k]
                            / max(
                                water_formation_volume_factor_grid[i, north_j, k], 1e-30
                            )
                        )
                    else:
                        alpha_gas_solubility_in_water_face = (
                            cell_alpha_gas_solubility_in_water
                        )

                    net_mass_water_flux += upwind_water_density * water_flux
                    net_mass_gas_total_flux += (
                        upwind_gas_density * gas_flux
                        + upwind_oil_density * alpha_solution_gor_face * oil_flux
                        + upwind_water_density
                        * alpha_gas_solubility_in_water_face
                        * water_flux
                    )
                    volumetric_outflow += abs(min(0.0, water_flux))
                    volumetric_outflow += abs(min(0.0, oil_flux))
                    volumetric_outflow += abs(min(0.0, gas_flux))
                else:
                    pressure_boundary = pressure_boundaries[pni, pnj, pnk]
                    T = (
                        face_transmissibilities_y[pni, pnj, pnk]
                        * md_per_cp_to_ft2_per_psi_per_day
                    )
                    if not np.isnan(pressure_boundary):
                        pressure_difference = pressure_boundary - cell_pressure
                        water_flux = cell_water_mobility * T * pressure_difference
                        oil_flux = cell_oil_mobility * T * pressure_difference
                        gas_flux = cell_gas_mobility * T * pressure_difference
                        net_mass_water_flux += cell_water_density * water_flux
                        net_mass_gas_total_flux += (
                            cell_gas_density * gas_flux
                            + cell_oil_density * cell_alpha_solution_gor * oil_flux
                            + cell_water_density
                            * cell_alpha_gas_solubility_in_water
                            * water_flux
                        )
                        volumetric_outflow += abs(min(0.0, water_flux))
                        volumetric_outflow += abs(min(0.0, oil_flux))
                        volumetric_outflow += abs(min(0.0, gas_flux))
                    else:
                        flux_boundary = flux_boundaries[pni, pnj, pnk]
                        if cell_total_mobility > 0.0:
                            water_fraction = cell_water_mobility / cell_total_mobility
                            oil_fraction = cell_oil_mobility / cell_total_mobility
                            gas_fraction = cell_gas_mobility / cell_total_mobility
                            water_flux = flux_boundary * water_fraction
                            oil_flux = flux_boundary * oil_fraction
                            gas_flux = flux_boundary * gas_fraction
                            net_mass_water_flux += cell_water_density * water_flux
                            net_mass_gas_total_flux += (
                                cell_gas_density * gas_flux
                                + cell_oil_density * cell_alpha_solution_gor * oil_flux
                                + cell_water_density
                                * cell_alpha_gas_solubility_in_water
                                * water_flux
                            )
                            volumetric_outflow += abs(min(0.0, water_flux))
                            volumetric_outflow += abs(min(0.0, oil_flux))
                            volumetric_outflow += abs(min(0.0, gas_flux))

                # BOTTOM (i, j, k+1)
                bottom_k = k + 1
                if bottom_k < cell_count_z:
                    (
                        water_flux,
                        oil_flux,
                        gas_flux,
                        upwind_water_density,
                        upwind_oil_density,
                        upwind_gas_density,
                    ) = compute_face_fluxes(
                        cell_indices=(i, j, k),
                        neighbour_indices=(i, j, bottom_k),
                        pressure_grid=pressure_grid,
                        face_transmissibility=face_transmissibilities_z[
                            i + 1, j + 1, k + 1
                        ],
                        water_relative_mobility_grid=water_relative_mobility_grid,
                        oil_relative_mobility_grid=oil_relative_mobility_grid,
                        gas_relative_mobility_grid=gas_relative_mobility_grid,
                        oil_water_capillary_pressure_grid=oil_water_capillary_pressure_grid,
                        gas_oil_capillary_pressure_grid=gas_oil_capillary_pressure_grid,
                        oil_density_grid=oil_density_grid,
                        water_density_grid=water_density_grid,
                        gas_density_grid=gas_density_grid,
                        elevation_grid=elevation_grid,
                        gravitational_constant=gravitational_constant,
                        md_per_cp_to_ft2_per_psi_per_day=md_per_cp_to_ft2_per_psi_per_day,
                    )
                    if oil_flux > 0.0:
                        alpha_solution_gor_face = (
                            solution_gas_to_oil_ratio_grid[i, j, bottom_k]
                            * gas_formation_volume_factor_grid[i, j, bottom_k]
                            / max(
                                oil_formation_volume_factor_grid[i, j, bottom_k], 1e-30
                            )
                        )
                    else:
                        alpha_solution_gor_face = cell_alpha_solution_gor

                    if water_flux > 0.0:
                        alpha_gas_solubility_in_water_face = (
                            gas_solubility_in_water_grid[i, j, bottom_k]
                            * gas_formation_volume_factor_grid[i, j, bottom_k]
                            / max(
                                water_formation_volume_factor_grid[i, j, bottom_k],
                                1e-30,
                            )
                        )
                    else:
                        alpha_gas_solubility_in_water_face = (
                            cell_alpha_gas_solubility_in_water
                        )

                    net_mass_water_flux += upwind_water_density * water_flux
                    net_mass_gas_total_flux += (
                        upwind_gas_density * gas_flux
                        + upwind_oil_density * alpha_solution_gor_face * oil_flux
                        + upwind_water_density
                        * alpha_gas_solubility_in_water_face
                        * water_flux
                    )
                    volumetric_outflow += abs(min(0.0, water_flux))
                    volumetric_outflow += abs(min(0.0, oil_flux))
                    volumetric_outflow += abs(min(0.0, gas_flux))
                else:
                    pbi, pbj, pbk = i + 1, j + 1, bottom_k + 1
                    pressure_boundary = pressure_boundaries[pbi, pbj, pbk]
                    T = (
                        face_transmissibilities_z[pbi, pbj, pbk]
                        * md_per_cp_to_ft2_per_psi_per_day
                    )
                    if not np.isnan(pressure_boundary):
                        pressure_difference = pressure_boundary - cell_pressure
                        water_flux = cell_water_mobility * T * pressure_difference
                        oil_flux = cell_oil_mobility * T * pressure_difference
                        gas_flux = cell_gas_mobility * T * pressure_difference
                        net_mass_water_flux += cell_water_density * water_flux
                        net_mass_gas_total_flux += (
                            cell_gas_density * gas_flux
                            + cell_oil_density * cell_alpha_solution_gor * oil_flux
                            + cell_water_density
                            * cell_alpha_gas_solubility_in_water
                            * water_flux
                        )
                        volumetric_outflow += abs(min(0.0, water_flux))
                        volumetric_outflow += abs(min(0.0, oil_flux))
                        volumetric_outflow += abs(min(0.0, gas_flux))
                    else:
                        flux_boundary = flux_boundaries[pbi, pbj, pbk]
                        if cell_total_mobility > 0.0:
                            water_fraction = cell_water_mobility / cell_total_mobility
                            oil_fraction = cell_oil_mobility / cell_total_mobility
                            gas_fraction = cell_gas_mobility / cell_total_mobility
                            water_flux = flux_boundary * water_fraction
                            oil_flux = flux_boundary * oil_fraction
                            gas_flux = flux_boundary * gas_fraction
                            net_mass_water_flux += cell_water_density * water_flux
                            net_mass_gas_total_flux += (
                                cell_gas_density * gas_flux
                                + cell_oil_density * cell_alpha_solution_gor * oil_flux
                                + cell_water_density
                                * cell_alpha_gas_solubility_in_water
                                * water_flux
                            )
                            volumetric_outflow += abs(min(0.0, water_flux))
                            volumetric_outflow += abs(min(0.0, oil_flux))
                            volumetric_outflow += abs(min(0.0, gas_flux))

                # TOP (i, j, k-1)
                top_k = k - 1
                pti, ptj, ptk = i + 1, j + 1, top_k + 1
                if top_k >= 0:
                    (
                        water_flux,
                        oil_flux,
                        gas_flux,
                        upwind_water_density,
                        upwind_oil_density,
                        upwind_gas_density,
                    ) = compute_face_fluxes(
                        cell_indices=(i, j, k),
                        neighbour_indices=(i, j, top_k),
                        pressure_grid=pressure_grid,
                        face_transmissibility=face_transmissibilities_z[pti, ptj, ptk],
                        water_relative_mobility_grid=water_relative_mobility_grid,
                        oil_relative_mobility_grid=oil_relative_mobility_grid,
                        gas_relative_mobility_grid=gas_relative_mobility_grid,
                        oil_water_capillary_pressure_grid=oil_water_capillary_pressure_grid,
                        gas_oil_capillary_pressure_grid=gas_oil_capillary_pressure_grid,
                        oil_density_grid=oil_density_grid,
                        water_density_grid=water_density_grid,
                        gas_density_grid=gas_density_grid,
                        elevation_grid=elevation_grid,
                        gravitational_constant=gravitational_constant,
                        md_per_cp_to_ft2_per_psi_per_day=md_per_cp_to_ft2_per_psi_per_day,
                    )
                    if oil_flux > 0.0:
                        alpha_solution_gor_face = (
                            solution_gas_to_oil_ratio_grid[i, j, top_k]
                            * gas_formation_volume_factor_grid[i, j, top_k]
                            / max(oil_formation_volume_factor_grid[i, j, top_k], 1e-30)
                        )
                    else:
                        alpha_solution_gor_face = cell_alpha_solution_gor

                    if water_flux > 0.0:
                        alpha_gas_solubility_in_water_face = (
                            gas_solubility_in_water_grid[i, j, top_k]
                            * gas_formation_volume_factor_grid[i, j, top_k]
                            / max(
                                water_formation_volume_factor_grid[i, j, top_k], 1e-30
                            )
                        )
                    else:
                        alpha_gas_solubility_in_water_face = (
                            cell_alpha_gas_solubility_in_water
                        )

                    net_mass_water_flux += upwind_water_density * water_flux
                    net_mass_gas_total_flux += (
                        upwind_gas_density * gas_flux
                        + upwind_oil_density * alpha_solution_gor_face * oil_flux
                        + upwind_water_density
                        * alpha_gas_solubility_in_water_face
                        * water_flux
                    )
                    volumetric_outflow += abs(min(0.0, water_flux))
                    volumetric_outflow += abs(min(0.0, oil_flux))
                    volumetric_outflow += abs(min(0.0, gas_flux))
                else:
                    pressure_boundary = pressure_boundaries[pti, ptj, ptk]
                    T = (
                        face_transmissibilities_z[pti, ptj, ptk]
                        * md_per_cp_to_ft2_per_psi_per_day
                    )
                    if not np.isnan(pressure_boundary):
                        pressure_difference = pressure_boundary - cell_pressure
                        water_flux = cell_water_mobility * T * pressure_difference
                        oil_flux = cell_oil_mobility * T * pressure_difference
                        gas_flux = cell_gas_mobility * T * pressure_difference
                        net_mass_water_flux += cell_water_density * water_flux
                        net_mass_gas_total_flux += (
                            cell_gas_density * gas_flux
                            + cell_oil_density * cell_alpha_solution_gor * oil_flux
                            + cell_water_density
                            * cell_alpha_gas_solubility_in_water
                            * water_flux
                        )
                        volumetric_outflow += abs(min(0.0, water_flux))
                        volumetric_outflow += abs(min(0.0, oil_flux))
                        volumetric_outflow += abs(min(0.0, gas_flux))
                    else:
                        flux_boundary = flux_boundaries[pti, ptj, ptk]
                        if cell_total_mobility > 0.0:
                            water_fraction = cell_water_mobility / cell_total_mobility
                            oil_fraction = cell_oil_mobility / cell_total_mobility
                            gas_fraction = cell_gas_mobility / cell_total_mobility
                            water_flux = flux_boundary * water_fraction
                            oil_flux = flux_boundary * oil_fraction
                            gas_flux = flux_boundary * gas_fraction
                            net_mass_water_flux += cell_water_density * water_flux
                            net_mass_gas_total_flux += (
                                cell_gas_density * gas_flux
                                + cell_oil_density * cell_alpha_solution_gor * oil_flux
                                + cell_water_density
                                * cell_alpha_gas_solubility_in_water
                                * water_flux
                            )
                            volumetric_outflow += abs(min(0.0, water_flux))
                            volumetric_outflow += abs(min(0.0, oil_flux))
                            volumetric_outflow += abs(min(0.0, gas_flux))

                net_water_mass_flux_grid[i, j, k] = net_mass_water_flux
                net_gas_total_mass_flux_grid[i, j, k] = net_mass_gas_total_flux
                net_volumetric_outflow_grid[i, j, k] = volumetric_outflow

    return (
        net_water_mass_flux_grid,
        net_gas_total_mass_flux_grid,
        net_volumetric_outflow_grid,
    )


@numba.njit(parallel=True, cache=True)
def apply_updates(
    updated_water_saturation_grid: ThreeDimensionalGrid,
    updated_oil_saturation_grid: ThreeDimensionalGrid,
    updated_gas_saturation_grid: ThreeDimensionalGrid,
    old_water_saturation_grid: ThreeDimensionalGrid,
    old_oil_saturation_grid: ThreeDimensionalGrid,
    old_gas_saturation_grid: ThreeDimensionalGrid,
    net_water_mass_flux_grid: ThreeDimensionalGrid,
    net_gas_total_mass_flux_grid: ThreeDimensionalGrid,
    net_volumetric_outflow_grid: ThreeDimensionalGrid,
    net_water_well_rate_grid: ThreeDimensionalGrid,
    net_oil_well_rate_grid: ThreeDimensionalGrid,
    net_gas_well_rate_grid: ThreeDimensionalGrid,
    net_water_well_mass_rate_grid: ThreeDimensionalGrid,
    net_oil_well_mass_rate_grid: ThreeDimensionalGrid,
    net_gas_well_mass_rate_grid: ThreeDimensionalGrid,
    old_water_density_grid: ThreeDimensionalGrid,
    old_oil_density_grid: ThreeDimensionalGrid,
    old_gas_density_grid: ThreeDimensionalGrid,
    current_water_density_grid: ThreeDimensionalGrid,
    current_oil_density_grid: ThreeDimensionalGrid,
    current_gas_density_grid: ThreeDimensionalGrid,
    solution_gas_to_oil_ratio_grid: ThreeDimensionalGrid,
    gas_solubility_in_water_grid: ThreeDimensionalGrid,
    gas_formation_volume_factor_grid: ThreeDimensionalGrid,
    oil_formation_volume_factor_grid: ThreeDimensionalGrid,
    water_formation_volume_factor_grid: ThreeDimensionalGrid,
    old_solution_gas_to_oil_ratio_grid: ThreeDimensionalGrid,
    old_gas_solubility_in_water_grid: ThreeDimensionalGrid,
    old_gas_formation_volume_factor_grid: ThreeDimensionalGrid,
    old_oil_formation_volume_factor_grid: ThreeDimensionalGrid,
    old_water_formation_volume_factor_grid: ThreeDimensionalGrid,
    cell_count_x: int,
    cell_count_y: int,
    cell_count_z: int,
    thickness_grid: ThreeDimensionalGrid,
    porosity_grid: ThreeDimensionalGrid,
    net_to_gross_grid: ThreeDimensionalGrid,
    cell_size_x: float,
    cell_size_y: float,
    time_step_in_days: float,
    cfl_threshold: float,
    dtype: npt.DTypeLike,
) -> typing.Tuple[
    ThreeDimensionalGrid,
    ThreeDimensionalGrid,
    ThreeDimensionalGrid,
    OneDimensionalGrid,
]:
    """
    Apply mass-based saturation updates using explicit forward Euler.

    The update sequence for each cell is:

    1. **Water** - advance water mass, divide by new water density to get `Sw_new`:

           M_w_old = old_water_density * Sw_old
           M_w_new = M_w_old + (dt/phi*V) * (net_mass_water_flux + cell_water_density * q_w_vol)
           Sw_new  = M_w_new / current_water_density

       The well water rate `q_w_vol` is volumetric (ft³/day); it is converted
       to a mass rate by multiplying by the cell water density.

    2. **Gas (total mass)** - advance total gas mass (free + dissolved):

           M_g_old = old_gas_density*Sg_old + old_oil_density*alpha_Rs_old*So_old + old_water_density*alpha_Rsw_old*Sw_old
           M_g_new = M_g_old + (dt/phi*V) * (net_mass_gas_total_flux + well_gas_mass_rate)

       The well mass gas rate includes solution gas carried in produced/injected
       oil and water:

           well_gas_mass_rate = cell_gas_density*q_g_vol + cel_oil_densityl*alpha_Rs_cell*q_o_vol
                                + cell_water_density*alpha_Rsw_cell*q_w_vol

    3. **Oil saturation (derived)** - `So_new = 1 - Sw_new - Sg_new`.

    4. **Free gas** - recover `Sg_new` from the new total gas mass minus the
       dissolved portion in the new oil and water:

           current_gas_density * Sg_new = M_g_new
                                - current_oil_density * alpha_Rs_new * So_new
                                - current_water_density * alpha_Rsw_new * Sw_new
           Sg_new = max(0, above) / current_gas_density

    Saturations are clamped to [0, 1] and a residual volume balance correction
    is applied to oil (any remaining tiny gap from `Sw + So + Sg != 1` is
    absorbed by `So`).

    The CFL check is performed on total volumetric outflow (from `net_volumetric_outflow_grid`)
    plus well outflows, relative to pore volume.

    :param updated_water_saturation_grid: Output water saturation (modified in-place).
    :param updated_oil_saturation_grid: Output oil saturation (modified in-place).
    :param updated_gas_saturation_grid: Output gas saturation (modified in-place).
    :param old_water_saturation_grid: Water saturation at start of time step.
    :param old_oil_saturation_grid: Oil saturation at start of time step.
    :param old_gas_saturation_grid: Gas saturation at start of time step.
    :param net_water_mass_flux_grid: Net mass water flux into each cell (lbm/day),
        from `assemble_flux_contributions`.
    :param net_gas_total_mass_flux_grid: Net total gas mass flux into each cell
        (lbm/day), from `assemble_flux_contributions`.
    :param net_volumetric_outflow_grid: Total volumetric outflow from face fluxes
        only (ft³/day, >= 0), used for CFL check.
    :param net_water_well_rate_grid: Volumetric water well rate per cell (ft³/day).
    :param net_oil_well_rate_grid: Volumetric oil well rate per cell (ft³/day).
    :param net_gas_well_rate_grid: Volumetric gas well rate per cell (ft³/day).
    :param old_water_density_grid: Water density at start-of-step pressure (lb/ft³).
    :param old_oil_density_grid: Oil effective density at start-of-step pressure (lb/ft³).
    :param old_gas_density_grid: Gas density at start-of-step pressure (lb/ft³).
    :param current_water_density_grid: Water density at new pressure (lb/ft³).
    :param current_oil_density_grid: Oil effective density at new pressure (lb/ft³).
    :param current_gas_density_grid: Gas density at new pressure (lb/ft³).
    :param solution_gas_to_oil_ratio_grid: Rs at new pressure (SCF/STB).
    :param gas_solubility_in_water_grid: Rsw at new pressure (SCF/STB).
    :param gas_formation_volume_factor_grid: Bg at new pressure (bbl/SCF).
    :param oil_formation_volume_factor_grid: Bo at new pressure (bbl/STB).
    :param water_formation_volume_factor_grid: Bw at new pressure (bbl/STB).
    :param cell_count_x: Number of cells in x-direction.
    :param cell_count_y: Number of cells in y-direction.
    :param cell_count_z: Number of cells in z-direction.
    :param thickness_grid: Cell thickness grid (ft).
    :param porosity_grid: Porosity grid (fraction).
    :param net_to_gross_grid: Net-to-gross ratio grid (fraction).
    :param cell_size_x: Cell size in x-direction (ft).
    :param cell_size_y: Cell size in y-direction (ft).
    :param time_step_in_days: Time step size (days).
    :param cfl_threshold: Maximum allowed CFL number for stability check.
    :param dtype: Numpy dtype for computations.
    :return: Tuple of (`updated_water_saturation_grid`, `updated_oil_saturation_grid`,
        `updated_gas_saturation_grid`, `cfl_violation_info`).
        `cfl_violation_info` is a 1-D array of length 6:
        [violated_flag, i, j, k, max_cfl_encountered, cfl_threshold].
    """
    # Store CFL values in a temporary grid to avoid race condition during parallel loop.
    cfl_grid = np.zeros((cell_count_x, cell_count_y, cell_count_z), dtype=dtype)

    for i in numba.prange(cell_count_x):  # type: ignore
        for j in range(cell_count_y):
            for k in range(cell_count_z):
                cell_pore_volume = (
                    cell_size_x
                    * cell_size_y
                    * thickness_grid[i, j, k]
                    * net_to_gross_grid[i, j, k]
                    * porosity_grid[i, j, k]
                )

                # CFL check
                water_well_outflow = abs(min(0.0, net_water_well_rate_grid[i, j, k]))
                oil_well_outflow = abs(min(0.0, net_oil_well_rate_grid[i, j, k]))
                gas_well_outflow = abs(min(0.0, net_gas_well_rate_grid[i, j, k]))
                total_volumetric_outflow = (
                    net_volumetric_outflow_grid[i, j, k]
                    + water_well_outflow
                    + oil_well_outflow
                    + gas_well_outflow
                )
                cfl_number = (
                    total_volumetric_outflow * time_step_in_days
                ) / cell_pore_volume
                cfl_grid[i, j, k] = cfl_number

                # PVT alpha factors at new pressure for dissolved gas accounting
                safe_oil_fvf = oil_formation_volume_factor_grid[i, j, k]
                safe_water_fvf = water_formation_volume_factor_grid[i, j, k]
                safe_gas_fvf = gas_formation_volume_factor_grid[i, j, k]
                if safe_oil_fvf < 1e-30:
                    safe_oil_fvf = 1e-30
                if safe_water_fvf < 1e-30:
                    safe_water_fvf = 1e-30
                if safe_gas_fvf < 1e-30:
                    safe_gas_fvf = 1e-30

                current_alpha_solution_gor = (
                    solution_gas_to_oil_ratio_grid[i, j, k]
                    * safe_gas_fvf
                    / safe_oil_fvf
                )
                current_alpha_gas_solubility_in_water = (
                    gas_solubility_in_water_grid[i, j, k]
                    * safe_gas_fvf
                    / safe_water_fvf
                )

                safe_old_oil_fvf = old_oil_formation_volume_factor_grid[i, j, k]
                safe_old_water_fvf = old_water_formation_volume_factor_grid[i, j, k]
                safe_old_gas_fvf = old_gas_formation_volume_factor_grid[i, j, k]
                if safe_old_oil_fvf < 1e-30:
                    safe_old_oil_fvf = 1e-30
                if safe_old_water_fvf < 1e-30:
                    safe_old_water_fvf = 1e-30
                if safe_old_gas_fvf < 1e-30:
                    safe_old_gas_fvf = 1e-30

                old_alpha_solution_gor = (
                    old_solution_gas_to_oil_ratio_grid[i, j, k]
                    * safe_old_gas_fvf
                    / safe_old_oil_fvf
                )
                old_alpha_gas_solubility_in_water = (
                    old_gas_solubility_in_water_grid[i, j, k]
                    * safe_old_gas_fvf
                    / safe_old_water_fvf
                )

                old_water_saturation = old_water_saturation_grid[i, j, k]
                old_oil_saturation = old_oil_saturation_grid[i, j, k]
                old_gas_aturation = old_gas_saturation_grid[i, j, k]

                old_water_density = old_water_density_grid[i, j, k]
                old_oil_density = old_oil_density_grid[i, j, k]
                old_gas_density = old_gas_density_grid[i, j, k]
                current_water_density = current_water_density_grid[i, j, k]
                current_oil_density = current_oil_density_grid[i, j, k]
                current_gas_density = current_gas_density_grid[i, j, k]

                if current_water_density < 1e-30:
                    current_water_density = 1e-30
                if current_oil_density < 1e-30:
                    current_oil_density = 1e-30
                if current_gas_density < 1e-30:
                    current_gas_density = 1e-30

                dt_over_pv = time_step_in_days / cell_pore_volume

                # Water mass update
                well_water_mass_rate = net_water_well_mass_rate_grid[i, j, k]

                old_water_mass = old_water_density * old_water_saturation
                new_water_mass = old_water_mass + dt_over_pv * (
                    net_water_mass_flux_grid[i, j, k] + well_water_mass_rate
                )
                new_water_saturation = new_water_mass / current_water_density

                # Total gas mass update
                # Old total gas mass per unit pore volume (lb/ft³ of pore space)
                old_total_gas_mass = (
                    old_gas_density * old_gas_aturation
                    + old_oil_density * old_alpha_solution_gor * old_oil_saturation
                    + old_water_density
                    * old_alpha_gas_solubility_in_water
                    * old_water_saturation
                )

                # Mass well rate for total gas: includes solution gas in produced/injected oil and water
                well_gas_mass_rate = (
                    net_gas_well_mass_rate_grid[i, j, k]
                    + current_alpha_solution_gor * net_oil_well_mass_rate_grid[i, j, k]
                    + current_alpha_gas_solubility_in_water
                    * net_water_well_mass_rate_grid[i, j, k]
                )
                new_total_gas_mass = old_total_gas_mass + dt_over_pv * (
                    net_gas_total_mass_flux_grid[i, j, k] + well_gas_mass_rate
                )

                # Oil saturation (derived from volume constraint)
                # Clamp Sw first; So will be derived after Sg
                if new_water_saturation < 0.0:
                    new_water_saturation = 0.0
                if new_water_saturation > 1.0:
                    new_water_saturation = 1.0

                # Free gas saturation from total gas mass
                # M_g_total = rho_g * Sg + rho_o * alpha_Rs * So + rho_w * alpha_Rsw * Sw
                # So = 1 - Sw - Sg  (derived), substitute and solve for Sg:
                # M_g_total = rho_g * Sg + rho_o * alpha_Rs * (1 - Sw - Sg) + rho_w * alpha_Rsw * Sw
                # M_g_total = (rho_g - rho_o * alpha_Rs) * Sg
                #             + rho_o * alpha_Rs * (1 - Sw)
                #             + rho_w * alpha_Rsw * Sw
                # Sg = (M_g_total - rho_o * alpha_Rs * (1 - Sw) - rho_w * alpha_Rsw * Sw)
                #      / (rho_g - rho_o * alpha_Rs)
                #
                # When rho_g >> rho_o * alpha_Rs (which holds for most reservoir conditions),
                # the denominator is positive. We guard against degenerate cases.
                gas_mass_dissolved_in_oil_and_water = (
                    current_oil_density
                    * current_alpha_solution_gor
                    * (1.0 - new_water_saturation)
                    + current_water_density
                    * current_alpha_gas_solubility_in_water
                    * new_water_saturation
                )
                free_gas_mass = new_total_gas_mass - gas_mass_dissolved_in_oil_and_water

                # Free gas mass cannot be negative (gas fully redissolves)
                if free_gas_mass < 0.0:
                    free_gas_mass = 0.0

                new_gas_saturation = free_gas_mass / current_gas_density

                # Clamp Sg
                if new_gas_saturation < 0.0:
                    new_gas_saturation = 0.0
                if new_gas_saturation > 1.0 - new_water_saturation:
                    new_gas_saturation = 1.0 - new_water_saturation

                # Derive So from volume constraint
                new_oil_saturation = 1.0 - new_water_saturation - new_gas_saturation
                if new_oil_saturation < 0.0:
                    new_oil_saturation = 0.0

                # Residual volume balance correction. Absorb tiny numerical gap into oil
                total_saturation = (
                    new_water_saturation + new_oil_saturation + new_gas_saturation
                )
                if abs(total_saturation - 1.0) > 1e-12:
                    new_oil_saturation += 1.0 - total_saturation
                    if new_oil_saturation < 0.0:
                        new_oil_saturation = 0.0

                updated_water_saturation_grid[i, j, k] = new_water_saturation
                updated_oil_saturation_grid[i, j, k] = new_oil_saturation
                updated_gas_saturation_grid[i, j, k] = new_gas_saturation

    # Sequential region for thread-safe CFL analysis
    cfl_violation_info = np.zeros(6, dtype=dtype)
    max_cfl = 0.0
    max_i, max_j, max_k = 0, 0, 0
    violated = False

    for i in range(cell_count_x):
        for j in range(cell_count_y):
            for k in range(cell_count_z):
                cfl_value = cfl_grid[i, j, k]

                if cfl_value > cfl_threshold:
                    violated = True

                if cfl_value > max_cfl:
                    max_cfl = cfl_value
                    max_i, max_j, max_k = i, j, k

    cfl_violation_info[0] = 1.0 if violated else 0.0
    cfl_violation_info[1] = float(max_i)
    cfl_violation_info[2] = float(max_j)
    cfl_violation_info[3] = float(max_k)
    cfl_violation_info[4] = max_cfl
    cfl_violation_info[5] = cfl_threshold
    return (
        updated_water_saturation_grid,
        updated_oil_saturation_grid,
        updated_gas_saturation_grid,
        cfl_violation_info,
    )

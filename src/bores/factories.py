import logging
import typing
import warnings

import numpy as np

from bores.constants import c
from bores.correlations.arrays import estimate_solution_gor
from bores.correlations.core import (
    compute_gas_gravity,
    validate_input_pressure,
    validate_input_temperature,
)
from bores.errors import ValidationError
from bores.fluids import Fluid
from bores.fractures import Fracture, apply_fractures
from bores.grids.base import build_uniform_grid
from bores.grids.pvt import (
    build_fluid_viscosity_grid,
    build_gas_compressibility_factor_grid,
    build_gas_compressibility_grid,
    build_gas_density_grid,
    build_gas_formation_volume_factor_grid,
    build_gas_free_water_formation_volume_factor_grid,
    build_gas_molecular_weight_grid,
    build_gas_solubility_in_water_grid,
    build_live_oil_density_grid,
    build_oil_api_gravity_grid,
    build_oil_bubble_point_pressure_grid,
    build_oil_compressibility_grid,
    build_oil_formation_volume_factor_grid,
    build_solution_gas_to_oil_ratio_grid,
    build_water_bubble_point_pressure_grid,
    build_water_compressibility_grid,
    build_water_density_grid,
    build_water_formation_volume_factor_grid,
    build_water_viscosity_grid,
)
from bores.models import (
    FluidProperties,
    ReservoirModel,
    RockPermeability,
    RockProperties,
    SaturationHistory,
)
from bores.tables.pvt import PVTTables
from bores.types import Coordinates, NDimension, NDimensionalGrid
from bores.wells import (
    InjectedFluid,
    InjectionWell,
    ProducedFluid,
    ProductionWell,
    WellControl,
    Wells,
)

logger = logging.getLogger(__name__)

__all__ = ["injection_well", "production_well", "reservoir_model", "wells_"]


def validate_saturation_grids(
    oil_saturation_grid: NDimensionalGrid[NDimension],
    water_saturation_grid: NDimensionalGrid[NDimension],
    gas_saturation_grid: NDimensionalGrid[NDimension],
    residual_oil_saturation_water_grid: NDimensionalGrid[NDimension],
    residual_oil_saturation_gas_grid: NDimensionalGrid[NDimension],
    porosity_grid: NDimensionalGrid[NDimension],
    normalize: bool = True,
) -> typing.Tuple[
    NDimensionalGrid[NDimension],
    NDimensionalGrid[NDimension],
    NDimensionalGrid[NDimension],
]:
    """
    Validate and optionally normalize saturation grids for reservoir modeling.

    This function performs comprehensive validation on saturation grids, considering
    only active cells (cells with finite and non-zero porosity). It checks:
    - Saturation sum equals 1.0 (with optional normalization)
    - Saturation values are within valid ranges [0, 1)
    - Residual saturations are physically valid
    - Residual oil saturation consistency (warning if sor_gas > sor_water)

    :param oil_saturation_grid: Oil saturation grid (fraction).
    :param water_saturation_grid: Water saturation grid (fraction).
    :param gas_saturation_grid: Gas saturation grid (fraction).
    :param residual_oil_saturation_water_grid: Residual oil saturation during water flooding grid (fraction).
    :param residual_oil_saturation_gas_grid: Residual oil saturation during gas flooding grid (fraction).
    :param residual_gas_saturation_grid: Residual gas saturation grid (fraction).
    :param connate_water_saturation_grid: Connate water saturation grid (fraction).
    :param irreducible_water_saturation_grid: Irreducible water saturation grid (fraction).
    :param porosity_grid: Porosity grid (fraction) used to identify active cells.
    :param normalize: If True, normalizes saturations to sum to 1.0 in active cells (default: True).
    :return: Tuple of (oil_saturation_grid, water_saturation_grid, gas_saturation_grid),
             potentially normalized if normalize=True.
    :raises ValueError: If saturation constraints are violated in active cells.
    :raises UserWarning: If residual oil saturation for gas is greater than for water.
    """
    # Identify active cells (finite and non-zero porosity)
    active_cells = np.isfinite(porosity_grid) & (porosity_grid > 0)

    # Check if saturations sum to 1.0 in active cells
    total_saturation = oil_saturation_grid + water_saturation_grid + gas_saturation_grid
    if not np.all(np.isclose(total_saturation[active_cells], 1.0)):
        if normalize:
            warnings.warn(
                "The sum of oil, water, and gas saturations does not equal 1 everywhere in active cells. "
                "Adjusting saturations to ensure they sum to 1.",
                UserWarning,
            )
            # Avoid division by zero
            total_saturation_safe = np.where(total_saturation == 0, 1, total_saturation)
            oil_saturation_grid = oil_saturation_grid / total_saturation_safe  # type: ignore
            water_saturation_grid = water_saturation_grid / total_saturation_safe  # type: ignore
            gas_saturation_grid = gas_saturation_grid / total_saturation_safe  # type: ignore
        else:
            raise ValidationError(
                "The sum of oil, water, and gas saturations does not equal 1 everywhere in active cells."
            )

    # Validate saturation ranges in active cells
    if np.any(
        (oil_saturation_grid[active_cells] < 0)
        | (oil_saturation_grid[active_cells] >= 1)
    ):
        raise ValidationError(
            "Oil saturation grid values must be in the range [0, 1) in active cells."
        )

    if np.any(
        (water_saturation_grid[active_cells] < 0)
        | (water_saturation_grid[active_cells] >= 1)
    ):
        raise ValidationError(
            "Water saturation grid values must be in the range [0, 1) in active cells."
        )

    if np.any(
        (gas_saturation_grid[active_cells] < 0)
        | (gas_saturation_grid[active_cells] >= 1)
    ):
        raise ValidationError(
            "Gas saturation grid values must be in the range [0, 1) in active cells."
        )

    # Check if residual oil saturation for gas is greater than for water
    if np.any(
        residual_oil_saturation_gas_grid[active_cells]
        > residual_oil_saturation_water_grid[active_cells]
    ):
        warnings.warn(
            "Residual oil saturation during gas flooding (Sor_g) is greater than residual oil "
            "saturation during water flooding (Sor_w) in some active cells. This may indicate "
            "that gas is less efficient at displacing oil than water, which is unusual. "
            "Please verify your residual saturation values.",
            UserWarning,
        )

    return oil_saturation_grid, water_saturation_grid, gas_saturation_grid


def reservoir_model(
    grid_shape: NDimension,
    cell_dimension: typing.Tuple[float, float],
    thickness_grid: NDimensionalGrid[NDimension],
    pressure_grid: NDimensionalGrid[NDimension],
    rock_compressibility: float,
    absolute_permeability: RockPermeability[NDimension],
    porosity_grid: NDimensionalGrid[NDimension],
    temperature_grid: NDimensionalGrid[NDimension],
    water_saturation_grid: NDimensionalGrid[NDimension],
    gas_saturation_grid: NDimensionalGrid[NDimension],
    oil_saturation_grid: NDimensionalGrid[NDimension],
    oil_viscosity_grid: NDimensionalGrid[NDimension],
    oil_bubble_point_pressure_grid: NDimensionalGrid[NDimension],
    residual_oil_saturation_water_grid: NDimensionalGrid[NDimension],
    residual_oil_saturation_gas_grid: NDimensionalGrid[NDimension],
    residual_gas_saturation_grid: NDimensionalGrid[NDimension],
    irreducible_water_saturation_grid: NDimensionalGrid[NDimension],
    connate_water_saturation_grid: NDimensionalGrid[NDimension],
    oil_specific_gravity_grid: typing.Optional[NDimensionalGrid[NDimension]] = None,
    oil_compressibility_grid: typing.Optional[NDimensionalGrid[NDimension]] = None,
    gas_gravity_grid: typing.Optional[NDimensionalGrid[NDimension]] = None,
    gas_viscosity_grid: typing.Optional[NDimensionalGrid[NDimension]] = None,
    gas_compressibility_grid: typing.Optional[NDimensionalGrid[NDimension]] = None,
    gas_compressibility_factor_grid: typing.Optional[
        NDimensionalGrid[NDimension]
    ] = None,
    gas_molecular_weight_grid: typing.Optional[NDimensionalGrid[NDimension]] = None,
    gas_density_grid: typing.Optional[NDimensionalGrid[NDimension]] = None,
    water_viscosity_grid: typing.Optional[NDimensionalGrid[NDimension]] = None,
    water_compressibility_grid: typing.Optional[NDimensionalGrid[NDimension]] = None,
    water_density_grid: typing.Optional[NDimensionalGrid[NDimension]] = None,
    water_bubble_point_pressure_grid: typing.Optional[
        NDimensionalGrid[NDimension]
    ] = None,
    solution_gas_to_oil_ratio_grid: typing.Optional[
        NDimensionalGrid[NDimension]
    ] = None,
    gas_solubility_in_water_grid: typing.Optional[NDimensionalGrid[NDimension]] = None,
    oil_formation_volume_factor_grid: typing.Optional[
        NDimensionalGrid[NDimension]
    ] = None,
    gas_formation_volume_factor_grid: typing.Optional[
        NDimensionalGrid[NDimension]
    ] = None,
    water_formation_volume_factor_grid: typing.Optional[
        NDimensionalGrid[NDimension]
    ] = None,
    net_to_gross_ratio_grid: typing.Optional[NDimensionalGrid[NDimension]] = None,
    water_salinity_grid: typing.Optional[NDimensionalGrid[NDimension]] = None,
    solvent_concentration_grid: typing.Optional[NDimensionalGrid[NDimension]] = None,
    oil_effective_viscosity_grid: typing.Optional[NDimensionalGrid[NDimension]] = None,
    oil_effective_density_grid: typing.Optional[NDimensionalGrid[NDimension]] = None,
    saturation_history: typing.Optional[SaturationHistory[NDimension]] = None,
    fractures: typing.Optional[typing.Iterable[Fracture]] = None,
    dip_angle: float = 0.0,
    dip_azimuth: float = 0.0,
    datum_depth: typing.Optional[float] = None,
    reservoir_gas: typing.Optional[typing.Union[str, Fluid]] = None,
    pvt_tables: typing.Optional[PVTTables] = None,
) -> ReservoirModel[NDimension]:
    """
    Constructs a N-Dimensional reservoir model with given rock and fluid properties.

    Note:

    - If both `oil_bubble_point_pressure_grid` and `solution_gas_to_oil_ratio_grid` are omitted,
      the function attempts to estimate these based on pressure, temperature, and oil properties
      using the iterative coupled Vazquez-Beggs/Standing solver (`estimate_solution_gor`), which
      accounts for temperature variation across the grid.
    - Saturation grids will be adjusted internally to ensure saturation sums to 1.0.
    - All fluid property grids are estimated from pressure and temperature if not provided.
    - Provide consistent and physically realistic input grids to ensure stable simulations.

    :param grid_shape: Number of grid cells (rows, columns).
    :param cell_dimension: Physical size of each cell (dx, dy) in feets.
    :param thickness_grid: Reservoir cells' thickness grid (ft).
    :param pressure_grid: Reservoir cells' pressure grid (psi).
    :param rock_compressibility: Rock compressibility in psi⁻¹.
    :param absolute_permeability_grid: Absolute permeability grid (mD).
    :param porosity_grid: Porosity grid (fraction).
    :param temperature_grid: Reservoir temperature grid (°F).
    :param oil_saturation_grid: Oil saturation grid (fraction).
    :param oil_viscosity_grid: Oil viscosity grid (cP).
    :param oil_compressibility_grid: Oil compressibility grid (psi⁻¹), optional.
        If omitted, computed from correlations. When both this and
        ``oil_formation_volume_factor_grid`` are omitted, an iterative bootstrap
        resolves their circular dependency (Bo needs Co above Pb, Co needs Bo
        below Pb) — typically converges in 2-3 iterations.
    :param oil_specific_gravity_grid: Oil specific gravity grid (dimensionless).
    :param oil_bubble_point_pressure_grid: Oil bubble point pressure grid (psi).
    :param gas_gravity_grid: Gas gravity grid (dimensionless), optional.
    :param residual_oil_saturation_water_grid: Residual oil saturation during water flooding grid (fraction).
    :param residual_oil_saturation_gas_grid: Residual oil saturation during gas flooding grid (fraction).
    :param gas_saturation_grid: Gas saturation grid (fraction), optional.
    :param gas_viscosity_grid: Gas viscosity grid (cP), optional.
    :param gas_compressibility_grid: Gas compressibility grid (psi⁻¹), optional.
    :param gas_compressibility_factor_grid: Gas compressibility factor grid (dimensionless), optional.
    :param gas_density_grid: Gas density grid (lb/ft³), optional.
    :param residual_gas_saturation_grid: Residual gas saturation grid (fraction), optional.
    :param water_saturation_grid: Water saturation grid (fraction), optional.
    :param water_viscosity_grid: Water viscosity grid (cP), optional.
    :param water_compressibility_grid: Water compressibility grid (psi⁻¹), optional.
    :param water_density_grid: Water density grid (lb/ft³), optional.
    :param water_bubble_point_pressure_grid: Water bubble point pressure grid (psi), optional.
    :param connate_water_saturation_grid: Connate water saturation grid (fraction).
    :param irreducible_water_saturation_grid: Irreducible water saturation grid (fraction).
    :param solution_gas_to_oil_ratio_grid: Solution gas to oil ratio grid (scf/bbl), optional.
    :param gas_solubility_in_water_grid: Gas solubility in water grid (scf/bbl), optional.
    :param oil_formation_volume_factor_grid: Oil formation volume factor grid (bbl/scf), optional.
    :param gas_formation_volume_factor_grid: Gas formation volume factor grid (bbl/scf), optional.
    :param water_formation_volume_factor_grid: Water formation volume factor grid (bbl/scf), optional.
    :param net_to_gross_ratio_grid: Net-to-gross ratio grid (fraction), optional.
    :param water_salinity_grid: Water salinity grid (ppm), optional.
    :param solvent_concentration_grid: Solvent concentration in oil phase (0=pure oil, 1=pure solvent), optional.
    :param oil_effective_viscosity_grid: Effective oil-solvent mixture viscosity using miscible model (e.g Todd Longstaff) (cP), optional.
    :param fractures: Iterable of fractures to be applied to the reservoir model, optional.
    :param reservoir_gas: Name of the reservoir gas, defaults to `c.RESERVOIR_GAS`. Can also be the name of the gas injected into the reservoir.
    :param pvt_tables: PVT tables for fluid properties, optional.
    :param datum_depth: Reference depth for reservoir model. Basically the reservoir top depth (below sea level)
    :return: The constructed N-Dimensional reservoir model with fluid and rock properties.
    """
    if not 2 <= len(grid_shape) <= 3:
        raise ValidationError(
            "`grid_shape` must be a tuple of two to three integers (rows, columns, [depth])."
        )
    if len(cell_dimension) < 2:
        raise ValidationError(
            "`cell_dimension` must be a tuple of two floats [cell width (dx), cell length (dy)]."
        )

    validate_input_pressure(pressure_grid)
    validate_input_temperature(temperature_grid)

    reservoir_gas = typing.cast(str, reservoir_gas or c.RESERVOIR_GAS)
    oil_pvt_table = pvt_tables.oil if pvt_tables is not None else None
    water_pvt_table = pvt_tables.water if pvt_tables is not None else None
    gas_pvt_table = pvt_tables.gas if pvt_tables is not None else None

    if water_salinity_grid is None:
        water_salinity_grid = build_uniform_grid(
            grid_shape=grid_shape,
            value=c.DEFAULT_WATER_SALINITY_PPM,
        )

    if net_to_gross_ratio_grid is None:
        # Assume uniform net-to-gross ratio of 1.0 (fully net)
        net_to_gross_ratio_grid = build_uniform_grid(
            grid_shape=grid_shape,
            value=1.0,
        )

    # Validate and normalize saturation grids
    oil_saturation_grid, water_saturation_grid, gas_saturation_grid = (
        validate_saturation_grids(
            oil_saturation_grid=oil_saturation_grid,
            water_saturation_grid=water_saturation_grid,
            gas_saturation_grid=gas_saturation_grid,
            residual_oil_saturation_water_grid=residual_oil_saturation_water_grid,
            residual_oil_saturation_gas_grid=residual_oil_saturation_gas_grid,
            porosity_grid=porosity_grid,
            normalize=True,
        )
    )

    if gas_gravity_grid is None:
        if gas_pvt_table is not None:
            gas_gravity_grid = typing.cast(
                typing.Optional[NDimensionalGrid[NDimension]],
                gas_pvt_table.specific_gravity(
                    pressure=pressure_grid,
                    temperature=temperature_grid,
                ),
            )
        if gas_gravity_grid is None:
            gas_gravity = compute_gas_gravity(gas=reservoir_gas)
            gas_gravity_grid = build_uniform_grid(
                grid_shape=grid_shape,
                value=gas_gravity,
            )

    # Viscosity Grids
    if gas_viscosity_grid is None:
        if gas_pvt_table is not None:
            gas_viscosity_grid = typing.cast(
                typing.Optional[NDimensionalGrid[NDimension]],
                gas_pvt_table.viscosity(
                    pressure=pressure_grid,
                    temperature=temperature_grid,
                ),
            )
        if gas_viscosity_grid is None:
            gas_viscosity_grid = build_fluid_viscosity_grid(
                pressure_grid=pressure_grid,
                temperature_grid=temperature_grid,
                fluid=reservoir_gas,
            )

    if water_viscosity_grid is None:
        if water_pvt_table is not None:
            water_viscosity_grid = typing.cast(
                typing.Optional[NDimensionalGrid[NDimension]],
                water_pvt_table.viscosity(
                    pressure=pressure_grid,
                    temperature=temperature_grid,
                    salinity=water_salinity_grid,
                ),
            )
        if water_viscosity_grid is None:
            water_viscosity_grid = build_water_viscosity_grid(
                pressure_grid=pressure_grid,
                temperature_grid=temperature_grid,
                salinity_grid=water_salinity_grid,
            )

    # Gas Z-factor grid must be built before `gas_compressibility_grid`,
    # `gas_density_grid`, and `gas_formation_volume_factor_grid`, all of which
    # depend on it. Also stored in `FluidProperties` so the simulator does not
    # have to recompute it each timestep.
    if gas_compressibility_factor_grid is None:
        if gas_pvt_table is not None:
            gas_compressibility_factor_grid = typing.cast(
                typing.Optional[NDimensionalGrid[NDimension]],
                gas_pvt_table.compressibility_factor(
                    pressure=pressure_grid,
                    temperature=temperature_grid,
                ),
            )
        if gas_compressibility_factor_grid is None:
            gas_compressibility_factor_grid = build_gas_compressibility_factor_grid(
                gas_gravity_grid=gas_gravity_grid,
                pressure_grid=pressure_grid,
                temperature_grid=temperature_grid,
            )

    # Gas compressibility grid
    # Note: the analytical dZ/dPr derivative used here is derived from
    # Papay's correlation regardless of which Z-factor method produced
    # `gas_compressibility_factor_grid`. This is an accepted engineering
    # approximation for the derivative; the Z values themselves come from
    # the selected method above.
    if gas_compressibility_grid is None:
        if gas_pvt_table is not None:
            gas_compressibility_grid = typing.cast(
                typing.Optional[NDimensionalGrid[NDimension]],
                gas_pvt_table.compressibility(
                    pressure=pressure_grid,
                    temperature=temperature_grid,
                ),
            )
        if gas_compressibility_grid is None:
            gas_compressibility_grid = build_gas_compressibility_grid(
                pressure_grid=pressure_grid,
                temperature_grid=temperature_grid,
                gas_gravity_grid=gas_gravity_grid,
                gas_compressibility_factor_grid=gas_compressibility_factor_grid,
            )

    if gas_molecular_weight_grid is None:
        if gas_pvt_table is not None:
            gas_molecular_weight_grid = typing.cast(
                typing.Optional[NDimensionalGrid[NDimension]],
                gas_pvt_table.molecular_weight(
                    pressure=pressure_grid,
                    temperature=temperature_grid,
                ),
            )
        if gas_molecular_weight_grid is None:
            gas_molecular_weight_grid = build_gas_molecular_weight_grid(
                gas_gravity_grid=gas_gravity_grid  # type: ignore
            )

    # Gas density
    if gas_density_grid is None:
        if gas_pvt_table is not None:
            gas_density_grid = typing.cast(
                typing.Optional[NDimensionalGrid[NDimension]],
                gas_pvt_table.density(
                    pressure=pressure_grid,
                    temperature=temperature_grid,
                ),
            )
        if gas_density_grid is None:
            gas_density_grid = build_gas_density_grid(
                pressure_grid=pressure_grid,
                temperature_grid=temperature_grid,
                gas_gravity_grid=gas_gravity_grid,  # type: ignore
                gas_compressibility_factor_grid=gas_compressibility_factor_grid,  # type: ignore
            )

    # Oil API gravity and specific gravity
    # Do not use PVT tables for oil api gravity and live oil density as these are
    # direct functions of oil specific gravity
    if oil_specific_gravity_grid is None:
        if oil_pvt_table is None:
            raise ValidationError(
                "`oil_specific_gravity_grid` must be provided if `oil_pvt_table` is not used."
            )
        else:
            oil_specific_gravity_grid = typing.cast(
                NDimensionalGrid[NDimension],
                oil_pvt_table.specific_gravity(
                    pressure=pressure_grid,
                    temperature=temperature_grid,
                ),
            )

    oil_api_gravity_grid = build_oil_api_gravity_grid(oil_specific_gravity_grid)

    # Computing Solution GOR and bubble point pressure, we have four cases:
    #  (a) Both provided            → use as-is
    #  (b) Pb provided, Rs missing  → derive Rs from Pb using correlations
    #  (c) Rs provided, Pb missing  → derive Pb from Rs (oil_pvt_table first)
    #  (d) Neither provided         → estimate using iterative coupled solver
    #      (estimate_solution_gor handles T variation; Standing alone does not)
    if (
        solution_gas_to_oil_ratio_grid is None
        and oil_bubble_point_pressure_grid is not None
    ):
        # Case (b): Pb given, derive Rs from it
        # User provided Pb but not Rs, use correlations to compute Rs from Pb
        # Note: We use correlations here even if `oil_pvt_table` is provided because
        # the user explicitly provided Pb and we need to honor that value
        solution_gas_to_oil_ratio_grid = build_solution_gas_to_oil_ratio_grid(
            pressure_grid=pressure_grid,
            temperature_grid=temperature_grid,
            bubble_point_pressure_grid=oil_bubble_point_pressure_grid,
            gas_gravity_grid=gas_gravity_grid,
            oil_api_gravity_grid=oil_api_gravity_grid,
        )
    elif (
        solution_gas_to_oil_ratio_grid is not None
        and oil_bubble_point_pressure_grid is None
    ):
        # Case (c): Rs given, derive Pb from it
        # When Rs is provided, try to get Pb from `oil_pvt_table` first, then fall back to correlations
        if oil_pvt_table is not None:
            oil_bubble_point_pressure_grid = typing.cast(
                typing.Optional[NDimensionalGrid[NDimension]],
                oil_pvt_table.bubble_point_pressure(
                    temperature=temperature_grid,
                    solution_gor=solution_gas_to_oil_ratio_grid,
                ),
            )
        if oil_bubble_point_pressure_grid is None:
            oil_bubble_point_pressure_grid = build_oil_bubble_point_pressure_grid(
                gas_gravity_grid=gas_gravity_grid,
                oil_api_gravity_grid=oil_api_gravity_grid,
                temperature_grid=temperature_grid,
                solution_gas_to_oil_ratio_grid=solution_gas_to_oil_ratio_grid,
            )
    elif (
        solution_gas_to_oil_ratio_grid is None
        and oil_bubble_point_pressure_grid is None
    ):
        # Case (d): Neither was provided so we use the iterative coupled solver.
        # estimate_solution_gor solves the coupled (Rs, Pb) system at each cell,
        # accounting for temperature via Vazquez-Beggs. This is more accurate than
        # the temperature-blind Standing standalone correlation.
        warnings.warn(
            "Both `oil_bubble_point_pressure_grid` and `solution_gas_to_oil_ratio_grid` are not "
            "provided. Estimating both using the iterative Vazquez-Beggs/Standing coupled solver. "
            "Results may be less accurate than providing measured values.",
            UserWarning,
        )

        # Try to use `oil_pvt_table` first to compute Rs and Pb, fall back to correlations
        if oil_pvt_table is not None:
            solution_gas_to_oil_ratio_grid = typing.cast(
                typing.Optional[NDimensionalGrid[NDimension]],
                oil_pvt_table.solution_gas_to_oil_ratio(
                    pressure=pressure_grid,
                    temperature=temperature_grid,
                    solution_gor=None,  # Will use 1D table if available
                ),
            )
        if solution_gas_to_oil_ratio_grid is None:
            solution_gas_to_oil_ratio_grid = estimate_solution_gor(
                pressure=pressure_grid,
                temperature=temperature_grid,
                oil_api_gravity=oil_api_gravity_grid,
                gas_gravity=gas_gravity_grid,
                max_iterations=20,
                tolerance=1e-4,
            )

        # Now derive Pb from the estimated Rs, trying `oil_pvt_table` first
        if oil_pvt_table is not None:
            oil_bubble_point_pressure_grid = typing.cast(
                typing.Optional[NDimensionalGrid[NDimension]],
                oil_pvt_table.bubble_point_pressure(
                    temperature=temperature_grid,
                    solution_gor=solution_gas_to_oil_ratio_grid,
                ),
            )
        if oil_bubble_point_pressure_grid is None:
            oil_bubble_point_pressure_grid = build_oil_bubble_point_pressure_grid(
                temperature_grid=temperature_grid,
                gas_gravity_grid=gas_gravity_grid,
                oil_api_gravity_grid=oil_api_gravity_grid,
                solution_gas_to_oil_ratio_grid=solution_gas_to_oil_ratio_grid,
            )

    solution_gas_to_oil_ratio_grid = typing.cast(
        NDimensionalGrid[NDimension], solution_gas_to_oil_ratio_grid
    )

    # Gas solubility in water
    if gas_solubility_in_water_grid is None:
        if gas_pvt_table is not None:
            gas_solubility_in_water_grid = typing.cast(
                typing.Optional[NDimensionalGrid[NDimension]],
                gas_pvt_table.solubility_in_water(
                    pressure=pressure_grid,
                    temperature=temperature_grid,
                    salinity=water_salinity_grid,
                ),
            )
        if gas_solubility_in_water_grid is None:
            gas_solubility_in_water_grid = build_gas_solubility_in_water_grid(
                pressure_grid=pressure_grid,
                temperature_grid=temperature_grid,
                salinity_grid=water_salinity_grid,
                gas=reservoir_gas,
            )

    # Formation Volume Factor Grids
    # Gas FVF must be built before oil FVF/Co because the oil compressibility
    # liberation correction term (below Pb) depends on gas FVF.
    if gas_formation_volume_factor_grid is None:
        if gas_pvt_table is not None:
            gas_formation_volume_factor_grid = typing.cast(
                typing.Optional[NDimensionalGrid[NDimension]],
                gas_pvt_table.formation_volume_factor(
                    pressure=pressure_grid,
                    temperature=temperature_grid,
                ),
            )
        if gas_formation_volume_factor_grid is None:
            gas_formation_volume_factor_grid = build_gas_formation_volume_factor_grid(
                pressure_grid=pressure_grid,
                temperature_grid=temperature_grid,
                gas_compressibility_factor_grid=gas_compressibility_factor_grid,  # type: ignore
            )

    # Oil FVF and oil compressibility have a circular dependency on each other.
    # Bo above Pb uses Co: Bo = Bob * exp(-Co * (P - Pb))
    # Co below Pb uses Bo: Co includes (Bg/Bo) * (dRs/dP) / 5.615 liberation term
    # When both are missing, iterate to convergence (typically 2-3 passes).
    if oil_formation_volume_factor_grid is None and oil_pvt_table is not None:
        oil_formation_volume_factor_grid = typing.cast(
            typing.Optional[NDimensionalGrid[NDimension]],
            oil_pvt_table.formation_volume_factor(
                pressure=pressure_grid,
                temperature=temperature_grid,
                solution_gor=solution_gas_to_oil_ratio_grid,
            ),
        )

    if oil_compressibility_grid is None and oil_pvt_table is not None:
        oil_compressibility_grid = typing.cast(
            typing.Optional[NDimensionalGrid[NDimension]],
            oil_pvt_table.compressibility(
                pressure=pressure_grid,
                temperature=temperature_grid,
            ),
        )

    needs_oil_fvf = oil_formation_volume_factor_grid is None
    need_oil_compressibility = oil_compressibility_grid is None
    if needs_oil_fvf and need_oil_compressibility:
        # Both are missing. We use iterative bootstrap.
        # Rs at bubble point is needed for Co below Pb.
        rs_at_bubble_point_grid = build_solution_gas_to_oil_ratio_grid(
            pressure_grid=oil_bubble_point_pressure_grid,
            temperature_grid=temperature_grid,
            bubble_point_pressure_grid=oil_bubble_point_pressure_grid,
            gas_gravity_grid=gas_gravity_grid,
            oil_api_gravity_grid=oil_api_gravity_grid,
        )

        max_iterations = 10
        convergence_tolerance = 1e-7  # psi⁻¹
        oil_compressibility_estimate = np.full_like(pressure_grid, 1e-5)

        logger.debug(
            "Beginning iterative Bo/Co bootstrap (max_iterations=%d, tolerance=%.2e).",
            max_iterations,
            convergence_tolerance,
        )
        max_change = 0.0
        for iteration in range(max_iterations):
            # Step 1: Build Bo using current Co estimate
            oil_formation_volume_factor_grid = build_oil_formation_volume_factor_grid(
                pressure_grid=pressure_grid,
                temperature_grid=temperature_grid,
                bubble_point_pressure_grid=oil_bubble_point_pressure_grid,
                oil_specific_gravity_grid=oil_specific_gravity_grid,
                gas_gravity_grid=gas_gravity_grid,
                solution_gas_to_oil_ratio_grid=solution_gas_to_oil_ratio_grid,
                oil_compressibility_grid=oil_compressibility_estimate,
            )

            # Step 2: Build Co using Bo from step 1
            oil_compressibility_updated = build_oil_compressibility_grid(
                pressure_grid=pressure_grid,
                temperature_grid=temperature_grid,
                bubble_point_pressure_grid=oil_bubble_point_pressure_grid,
                oil_api_gravity_grid=oil_api_gravity_grid,
                gas_gravity_grid=gas_gravity_grid,
                gor_at_bubble_point_pressure_grid=rs_at_bubble_point_grid,
                gas_formation_volume_factor_grid=gas_formation_volume_factor_grid,
                oil_formation_volume_factor_grid=oil_formation_volume_factor_grid,
            )

            # Step 3: Check convergence
            max_change = float(
                np.max(
                    np.abs(oil_compressibility_updated - oil_compressibility_estimate)
                )
            )
            logger.debug(
                f"Bo/Co iteration {iteration + 1}/{max_iterations}: max ΔCo = {max_change:.3e} psi⁻¹.",
            )

            oil_compressibility_estimate = oil_compressibility_updated
            if max_change < convergence_tolerance:
                logger.debug(
                    f"Bo/Co bootstrap converged in {iteration + 1} iteration(s) "
                    f"(max ΔCo = {max_change:.3e} psi⁻¹ < tolerance {convergence_tolerance:.3e} psi⁻¹).",
                )
                break
        else:
            warnings.warn(
                f"Bo/Co bootstrap did not converge within {max_iterations} iterations "
                f"(final max ΔCo = {max_change:.3e} psi⁻¹). "
                "Results may be less accurate for volatile or heavy oils. "
                "Consider providing pre-computed `oil_compressibility_table`.",
                UserWarning,
            )

        oil_compressibility_grid = oil_compressibility_estimate

    elif needs_oil_fvf and not need_oil_compressibility:
        # Co provided, build Bo in a single pass
        oil_formation_volume_factor_grid = build_oil_formation_volume_factor_grid(
            pressure_grid=pressure_grid,
            temperature_grid=temperature_grid,
            bubble_point_pressure_grid=oil_bubble_point_pressure_grid,
            oil_specific_gravity_grid=oil_specific_gravity_grid,
            gas_gravity_grid=gas_gravity_grid,
            solution_gas_to_oil_ratio_grid=solution_gas_to_oil_ratio_grid,
            oil_compressibility_grid=oil_compressibility_grid,
        )

    elif need_oil_compressibility and not needs_oil_fvf:
        # Bo provided, build Co in a single pass
        rs_at_bubble_point_grid = build_solution_gas_to_oil_ratio_grid(
            pressure_grid=oil_bubble_point_pressure_grid,
            temperature_grid=temperature_grid,
            bubble_point_pressure_grid=oil_bubble_point_pressure_grid,
            gas_gravity_grid=gas_gravity_grid,
            oil_api_gravity_grid=oil_api_gravity_grid,
        )
        oil_compressibility_grid = build_oil_compressibility_grid(
            pressure_grid=pressure_grid,
            temperature_grid=temperature_grid,
            bubble_point_pressure_grid=oil_bubble_point_pressure_grid,
            oil_api_gravity_grid=oil_api_gravity_grid,
            gas_gravity_grid=gas_gravity_grid,
            gor_at_bubble_point_pressure_grid=rs_at_bubble_point_grid,
            gas_formation_volume_factor_grid=gas_formation_volume_factor_grid,
            oil_formation_volume_factor_grid=oil_formation_volume_factor_grid,
        )

    # Gas-free water FVF
    # This is always built from correlations (not a standard PVT table entry).
    # It is used internally for water density and water compressibility only.
    # Which is consistent with those two calculations because all three use the
    # same McCain correlation. It is not the same as `water_formation_volume_factor_grid`
    # (which may come from `pvt_tables`) and should not be confused with it.
    gas_free_water_formation_volume_factor_grid = (
        build_gas_free_water_formation_volume_factor_grid(
            pressure_grid=pressure_grid, temperature_grid=temperature_grid
        )
    )

    # Water density
    if water_density_grid is None:
        if water_pvt_table is not None:
            water_density_grid = typing.cast(
                typing.Optional[NDimensionalGrid[NDimension]],
                water_pvt_table.density(
                    pressure=pressure_grid,
                    temperature=temperature_grid,
                    salinity=water_salinity_grid,
                ),
            )
        if water_density_grid is None:
            water_density_grid = build_water_density_grid(
                pressure_grid=pressure_grid,
                temperature_grid=temperature_grid,
                gas_gravity_grid=gas_gravity_grid,  # type: ignore
                salinity_grid=water_salinity_grid,
                gas_solubility_in_water_grid=gas_solubility_in_water_grid,  # type: ignore
                gas_free_water_formation_volume_factor_grid=gas_free_water_formation_volume_factor_grid,
            )

    # Water FVF
    if water_formation_volume_factor_grid is None:
        if water_pvt_table is not None:
            water_formation_volume_factor_grid = typing.cast(
                typing.Optional[NDimensionalGrid[NDimension]],
                water_pvt_table.formation_volume_factor(
                    pressure=pressure_grid,
                    temperature=temperature_grid,
                    salinity=water_salinity_grid,
                ),
            )
        if water_formation_volume_factor_grid is None:
            water_formation_volume_factor_grid = (
                build_water_formation_volume_factor_grid(
                    water_density_grid=water_density_grid,  # type: ignore
                    salinity_grid=water_salinity_grid,
                )
            )

    # Water bubble point pressure
    if water_bubble_point_pressure_grid is None:
        if water_pvt_table is not None:
            water_bubble_point_pressure_grid = typing.cast(
                typing.Optional[NDimensionalGrid[NDimension]],
                water_pvt_table.bubble_point_pressure(
                    pressure=pressure_grid,
                    temperature=temperature_grid,
                    salinity=water_salinity_grid,
                ),
            )
        if water_bubble_point_pressure_grid is None:
            water_bubble_point_pressure_grid = build_water_bubble_point_pressure_grid(
                temperature_grid=temperature_grid,
                gas_solubility_in_water_grid=gas_solubility_in_water_grid,  # type: ignore
                salinity_grid=water_salinity_grid,
                gas=reservoir_gas,
            )

    # Water compressibility
    if water_compressibility_grid is None:
        if water_pvt_table is not None:
            water_compressibility_grid = typing.cast(
                typing.Optional[NDimensionalGrid[NDimension]],
                water_pvt_table.compressibility(
                    pressure=pressure_grid,
                    temperature=temperature_grid,
                    salinity=water_salinity_grid,
                ),
            )
        if water_compressibility_grid is None:
            water_compressibility_grid = build_water_compressibility_grid(
                pressure_grid=pressure_grid,
                temperature_grid=temperature_grid,
                bubble_point_pressure_grid=water_bubble_point_pressure_grid,  # type: ignore
                gas_formation_volume_factor_grid=gas_formation_volume_factor_grid,  # type: ignore
                gas_solubility_in_water_grid=gas_solubility_in_water_grid,  # type: ignore
                gas_free_water_formation_volume_factor_grid=gas_free_water_formation_volume_factor_grid,
                salinity=water_salinity_grid,
            )

    # Live oil density
    oil_density_grid = None
    if oil_pvt_table is not None:
        oil_density_grid = typing.cast(
            typing.Optional[NDimensionalGrid[NDimension]],
            oil_pvt_table.density(pressure=pressure_grid, temperature=temperature_grid),
        )

    if oil_density_grid is None:
        oil_density_grid = build_live_oil_density_grid(
            oil_api_gravity_grid=oil_api_gravity_grid,
            gas_gravity_grid=gas_gravity_grid,  # type: ignore
            solution_gas_to_oil_ratio_grid=solution_gas_to_oil_ratio_grid,
            formation_volume_factor_grid=oil_formation_volume_factor_grid,  # type: ignore
        )

    # Miscible flood grids (defaults to pure oil if not provided)
    solvent_concentration_grid = solvent_concentration_grid or build_uniform_grid(
        grid_shape=grid_shape, value=0.0
    )
    oil_effective_viscosity_grid = (
        oil_effective_viscosity_grid or oil_viscosity_grid.copy()
    )
    oil_effective_density_grid = oil_effective_density_grid or oil_density_grid.copy()

    # Assemble FluidProperties
    fluid_properties = FluidProperties(
        pressure_grid=pressure_grid,
        temperature_grid=temperature_grid,
        oil_bubble_point_pressure_grid=oil_bubble_point_pressure_grid,
        oil_saturation_grid=oil_saturation_grid,
        oil_viscosity_grid=oil_viscosity_grid,
        oil_compressibility_grid=oil_compressibility_grid,  # type: ignore[arg-type]
        oil_specific_gravity_grid=oil_specific_gravity_grid,
        oil_api_gravity_grid=oil_api_gravity_grid,
        oil_density_grid=oil_density_grid,
        gas_saturation_grid=gas_saturation_grid,
        gas_viscosity_grid=gas_viscosity_grid,  # type: ignore
        gas_compressibility_grid=gas_compressibility_grid,  # type: ignore
        gas_compressibility_factor_grid=gas_compressibility_factor_grid,  # type: ignore
        gas_gravity_grid=gas_gravity_grid,  # type: ignore
        gas_molecular_weight_grid=gas_molecular_weight_grid,  # type: ignore
        gas_density_grid=gas_density_grid,  # type: ignore
        water_saturation_grid=water_saturation_grid,
        water_viscosity_grid=water_viscosity_grid,  # type: ignore
        water_compressibility_grid=water_compressibility_grid,  # type: ignore
        water_density_grid=water_density_grid,  # type: ignore
        water_bubble_point_pressure_grid=water_bubble_point_pressure_grid,  # type: ignore
        solution_gas_to_oil_ratio_grid=solution_gas_to_oil_ratio_grid,
        gas_solubility_in_water_grid=gas_solubility_in_water_grid,  # type: ignore
        oil_formation_volume_factor_grid=oil_formation_volume_factor_grid,  # type: ignore
        gas_formation_volume_factor_grid=gas_formation_volume_factor_grid,  # type: ignore
        water_formation_volume_factor_grid=water_formation_volume_factor_grid,  # type: ignore
        water_salinity_grid=water_salinity_grid,
        solvent_concentration_grid=solvent_concentration_grid,
        oil_effective_viscosity_grid=oil_effective_viscosity_grid,
        oil_effective_density_grid=oil_effective_density_grid,
        reservoir_gas=reservoir_gas,
    )
    rock_properties = RockProperties(
        compressibility=rock_compressibility,
        absolute_permeability=absolute_permeability,
        net_to_gross_ratio_grid=net_to_gross_ratio_grid,
        porosity_grid=porosity_grid,
        residual_oil_saturation_water_grid=residual_oil_saturation_water_grid,
        residual_oil_saturation_gas_grid=residual_oil_saturation_gas_grid,
        residual_gas_saturation_grid=residual_gas_saturation_grid,
        irreducible_water_saturation_grid=irreducible_water_saturation_grid,
        connate_water_saturation_grid=connate_water_saturation_grid,
    )

    if saturation_history is None:
        # Just store the initial saturations as the max saturations
        saturation_history = SaturationHistory.from_initial_saturations(
            water_saturation_grid=water_saturation_grid.copy(),
            gas_saturation_grid=gas_saturation_grid.copy(),
        )

    model = ReservoirModel(
        grid_shape=grid_shape,
        cell_dimension=cell_dimension,
        thickness_grid=thickness_grid,
        fluid_properties=fluid_properties,
        rock_properties=rock_properties,
        saturation_history=saturation_history,
        dip_angle=dip_angle,
        dip_azimuth=dip_azimuth,
        datum_depth=datum_depth,
    )
    if fractures is not None and len(model.grid_shape) == 3:
        return apply_fractures(model, *fractures)  # type: ignore[return-value]
    return model


def injection_well(
    well_name: str,
    perforating_intervals: typing.Sequence[typing.Tuple[Coordinates, Coordinates]],
    radius: float,
    control: WellControl,
    injected_fluid: InjectedFluid,
    **kwargs: typing.Any,
) -> InjectionWell[Coordinates]:
    """
    Constructs an injection well with the given parameters.

    :param well_name: Name or identifier for the well
    :param perforating_intervals: Sequence of tuples representing the start and end locations of each interval in the grid
    :param radius: Radius of the well (ft)
    :param injected_fluid: The fluid being injected into the well, represented as a `WellFluid` instance.
    :param control: Control parameters for the well, represented as a `WellControl` instance.
    :param kwargs: Additional keyword arguments to be passed to the `InjectionWell` constructor
    :return: `InjectionWell` instance
    """
    return InjectionWell(
        name=well_name,
        perforating_intervals=perforating_intervals,
        radius=radius,
        control=control,
        injected_fluid=injected_fluid,
        **kwargs,
    )


def production_well(
    well_name: str,
    perforating_intervals: typing.Sequence[typing.Tuple[Coordinates, Coordinates]],
    radius: float,
    control: WellControl,
    produced_fluids: typing.Sequence[ProducedFluid],
    skin_factor: float = 0.0,
    **kwargs: typing.Any,
) -> ProductionWell[Coordinates]:
    """
    Constructs a production well with the given parameters.

    :param well_name: Name or identifier for the well
    :param perforating_intervals: Sequence of tuples representing the start and end locations of each interval in the grid
    :param radius: Radius of the well (ft)
    :param produced_fluids: List of fluids being produced by the well, represented as a sequence of `WellFluid` instances.
    :param skin_factor: Skin factor for the well, default is 0.0
    :param control: Control parameters for the well, represented as a `WellControl` instance.
    :param kwargs: Additional keyword arguments to be passed to the `ProductionWell` constructor
    :return: `ProductionWell` instance
    """
    if not produced_fluids:
        raise ValueError("Produced fluids sequence must not be empty.")
    return ProductionWell(
        name=well_name,
        perforating_intervals=perforating_intervals,
        radius=radius,
        skin_factor=skin_factor,
        control=control,
        produced_fluids=produced_fluids,
        **kwargs,
    )


def wells_(
    injectors: typing.Optional[typing.Sequence[InjectionWell[Coordinates]]] = None,
    producers: typing.Optional[typing.Sequence[ProductionWell[Coordinates]]] = None,
    **kwargs: typing.Any,
) -> Wells[Coordinates]:
    """
    Constructs a ``Wells`` instance containing both injection and production wells.

    :param injectors: Sequence of injection wells
    :param producers: Sequence of production wells
    :param kwargs: Additional keyword arguments to be passed to the `Wells` constructor
    :return: ``Wells`` instance
    """
    return Wells(
        injection_wells=injectors or [],
        production_wells=producers or [],
        **kwargs,
    )

import logging
import typing

from bores.boundary_conditions import BoundaryConditions, BoundaryMetadata
from bores.models import FluidProperties
from bores.types import NDimensionalGrid, ThreeDimensionalGrid, ThreeDimensions

logger = logging.getLogger(__name__)


def apply_pressure_boundary_condition(
    padded_pressure_grid: ThreeDimensionalGrid,
    boundary_conditions: BoundaryConditions[ThreeDimensions],
    cell_dimension: typing.Tuple[float, float],
    grid_shape: typing.Tuple[int, int, int],
    thickness_grid: NDimensionalGrid[ThreeDimensions],
    time: float,
    pad_width: int = 1,
) -> ThreeDimensionalGrid:
    """
    Applies presure boundary condition to the pressure grid (in-place).

    :param padded_pressure_grid: The padded pressure grid.
    :param boundary_conditions: The boundary conditions to apply.
    :param cell_dimension: The dimensions of each grid cell.
    :param grid_shape: The shape of the simulation grid.
    :param thickness_grid: The (unpadded) thickness grid of the reservoir.
    :param pad_width: Number of ghost cells used for grid padding.
    :param time: The current simulation time.
    """
    boundary_conditions["pressure"].apply(
        padded_pressure_grid,
        metadata=BoundaryMetadata(
            cell_dimension=cell_dimension,
            thickness_grid=thickness_grid,
            time=time,
            grid_shape=grid_shape,
            property_name="pressure",
        ),
        pad_width=pad_width,
    )
    return padded_pressure_grid


def apply_saturation_boundary_conditions(
    padded_water_saturation_grid: ThreeDimensionalGrid,
    padded_oil_saturation_grid: ThreeDimensionalGrid,
    padded_gas_saturation_grid: ThreeDimensionalGrid,
    boundary_conditions: BoundaryConditions[ThreeDimensions],
    cell_dimension: typing.Tuple[float, float],
    grid_shape: typing.Tuple[int, int, int],
    thickness_grid: NDimensionalGrid[ThreeDimensions],
    time: float,
    pad_width: int = 1,
) -> typing.Tuple[ThreeDimensionalGrid, ThreeDimensionalGrid, ThreeDimensionalGrid]:
    """
    Applies boundary conditions to the fluid saturation grids (in-place).

    :param padded_water_saturation_grid: The padded water saturation grid.
    :param padded_oil_saturation_grid: The padded oil saturation grid.
    :param padded_gas_saturation_grid: The padded gas saturation grid.
    :param boundary_conditions: The boundary conditions to apply.
    :param cell_dimension: The dimensions of each grid cell.
    :param grid_shape: The shape of the simulation grid.
    :param thickness_grid: The (unpadded) thickness grid of the reservoir.
    :param pad_width: Number of ghost cells used for grid padding.
    :param time: The current simulation time.
    """
    boundary_conditions["oil_saturation"].apply(
        padded_oil_saturation_grid,
        metadata=BoundaryMetadata(
            cell_dimension=cell_dimension,
            thickness_grid=thickness_grid,
            time=time,
            grid_shape=grid_shape,
            property_name="oil_saturation",
        ),
        pad_width=pad_width,
    )
    boundary_conditions["water_saturation"].apply(
        padded_water_saturation_grid,
        metadata=BoundaryMetadata(
            cell_dimension=cell_dimension,
            thickness_grid=thickness_grid,
            time=time,
            grid_shape=grid_shape,
            property_name="water_saturation",
        ),
        pad_width=pad_width,
    )
    boundary_conditions["gas_saturation"].apply(
        padded_gas_saturation_grid,
        metadata=BoundaryMetadata(
            cell_dimension=cell_dimension,
            thickness_grid=thickness_grid,
            time=time,
            grid_shape=grid_shape,
            property_name="gas_saturation",
        ),
        pad_width=pad_width,
    )
    return (
        padded_water_saturation_grid,
        padded_oil_saturation_grid,
        padded_gas_saturation_grid,
    )


def apply_temperature_boundary_condition(
    padded_temperature_grid: ThreeDimensionalGrid,
    boundary_conditions: BoundaryConditions[ThreeDimensions],
    cell_dimension: typing.Tuple[float, float],
    grid_shape: typing.Tuple[int, int, int],
    thickness_grid: NDimensionalGrid[ThreeDimensions],
    time: float,
    pad_width: int = 1,
) -> ThreeDimensionalGrid:
    """
    Applies presure boundary condition to the temperature grid (in-place).

    :param padded_temperature_grid: The padded temperature grid.
    :param boundary_conditions: The boundary conditions to apply.
    :param cell_dimension: The dimensions of each grid cell.
    :param grid_shape: The shape of the simulation grid.
    :param thickness_grid: The (unpadded) thickness grid of the reservoir.
    :param pad_width: Number of ghost cells used for grid padding.
    :param time: The current simulation time.
    """
    boundary_conditions["temperature"].apply(
        padded_temperature_grid,
        metadata=BoundaryMetadata(
            cell_dimension=cell_dimension,
            thickness_grid=thickness_grid,
            time=time,
            grid_shape=grid_shape,
            property_name="temperature",
        ),
        pad_width=pad_width,
    )
    return padded_temperature_grid


def apply_boundary_conditions(
    padded_fluid_properties: FluidProperties[ThreeDimensions],
    boundary_conditions: BoundaryConditions[ThreeDimensions],
    cell_dimension: typing.Tuple[float, float],
    grid_shape: typing.Tuple[int, int, int],
    thickness_grid: NDimensionalGrid[ThreeDimensions],
    time: float,
    pad_width: int = 1,
) -> FluidProperties:
    """
    Applies boundary conditions to the (padded) fluid property grids (in-plcae).

    :param padded_fluid_properties: The padded fluid properties.
    :param boundary_conditions: The boundary conditions to apply.
    :param cell_dimension: The dimensions of each grid cell.
    :param grid_shape: The shape of the simulation grid.
    :param thickness_grid: The (unpadded) thickness grid of the reservoir.
    :param pad_width: Number of ghost cells used for grid padding.
    :param time: The current simulation time.
    """
    apply_pressure_boundary_condition(
        padded_pressure_grid=padded_fluid_properties.pressure_grid,
        boundary_conditions=boundary_conditions,
        cell_dimension=cell_dimension,
        grid_shape=grid_shape,
        thickness_grid=thickness_grid,
        time=time,
        pad_width=pad_width,
    )
    apply_saturation_boundary_conditions(
        padded_water_saturation_grid=padded_fluid_properties.water_saturation_grid,
        padded_oil_saturation_grid=padded_fluid_properties.oil_saturation_grid,
        padded_gas_saturation_grid=padded_fluid_properties.gas_saturation_grid,
        boundary_conditions=boundary_conditions,
        cell_dimension=cell_dimension,
        grid_shape=grid_shape,
        thickness_grid=thickness_grid,
        time=time,
        pad_width=pad_width,
    )
    apply_temperature_boundary_condition(
        padded_temperature_grid=padded_fluid_properties.temperature_grid,
        boundary_conditions=boundary_conditions,
        cell_dimension=cell_dimension,
        grid_shape=grid_shape,
        thickness_grid=thickness_grid,
        time=time,
        pad_width=pad_width,
    )
    return padded_fluid_properties

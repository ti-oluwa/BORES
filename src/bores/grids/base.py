import typing
import warnings

import attrs
import numba  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from bores._precision import get_dtype
from bores.errors import ValidationError
from bores.grids.utils import pad_grid, unpad_grid
from bores.serialization import Serializable
from bores.types import ArrayLike, NDimension, NDimensionalGrid, Orientation

__all__ = [
    "CapillaryPressureGrids",
    "RelativeMobilityGrids",
    "apply_structural_dip",
    "array",
    "build_depth_grid",
    "build_elevation_grid",
    "build_layered_grid",
    "build_uniform_grid",
    "depth_grid",
    "elevation_grid",
    "layered_grid",
    "uniform_grid",
]


def array(obj: typing.Any, **kwargs: typing.Any):
    """
    Wrapper around np.array to enforce global dtype.

    :param obj: Object to convert to numpy array
    :param kwargs: Additional keyword arguments for `np.array`
    :return: return value of `np.array`
    """
    kwargs.setdefault("dtype", get_dtype())
    return np.array(obj, **kwargs)


def build_uniform_grid(
    grid_shape: NDimension,
    value: float = 0.0,
) -> NDimensionalGrid[NDimension]:
    """
    Constructs a N-Dimensional uniform grid with the specified initial value.

    :param grid_shape: Tuple of number of cells in all directions (x, y, z).
    :param value: Initial value to fill the grid with
    :return: Numpy array representing the grid
    """
    return np.full(  # type: ignore
        grid_shape,
        fill_value=value,
        dtype=get_dtype(),
        order="C",
    )


uniform_grid = build_uniform_grid  # Alias for convenience


def build_layered_grid(
    grid_shape: NDimension,
    layer_values: ArrayLike[float],
    orientation: typing.Union[Orientation, typing.Literal["x", "y", "z"]],
) -> NDimensionalGrid[NDimension]:
    """
    Constructs a N-Dimensional layered grid with specified layer values.

    :param grid_shape: Tuple of number of cells in x, y, and z directions (cell_count_x, cell_count_y, cell_count_z)
    :param orientation: Direction or axis along which layers are defined ('x', 'y', or 'z')
    :param layer_values: Values for each layer (must match number of layers).
        The number of values should match the number of cells in that direction.
        If the grid NDimension is (50, 30, 10) and orientation is 'horizontal',
        then values should have exactly 50 values.
        If orientation is 'vertical', then values should have exactly 30 values.

    :return: N-Dimensional numpy array representing the grid
    """
    if len(layer_values) < 1:
        raise ValidationError("At least one layer value must be provided.")

    orientation = (
        Orientation(orientation) if isinstance(orientation, str) else orientation
    )
    dtype = get_dtype()
    layered_grid = build_uniform_grid(grid_shape=grid_shape, value=0.0)
    if orientation == Orientation.X:  # Layering along x-axis
        if len(layer_values) != grid_shape[0]:
            raise ValidationError(
                "Number of layer values must match number of cells in x direction."
            )

        for i, layer_value in enumerate(layer_values):
            layered_grid[i, :, :] = layer_value
        return layered_grid.astype(dtype, copy=False)

    elif orientation == Orientation.Y:  # Layering along y-axis
        if len(layer_values) != grid_shape[1]:
            raise ValidationError(
                "Number of layer values must match number of cells in y direction."
            )

        for j, layer_value in enumerate(layer_values):
            layered_grid[:, j, :] = layer_value
        return layered_grid.astype(dtype, copy=False)

    elif orientation == Orientation.Z:  # Layering along z-axis
        if len(grid_shape) != 3:
            raise ValidationError(
                "Grid dimension must be N-Dimensional for z-direction layering."
            )

        if len(layer_values) != grid_shape[2]:
            raise ValidationError(
                "Number of layer values must match number of cells in z direction."
            )

        for k, layer_value in enumerate(layer_values):
            layered_grid[:, :, k] = layer_value
        return layered_grid.astype(dtype, copy=False)

    raise ValidationError(
        "Invalid layering direction. Must be one of 'x', 'y', or 'z'."
    )


layered_grid = build_layered_grid  # Alias for convenience


@numba.njit(cache=True)
def _compute_elevation_downward(
    thickness_grid: NDimensionalGrid[NDimension],
    dtype: npt.DTypeLike,
    datum: float = 0.0,
) -> NDimensionalGrid[NDimension]:
    """
    Compute elevation grid in downward direction (depth from top).

    :param thickness_grid: 3D array of cell thicknesses (ft)
    :param dtype: NumPy dtype for array allocation
    :param datum: Reference elevation/depth for the bottom/top of the grid (ft).
    :return: 3D elevation grid (ft)
    """
    _, _, nz = thickness_grid.shape
    elevation_grid = np.zeros_like(thickness_grid, dtype=dtype)

    # Start from top layer
    elevation_grid[:, :, 0] = thickness_grid[:, :, 0] / 2
    for k in range(1, nz):
        elevation_grid[:, :, k] = (
            elevation_grid[:, :, k - 1]
            + thickness_grid[:, :, k - 1] / 2
            + thickness_grid[:, :, k] / 2
        )

    return elevation_grid + datum  # type: ignore


@numba.njit(cache=True)
def _compute_elevation_upward(
    thickness_grid: NDimensionalGrid[NDimension],
    dtype: npt.DTypeLike,
    datum: float = 0.0,
) -> NDimensionalGrid[NDimension]:
    """
    Compute elevation grid in upward direction (elevation from bottom).

    :param thickness_grid: 3D array of cell thicknesses (ft)
    :param dtype: NumPy dtype for array allocation
    :param datum: Reference elevation/depth for the bottom/top of the grid (ft).
    :return: 3D elevation grid (ft)
    """
    _, _, nz = thickness_grid.shape
    elevation_grid = np.zeros_like(thickness_grid, dtype=dtype)

    # Start from bottom layer
    elevation_grid[:, :, -1] = thickness_grid[:, :, -1] / 2
    for k in range(nz - 2, -1, -1):
        elevation_grid[:, :, k] = (
            elevation_grid[:, :, k + 1]
            + thickness_grid[:, :, k + 1] / 2
            + thickness_grid[:, :, k] / 2
        )

    return elevation_grid + datum  # type: ignore


def _build_elevation_grid(
    thickness_grid: NDimensionalGrid[NDimension],
    direction: typing.Literal["downward", "upward"] = "downward",
    datum: float = 0.0,
) -> NDimensionalGrid[NDimension]:
    """
    Convert a cell thickness (height) grid into an absolute elevation grid (cell center z-coordinates).

    The elevation grid is generated based on the thickness of each cell, starting from the top or bottom
    of the reservoir, depending on the specified direction.

    :param thickness_grid: N-dimensional numpy array representing the thickness of each cell in the reservoir (ft).
    :param direction: Direction to generate the elevation grid.
        Can be "downward" (from top to bottom) or "upward" (from bottom to top).
        "downward" basically gives a depth grid from the top of the reservoir.
    :param datum: Reference elevation/depth for the bottom/top of the grid (ft).
    :return: N-dimensional numpy array representing the elevation of each cell in the reservoir (ft).
    """
    if direction not in {"downward", "upward"}:
        raise ValidationError("direction must be 'downward' or 'upward'")

    dtype = get_dtype()

    if direction == "downward":
        return _compute_elevation_downward(thickness_grid, dtype=dtype, datum=datum)
    return _compute_elevation_upward(thickness_grid, dtype=dtype, datum=datum)


def build_elevation_grid(
    thickness_grid: NDimensionalGrid[NDimension], datum: float = 0.0
) -> NDimensionalGrid[NDimension]:
    """
    Build an upward elevation grid from a thickness grid.

    Elevation is measured upward from a datum level, where positive elevation
    means above the datum. The datum typically represents:
    - Sea level (datum = 0)
    - Base of reservoir (datum = base elevation)
    - Reference surface (datum = reference elevation)

    :param thickness_grid: N-dimensional numpy array representing the thickness
        of each cell in the reservoir (ft).
    :param datum: Reference elevation for the bottom of the grid (ft). Elevation of the bottom surface of the grid.
        - datum = 0: Bottom layer starts at elevation 0 (e.g., sea level)
        - datum > 0: Bottom layer starts above the reference (e.g., above sea level)
        - datum < 0: Bottom layer starts below the reference (e.g., subsea)
        Default is 0.0 (bottom layer at reference level).
        - Negative if bottom is subsea (most common)
        - Positive if bottom is above sea level
        - Zero if bottom is exactly at sea level
    :return: N-dimensional numpy array representing the elevation of each cell
        center in the reservoir (ft), measured upward from datum.

    Example:
    ```python
    # Reservoir from -2000 to -1000 ft (subsea)
    thickness = np.full((10, 10, 20), 50.0)  # 20 layers, 50 ft each

    # Datum at base of reservoir
    elev_grid = build_elevation_grid(thickness, datum=-2000.0)
    # elev_grid[0,0,-1] = -1975.0 ft (center of bottom 50-ft layer)
    # elev_grid[0,0,0] = -1025.0 ft  (center of top layer)

    # Datum at sea level (bottom at -1000 ft from top)
    elev_grid = build_elevation_grid(thickness, datum=-1000.0)
    # elev_grid[0,0,-1] = -975.0 ft
    # elev_grid[0,0,0] = -25.0 ft
    ```

    Notes:
        - Elevation increases upward (k=-1 is lowest, k=0 is highest)
        - For depth (downward-positive), use `build_depth_grid()` instead
        - Datum represents the elevation of the BOTTOM of the grid
    """
    return _build_elevation_grid(thickness_grid, direction="upward", datum=datum)


elevation_grid = build_elevation_grid  # Alias for convenience


def build_depth_grid(
    thickness_grid: NDimensionalGrid[NDimension], datum: float = 0.0
) -> NDimensionalGrid[NDimension]:
    """
    Build a downward depth grid from a thickness grid.

    Depth is measured downward from a datum level, where positive depth means
    below the datum. The datum typically represents:
    - Sea level (datum = 0)
    - Ground surface (datum = surface elevation)
    - Top of reservoir (datum = top depth)

    :param thickness_grid: N-dimensional numpy array representing the thickness
        of each cell in the reservoir (ft).
    :param datum: Reference depth for the top of the grid (ft). Depth of the top surface of the grid.
        - datum = 0: Top layer starts at depth 0 (e.g., sea level)
        - datum > 0: Top layer starts below the reference (e.g., subsea depth)
        - datum < 0: Top layer starts above the reference (e.g., above sea level)
        Default is 0.0 (top layer at reference level).
        - Always positive (depth increases downward)
        - datum = 1000.0 means top is at 1000 ft depth
    :return: N-dimensional numpy array representing the depth of each cell
        center in the reservoir (ft), measured downward from datum.

    Example:
    ```python
    # Reservoir 1000-2000 ft subsea depth
    thickness = np.full((10, 10, 20), 50.0)  # 20 layers, 50 ft each

    # Option 1: Datum at sea level, specify top depth
    depth_grid = build_depth_grid(thickness, datum=1000.0)
    # depth_grid[0,0,0] = 1025.0 ft  (center of first 50-ft layer)
    # depth_grid[0,0,-1] = 1975.0 ft (center of last layer)

    # Option 2: Datum at top of reservoir
    depth_grid = build_depth_grid(thickness, datum=0.0)
    # depth_grid[0,0,0] = 25.0 ft  (relative to top)
    # depth_grid[0,0,-1] = 975.0 ft
    ```

    Notes:
        - Depth increases downward (k=0 is shallowest, k=-1 is deepest)
        - For elevation (upward-positive), use `build_elevation_grid()` instead
        - Datum represents the depth/elevation of the TOP of the grid
    """
    return _build_elevation_grid(thickness_grid, direction="downward", datum=datum)


depth_grid = build_depth_grid  # Alias for convenience


@numba.njit(parallel=True, cache=True)
def _apply_dip_upward(
    dipped_elevation_grid: NDimensionalGrid[NDimension],
    grid_dimensions: typing.Tuple[int, int],
    cell_dimensions: typing.Tuple[float, float],
    dip_components: typing.Tuple[float, float, float],
) -> NDimensionalGrid[NDimension]:
    """
    Apply structural dip for upward elevation convention (parallel).

    Each (i,j) column is processed independently, allowing parallelization.

    :param dipped_elevation_grid: Grid to modify in-place
    :param grid_dimensions: (nx, ny) - number of cells in x and y directions
    :param cell_dimensions: (cell_size_x, cell_size_y) - cell sizes in feet
    :param dip_components: (dx_component, dy_component, tan_dip_angle) - pre-computed dip parameters
    :return: Modified elevation grid
    """
    nx, ny = grid_dimensions
    cell_size_x, cell_size_y = cell_dimensions
    dx_component, dy_component, tan_dip_angle = dip_components

    for i in numba.prange(nx):  # type: ignore  # Parallel outer loop
        for j in range(ny):
            x_distance = i * cell_size_x
            y_distance = j * cell_size_y
            distance_along_dip = (x_distance * dx_component) + (
                y_distance * dy_component
            )
            dip_offset = distance_along_dip * tan_dip_angle
            # Upward: moving in dip direction decreases elevation
            dipped_elevation_grid[i, j, :] -= dip_offset

    return dipped_elevation_grid


@numba.njit(parallel=True, cache=True)
def _apply_dip_downward(
    dipped_elevation_grid: NDimensionalGrid[NDimension],
    grid_dimensions: typing.Tuple[int, int],
    cell_dimensions: typing.Tuple[float, float],
    dip_components: typing.Tuple[float, float, float],
) -> NDimensionalGrid[NDimension]:
    """
    Apply structural dip for downward depth convention (parallel).

    Each (i,j) column is processed independently, allowing parallelization.

    :param dipped_elevation_grid: Grid to modify in-place
    :param grid_dimensions: (nx, ny) - number of cells in x and y directions
    :param cell_dimensions: (cell_size_x, cell_size_y) - cell sizes in feet
    :param dip_components: (dx_component, dy_component, tan_dip_angle) - pre-computed dip parameters
    :return: Modified elevation grid
    """
    nx, ny = grid_dimensions
    cell_size_x, cell_size_y = cell_dimensions
    dx_component, dy_component, tan_dip_angle = dip_components

    for i in numba.prange(nx):  # type: ignore  # Parallel outer loop
        for j in range(ny):
            x_distance = i * cell_size_x
            y_distance = j * cell_size_y
            distance_along_dip = (x_distance * dx_component) + (
                y_distance * dy_component
            )
            dip_offset = distance_along_dip * tan_dip_angle
            # Downward: moving in dip direction increases depth
            dipped_elevation_grid[i, j, :] += dip_offset

    return dipped_elevation_grid


def apply_structural_dip(
    elevation_grid: NDimensionalGrid[NDimension],
    cell_dimension: typing.Tuple[float, float],
    elevation_direction: typing.Literal["downward", "upward"],
    dip_angle: float,
    dip_azimuth: float,
) -> NDimensionalGrid[NDimension]:
    """
    Apply structural dip to a base elevation grid using azimuth convention.

    The dip is applied by adding a planar gradient in the specified azimuth direction.
    The dip angle represents the angle of the reservoir surface from horizontal.

    ---

    ## **Azimuth Convention:**
    ```
    Grid Coordinate System:
    North (0°/360°)
         ↑ (+y)
         |
         |
    West ←─────┼─────→ East (90°)
    (270°)  |    (+x)
         |
         ↓ (-y)
    South (180°)
    ```

    Azimuth Examples:
    - 0° (North): Dips toward North
    - 90° (East): Dips toward East
    - 180° (South): Dips toward South
    - 270° (West): Dips toward West
    - 45° (NE): Dips toward Northeast

    The surface tilts DOWN in the azimuth direction, meaning elevation
    DECREASES in that direction (or depth INCREASES for downward convention).

    :param elevation_grid: Base flat elevation grid (shape: [nx, ny, nz])
    :param cell_dimension: Tuple of (cell_size_x, cell_size_y) in feet
    :param elevation_direction: Whether elevation is "upward" (elevation) or "downward" (depth)
    :param dip_angle: Dip angle in degrees (0-90)
    :param dip_azimuth: Dip azimuth in degrees (0-360), measured clockwise from North
    :return: Elevation grid with structural dip applied
    """
    if elevation_direction not in {"downward", "upward"}:
        raise ValidationError("`elevation_direction` must be 'downward' or 'upward'")

    if not (0.0 <= dip_angle <= 90.0):
        raise ValidationError("`dip_angle` must be between 0 and 90 degrees")

    if not (0.0 <= dip_azimuth < 360.0):
        raise ValidationError("`dip_azimuth` must be between 0 and 360 degrees")

    dtype = get_dtype()
    dipped_elevation_grid = elevation_grid.copy().astype(dtype, copy=False)
    dip_angle_radians = np.radians(dip_angle, dtype=dtype)
    dip_azimuth_radians = np.radians(dip_azimuth, dtype=dtype)

    grid_shape = elevation_grid.shape
    nx, ny = grid_shape[0], grid_shape[1]

    # Convert azimuth to directional components
    # Azimuth: 0° = North (+y), 90° = East (+x), 180° = South (-y), 270° = West (-x)
    dx_component = np.sin(dip_azimuth_radians)  # Positive = East
    dy_component = np.cos(dip_azimuth_radians)  # Positive = North
    tan_dip_angle = np.tan(dip_angle_radians)

    grid_dimensions = (nx, ny)
    dip_components = (dx_component, dy_component, tan_dip_angle)

    if elevation_direction == "upward":
        return _apply_dip_upward(
            dipped_elevation_grid=dipped_elevation_grid,
            grid_dimensions=grid_dimensions,
            cell_dimensions=cell_dimension,
            dip_components=dip_components,
        )
    return _apply_dip_downward(
        dipped_elevation_grid=dipped_elevation_grid,
        grid_dimensions=grid_dimensions,
        cell_dimensions=cell_dimension,
        dip_components=dip_components,
    )


class PadMixin(typing.Generic[NDimension]):
    """Mixin class to add padding functionality to attrs classes with numpy array fields."""

    def get_paddable_fields(self) -> typing.Iterable[typing.Any]:
        """Return iterable of attrs fields that can be padded."""
        raise NotImplementedError

    def pad(
        self,
        pad_width: int = 1,
        hook: typing.Optional[
            typing.Callable[
                [NDimensionalGrid[NDimension]], NDimensionalGrid[NDimension]
            ]
        ] = None,
        exclude: typing.Optional[typing.Iterable[str]] = None,
    ) -> Self:
        """
        Pad all numpy array fields in the attrs class.

        :param pad_width: Number of cells to pad on each side of each dimension.
        :param hook: Optional callable to apply additional processing to each padded grid.
        :param exclude: Optional iterable of field names to exclude from hooking.
        :return: New instance of the attrs class with padded numpy array fields.
        """
        if not attrs.has(type(self)):
            raise TypeError(
                f"{self.__class__.__name__} can only be used with attrs classes"
            )

        target_fields = self.get_paddable_fields()
        padded_fields_values = {}
        non_init_fields_values = {}
        for field in target_fields:
            value = getattr(self, field.name)
            if not isinstance(value, np.ndarray):
                raise TypeError(
                    f"Field '{field.name}' is not a numpy array and cannot be padded"
                )

            padded_value = pad_grid(grid=value, pad_width=pad_width)
            if hook and (not exclude or field.name not in exclude):
                padded_value = hook(padded_value)

            if not field.init:
                non_init_fields_values[field.name] = padded_value
            else:
                padded_fields_values[field.name] = padded_value

        instance = attrs.evolve(self, **padded_fields_values)  # type: ignore[misc]
        for name, value in non_init_fields_values:
            object.__setattr__(instance, name, value)
        return instance

    def unpad(self, pad_width: int = 1) -> Self:
        """
        Remove padding from all numpy array fields in the attrs class.

        :param pad_width: Number of cells to remove from each side of each dimension.
        :return: New instance of the attrs class with unpadded numpy array fields.
        """
        if not attrs.has(type(self)):
            raise TypeError(
                f"{self.__class__.__name__} can only be used with attrs classes"
            )

        target_fields = self.get_paddable_fields()
        unpadded_fields_values = {}
        non_init_fields_values = {}
        for field in target_fields:
            value = getattr(self, field.name)
            if not isinstance(value, np.ndarray):
                raise TypeError(
                    f"Field '{field.name}' is not a numpy array and cannot be padded"
                )

            padded_value = unpad_grid(grid=value, pad_width=pad_width)
            if not field.init:
                non_init_fields_values[field.name] = padded_value
            else:
                unpadded_fields_values[field.name] = padded_value

        instance = attrs.evolve(self, **unpadded_fields_values)  # type: ignore[misc]
        for name, value in non_init_fields_values:
            object.__setattr__(instance, name, value)
        return instance

    def apply_hook(
        self,
        hook: typing.Callable[
            [NDimensionalGrid[NDimension]], NDimensionalGrid[NDimension]
        ],
        exclude: typing.Optional[typing.Iterable[str]] = None,
    ) -> Self:
        """
        Apply a hook function to all numpy array fields in the attrs class.

        :param hook: Callable to apply to each numpy array field.
        :param exclude: Optional iterable of field names to exclude from hooking.
        :return: New instance of the attrs class with hooked numpy array fields.
        """
        if not attrs.has(type(self)):
            raise TypeError(
                f"{self.__class__.__name__} can only be used with attrs classes"
            )

        target_fields = self.get_paddable_fields()
        hooked_fields = {}
        for field in target_fields:
            if exclude and field.name in exclude:
                continue
            value = getattr(self, field.name)
            if not isinstance(value, np.ndarray):
                raise TypeError(
                    f"Field '{field.name}' is not a numpy array and cannot be padded"
                )
            hooked_value = hook(value)
            hooked_fields[field.name] = hooked_value
        return attrs.evolve(self, **hooked_fields)  # type: ignore[misc]


@attrs.frozen(slots=True)
class RelPermGrids(PadMixin[NDimension], Serializable):  # type: ignore[override]
    """
    Wrapper for n-dimensional grids representing relative permeabilities
    for different fluid phases (oil, water, gas).
    """

    oil_relative_permeability: NDimensionalGrid[NDimension]
    """Grid representing oil relative permeability."""
    water_relative_permeability: NDimensionalGrid[NDimension]
    """Grid representing water relative permeability."""
    gas_relative_permeability: NDimensionalGrid[NDimension]
    """Grid representing gas relative permeability."""

    @property
    def kro(self) -> NDimensionalGrid[NDimension]:
        return self.oil_relative_permeability

    @property
    def krw(self) -> NDimensionalGrid[NDimension]:
        return self.water_relative_permeability

    @property
    def krg(self) -> NDimensionalGrid[NDimension]:
        return self.gas_relative_permeability

    Kro = kro
    Krw = krw
    Krg = krg

    def __iter__(self) -> typing.Iterator[NDimensionalGrid[NDimension]]:
        yield self.water_relative_permeability
        yield self.oil_relative_permeability
        yield self.gas_relative_permeability

    def get_paddable_fields(self) -> typing.Iterable[typing.Any]:
        return attrs.fields(self.__class__)


@attrs.frozen(slots=True)
class RelativeMobilityGrids(PadMixin[NDimension], Serializable):
    """
    Wrapper for n-dimensional grids representing relative mobilities
    for different fluid phases (oil, water, gas).
    """

    oil_relative_mobility: NDimensionalGrid[NDimension]
    """Grid representing oil relative mobility."""
    water_relative_mobility: NDimensionalGrid[NDimension]
    """Grid representing water relative mobility."""
    gas_relative_mobility: NDimensionalGrid[NDimension]
    """Grid representing gas relative mobility."""

    @property
    def λo(self) -> NDimensionalGrid[NDimension]:
        return self.oil_relative_mobility

    @property
    def λw(self) -> NDimensionalGrid[NDimension]:
        return self.water_relative_mobility

    @property
    def λg(self) -> NDimensionalGrid[NDimension]:
        return self.gas_relative_mobility

    def __iter__(self) -> typing.Iterator[NDimensionalGrid[NDimension]]:
        yield self.water_relative_mobility
        yield self.oil_relative_mobility
        yield self.gas_relative_mobility

    def get_paddable_fields(self) -> typing.Iterable[typing.Any]:
        return attrs.fields(self.__class__)


@attrs.frozen(slots=True)
class CapillaryPressureGrids(PadMixin[NDimension], Serializable):
    """
    Wrapper for n-dimensional grids representing capillary pressures
    for different fluid phases (oil-water, oil-gas).
    """

    oil_water_capillary_pressure: NDimensionalGrid[NDimension]
    """Grid representing oil-water capillary pressure."""
    gas_oil_capillary_pressure: NDimensionalGrid[NDimension]
    """Grid representing gas-oil capillary pressure."""

    @property
    def pcow(self) -> NDimensionalGrid[NDimension]:
        return self.oil_water_capillary_pressure

    @property
    def pcgo(self) -> NDimensionalGrid[NDimension]:
        return self.gas_oil_capillary_pressure

    Pcow = pcow
    Pcgo = pcgo

    def __iter__(self) -> typing.Iterator[NDimensionalGrid[NDimension]]:
        yield self.oil_water_capillary_pressure
        yield self.gas_oil_capillary_pressure

    def get_paddable_fields(self) -> typing.Iterable[typing.Any]:
        return attrs.fields(self.__class__)

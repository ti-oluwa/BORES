"""Gridding Utilities"""

import typing
import warnings

import numba
import numpy as np
import numpy.typing as npt

from bores.errors import ValidationError
from bores.precision import get_dtype
from bores.types import (
    NDimension,
    NDimensionalGrid,
    Orientation,
    Spacing,
    ThreeDimensionalGrid,
    ThreeDimensions,
    TwoDimensionalGrid,
    TwoDimensions,
)

__all__ = [
    "coarsen_grid",
    "coarsen_permeability_grids",
    "flatten_multilayer_grid_to_surface",
    "flatten_multilayer_grids",
    "get_pad_mask",
    "layer_to_link_permeability",
    "link_to_layer_permeability",
    "make_saturation_grid",
    "pad_grid",
    "unpad_grid",
]


def pad_grid(
    grid: NDimensionalGrid[NDimension], pad_width: int = 1
) -> NDimensionalGrid[NDimension]:
    """
    Pads a N-Dimensional grid with the edge values to create a border around the grid.

    This is useful for finite difference methods where boundary conditions are applied.

    :param grid: N-Dimensional numpy array representing the grid to be padded
    :param pad_width: Width of the padding to be applied on all sides of the grid
    :return: Padded N-Dimensional numpy array
    """
    return np.pad(grid, pad_width=pad_width, mode="edge")  # type: ignore[return-value]


@numba.njit(cache=True)
def get_pad_mask(grid_shape: typing.Tuple[int, ...], pad_width: int = 1) -> np.ndarray:
    """
    Generate a boolean mask for the padded grid indicating the padded regions.

    :param grid_shape: Shape of the original grid before padding
    :param pad_width: Width of the padding applied on all sides of the grid
    :return: Boolean mask numpy array where True indicates padded regions
    """
    padded_shape = tuple(dim + 2 * pad_width for dim in grid_shape)
    mask = np.zeros(padded_shape, dtype=bool)

    # Set padded regions to True
    slices = tuple(
        slice(0, pad_width)
        if i == 0
        else slice(-pad_width, None)
        if i == 1
        else slice(pad_width, -pad_width)
        for i, _ in enumerate(padded_shape)
    )
    mask[slices] = True
    return mask


@numba.njit(cache=True)
def unpad_grid(
    grid: NDimensionalGrid[NDimension], pad_width: int = 1
) -> NDimensionalGrid[NDimension]:
    """
    Remove padding from a N-Dimensional grid.

    :param grid: Padded N-Dimensional numpy array representing the grid
    :param pad_width: Width of the padding to be removed from all sides of the grid
    :return: N-Dimensional numpy array with padding removed
    """
    # Build slices list explicitly (generator expressions not supported in Numba)
    ndim = grid.ndim
    if ndim == 2:
        unpadded_grid = grid[pad_width:-pad_width, pad_width:-pad_width]
    elif ndim == 3:
        unpadded_grid = grid[
            pad_width:-pad_width, pad_width:-pad_width, pad_width:-pad_width
        ]
    else:
        raise ValueError(
            f"Unsupported grid dimension: {ndim}. Only 2D and 3D grids are supported."
        )

    return unpadded_grid  # type: ignore


def coarsen_grid(
    data: np.ndarray,
    batch_size: typing.Tuple[int, ...],
    method: typing.Literal["mean", "sum", "max", "min", "harmonic"] = "mean",
    epsilon: float = 1e-10,
) -> np.ndarray:
    """
    Coarsen (downsample) a 2D or 3D grid by aggregating blocks of adjacent cells.

    Pads the grid if necessary to make dimensions divisible by `batch_size`.

    :param data: 2D or 3D numpy array to coarsen. Shape can be (nx, ny) or (nx, ny, nz).
    :param batch_size: Tuple of ints representing the coarsening factor along each dimension.
        Length must match `data.ndim`. Example: (2,2) for 2D, (2,2,2) for 3D.
    :param method: Aggregation method to use on each block.
        - 'mean': Arithmetic mean (for porosity, saturation)
        - 'sum': Sum (for total volume, pore volume)
        - 'max': Maximum value in block
        - 'min': Minimum value in block
        - 'harmonic': Harmonic mean (WARNING: only valid for isotropic averaging)
    :param epsilon: Small value to avoid division by zero in harmonic mean (default: 1e-10)
    :return: Coarsened numpy array.
    :raises ValidationError: if `batch_size` length does not match `data.ndim` or if method is unsupported.

    Note:
        For permeability coarsening, use `coarsen_permeability_grids()` instead, which
        applies direction-appropriate averaging (harmonic in flow direction, arithmetic
        perpendicular).

    Example:
    ```python
    data2d = np.arange(16, dtype=float).reshape(4,4)
    coarsen_grid(data2d, batch_size=(2,2))
    # array([[ 2.5,  4.5],
    #        [10.5, 12.5]])

    data3d = np.arange(64, dtype=float).reshape(4,4,4)
    coarsen_grid(data3d, batch_size=(2,2,2), method='max')
    # array([[[ 5.,  7.],
    #         [13., 15.]],
    #        [[21., 23.],
    #         [29., 31.]]])
    ```
    """
    if len(batch_size) != data.ndim:
        raise ValidationError(
            f"batch_size length {len(batch_size)} must match data.ndim {data.ndim}"
        )

    # Validate method
    valid_methods = ("mean", "sum", "max", "min", "harmonic")
    if method not in valid_methods:
        raise ValidationError(
            f"Unsupported method '{method}'. Must be one of {valid_methods}"
        )

    # Calculate padding needed
    pad_width = []
    for dim, b in zip(data.shape, batch_size):
        remainder = dim % b
        if remainder == 0:
            pad_width.append((0, 0))
        else:
            pad_width.append((0, b - remainder))

    # Pad with appropriate value based on method
    # Use NaN for methods that support it (will be ignored in aggregation)
    if method in ("mean", "max", "min", "harmonic"):
        pad_value = np.nan
    elif method == "sum":
        pad_value = 0.0

    data_padded = np.pad(
        data, pad_width=pad_width, mode="constant", constant_values=pad_value
    )

    # Reshape to group blocks along each dimension
    # E.g., (100, 50) with batch (2, 5) → (50, 2, 10, 5) → aggregate over axes (1, 3)
    reshape_shape = []
    for dim, b in zip(data_padded.shape, batch_size):
        reshape_shape.extend([dim // b, b])

    data_reshaped = data_padded.reshape(reshape_shape)

    # Axes to aggregate over: every second axis (the block dimensions)
    agg_axes = tuple(range(1, data_reshaped.ndim, 2))

    dtype = get_dtype()

    # Apply aggregation
    if method == "mean":
        coarsened = np.nanmean(data_reshaped, axis=agg_axes).astype(dtype)

    elif method == "sum":
        coarsened = data_reshaped.sum(axis=agg_axes, dtype=dtype)

    elif method == "max":
        coarsened = np.nanmax(data_reshaped, axis=agg_axes).astype(dtype)

    elif method == "min":
        coarsened = np.nanmin(data_reshaped, axis=agg_axes).astype(dtype)

    elif method == "harmonic":
        # Harmonic mean: H = n / sum(1/x_i)
        # Add epsilon to avoid division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            reciprocals = 1.0 / (data_reshaped + epsilon)
            reciprocals = np.where(np.isnan(data_reshaped), np.nan, reciprocals)

            # Count non-NaN values per block
            counts = np.sum(~np.isnan(data_reshaped), axis=agg_axes)

            # Sum of reciprocals, ignoring NaNs
            sum_reciprocals = np.nansum(reciprocals, axis=agg_axes)

            # Harmonic mean = n / sum(1/x)
            coarsened = counts / sum_reciprocals

            # Handle edge cases
            coarsened = np.where(counts == 0, np.nan, coarsened)
            coarsened = np.where(np.isinf(coarsened), 0.0, coarsened)
            coarsened = coarsened.astype(dtype)

    return coarsened


def _coarsen_2d_permeability_grids(
    kx: TwoDimensionalGrid,
    ky: TwoDimensionalGrid,
    batch_size: TwoDimensions,
    epsilon: float = 1e-10,
) -> typing.Tuple[TwoDimensionalGrid, TwoDimensionalGrid]:
    """
    Coarsen 2D permeability grids using direction-appropriate averaging.

    Uses Cardwell-Parsons averaging:
    - k_x: harmonic mean in x-direction, arithmetic mean in y-direction
    - k_y: harmonic mean in y-direction, arithmetic mean in x-direction

    This preserves the effective permeability for flow in each direction.

    :param kx: X-direction permeability grid (mD), shape (nx, ny)
    :param ky: Y-direction permeability grid (mD), shape (nx, ny)
    :param batch_size: Coarsening factors (bx, by) for each direction
    :param epsilon: Small value to avoid division by zero (default: 1e-10)
    :return: Tuple of (coarsened_kx, coarsened_ky)
    :raises ValidationError: If grids have different shapes or batch_size is invalid

    Example:
    ```python
    # 4x4 grid with layered permeability
    kx = np.array([[100, 100, 100, 100],
                    [  1,   1,   1,   1],
                    [100, 100, 100, 100],
                    [  1,   1,   1,   1]], dtype=float)
    ky = kx.copy()

    kx_c, ky_c = _coarsen_2d_permeability_grids(kx, ky, batch_size=(2, 2))
    # kx_c uses harmonic mean in x (across layers), arithmetic in y
    # ky_c uses harmonic mean in y (across layers), arithmetic in x
    ```

    References:
        Cardwell, W. T., & Parsons, R. L. (1945). "Average Permeabilities of
        Heterogeneous Oil Sands." Transactions of the AIME, 160(01), 34-42.
    """
    if kx.shape != ky.shape:
        raise ValidationError(
            f"Permeability grids must have same shape. Got kx: {kx.shape}, ky: {ky.shape}"
        )

    if kx.ndim != 2:
        raise ValidationError(
            f"Expected 2D grids, got {kx.ndim}D. Use _coarsen_3d_permeability_grids instead."
        )

    if len(batch_size) != 2:
        raise ValidationError(
            f"batch_size must have 2 elements for 2D grids, got {len(batch_size)}"
        )

    bx, by = batch_size
    if bx < 1 or by < 1:
        raise ValidationError(f"batch_size elements must be >= 1, got ({bx}, {by})")

    nx, ny = kx.shape

    # Compute padding
    pad_x = (bx - nx % bx) % bx
    pad_y = (by - ny % by) % by

    # Pad grids with NaN
    if pad_x > 0 or pad_y > 0:
        kx_padded = np.pad(
            kx, ((0, pad_x), (0, pad_y)), mode="constant", constant_values=np.nan
        )
        ky_padded = np.pad(
            ky, ((0, pad_x), (0, pad_y)), mode="constant", constant_values=np.nan
        )
    else:
        kx_padded = kx
        ky_padded = ky

    nx_new, ny_new = kx_padded.shape
    nx_coarse = nx_new // bx
    ny_coarse = ny_new // by

    dtype = get_dtype()

    # Initialize output arrays
    kx_coarse = np.zeros((nx_coarse, ny_coarse), dtype=dtype)
    ky_coarse = np.zeros((nx_coarse, ny_coarse), dtype=dtype)

    # Coarsen k_x: harmonic in x, arithmetic in y
    for i in range(nx_coarse):
        for j in range(ny_coarse):
            # Extract block
            block_kx = kx_padded[i * bx : (i + 1) * bx, j * by : (j + 1) * by]

            # First, take harmonic mean along x-direction (axis=0)
            kx_harmonic_x = _axis_harmonic_mean(block_kx, axis=0, epsilon=epsilon)

            # Then, take arithmetic mean along y-direction
            kx_coarse[i, j] = np.nanmean(kx_harmonic_x)

    # Coarsen k_y: harmonic in y, arithmetic in x
    for i in range(nx_coarse):
        for j in range(ny_coarse):
            # Extract block
            block_ky = ky_padded[i * bx : (i + 1) * bx, j * by : (j + 1) * by]

            # First, take harmonic mean along y-direction (axis=1)
            ky_harmonic_y = _axis_harmonic_mean(block_ky, axis=1, epsilon=epsilon)

            # Then, take arithmetic mean along x-direction
            ky_coarse[i, j] = np.nanmean(ky_harmonic_y)

    return kx_coarse, ky_coarse


def _coarsen_3d_permeability_grids(
    kx: ThreeDimensionalGrid,
    ky: ThreeDimensionalGrid,
    kz: ThreeDimensionalGrid,
    batch_size: ThreeDimensions,
    epsilon: float = 1e-10,
) -> typing.Tuple[ThreeDimensionalGrid, ThreeDimensionalGrid, ThreeDimensionalGrid]:
    """
    Coarsen 3D permeability grids using direction-appropriate averaging.

    Uses Cardwell-Parsons averaging extended to 3D:
    - k_x: harmonic mean in x, arithmetic mean in y and z
    - k_y: harmonic mean in y, arithmetic mean in x and z
    - k_z: harmonic mean in z, arithmetic mean in x and y

    This preserves the effective permeability for flow in each direction.

    :param kx: X-direction permeability grid (mD), shape (nx, ny, nz)
    :param ky: Y-direction permeability grid (mD), shape (nx, ny, nz)
    :param kz: Z-direction permeability grid (mD), shape (nx, ny, nz)
    :param batch_size: Coarsening factors (bx, by, bz) for each direction
    :param epsilon: Small value to avoid division by zero (default: 1e-10)
    :return: Tuple of (coarsened_kx, coarsened_ky, coarsened_kz)
    :raises ValidationError: If grids have different shapes or batch_size is invalid

    Example:
    ```python
    # 4x4x4 grid
    kx = np.random.uniform(10, 100, (4, 4, 4))
    ky = np.random.uniform(10, 100, (4, 4, 4))
    kz = np.random.uniform(1, 10, (4, 4, 4))  # Lower vertical perm

    kx_c, ky_c, kz_c = _coarsen_3d_permeability_grids(
        kx, ky, kz,
        batch_size=(2, 2, 2)
    )
    # Result: 2x2x2 coarsened grids
    ```

    References:
        Cardwell, W. T., & Parsons, R. L. (1945). "Average Permeabilities of
        Heterogeneous Oil Sands." Transactions of the AIME, 160(01), 34-42.

        Deutsch, C. V. (1989). "Calculating Effective Absolute Permeability in
        Sandstone/Shale Sequences." SPE Formation Evaluation, 4(03), 343-348.
    """
    if not (kx.shape == ky.shape == kz.shape):
        raise ValidationError(
            f"All permeability grids must have same shape. "
            f"Got kx: {kx.shape}, ky: {ky.shape}, kz: {kz.shape}"
        )

    if kx.ndim != 3:
        raise ValidationError(
            f"Expected 3D grids, got {kx.ndim}D. Use _coarsen_2d_permeability_grids instead."
        )

    if len(batch_size) != 3:
        raise ValidationError(
            f"batch_size must have 3 elements for 3D grids, got {len(batch_size)}"
        )

    bx, by, bz = batch_size
    if bx < 1 or by < 1 or bz < 1:
        raise ValidationError(
            f"All batch_size elements must be >= 1, got ({bx}, {by}, {bz})"
        )

    nx, ny, nz = kx.shape

    # Compute padding
    pad_x = (bx - nx % bx) % bx
    pad_y = (by - ny % by) % by
    pad_z = (bz - nz % bz) % bz

    # Pad grids with NaN
    if pad_x > 0 or pad_y > 0 or pad_z > 0:
        pad_width = ((0, pad_x), (0, pad_y), (0, pad_z))
        kx_padded = np.pad(kx, pad_width, mode="constant", constant_values=np.nan)
        ky_padded = np.pad(ky, pad_width, mode="constant", constant_values=np.nan)
        kz_padded = np.pad(kz, pad_width, mode="constant", constant_values=np.nan)
    else:
        kx_padded = kx
        ky_padded = ky
        kz_padded = kz

    nx_new, ny_new, nz_new = kx_padded.shape
    nx_coarse = nx_new // bx
    ny_coarse = ny_new // by
    nz_coarse = nz_new // bz

    dtype = get_dtype()

    # Initialize output arrays
    kx_coarse = np.zeros((nx_coarse, ny_coarse, nz_coarse), dtype=dtype)
    ky_coarse = np.zeros((nx_coarse, ny_coarse, nz_coarse), dtype=dtype)
    kz_coarse = np.zeros((nx_coarse, ny_coarse, nz_coarse), dtype=dtype)

    # Coarsen k_x: harmonic in x, arithmetic in y and z
    for i in range(nx_coarse):
        for j in range(ny_coarse):
            for k in range(nz_coarse):
                # Extract block
                block_kx = kx_padded[
                    i * bx : (i + 1) * bx, j * by : (j + 1) * by, k * bz : (k + 1) * bz
                ]

                # Harmonic mean in x (axis=0), then arithmetic in y and z
                kx_harmonic_x = _axis_harmonic_mean(block_kx, axis=0, epsilon=epsilon)
                kx_coarse[i, j, k] = np.nanmean(kx_harmonic_x)

    # Coarsen k_y: harmonic in y, arithmetic in x and z
    for i in range(nx_coarse):
        for j in range(ny_coarse):
            for k in range(nz_coarse):
                # Extract block
                block_ky = ky_padded[
                    i * bx : (i + 1) * bx, j * by : (j + 1) * by, k * bz : (k + 1) * bz
                ]

                # Harmonic mean in y (axis=1), then arithmetic in x and z
                ky_harmonic_y = _axis_harmonic_mean(block_ky, axis=1, epsilon=epsilon)
                ky_coarse[i, j, k] = np.nanmean(ky_harmonic_y)

    # Coarsen k_z: harmonic in z, arithmetic in x and y
    for i in range(nx_coarse):
        for j in range(ny_coarse):
            for k in range(nz_coarse):
                # Extract block
                block_kz = kz_padded[
                    i * bx : (i + 1) * bx, j * by : (j + 1) * by, k * bz : (k + 1) * bz
                ]

                # Harmonic mean in z (axis=2), then arithmetic in x and y
                kz_harmonic_z = _axis_harmonic_mean(block_kz, axis=2, epsilon=epsilon)
                kz_coarse[i, j, k] = np.nanmean(kz_harmonic_z)

    return kx_coarse, ky_coarse, kz_coarse


def _axis_harmonic_mean(
    data: npt.NDArray, axis: int, epsilon: float = 1e-10
) -> npt.NDArray:
    """
    Compute harmonic mean along a specific axis, handling NaN values.

    Harmonic mean: H = n / sum(1/x_i) where n = count of non-NaN values

    :param data: Input array
    :param axis: Axis along which to compute harmonic mean
    :param epsilon: Small value added to denominator to avoid division by zero
    :return: Array with harmonic mean computed along specified axis
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        # Compute reciprocals, preserving NaN
        reciprocals = np.where(np.isnan(data), np.nan, 1.0 / (data + epsilon))

        # Count non-NaN values along axis
        counts = np.sum(~np.isnan(data), axis=axis)

        # Sum reciprocals, ignoring NaN
        sum_reciprocals = np.nansum(reciprocals, axis=axis)

        # Harmonic mean = n / sum(1/x)
        result = counts / (sum_reciprocals + epsilon)

        # Handle edge cases
        result = np.where(counts == 0, np.nan, result)
        result = np.where(np.isinf(result), 0.0, result)

    return result


def coarsen_permeability_grids(
    kx: typing.Union[TwoDimensionalGrid, ThreeDimensionalGrid],
    ky: typing.Union[TwoDimensionalGrid, ThreeDimensionalGrid],
    kz: typing.Optional[ThreeDimensionalGrid] = None,
    batch_size: typing.Union[TwoDimensions, ThreeDimensions, None] = None,
    epsilon: float = 1e-10,
) -> typing.Union[
    typing.Tuple[TwoDimensionalGrid, TwoDimensionalGrid],
    typing.Tuple[ThreeDimensionalGrid, ThreeDimensionalGrid, ThreeDimensionalGrid],
]:
    """
    Coarsen permeability grids using direction-appropriate averaging.

    Automatically dispatches to 2D or 3D version based on input dimensions.

    :param kx: X-direction permeability grid (mD)
    :param ky: Y-direction permeability grid (mD)
    :param kz: Z-direction permeability grid (mD), required for 3D
    :param batch_size: Coarsening factors for each direction
    :param epsilon: Small value to avoid division by zero (default: 1e-10)
    :return: Coarsened permeability grids (kx, ky) for 2D or (kx, ky, kz) for 3D

    Example:
    ```python
    # 2D case
    kx, ky = coarsen_permeability_grids(kx_2d, ky_2d, batch_size=(2, 2))

    # 3D case
    kx, ky, kz = coarsen_permeability_grids(kx_3d, ky_3d, kz_3d, batch_size=(2, 2, 2))
    ```
    """
    if batch_size is None:
        raise ValidationError("batch_size must be provided")

    if kx.ndim == 2:
        if kz is not None:
            raise ValidationError("kz should not be provided for 2D grids")
        if len(batch_size) != 2:
            raise ValidationError(
                f"`batch_size` must have 2 elements for 2D grids, got {len(batch_size)}"
            )
        return _coarsen_2d_permeability_grids(
            kx=kx,  # type: ignore[arg-type]
            ky=ky,  # type: ignore[arg-type]
            batch_size=batch_size,  # type: ignore[arg-type]
            epsilon=epsilon,
        )

    elif kx.ndim == 3:
        if kz is None:
            raise ValidationError("kz must be provided for 3D grids")
        if len(batch_size) != 3:
            raise ValidationError(
                f"`batch_size` must have 3 elements for 3D grids, got {len(batch_size)}"
            )
        return _coarsen_3d_permeability_grids(
            kx=kx,  # type: ignore[arg-type]
            ky=ky,  # type: ignore[arg-type]
            kz=kz,
            batch_size=batch_size,  # type: ignore[arg-type]
            epsilon=epsilon,
        )

    raise ValidationError(f"Permeability grids must be 2D or 3D, got {kx.ndim}D")


FlattenStrategy = typing.Union[
    typing.Callable[[npt.NDArray], typing.Union[float, np.floating, npt.NDArray]],
    typing.Literal["max", "min", "mean", "sum", "top", "bottom", "weighted_mean"],
]


def flatten_multilayer_grid_to_surface(
    multilayer_grid: ThreeDimensionalGrid,
    strategy: FlattenStrategy = "max",
    weights: typing.Optional[ThreeDimensionalGrid] = None,
    ignore_nan: bool = True,
) -> TwoDimensionalGrid:
    """
    Flatten a 3D multilayer grid to a 2D surface by collapsing the z-axis (depth).

    This is useful for creating 2D property maps from 3D reservoir grids, such as:
    - Maximum saturation across all layers
    - Average pressure over reservoir thickness
    - Top-of-reservoir property maps

    The z-axis (axis=2) corresponds to depth, where k=0 is the top layer and
    k=nz-1 is the bottom layer.

    :param multilayer_grid: 3D grid with shape (nx, ny, nz) to flatten
    :param strategy: Flattening method to use:
        - "max": Maximum value across layers (useful for saturation, pressure)
        - "min": Minimum value across layers
        - "mean": Arithmetic mean across layers (useful for average properties)
        - "sum": Sum across layers (useful for volumes, totals)
        - "top": Value at top layer (k=0)
        - "bottom": Value at bottom layer (k=nz-1)
        - "weighted_mean": Weighted average (requires `weights` parameter)
        - callable: Custom function that takes 1D array and returns scalar
    :param weights: Optional 3D weight grid (same shape as multilayer_grid).
        Only used when strategy="weighted_mean". Typically layer thickness.
    :param ignore_nan: If True, use NaN-aware operations (nanmax, nanmean, etc.).
        If False, NaN values will propagate to the output. Default: True.
    :return: 2D grid with shape (nx, ny) after flattening
    :raises ValidationError: If input is not 3D, weights shape mismatch, or invalid strategy

    Examples:
    ```python
    # Maximum oil saturation across all layers
    so_max = flatten_multilayer_grid_to_surface(so_grid, strategy="max")

    # Average pressure (thickness-weighted)
    p_avg = flatten_multilayer_grid_to_surface(
        pressure_grid,
        strategy="weighted_mean",
        weights=thickness_grid
    )

    # Top-of-reservoir porosity
    phi_top = flatten_multilayer_grid_to_surface(porosity_grid, strategy="top")

    # Custom strategy: 90th percentile
    p90 = flatten_multilayer_grid_to_surface(
        perm_grid,
        strategy=lambda z: np.nanpercentile(z, 90)
    )
    ```

    Notes:
    - For permeability, consider using directional averaging instead of simple flattening
    - NaN values are handled appropriately if `ignore_nan=True`
    - Weighted mean normalizes weights automatically (no need to pre-normalize)
    """
    if multilayer_grid.ndim != 3:
        raise ValidationError(
            f"`multilayer_grid` must be 3D with shape (nx, ny, nz), got {multilayer_grid.ndim}D"
        )

    nx, ny, nz = multilayer_grid.shape
    dtype = get_dtype()

    # Handle weighted mean separately
    if strategy == "weighted_mean":
        if weights is None:
            raise ValidationError(
                "`weights` parameter is required when `strategy='weighted_mean'`"
            )
        if weights.shape != multilayer_grid.shape:
            raise ValidationError(
                f"`weights` shape {weights.shape} must match `multilayer_grid` shape {multilayer_grid.shape}"
            )

        # Weighted average: sum(w * x) / sum(w)
        if ignore_nan:
            # Handle NaN in both data and weights
            weighted_sum = np.nansum(weights * multilayer_grid, axis=2)
            weight_sum = np.nansum(weights, axis=2)  # type: ignore[arg-type]
        else:
            weighted_sum = np.sum(weights * multilayer_grid, axis=2)
            weight_sum = np.sum(weights, axis=2)  # type: ignore[arg-type]

        # Avoid division by zero
        result = np.divide(
            weighted_sum,
            weight_sum,
            out=np.full((nx, ny), np.nan, dtype=dtype),
            where=(weight_sum != 0),
        )
        return result.astype(dtype)  # type: ignore[return-value]

    if isinstance(strategy, str):
        if strategy == "max":
            func = np.nanmax if ignore_nan else np.max
            return func(multilayer_grid, axis=2).astype(dtype)

        elif strategy == "min":
            func = np.nanmin if ignore_nan else np.min
            return func(multilayer_grid, axis=2).astype(dtype)

        elif strategy == "mean":
            func = np.nanmean if ignore_nan else np.mean
            return func(multilayer_grid, axis=2, dtype=dtype)

        elif strategy == "sum":
            func = np.nansum if ignore_nan else np.sum
            return func(multilayer_grid, axis=2, dtype=dtype)

        elif strategy == "top":
            # k=0 is top layer
            return multilayer_grid[:, :, 0].astype(dtype)  # type: ignore[return-value]

        elif strategy == "bottom":
            # k=nz-1 is bottom layer
            return multilayer_grid[:, :, -1].astype(dtype)  # type: ignore[return-value]

        else:
            raise ValidationError(
                f"Unknown strategy '{strategy}'. Valid options: "
                "'max', 'min', 'mean', 'sum', 'top', 'bottom', 'weighted_mean'"
            )

    elif callable(strategy):
        # Check if function is vectorized (much faster)
        try:
            # Test with a small slice
            test_slice = multilayer_grid[0, 0, :]
            result_scalar = strategy(test_slice)

            # Check if result is scalar
            if not np.isscalar(result_scalar):
                raise ValidationError(
                    f"Custom strategy function must return a scalar, got {type(result_scalar)}"
                )

            # Try vectorized approach first
            # Reshape to (nx*ny, nz) for efficient processing
            reshaped = multilayer_grid.reshape(-1, nz)

            # Check if function works on 2D array (vectorized)
            try:
                # Some numpy functions can handle 2D input
                result_flat = strategy(reshaped)
                if result_flat.shape == (nx * ny,):  # type: ignore[union-attr]
                    return result_flat.reshape(nx, ny).astype(dtype)  # type: ignore[union-attr]
            except (ValueError, TypeError):
                pass  # Fall back to `apply_along_axis`

            # Fall back to slower `apply_along_axis`
            warnings.warn(
                "Using `apply_along_axis` for custom strategy. "
                "For better performance, use vectorized numpy functions or built-in strategies.",
                UserWarning,
                stacklevel=2,
            )
            result = np.apply_along_axis(strategy, axis=2, arr=multilayer_grid)
            return result.astype(dtype)  # type: ignore[return-value]

        except Exception as exc:
            raise ValidationError(f"Custom strategy function failed: {exc}") from exc

    raise ValidationError(
        f"`strategy` must be a string or callable, got {type(strategy)}"
    )


def flatten_multilayer_grids(
    grids: typing.Dict[str, ThreeDimensionalGrid],
    strategy: typing.Union[FlattenStrategy, typing.Dict[str, FlattenStrategy]] = "max",
    weights: typing.Optional[typing.Dict[str, ThreeDimensionalGrid]] = None,
    ignore_nan: bool = True,
) -> typing.Dict[str, TwoDimensionalGrid]:
    """
    Flatten multiple 3D grids to 2D surfaces using specified strategies.

    Convenient wrapper for flattening multiple related grids (e.g., all saturations,
    all pressures) with a single call.

    :param grids: Dictionary of {name: 3D_grid} to flatten
    :param strategy: Single strategy for all grids, or dict of {name: strategy}
    :param weights: Optional dict of {name: weight_grid} for weighted averaging
    :param ignore_nan: If True, use NaN-aware operations
    :return: Dictionary of {name: 2D_grid} after flattening

    Example:
    ```python
    grids_3d = {
        'oil_saturation': so_grid,
        'water_saturation': sw_grid,
        'pressure': p_grid,
    }

    # Use same strategy for all
    grids_2d = flatten_multilayer_grids(grids_3d, strategy="max")

    # Use different strategies per grid
    strategies = {
        'oil_saturation': 'max',
        'water_saturation': 'mean',
        'pressure': 'weighted_mean',
    }
    weights_dict = {
        'pressure': thickness_grid,
    }
    grids_2d = flatten_multilayer_grids(
        grids_3d,
        strategy=strategies,
        weights=weights_dict
    )
    ```
    """
    result = {}
    for name, grid in grids.items():
        if isinstance(strategy, dict):
            grid_strategy = strategy.get(name, "max")  # type: ignore
        else:
            grid_strategy = strategy

        grid_weights = None
        if weights is not None and name in weights:
            grid_weights = weights[name]

        result[name] = flatten_multilayer_grid_to_surface(
            grid,
            strategy=grid_strategy,
            weights=grid_weights,
            ignore_nan=ignore_nan,
        )
    return result


_ORIENTATION_TO_AXIS: typing.Dict[Orientation, int] = {
    Orientation.X: 0,
    Orientation.Y: 1,
    Orientation.Z: 2,
}


def _resolve_axis(
    orientation: typing.Union[Orientation, typing.Literal["x", "y", "z"]],
    ndim: int,
) -> int:
    """Resolve an `Orientation` (or string) to a numpy axis index."""
    if isinstance(orientation, str):
        orientation = Orientation(orientation)
    axis = _ORIENTATION_TO_AXIS.get(orientation)
    if axis is None:
        raise ValueError(f"`orientation` must be one of X, Y, Z, got {orientation!r}.")
    if axis >= ndim:
        raise ValueError(
            f"`orientation` {orientation!r} maps to axis {axis}, but the array "
            f"only has {ndim} dimension(s)."
        )
    return axis


def layer_to_link_permeability(
    cell_permeability: npt.NDArray,
    cell_lenghts: npt.NDArray,
    orientation: typing.Union[
        Orientation, typing.Literal["x", "y", "z"]
    ] = Orientation.Z,
) -> npt.NDArray:
    """
    Convert per-cell permeability to inter-cell interface (link) permeability.

    Works for any flow direction. Pass kx with dx and `Orientation.X`, ky with
    dy and `Orientation.Y`, or kz with dz and `Orientation.Z`.

    For each pair of adjacent cells `i` and `i+1` along `orientation`, the
    dimension-weighted harmonic mean link permeability is:

    ```
    k_link[i] = (d[i] + d[i+1]) / (d[i]/k[i] + d[i+1]/k[i+1])
    ```

    where `d[i]` is the cell dimension in the direction of flow (ft).

    The input grid can have any number of dimensions. All axes other than the
    one selected by `orientation` are treated as independent spatial locations
    and are passed through unchanged. The output has the same shape as the
    input except that the selected axis is reduced by one (n_cells → n_cells - 1).

    :param cell_permeability: Array of per-cell permeability values (mD).
        The size along the axis selected by `orientation` is `n_cells`.
        All values must be strictly positive.

        Accepted shapes (with `orientation=Orientation.Z`):

        - `(n_cells,)` - 1-D, one value per cell
        - `(nx, n_cells)` - 2-D, spatially varying in x
        - `(nx, ny, n_cells)` - 3-D full grid

    :param cell_lenghts: 1-D array of cell lengths/thicknesses in the flow direction
        (ft), length `n_cells`. Pass dx for `Orientation.X`, dy for
        `Orientation.Y`, dz for `Orientation.Z`. Must be strictly positive
        and match the size of `cell_permeability` along the selected axis.
    :param orientation: Flow direction. Accepts `Orientation.X/Y/Z` or the
        string literals `"x"`, `"y"`, `"z"`. Defaults to `Orientation.Z`.
        Follows the same convention as `build_layered_grid`:
        X → axis 0, Y → axis 1, Z → axis 2.
    :return: Array of interface link permeability values (mD) with the same
        shape as `cell_permeability` except the selected axis is reduced from
        `n_cells` to `n_cells - 1`.

    Example:

    ```python
    import numpy as np
    from bores.types import Orientation
    from bores.grids.utils import layer_to_link_permeability

    # Vertical (z) direction - SPE1 layers
    kz = np.array([250.0, 25.0, 50.0])   # mD
    dz = np.array([ 20.0, 30.0, 50.0])   # ft
    kz_links = layer_to_link_permeability(kz, dz, Orientation.Z)
    # array([39.0625, 36.3636])

    # Horizontal (x) direction
    kx = np.array([500.0, 50.0, 200.0])   # mD
    dx = np.array([1000.0, 1000.0, 1000.0])   # ft
    kx_links = layer_to_link_permeability(kx, dx, Orientation.X)
    # array([90.9091, 80.0])

    # 3-D grid - kz varies along last axis
    kz_grid = np.broadcast_to(kz, (10, 10, 3)).copy()
    kz_links_grid = layer_to_link_permeability(kz_grid, dz, Orientation.Z)
    # shape (10, 10, 2)
    ```
    """
    cell_permeability = np.asarray(cell_permeability, dtype=float)
    cell_lenghts = np.asarray(cell_lenghts, dtype=float)

    if cell_lenghts.ndim != 1:
        raise ValueError("`cell_lenghts` must be a 1-D array.")

    axis = _resolve_axis(orientation, cell_permeability.ndim)
    n_cells = cell_permeability.shape[axis]

    if n_cells != len(cell_lenghts):
        raise ValueError(
            f"`cell_permeability` has {n_cells} cells along {orientation!r} (axis {axis}) "
            f"but `cell_lenghts` has length {len(cell_lenghts)}."
        )
    if n_cells < 2:
        raise ValueError(
            f"Need at least 2 cells to compute interface links, got {n_cells}."
        )
    if np.any(cell_permeability <= 0):
        raise ValueError("All `cell_permeability` values must be strictly positive.")
    if np.any(cell_lenghts <= 0):
        raise ValueError("All `cell_lenghts` values must be strictly positive.")

    # Reshape `cell_lenghts` so it broadcasts against `cell_permeability` along axis.
    broadcast_shape = [1] * cell_permeability.ndim
    broadcast_shape[axis] = n_cells
    d_broadcast = cell_lenghts.reshape(broadcast_shape)

    upper_slice = [slice(None)] * cell_permeability.ndim
    lower_slice = [slice(None)] * cell_permeability.ndim
    upper_slice[axis] = slice(None, -1)  # cells 0 … n-2
    lower_slice[axis] = slice(1, None)  # cells 1 … n-1

    d_upper = d_broadcast[tuple(upper_slice)]
    d_lower = d_broadcast[tuple(lower_slice)]
    k_upper = cell_permeability[tuple(upper_slice)]
    k_lower = cell_permeability[tuple(lower_slice)]

    return (d_upper + d_lower) / (d_upper / k_upper + d_lower / k_lower)


def link_to_layer_permeability(
    interface_permeability: npt.NDArray,
    cell_lenghts: npt.NDArray,
    anchor_permeability: typing.Union[float, npt.NDArray],
    anchor_index: int = 0,
    orientation: typing.Union[
        Orientation, typing.Literal["x", "y", "z"]
    ] = Orientation.Z,
) -> npt.NDArray:
    """
    Recover per-cell permeability from inter-cell interface (link) permeability.

    Works for any flow direction. Pass kx links with dx and `Orientation.X`,
    ky links with dy and `Orientation.Y`, or kz links with dz and `Orientation.Z`.
    The inverse of `layer_to_link_permeability`.

    Because there are `n_cells - 1` equations for `n_cells` unknowns the
    system is underdetermined. You must fix one cell's permeability as an
    anchor. The remaining values are recovered by propagating the
    harmonic-mean constraint forward and backward from that anchor:

    ```
    # forward  (recovering cell i+1 from cell i)
    k[i+1] = d[i+1] / ((d[i] + d[i+1]) / k_link[i] - d[i] / k[i])

    # backward (recovering cell i from cell i+1)
    k[i] = d[i] / ((d[i] + d[i+1]) / k_link[i] - d[i+1] / k[i+1])
    ```

    The output has the same shape as `interface_permeability` except that the
    axis selected by `orientation` grows by one (n_cells - 1 → n_cells).

    :param interface_permeability: Array of interface link permeability values
        (mD).  The size along the axis selected by `orientation` is
        `n_cells - 1`. All values must be strictly positive.
    :param cell_lenghts: 1-D array of cell lengths/thicknesses in the flow direction
        (ft), length `n_cells`. Pass dx for `Orientation.X`, dy for
        `Orientation.Y`, dz for `Orientation.Z`. Must be strictly positive.
    :param anchor_permeability: Known permeability (mD) of the anchor cell.
        Can be a scalar (same value everywhere) or an array whose shape matches
        a slice of `interface_permeability` with the orientation axis removed
        (one value per spatial location).
    :param anchor_index: Index of the cell whose permeability is known along
        the orientation axis. Defaults to `0` (first cell).
    :param orientation: Flow direction.  Accepts `Orientation.X/Y/Z` or the
        string literals `"x"`, `"y"`, `"z"`. Defaults to `Orientation.Z`.
    :return: Array of per-cell permeability values (mD) with the same shape
        as `interface_permeability` except the selected axis grows from
        `n_cells - 1` to `n_cells`.

    Example:

    ```python
    import numpy as np
    from bores.types import Orientation
    from bores.grids.utils import link_to_layer_permeability

    # 1-D round-trip - vertical (z) direction
    dz = np.array([20.0, 30.0, 50.0])
    kz_links = np.array([39.0625, 36.3636])
    kz = link_to_layer_permeability(
        kz_links, dz, anchor_permeability=250.0, orientation=Orientation.Z
    )
    # array([250.,  25.,  50.])

    # 3-D: recover a (10, 10, 3) kz grid from a (10, 10, 2) link grid
    links_grid = np.broadcast_to(kz_links, (10, 10, 2)).copy()
    kz_grid = link_to_layer_permeability(
        links_grid, dz, anchor_permeability=250.0, orientation=Orientation.Z
    )
    # shape (10, 10, 3)
    ```
    """
    interface_permeability = np.asarray(interface_permeability, dtype=float)
    cell_lenghts = np.asarray(cell_lenghts, dtype=float)

    if cell_lenghts.ndim != 1:
        raise ValueError("`cell_lenghts` must be a 1-D array.")

    n_cells = len(cell_lenghts)
    n_interfaces = n_cells - 1
    axis = _resolve_axis(orientation, interface_permeability.ndim)

    if interface_permeability.shape[axis] != n_interfaces:
        raise ValueError(
            f"`interface_permeability` has {interface_permeability.shape[axis]} values "
            f"along {orientation!r} (axis {axis}) but `cell_lenghts` implies "
            f"{n_interfaces} interfaces ({n_cells} cells)."
        )
    if not (0 <= anchor_index < n_cells):
        raise ValueError(
            f"`anchor_index` must be in [0, {n_cells - 1}], got {anchor_index}."
        )
    if np.any(np.asarray(anchor_permeability) <= 0):
        raise ValueError("`anchor_permeability` must be strictly positive.")
    if np.any(cell_lenghts <= 0):
        raise ValueError("All `cell_lenghts` values must be strictly positive.")
    if np.any(interface_permeability <= 0):
        raise ValueError(
            "All `interface_permeability` values must be strictly positive."
        )

    output_shape = list(interface_permeability.shape)
    output_shape[axis] = n_cells
    recovered = np.empty(output_shape, dtype=float)

    def _get(array: npt.NDArray, index: int) -> npt.NDArray:
        idx = [slice(None)] * array.ndim
        idx[axis] = index  # type: ignore
        return array[tuple(idx)]

    def _set(array: npt.NDArray, index: int, values: npt.NDArray) -> None:
        idx = [slice(None)] * array.ndim
        idx[axis] = index  # type: ignore
        array[tuple(idx)] = values

    # Seed anchor
    _set(
        recovered,
        anchor_index,
        np.broadcast_to(
            anchor_permeability, _get(recovered, anchor_index).shape
        ).copy(),
    )

    # Propagate forward: anchor_index → last cell
    for i in range(anchor_index, n_cells - 1):
        d_upper = cell_lenghts[i]
        d_lower = cell_lenghts[i + 1]
        link = _get(interface_permeability, i)
        k_upper = _get(recovered, i)
        total_resistance = (d_upper + d_lower) / link
        lower_resistance = total_resistance - d_upper / k_upper

        if np.any(lower_resistance <= 0):
            raise ValueError(
                f"Cannot recover permeability at cell {i + 1}: interface {i} implies "
                f"a non-positive or infinite value at one or more spatial locations. "
                f"Verify that `anchor_permeability` at cell {anchor_index} is "
                f"consistent with the provided link values."
            )
        _set(recovered, i + 1, d_lower / lower_resistance)

    # Propagate backward: anchor_index → first cell
    for i in range(anchor_index - 1, -1, -1):
        d_upper = cell_lenghts[i]
        d_lower = cell_lenghts[i + 1]
        link = _get(interface_permeability, i)
        k_lower = _get(recovered, i + 1)
        total_resistance = (d_upper + d_lower) / link
        upper_resistance = total_resistance - d_lower / k_lower

        if np.any(upper_resistance <= 0):
            raise ValueError(
                f"Cannot recover permeability at cell {i}: interface {i} implies "
                f"a non-positive or infinite value at one or more spatial locations. "
                f"Verify that `anchor_permeability` at cell {anchor_index} is "
                f"consistent with the provided link values."
            )
        _set(recovered, i, d_upper / upper_resistance)

    return recovered


@numba.njit(cache=True)
def make_saturation_grid(
    n_points: int = 200,
    s_min: float = 0.0,
    s_max: float = 1.0,
    spacing: Spacing = "cosine",
    dtype: npt.DTypeLike = np.float64,
) -> npt.NDArray:
    """
    Build a 1-D saturation grid over `[s_min, s_max]`.

    :param n_points: Number of grid points (≥ 2).
    :param s_min: Physical saturation range. Must satisfy `0 ≤ s_min < s_max ≤ 1`.
    :param s_max: Physical saturation range. Must satisfy `0 ≤ s_min < s_max ≤ 1`.
    :param spacing: `"cosine"` (default) — Chebyshev-cosine spacing, denser at the
        endpoints. `"linspace"` — uniform spacing.
    :return: NDArray of shape `(n_points,)` with values in `[s_min, s_max]`,
    monotonically increasing.
    """
    if n_points < 2:
        raise ValueError(f"`n_points` must be ≥ 2, got {n_points}")
    if not (0.0 <= s_min < s_max <= 1.0):
        raise ValueError(
            f"Require 0 ≤ s_min < s_max ≤ 1, got `s_min={s_min}`, `s_max={s_max}`"
        )

    if spacing == "cosine":
        i = np.arange(n_points, dtype=dtype)
        unit: npt.NDArray = 0.5 * (1.0 - np.cos(np.pi * i / (n_points - 1)))
    elif spacing == "linspace":
        unit: npt.NDArray = np.linspace(0.0, 1.0, n_points)
    else:
        raise ValueError(f"`spacing` must be 'cosine' or 'linspace', got '{spacing}'")

    return (s_min + unit * (s_max - s_min)).astype(dtype)

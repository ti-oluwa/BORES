"""
Shared utilities for visualization modules.
"""

import itertools
import logging
import os
import typing
from pathlib import Path

import attrs
import imageio
import numpy as np
import plotly.graph_objects as go
from typing_extensions import TypedDict

from bores._precision import get_dtype
from bores.errors import ValidationError
from bores.grids.utils import coarsen_grid
from bores.models import ReservoirModel
from bores.states import ModelState
from bores.types import (
    NDimension,
    NDimensionalGrid,
    OneDimensionalGrid,
    ThreeDimensionalGrid,
    ThreeDimensions,
)
from bores.visualization.base import PropertyMeta

logger = logging.getLogger(__name__)

__all__ = [
    "GifExporter",
    "HtmlExporter",
    "Label",
    "LabelCoordinate",
    "Labels",
    "Mp4Exporter",
    "WebPExporter",
]


def _format_value(value: float, metadata: PropertyMeta) -> str:
    """
    Format a value for display with appropriate precision and scientific notation.

    :param value: The numeric value to format
    :param metadata: Property metadata for context
    :return: Formatted string representation
    """
    if np.isnan(value) or np.isinf(value):
        return "N/A"

    # Convert to absolute value to check decimal places
    abs_val = abs(value)

    # If value is very small or would have more than 6 decimal places, use scientific notation
    if abs_val == 0:
        return "0.000"
    elif (
        abs_val < 1e-6
        or (abs_val < 1 and len(f"{abs_val:.10f}".rstrip("0").split(".")[1]) > 6)
        or abs_val >= 1e6
    ):
        return f"{value:.4e}"
    elif abs_val >= 1000:
        return f"{value:.1f}"
    elif abs_val >= 1:
        return f"{value:.3f}"
    # For values between 0 and 1, show up to 6 decimal places
    formatted = f"{value:.6f}".rstrip("0").rstrip(".")
    return formatted if formatted else "0"


def _invert_z_axis(arr: np.typing.NDArray) -> np.typing.NDArray:
    """
    Invert the Z axis (last axis) of a 3D array so that data[:,:,0] becomes data[:,:,nz-1].
    This ensures numpy convention (top layer as k=0) matches plotly's rendering (bottom as k=0).
    """
    if arr.ndim == 3:
        return arr[:, :, ::-1]
    return arr


@attrs.frozen
class LabelCoordinate:
    """Represents a 3D position for placing labels."""

    x: int
    y: int
    z: int

    def as_physical(
        self,
        cell_dimension: typing.Tuple[float, float],
        depth_grid: ThreeDimensionalGrid,
        coordinate_offsets: typing.Optional[typing.Tuple[int, int, int]] = None,
    ) -> typing.Tuple[float, float, float]:
        """
        Convert index coordinates to physical coordinates.

        :param cell_dimension: Physical size of each cell in x and y directions (feet)
        :param depth_grid: 3D array with depth of each cell center (feet, positive downward)
        :param coordinate_offsets: Optional cell index offsets to apply to the physical coordinates
        :return: Tuple of (x_physical, y_physical, z_physical) coordinates
        """
        offsets = coordinate_offsets or (0, 0, 0)
        dx, dy = cell_dimension

        # Calculate actual grid indices
        actual_x = offsets[0] + self.x
        actual_y = offsets[1] + self.y
        actual_z = offsets[2] + self.z

        # Convert to physical coordinates
        x_physical = actual_x * dx
        y_physical = actual_y * dy

        if (
            actual_x < depth_grid.shape[0]
            and actual_y < depth_grid.shape[1]
            and actual_z < depth_grid.shape[2]
        ):
            # Use depth grid directly - negate because depth is positive downward
            z_physical = -depth_grid[actual_x, actual_y, actual_z]
        else:
            # Fallback to simple index-based positioning
            z_physical = (
                -(offsets[2] + self.z) * 10.0
            )  # Assume 10 ft average depth per layer

        return x_physical, y_physical, typing.cast(float, z_physical)


class _SafeValuesDict(dict):
    """A dictionary that safely handles missing keys by returning a formatted string."""

    def __missing__(self, key) -> str:
        return f"{{{key}}}"


class LabelFormatValues(TypedDict):
    """Format values for label text generation."""

    x_index: typing.Union[int, float]
    y_index: typing.Union[int, float]
    z_index: typing.Union[int, float]
    x_physical: typing.Optional[float]
    y_physical: typing.Optional[float]
    z_physical: typing.Optional[float]
    value: typing.Union[int, float, str]
    formatted_value: typing.Optional[str]
    property_name: typing.Optional[str]
    unit: typing.Optional[str]


@attrs.frozen
class Label:
    """
    A flexible label that can be positioned in 3D space on a 3D grid and extract data dynamically.
    """

    position: LabelCoordinate
    text_template: str = """
    {name}
    <br>
    Cell Coordinates: ({x_index}, {y_index}, {z_index})
    <br>
    Physical Coordinates: ({x_physical:.2f}, {y_physical:.2f}, {z_physical:.2f}) ft
    <br>
    Value: {value}
    <br>
    Formatted Value: {formatted_value}
    <br>
    Property: {property_name}
    <br>
    Unit: {unit}
    <br>
    """

    font_size: int = 12
    font_color: str = "#333333"  # Default dark gray
    background_color: typing.Optional[str] = "rgba(240, 240, 240, 0.9)"
    border_color: typing.Optional[str] = None
    border_width: int = 1
    offset: typing.Tuple[float, float, float] = (0, 0, 0)
    visible: bool = True
    name: typing.Optional[str] = None

    def get_text(
        self,
        data_grid: typing.Optional[ThreeDimensionalGrid] = None,
        cell_dimension: typing.Optional[typing.Tuple[float, float]] = None,
        depth_grid: typing.Optional[ThreeDimensionalGrid] = None,
        metadata: typing.Optional[PropertyMeta] = None,
        format_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
    ) -> str:
        """
        Generate the label text based on data at the label position.

        :param data_grid: 3D data array to extract values from
        :param cell_dimension: Physical size of each cell in x and y directions (feet)
        :param depth_grid: 3D array with depth of each cell center (feet, positive downward)
        :param metadata: Property metadata for formatting
        :param format_kwargs: Additional values for string formatting
        :return: Formatted label text
        """
        values = {
            "x_index": self.position.x,
            "y_index": self.position.y,
            "z_index": self.position.z,
            "x_physical": None,
            "y_physical": None,
            "z_physical": None,
            "value": "N/A",
            "formatted_value": "N/A",
            "property_name": "N/A",
            "unit": "N/A",
        }

        raw_value = None
        formatted_value = None
        x_index = int(self.position.x)
        y_index = int(self.position.y)
        z_index = int(self.position.z)

        # Extract data value if grid is provided
        if data_grid is not None:
            raw_value = data_grid[x_index, y_index, z_index]

        if cell_dimension is not None and depth_grid is not None:
            x_physical, y_physical, z_physical = self.position.as_physical(
                cell_dimension=cell_dimension,
                depth_grid=depth_grid,
                coordinate_offsets=None,  # Don't pass offset here, handle separately
            )
            # Apply the label's offset to the calculated physical coordinates
            x_physical += self.offset[0]
            y_physical += self.offset[1]
            z_physical += self.offset[2]

            values["x_physical"] = x_physical  # type: ignore
            values["y_physical"] = y_physical  # type: ignore
            values["z_physical"] = z_physical  # type: ignore

        if metadata is not None:
            if raw_value is not None:
                formatted_value = _format_value(raw_value, metadata)
            values["unit"] = metadata.unit
            values["property_name"] = metadata.display_name

        values["value"] = raw_value if raw_value is not None else "N/A"
        values["formatted_value"] = (
            formatted_value if formatted_value is not None else "N/A"
        )

        # Add any additional format kwargs
        if format_kwargs:
            values.update(format_kwargs)
        return self.text_template.format(**_SafeValuesDict(values))

    def as_annotation(
        self,
        data_grid: typing.Optional[ThreeDimensionalGrid] = None,
        cell_dimension: typing.Optional[typing.Tuple[float, float]] = None,
        depth_grid: typing.Optional[ThreeDimensionalGrid] = None,
        metadata: typing.Optional[PropertyMeta] = None,
        format_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
        coordinate_offsets: typing.Optional[typing.Tuple[int, int, int]] = None,
    ) -> typing.Dict[str, typing.Any]:
        """
        Convert label to Plotly 3D annotation format.

        :param data_grid: 3D data array for value extraction
        :param cell_dimension: Physical size of each cell in x and y directions (feet)
        :param depth_grid: 3D array with depth of each cell center (feet, positive downward)
        :param metadata: Property metadata for formatting
        :param format_kwargs: Additional formatting values
        :param coordinate_offsets: Coordinate offsets for sliced data
        :return: Plotly annotation dictionary
        """
        if not self.visible:
            return {}

        # Determine if we should use physical coordinates
        use_physical = cell_dimension is not None and depth_grid is not None

        if use_physical:
            cell_dimension = typing.cast(typing.Tuple[float, float], cell_dimension)
            depth_grid = typing.cast(ThreeDimensionalGrid, depth_grid)
            # Convert to physical coordinates
            try:
                x_physical, y_physical, z_physical = self.position.as_physical(
                    cell_dimension, depth_grid, coordinate_offsets
                )
                # Apply offset in physical space
                x_position = x_physical + self.offset[0]
                y_position = y_physical + self.offset[1]
                z_position = z_physical + self.offset[2]

                # Log physical coordinates for debugging
                logger.debug(
                    f"Label at index ({self.position.x}, {self.position.y}, {self.position.z}) "
                    f"-> physical ({x_position:.2f}, {y_position:.2f}, {z_position:.2f})"
                )

            except Exception as exc:
                logger.warning(
                    f"Failed to convert to physical coordinates: {exc}. Using index coordinates."
                )
                use_physical = False

        if not use_physical:
            # Use index coordinates with offsets
            offsets = coordinate_offsets or (0, 0, 0)
            x_position = self.position.x + offsets[0] + self.offset[0]
            y_position = self.position.y + offsets[1] + self.offset[1]
            z_position = self.position.z + offsets[2] + self.offset[2]

            logger.debug(
                f"Label using index coordinates: ({x_position}, {y_position}, {z_position})"
            )

        annotation = {
            "x": float(x_position),  # type: ignore
            "y": float(y_position),  # type: ignore
            "z": float(z_position),  # type: ignore
            "text": self.get_text(
                data_grid=data_grid,
                cell_dimension=cell_dimension,
                depth_grid=depth_grid,
                metadata=metadata,
                format_kwargs=format_kwargs,
            ),
            "showarrow": True,
            "font": {"size": self.font_size, "color": self.font_color},
        }

        # Add optional styling only if values are provided
        if self.background_color:
            annotation["bgcolor"] = self.background_color
        if self.border_color:
            annotation["bordercolor"] = self.border_color
        if self.border_width > 0:
            annotation["borderwidth"] = self.border_width

        logger.debug(f"Created annotation: {annotation}")
        return annotation


class Labels:
    """
    A collection of labels for 3D plots, allowing dynamic label creation and management.
    """

    def __init__(self, labels: typing.Optional[typing.Iterable[Label]] = None):
        self._labels = list(labels) if labels is not None else []

    def add(self, label: Label) -> None:
        """Add a label to the collection."""
        self._labels.append(label)

    def add_grid_labels(
        self,
        data_shape: typing.Tuple[int, int, int],
        spacing: typing.Tuple[int, int, int] = (10, 10, 5),
        template: str = "({x_index}, {y_index}, {z_index})",
        **label_kwargs,
    ) -> None:
        """
        Add grid labels at regular intervals.

        :param data_shape: Shape of the 3D data grid
        :param spacing: Spacing between labels in each dimension
        :param template: Text template for labels
        :param label_kwargs: Additional Label constructor arguments
        """
        nx, ny, nz = data_shape
        x_spacing, y_spacing, z_spacing = spacing

        for x, y, z in itertools.product(
            range(0, nx, x_spacing), range(0, ny, y_spacing), range(0, nz, z_spacing)
        ):
            position = LabelCoordinate(x, y, z)
            label = Label(position=position, text_template=template, **label_kwargs)
            self.add(label)

    def add_boundary_labels(
        self,
        data_shape: typing.Tuple[int, int, int],
        template: str = "Boundary ({x_index}, {y_index}, {z_index})",
        **label_kwargs,
    ) -> None:
        """
        Add labels at the boundaries of the data grid.

        :param data_shape: Shape of the 3D data grid
        :param template: Text template for boundary labels
        :param label_kwargs: Additional Label constructor arguments
        """
        nx, ny, nz = data_shape

        # Corner positions
        corners = [
            (0, 0, 0),
            (nx - 1, 0, 0),
            (0, ny - 1, 0),
            (0, 0, nz - 1),
            (nx - 1, ny - 1, 0),
            (nx - 1, 0, nz - 1),
            (0, ny - 1, nz - 1),
            (nx - 1, ny - 1, nz - 1),
        ]

        for x, y, z in corners:
            position = LabelCoordinate(x, y, z)
            label = Label(
                position=position,
                text_template=template,
                name=f"corner_{x}_{y}_{z}",
                **label_kwargs,
            )
            self.add(label)

    def add_well_labels(
        self,
        well_positions: typing.List[typing.Tuple[int, int, int]],
        well_names: typing.Optional[typing.List[str]] = None,
        template: str = "Well - '{name}' ({x_index}, {y_index}, {z_index}): {formatted_value} ({unit})",
        **label_kwargs,
    ) -> None:
        """
        Add labels for well positions.

        :param well_positions: List of (x, y, z) well positions
        :param well_names: Optional well names (defaults to Well_1, Well_2, etc.)
        :param template: Text template for well labels
        :param label_kwargs: Additional Label constructor arguments
        """
        for i, (x, y, z) in enumerate(well_positions):
            name = (
                well_names[i] if well_names and i < len(well_names) else f"Well_{i + 1}"
            )
            position = LabelCoordinate(x, y, z)
            label_kwargs.setdefault("font_size", 12)
            label_kwargs.setdefault("font_color", "#333")
            label = Label(
                position=position,
                text_template=template,
                name=name,
                **label_kwargs,
            )
            self.add(label)

    def visible(self) -> typing.Generator[Label, None, None]:
        """Return only visible labels."""
        return (label for label in self._labels if label.visible)

    def as_annotations(
        self,
        data_grid: typing.Optional[ThreeDimensionalGrid] = None,
        metadata: typing.Optional[PropertyMeta] = None,
        cell_dimension: typing.Optional[typing.Tuple[float, float]] = None,
        depth_grid: typing.Optional[ThreeDimensionalGrid] = None,
        coordinate_offsets: typing.Optional[typing.Tuple[int, int, int]] = None,
        format_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
    ) -> typing.List[typing.Dict[str, typing.Any]]:
        """
        Convert all visible labels to Plotly annotations.

        :param data_grid: 3D data array for value extraction
        :param metadata: Property metadata for formatting
        :param cell_dimension: Physical cell dimensions for coordinate conversion
        :param depth_grid: Depth grid for physical coordinate conversion (feet, positive downward)
        :param coordinate_offsets: Coordinate offsets for sliced data
        :param format_kwargs: Additional formatting values
        :return: List of Plotly annotation dictionaries
        """
        annotations = []

        for label in self.visible():
            try:
                # Check if label position is within the sliced data bounds
                if coordinate_offsets is not None and data_grid is not None:  # noqa
                    # Check if label falls within the current slice bounds
                    # Label position should be relative to the sliced data (0-indexed)
                    if (
                        label.position.x < 0
                        or label.position.x >= data_grid.shape[0]
                        or label.position.y < 0
                        or label.position.y >= data_grid.shape[1]
                        or label.position.z < 0
                        or label.position.z >= data_grid.shape[2]
                    ):
                        # Label is outside the sliced data bounds, skip it
                        continue

                # Add custom values for well names, etc.
                kwargs = {"name": label.name} if label.name else {}
                kwargs.update(format_kwargs or {})
                annotation = label.as_annotation(
                    data_grid=data_grid,
                    cell_dimension=cell_dimension,
                    depth_grid=depth_grid,
                    metadata=metadata,
                    format_kwargs=kwargs,
                    coordinate_offsets=coordinate_offsets,
                )

                if annotation:
                    annotations.append(annotation)

            except Exception as exc:
                logger.warning(
                    f"Failed to create annotation for label {label.name}: {exc}"
                )
                continue
        return annotations

    def clear(self) -> None:
        """Clear all labels."""
        self._labels.clear()

    def __len__(self) -> int:
        """Get the number of labels."""
        return len(self._labels)

    def __iter__(self) -> typing.Iterator[Label]:
        """Iterate over all labels."""
        return iter(self._labels)

    def __getitem__(self, index: int) -> Label:
        """Get a label by index."""
        return self._labels[index]


@typing.overload
def coarsen_grid_and_coords(
    data: NDimensionalGrid[NDimension],
    x_coords: OneDimensionalGrid,
    y_coords: OneDimensionalGrid,
    z_coords: None,
    batch_size: typing.Optional[NDimension],
    method: typing.Literal["mean", "sum", "max", "min"],
) -> typing.Tuple[
    NDimensionalGrid[NDimension], OneDimensionalGrid, OneDimensionalGrid, None
]: ...


@typing.overload
def coarsen_grid_and_coords(
    data: NDimensionalGrid[NDimension],
    x_coords: OneDimensionalGrid,
    y_coords: OneDimensionalGrid,
    z_coords: NDimensionalGrid[NDimension],
    batch_size: typing.Optional[NDimension],
    method: typing.Literal["mean", "sum", "max", "min"],
) -> typing.Tuple[
    NDimensionalGrid[NDimension],
    OneDimensionalGrid,
    OneDimensionalGrid,
    NDimensionalGrid[NDimension],
]: ...


def coarsen_grid_and_coords(
    data: NDimensionalGrid[NDimension],
    x_coords: OneDimensionalGrid,
    y_coords: OneDimensionalGrid,
    z_coords: typing.Optional[NDimensionalGrid[NDimension]] = None,
    batch_size: typing.Optional[NDimension] = None,
    method: typing.Literal["mean", "sum", "max", "min"] = "mean",
) -> typing.Tuple[
    NDimensionalGrid[NDimension],
    OneDimensionalGrid,
    OneDimensionalGrid,
    typing.Optional[NDimensionalGrid[NDimension]],
]:
    """
    Coarsen a 2D or 3D grid and compute coarsened coordinates using padding instead of trimming.
    Cell coordinates represent **cell boundaries**.

    :param data: 2D or 3D numpy array to coarsen.
    :param x_coords: 1D array of x-axis cell boundaries.
    :param y_coords: 1D array of y-axis cell boundaries.
    :param z_coords: 1D or 3D array of z-axis cell boundaries for 3D grid. Required if data.ndim==3.
    :param batch_size: Coarsening factor per dimension.
    :param method: Aggregation method for data blocks.
    :return: Tuple (coarsened_data, x_batch, y_batch, z_batch)
    """
    if batch_size is None:
        batch_size = typing.cast(NDimension, (2,) * data.ndim)

    # Pad data to be divisible by `batch_size`
    pad_width = []
    for dim, b in zip(data.shape, batch_size):
        remainder = dim % b
        pad_width.append((0, b - remainder if remainder > 0 else 0))

    if method == "mean":
        pad_value = np.nan
    elif method == "sum":
        pad_value = 0
    elif method == "max":
        pad_value = -np.inf
    elif method == "min":
        pad_value = np.inf
    else:
        raise ValidationError(f"Unsupported method '{method}'")

    data_padded = np.pad(
        data, pad_width=pad_width, mode="constant", constant_values=pad_value
    )

    # Coarsen data
    coarsened_data = coarsen_grid(data_padded, batch_size=batch_size, method=method)

    # Coarsen 1D coordinates (x, y)
    def coarsen_1d(coords: np.ndarray, b: int) -> np.ndarray:
        remainder = len(coords) % b
        if remainder > 0:
            coords = np.pad(coords, (0, b - remainder), mode="edge")
        # Coarsen using mean to preserve boundary positions
        return coords.reshape(-1, b).mean(axis=1)

    x_batch = coarsen_1d(x_coords, batch_size[0])
    y_batch = coarsen_1d(y_coords, batch_size[1])

    # Coarsen z-coordinates
    z_batch = None
    if z_coords is not None:
        if data.ndim != 3:
            raise ValidationError("z_coords is only valid for 3D data")

        bx, by, bz = batch_size

        if z_coords.ndim == 1:
            # 1D vertical boundaries
            z_batch = coarsen_1d(z_coords, bz)
            # Append last boundary to maintain nz+1
            z_batch = np.append(z_batch, z_batch[-1] + (z_batch[-1] - z_batch[-2]))
        elif z_coords.ndim == 3:
            # Full 3D boundaries: (nx, ny, nz+1)
            nx, ny, nzp1 = z_coords.shape
            pad_x = bx - nx % bx if nx % bx > 0 else 0
            pad_y = by - ny % by if ny % by > 0 else 0
            pad_z = bz - nzp1 % bz if nzp1 % bz > 0 else 0

            z_padded = np.pad(
                z_coords, ((0, pad_x), (0, pad_y), (0, pad_z)), mode="edge"
            )
            nxp, nyp, nzp1p = z_padded.shape

            # Reshape and coarsen along all axes
            z_reshaped = z_padded.reshape(nxp // bx, bx, nyp // by, by, nzp1p // bz, bz)
            z_batch = z_reshaped.mean(axis=(1, 3, 5))

            # Append last boundary layer along Z to maintain nz+1
            z_batch = np.concatenate([z_batch, z_batch[:, :, -1:]], axis=2)
        else:
            raise ValidationError("z_coords must be 1D or 3D array for 3D data")

    return (
        typing.cast(NDimensionalGrid[NDimension], coarsened_data),
        typing.cast(OneDimensionalGrid, x_batch),
        typing.cast(OneDimensionalGrid, y_batch),
        typing.cast(typing.Optional[NDimensionalGrid[NDimension]], z_batch),
    )


def slice_3d_grid(
    data: ThreeDimensionalGrid,
    x_slice: typing.Optional[typing.Union[int, slice, typing.Tuple[int, int]]] = None,
    y_slice: typing.Optional[typing.Union[int, slice, typing.Tuple[int, int]]] = None,
    z_slice: typing.Optional[typing.Union[int, slice, typing.Tuple[int, int]]] = None,
) -> typing.Tuple[ThreeDimensionalGrid, typing.Tuple[slice, slice, slice]]:
    """
    Apply slicing operations to a 3D grid.

    :param data: 3D array to slice (nx, ny, nz)
    :param x_slice: X-axis slice specification (int, slice, or tuple of (start, end))
    :param y_slice: Y-axis slice specification (int, slice, or tuple of (start, end))
    :param z_slice: Z-axis slice specification (int, slice, or tuple of (start, end))
    :return: Tuple of (sliced_data, (x_slice_obj, y_slice_obj, z_slice_obj))
    :raises ValidationError: If slice specifications are invalid or result in empty grid

    Examples:

    ```python
    data = np.random.rand(10, 10, 10)
    # Extract single plane at x=5
    sliced, _ = slice_3d_grid(data, x_slice=5)
    # Extract range from x=2 to x=8
    sliced, _ = slice_3d_grid(data, x_slice=(2, 8))
    # Extract middle section in all dimensions
    sliced, _ = slice_3d_grid(data, x_slice=(2, 8), y_slice=(3, 7), z_slice=(1, 9))
    ```
    """
    nx, ny, nz = data.shape

    def normalize_slice_spec(
        spec: typing.Optional[typing.Union[int, slice, typing.Tuple[int, int]]],
        dimension_size: int,
        axis_name: str,
    ) -> slice:
        """
        Convert various slice specifications to a standard slice object.

        :param spec: Slice specification (None, int, slice, or (start, end) tuple)
        :param dimension_size: Size of the dimension being sliced
        :param axis_name: Name of axis for error messages ('x', 'y', or 'z')
        :return: Normalized slice object
        :raises ValidationError: If specification is invalid
        """
        if spec is None:
            return slice(None)

        if isinstance(spec, int):
            # Convert negative indices
            if spec < 0:
                spec += dimension_size
            # Validate range
            if not (0 <= spec < dimension_size):
                raise ValidationError(
                    f"{axis_name}_slice {spec} out of range [0, {dimension_size - 1}]"
                )
            # Single index -> slice with width 1
            return slice(spec, spec + 1)

        if isinstance(spec, tuple) and len(spec) == 2:
            start, end = spec
            # Handle negative indices
            if start < 0:
                start += dimension_size
            if end < 0:
                end += dimension_size
            return slice(start, end)

        if isinstance(spec, slice):
            return spec

        raise ValidationError(
            f"Invalid '{axis_name}_slice': {spec}. "
            f"Expected int, slice, or (start, end) tuple"
        )

    # Normalize all slice specifications
    slice_x = normalize_slice_spec(x_slice, nx, "x")
    slice_y = normalize_slice_spec(y_slice, ny, "y")
    slice_z = normalize_slice_spec(z_slice, nz, "z")

    # Apply slicing
    sliced_data = data[slice_x, slice_y, slice_z]

    # Validate result
    if sliced_data.ndim != 3 or any(d < 1 for d in sliced_data.shape):
        raise ValidationError(
            f"Slice operation produced invalid shape {sliced_data.shape}. "
            f"Original shape: {data.shape}, "
            f"Slices: x={slice_x}, y={slice_y}, z={slice_z}"
        )

    return typing.cast(ThreeDimensionalGrid, sliced_data), (slice_x, slice_y, slice_z)


_missing = object()


def get_data(
    source: typing.Union[ModelState[ThreeDimensions], ReservoirModel],
    name: str,
) -> ThreeDimensionalGrid:
    """
    Extract property data from model state or reservoir model.

    Supports nested property access via dot notation (e.g., "permeability.x").

    :param source: The model or model state containing the property
    :param name: Property name as defined by `PropertyMeta.name`, supports dot notation
    :return: 3D numpy array containing the property data
    :raises ValidationError: If property name is invalid for the source type
    :raises AttributeError: If property is not found or has invalid value
    :raises TypeError: If property is not a 3-dimensional array

    Examples:

    ```python
    state = ModelState(...)
    pressure = get_data(state, "pressure")
    perm_x = get_data(state, "permeability.x")
    porosity = get_data(model, "model.porosity")
    ```
    """
    source_type = "model state"
    if isinstance(source, ReservoirModel):
        if not name.startswith("model."):
            raise ValidationError(
                f"Property {name.split('.')[-1]} not available on model. "
                f"Model properties must be prefixed with 'model.'"
            )
        name = name.removeprefix("model.")
        source_type = "reservoir model"

    # Navigate through nested attributes using dot notation
    obj = source
    for part in name.split("."):
        val = getattr(obj, part, _missing)
        if val is _missing:
            raise AttributeError(f"'{name}' not found in {source_type}")
        obj = val

    if obj is None or obj is _missing:
        raise AttributeError(f"Property '{name}' is invalid")

    if not isinstance(obj, np.ndarray):
        obj = np.array(obj, dtype=get_dtype())

    if obj.ndim != 3:
        raise TypeError(f"Property '{name}' is not a 3-D array")

    return typing.cast(ThreeDimensionalGrid, obj)


class FrameExporter(typing.Protocol):
    """
    Protocol for animation frame exporters.

    Implementations receive a list of captured frame images (numpy arrays)
    and write them to a file in the desired format.

    Example:
    ```python
    class CustomExporter:
        def __init__(self, path: str) -> None:
            self.path = path

        def write(self, frames: list[np.ndarray], fps: float) -> None:
            # write frames to self.path
            ...
    ```
    """

    def write(self, frames: typing.List[np.typing.NDArray], fps: float) -> None:
        """
        Write captured frames to disk.

        :param frames: List of RGBA image arrays (H, W, 4)
        :param fps: Frames per second
        """
        ...


class GifExporter:
    """Export animation frames as an animated GIF."""

    def __init__(self, path: typing.Union[str, os.PathLike], loop: int = 0) -> None:
        """
        :param path: Output file path (e.g. `"animation.gif"`)
        :param loop: Number of loops (0 = infinite)
        """
        self.path = Path(path).resolve()
        self.loop = loop

    def write(self, frames: typing.List[np.typing.NDArray], fps: float) -> None:
        imageio.mimsave(self.path, frames, duration=1.0 / fps, loop=self.loop)  # type: ignore
        logger.info("Wrote GIF (%d frames) to %s", len(frames), self.path)


class Mp4Exporter:
    """Export animation frames as an MP4 video (requires `ffmpeg`)."""

    def __init__(
        self,
        path: typing.Union[str, os.PathLike],
        codec: str = "libx264",
        quality: int = 8,
    ) -> None:
        """
        :param path: Output file path (e.g. `"animation.mp4"`)
        :param codec: Video codec (default `"libx264"`)
        :param quality: Quality level 0-10, higher is better (default 8)
        """
        self.path = Path(path).resolve()
        self.codec = codec
        self.quality = quality

    def write(self, frames: typing.List[np.typing.NDArray], fps: float) -> None:
        writer = imageio.get_writer(
            self.path,
            format="FFMPEG",  # type: ignore
            fps=fps,
            codec=self.codec,
            quality=self.quality,
        )
        for frame in frames:
            writer.append_data(frame)
        writer.close()
        logger.info("Wrote MP4 (%d frames) to %s", len(frames), self.path)


class WebPExporter:
    """Export animation frames as an animated WebP image."""

    def __init__(self, path: typing.Union[str, os.PathLike], loop: int = 0) -> None:
        """
        :param path: Output file path (e.g. `"animation.webp"`)
        :param loop: Number of loops (0 = infinite)
        """
        self.path = Path(path).resolve()
        self.loop = loop

    def write(self, frames: typing.List[np.typing.NDArray], fps: float) -> None:
        imageio.mimsave(self.path, frames, duration=1.0 / fps, loop=self.loop)  # type: ignore
        logger.info("Wrote WebP (%d frames) to %s", len(frames), self.path)


class HtmlExporter:
    """Export a Plotly figure as an interactive HTML file."""

    def __init__(
        self,
        path: typing.Union[str, os.PathLike],
        auto_open: bool = False,
        include_plotlyjs: typing.Union[bool, str] = True,
    ) -> None:
        """
        :param path: Output file path (e.g. `"animation.html"`)
        :param auto_open: Open in browser after saving
        :param include_plotlyjs: Include plotly.js in HTML (True, False, or `"cdn"`)
        """
        self.path = Path(path).resolve()
        self.auto_open = auto_open
        self.include_plotlyjs = include_plotlyjs

    def write(self, figure: go.Figure) -> None:
        """
        Write a Plotly figure (with animation frames) to an HTML file.

        :param figure: A `plotly.graph_objects.Figure`
        """
        figure.write_html(
            str(self.path),
            auto_open=self.auto_open,
            include_plotlyjs=self.include_plotlyjs,
        )
        logger.info("Wrote HTML animation to %s", self.path)


_SAVER_REGISTRY: typing.Dict[str, type] = {
    ".gif": GifExporter,
    ".mp4": Mp4Exporter,
    ".webp": WebPExporter,
    ".html": HtmlExporter,
}


def resolve_exporter(
    save: typing.Union[FrameExporter, HtmlExporter, str, None],
    output_gif: typing.Optional[str] = None,
) -> typing.Union[FrameExporter, HtmlExporter, None]:
    """
    Resolve a *save* argument into a concrete exporter.

    Accepts:
    - A `FrameExporter` or `HtmlExporter` instance (returned as-is).
    - A file path string (e.g. `"out.mp4"`, `"out.html"`) and infers format
        from extension.
    - `None` with `output_gif` to use `GifExporter`.

    :raises ValidationError: If the file extension is not supported.
    """
    if save is None and output_gif is not None:
        return GifExporter(output_gif)

    if save is None:
        return None

    if isinstance(save, str):
        ext = os.path.splitext(save)[1].lower()
        cls = _SAVER_REGISTRY.get(ext)
        if cls is None:
            raise ValidationError(
                f"Unsupported animation format '{ext}'. "
                f"Supported: {', '.join(_SAVER_REGISTRY)}"
            )
        return cls(save)  # type: ignore[call-arg]

    return save

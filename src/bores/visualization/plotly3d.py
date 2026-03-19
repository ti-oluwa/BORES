"""
Plotly-based 3D Visualization Suite for Reservoir Simulation Data and Results.
"""

import atexit
import itertools
import logging
import typing
import weakref
from abc import ABC, abstractmethod
from enum import Enum

import attrs
import numpy as np
import plotly.graph_objects as go
from typing_extensions import TypedDict, Unpack

from bores._precision import get_dtype
from bores.errors import ValidationError
from bores.grids.utils import coarsen_grid
from bores.models import ReservoirModel
from bores.states import ModelState
from bores.types import (
    OneDimensionalGrid,
    ThreeDimensionalGrid,
    ThreeDimensions,
)
from bores.visualization.base import (
    ColorScheme,
    PropertyMeta,
    PropertyRegistry,
    property_registry,
)
from bores.visualization.config import MAX_VOLUME_CELLS_3D, RECOMMENDED_VOLUME_CELLS_3D
from bores.visualization.utils import (
    HtmlExporter,
    Label,
    Labels,
    _format_value,
    _invert_z_axis,
    get_data,
    slice_grid,
)
from bores.wells import Wells

logger = logging.getLogger(__name__)


_active_figures: typing.List[weakref.ref] = []


def _register_figure(fig: go.Figure) -> None:
    """Track active figures for memory cleanup."""
    _active_figures.append(weakref.ref(fig, _on_figure_deleted))


def _on_figure_deleted(ref: typing.Any) -> None:
    """Remove dead weakref from registry."""
    try:
        _active_figures.remove(ref)
    except ValueError:
        pass


def cleanup_figures() -> None:
    """
    Release memory from all tracked figures.

    Clears figure data to free memory. Useful after creating large
    animations or multiple figures. Call this when figures are no
    longer needed.

    Example:
    ```python
    from bores.visualization.plotly3d import cleanup_figures

    # Create many figures
    for i in range(100):
        viz.make_plot(...)

    # Free memory when done
    cleanup_figures()
    ```
    """
    for ref in _active_figures[:]:
        fig = ref()
        if fig is not None:
            try:
                fig.data = []
                fig.frames = []
            except (AttributeError, RuntimeError) as e:
                logger.debug(f"Failed to cleanup figure: {e}")
    _active_figures.clear()


atexit.register(cleanup_figures)


class PlotType(str, Enum):
    """Types of 3D plots available."""

    VOLUME = "volume"
    """Volume rendering for scalar fields, showing continuous data distribution."""
    ISOSURFACE = "isosurface"
    """Isosurface plots for discrete value thresholds, showing surfaces at specific data values."""
    SCATTER_3D = "scatter_3d"
    """3D scatter plots for visualizing point data in 3D space."""


class CameraPosition(TypedDict):
    """Camera position and orientation for 3D plots."""

    eye: dict[str, float]
    """Camera position in 3D space (x, y, z coordinates)."""

    center: dict[str, float]
    """Point in 3D space that the camera is looking at (x, y, z coordinates)."""

    up: dict[str, float]
    """Up direction vector for the camera (x, y, z coordinates)."""


class Lighting(TypedDict, total=False):
    """Lighting configuration for 3D plots."""

    ambient: float
    """Ambient light intensity (0.0 to 1.0)."""

    diffuse: float
    """Diffuse light intensity (0.0 to 1.0)."""

    specular: float
    """Specular light intensity (0.0 to 1.0)."""

    roughness: float
    """Surface roughness (0.0 to 1.0)."""

    fresnel: float
    """Fresnel effect intensity (0.0 to 1.0)."""

    facenormalsepsilon: float
    """Epsilon value for face normals (to avoid numerical issues)."""


class LightPosition(TypedDict):
    """Position of the light source in 3D space."""

    x: float
    """X coordinate of the light source."""

    y: float
    """Y coordinate of the light source."""

    z: float
    """Z coordinate of the light source."""


class WellKwargs(TypedDict, total=False):
    """
    Configuration options for well visualization in 3D plots.

    All fields are optional. Default values are used if not specified.
    """

    show_wellbore: bool
    """Whether to show wellbore trajectory as colored tubes (default: True)."""

    show_surface_marker: bool
    """Whether to show arrows at surface location (default: True)."""

    show_well_labels: bool
    """Whether to show well name labels at surface (default: True)."""

    show_perforations: bool
    """Whether to highlight perforated intervals with thicker lines (default: False)."""

    injection_color: str
    """Color for injection wells - CSS color, hex, or rgb (default: "#ff4444")."""

    production_color: str
    """Color for production wells - CSS color, hex, or rgb (default: "#44dd44")."""

    shut_in_color: str
    """Color for shut-in/inactive wells - CSS color, hex, or rgb (default: "#888888")."""

    wellbore_width: float
    """Width of wellbore line representation in pixels (default: 15.0)."""

    surface_marker_size: float
    """Thickness multiplier for surface marker arrows (default: 3.0)."""


DEFAULT_CAMERA_POSITION = CameraPosition(
    eye=dict(x=2.2, y=2.2, z=1.8),  # type: ignore
    center=dict(x=0.0, y=0.0, z=0.0),  # type: ignore
    up=dict(x=0.0, y=0.0, z=1.0),  # type: ignore
)

DEFAULT_OPACITY_SCALE_VALUES = [
    [0, 0.8],
    [0.5, 0.9],
    [1, 1.0],
]

DEFAULT_LIGHTING = Lighting(
    ambient=0.5,
    diffuse=0.8,
    specular=0.2,
    roughness=0.5,
    fresnel=0.2,
    facenormalsepsilon=0.000001,
)


@attrs.frozen
class PlotConfig:
    """Configuration for 3D plots."""

    width: int = 1200
    """Plot width in pixels. Larger values provide higher resolution but may impact performance."""

    height: int = 960
    """Plot height in pixels. Larger values provide higher resolution but may impact performance."""

    plot_type: PlotType = PlotType.VOLUME
    """Default plot type to use when no specific type is requested. Different plot types offer 
    different visualization perspectives of the same 3D data."""

    color_scheme: ColorScheme = ColorScheme.VIRIDIS
    """Default color scheme for data visualization. Professional color schemes are optimized
    for scientific data visualization and accessibility."""

    opacity: float = 0.95
    """Default opacity level for plot elements (0.0 = transparent, 1.0 = opaque).
    Lower values allow better visualization of internal structures."""

    show_colorbar: bool = True
    """Whether to display the colorbar legend showing the data value to color mapping.
    Useful for quantitative analysis of the visualization."""

    show_axes: bool = True
    """Whether to display 3D axes labels and grid lines. Helps with spatial orientation
    and coordinate reference."""

    camera_position: CameraPosition = DEFAULT_CAMERA_POSITION
    """3D camera position and orientation. If None, uses default isometric view.
    Dict should contain 'eye', 'center', and 'up' keys with x,y,z coordinates."""

    title: str = ""
    """Plot title to display above the visualization. Empty string shows no title."""

    use_opacity_scaling: bool = False
    """Whether to apply data-driven opacity scaling for better depth perception.
    Higher data values become more opaque, lower values more transparent."""

    opacity_scale_values: typing.Sequence[typing.Sequence[float]] = (
        DEFAULT_OPACITY_SCALE_VALUES
    )
    """Custom opacity scaling values for volume rendering. List of [data_fraction, opacity] pairs.
    If None, uses default scaling optimized for reservoir data visualization.
    Example: [[0, 0.05], [0.2, 0.3], [0.5, 0.6], [0.8, 0.8], [1, 1.0]]"""

    aspect_mode: typing.Optional[typing.Literal["cube", "data", "auto"]] = None
    """Aspect mode for the 3D plot. Determines how the plot is scaled and displayed.
    Options are:
    - "cube": Equal scaling for all axes (default).
    - "data": Automatic scaling based on data extents.
    - "auto": Automatic aspect ratio adjustment."""

    lighting: Lighting = DEFAULT_LIGHTING
    """Lighting configuration for 3D plots. Controls how light interacts with surfaces,

    including ambient, diffuse, and specular reflections, roughness, and fresnel effects.
    This affects how surfaces appear under different lighting conditions, enhancing realism."""

    light_position: typing.Optional[LightPosition] = None
    """Position of the light source in 3D space. Controls where the light comes from,
    affecting shadows and highlights on surfaces. This can be adjusted to simulate different
    lighting conditions, such as overhead sunlight or side lighting for dramatic effects."""

    paper_bgcolor: str = "#ffffff"
    """Background color of entire figure (outer area)"""

    scene_bgcolor: str = "#f8f9fa"
    """Background color of the 3D scene (light gray for subtle contrast)"""

    show_labels: bool = True
    """Global toggle for all labels in plots. When False, disables labels even if labels parameter is provided.
    Useful for clean visualizations without text annotations."""


class BaseRenderer(ABC):
    """Base class for 3D renders."""

    supports_physical_dimensions: typing.ClassVar[bool] = False
    """Whether this renderer supports physical cell dimensions and depth grids."""

    def __init__(self, config: PlotConfig) -> None:
        self.config = config

    @abstractmethod
    def render(
        self,
        figure: go.Figure,
        data: ThreeDimensionalGrid,
        metadata: PropertyMeta,
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> go.Figure:
        """
        Render a 3D plot from data.

        Subclasses should override this with their specific keyword arguments.
        """
        ...

    def get_colorscale(self, color_scheme: typing.Union[ColorScheme, str]) -> str:
        """
        Get plotly colorscale string from ColorScheme enum.

        :param color_scheme: The color scheme enum to convert
        :return: Plotly colorscale string
        """
        return ColorScheme(color_scheme).value

    def get_scene_config(
        self,
        x_title: str,
        y_title: str,
        z_title: str,
        aspect_mode: str,
        z_scale: float = 1.0,
        x_range: typing.Optional[typing.Tuple[float, float]] = None,
        y_range: typing.Optional[typing.Tuple[float, float]] = None,
        z_range: typing.Optional[typing.Tuple[float, float]] = None,
    ) -> typing.Dict[str, typing.Any]:
        """
        Prepare scene configuration with proper aspect ratio handling.

        :param x_title: Title for X axis
        :param y_title: Title for Y axis
        :param z_title: Title for Z axis
        :param aspect_mode: Base aspect mode ("auto", "cube", "data", or "manual")
        :param z_scale: Scale factor for Z-axis visual spacing
        :param x_range: Optional (min, max) range for X coordinates
        :param y_range: Optional (min, max) range for Y coordinates
        :param z_range: Optional (min, max) range for Z coordinates
        :return: Scene configuration dictionary
        """
        scene_config: typing.Dict[str, typing.Any] = {
            "xaxis_title": x_title,
            "yaxis_title": y_title,
            "zaxis_title": z_title,
            "camera": self.config.camera_position,
            "dragmode": "orbit",
            "bgcolor": self.config.scene_bgcolor,
            "xaxis": {
                "backgroundcolor": "rgba(0,0,0,0)",
                "gridcolor": "lightgray",
                "showbackground": True,
                "zerolinecolor": "gray",
            },
            "yaxis": {
                "backgroundcolor": "rgba(0,0,0,0)",
                "gridcolor": "lightgray",
                "showbackground": True,
                "zerolinecolor": "gray",
            },
            "zaxis": {
                "backgroundcolor": "rgba(0,0,0,0)",
                "gridcolor": "lightgray",
                "showbackground": True,
                "zerolinecolor": "gray",
            },
        }

        # Apply aspect ratio based on mode and z_scale
        if z_scale != 1.0:
            if aspect_mode == "data" and all(
                r is not None for r in [x_range, y_range, z_range]
            ):
                # Calculate aspect ratio from actual data extents
                x_extent = x_range[1] - x_range[0]  # type: ignore
                y_extent = y_range[1] - y_range[0]  # type: ignore
                z_extent = z_range[1] - z_range[0]  # type: ignore

                # Normalize to largest extent
                max_extent = max(x_extent, y_extent, z_extent)
                if max_extent > 0:
                    x_ratio = x_extent / max_extent
                    y_ratio = y_extent / max_extent
                    z_ratio = (z_extent / max_extent) * z_scale
                    scene_config["aspectmode"] = "manual"
                    scene_config["aspectratio"] = {
                        "x": x_ratio,
                        "y": y_ratio,
                        "z": z_ratio,
                    }  # type: ignore[typeddict-item]
                else:
                    # Fallback to cube with z_scale
                    scene_config["aspectmode"] = "manual"
                    scene_config["aspectratio"] = {"x": 1, "y": 1, "z": z_scale}  # type: ignore[typeddict-item]
            elif aspect_mode == "cube":
                # Cube mode: equal ratios but scale z
                scene_config["aspectmode"] = "manual"
                scene_config["aspectratio"] = {"x": 1, "y": 1, "z": z_scale}  # type: ignore[typeddict-item]
            else:
                # Auto mode: let Plotly decide x and y, but scale z
                # Use manual with equal x/y ratios
                scene_config["aspectmode"] = "manual"
                scene_config["aspectratio"] = {"x": 1, "y": 1, "z": z_scale}  # type: ignore[typeddict-item]
        else:
            # No z_scale, use the requested aspect mode as-is
            scene_config["aspectmode"] = aspect_mode

        return scene_config

    def normalize_data(
        self,
        data: ThreeDimensionalGrid,
        metadata: PropertyMeta,
        normalize_range: bool = False,
    ) -> typing.Tuple[ThreeDimensionalGrid, ThreeDimensionalGrid]:
        """
        Prepare data for plotting (handle log scale, clipping, etc.).

        :param data: Input data array to prepare
        :param metadata: Property metadata containing scaling and range information
        :param normalize_range: Whether to normalize data to 0-1 range (only for volume rendering)
        :return: Tuple of (processed_data_for_plotting, original_data_for_display)
        """
        # Keep original data for display purposes (hover text, colorbar)
        dtype = get_dtype()
        display_data = data.astype(dtype, copy=False)

        # Process data for plotting
        processed_data = data.astype(dtype, copy=False)

        if metadata.log_scale:
            # Handle zero/negative values for log scale
            min_positive = np.nanmin(data[data > 0])
            processed_data = np.where(data <= 0, min_positive * 0.1, data)
            processed_data = np.log10(processed_data)

        # Apply clipping if specified
        if metadata.min_val is not None and metadata.max_val is not None:
            if metadata.log_scale:
                # Clip original data, then apply log to processed data
                display_data = np.clip(display_data, metadata.min_val, metadata.max_val)
                processed_data = np.log10(
                    np.where(display_data <= 0, metadata.min_val * 0.1, display_data)
                )
            else:
                processed_data = np.clip(
                    processed_data, metadata.min_val, metadata.max_val
                )
                display_data = processed_data.copy()

        elif metadata.min_val is not None:
            if metadata.log_scale:
                display_data = np.clip(display_data, metadata.min_val, None)
                processed_data = np.log10(
                    np.where(display_data <= 0, metadata.min_val * 0.1, display_data)
                )
            else:
                processed_data = np.clip(processed_data, metadata.min_val, None)
                display_data = processed_data.copy()

        elif metadata.max_val is not None:
            if metadata.log_scale:
                display_data = np.clip(display_data, None, metadata.max_val)
                processed_data = np.log10(
                    np.where(display_data <= 0, metadata.max_val * 0.1, display_data)
                )
            else:
                processed_data = np.clip(processed_data, None, metadata.max_val)
                display_data = processed_data.copy()

        if normalize_range:
            # Only normalize to 0-1 range when explicitly requested (e.g for volume rendering)
            data_min = float(np.nanmin(processed_data))
            data_max = float(np.nanmax(processed_data))
            if data_max > data_min:
                processed_data = (processed_data - data_min) / (data_max - data_min)

        processed_data = _invert_z_axis(processed_data)
        display_data = _invert_z_axis(display_data)
        return typing.cast(ThreeDimensionalGrid, processed_data), typing.cast(
            ThreeDimensionalGrid, display_data
        )

    def get_physical_coordinates(
        self,
        cell_dimension: typing.Tuple[float, float],
        depth_grid: ThreeDimensionalGrid,
        coordinate_offsets: typing.Optional[typing.Tuple[int, int, int]] = None,
    ) -> typing.Tuple[OneDimensionalGrid, OneDimensionalGrid, ThreeDimensionalGrid]:
        """
        Prepare physical coordinate arrays based on cell dimensions and depth grid.

        Returns cell boundary coordinates for all dimensions (not cell centers).
        For a grid with shape (nx, ny, nz), returns:
        - X boundaries: 1D array of length nx+1
        - Y boundaries: 1D array of length ny+1
        - Z boundaries: 3D array of shape (nx, ny, nz+1)

        Implements numpy array indexing convention:
        - data[:, :, 0] (first array layer) appears at the TOP of the visualization
        - data[:, :, k] layers stack downward as k increases
        - data[:, :, -1] (last array layer) appears at the BOTTOM of the visualization

        :param cell_dimension: Physical size of each cell in x and y directions (feet)
        :param depth_grid: 3D array with depth of each cell center (feet, positive downward).
                          Includes structural dip if model has dip_angle > 0.
        :param coordinate_offsets: Optional cell index offsets for sliced data (x_offset, y_offset, z_offset)
        :return: Tuple of (X, Y, Z) coordinate arrays in physical units (all boundaries, not centers)
        """
        nx, ny, nz = depth_grid.shape
        dx, dy = cell_dimension

        # Calculate physical coordinate offsets from cell index offsets
        physical_offsets = (0.0, 0.0, 0.0)
        if coordinate_offsets is not None:
            x_index_offset, y_index_offset, _ = coordinate_offsets
            x_start_offset = x_index_offset * dx
            y_start_offset = y_index_offset * dy

            # No need to calculate Z offset from depth_grid - depths are absolute
            physical_offsets = (x_start_offset, y_start_offset, 0.0)

        # Apply physical offsets (for X and Y only, Z uses depth values directly)
        x_offset, y_offset, _ = physical_offsets

        # Create base coordinate grids with offsets
        x_coords = x_offset + np.arange(nx + 1) * dx  # Cell boundaries
        y_coords = y_offset + np.arange(ny + 1) * dy  # Cell boundaries

        # Calculate Z cell boundaries from depth grid (which contains cell centers)
        # depth_grid[i,j,k] is the depth of the center of cell k
        # We need to calculate boundaries (nz+1 values) from centers (nz values)
        z_boundaries = np.zeros((nx, ny, nz + 1))

        for i, j in itertools.product(range(nx), range(ny)):
            # For interior boundaries: midpoint between adjacent cell centers
            for k in range(nz - 1):
                z_boundaries[i, j, k + 1] = (
                    depth_grid[i, j, k] + depth_grid[i, j, k + 1]
                ) / 2

            # First boundary (top): extrapolate from first two centers
            if nz > 1:
                z_boundaries[i, j, 0] = (
                    depth_grid[i, j, 0]
                    - (depth_grid[i, j, 1] - depth_grid[i, j, 0]) / 2
                )
            else:
                # Only one layer - assume unit thickness
                z_boundaries[i, j, 0] = depth_grid[i, j, 0] - 0.5

            # Last boundary (bottom): extrapolate from last two centers
            if nz > 1:
                z_boundaries[i, j, nz] = (
                    depth_grid[i, j, nz - 1]
                    + (depth_grid[i, j, nz - 1] - depth_grid[i, j, nz - 2]) / 2
                )
            else:
                z_boundaries[i, j, 1] = depth_grid[i, j, 0] + 0.5

        # Negate depths for Z-axis (positive upward in Plotly)
        z_coords = -z_boundaries

        return (
            typing.cast(OneDimensionalGrid, x_coords),
            typing.cast(OneDimensionalGrid, y_coords),
            typing.cast(ThreeDimensionalGrid, z_coords),
        )

    def apply_labels(
        self,
        figure: go.Figure,
        labels: typing.Optional[
            typing.Union["Labels", typing.Iterable["Label"]]
        ] = None,
        data: typing.Optional[ThreeDimensionalGrid] = None,
        metadata: typing.Optional[PropertyMeta] = None,
        cell_dimension: typing.Optional[typing.Tuple[float, float]] = None,
        depth_grid: typing.Optional[ThreeDimensionalGrid] = None,
        coordinate_offsets: typing.Optional[typing.Tuple[int, int, int]] = None,
        format_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
    ) -> None:
        """
        Apply labels to a 3D plot using the Label system.

        :param figure: Plotly figure to add labels to
        :param labels: Labels to apply (Labels or list of Label objects)
        :param data: 3D data array for value extraction
        :param metadata: Property metadata for formatting
        :param cell_dimension: Physical cell dimensions for coordinate conversion
        :param depth_grid: Depth grid for physical coordinate conversion (feet, positive downward)
        :param coordinate_offsets: Coordinate offsets for sliced data
        :param format_kwargs: Additional formatting values
        """
        if not self.config.show_labels:
            return

        if labels is None:
            return

        # Handle both Labels and list of Labels
        if isinstance(labels, Labels):
            label_collection = labels
        else:
            label_collection = Labels(labels)

        # Get Plotly annotations
        annotations = label_collection.as_annotations(
            data_grid=data,
            metadata=metadata,
            cell_dimension=cell_dimension,
            depth_grid=depth_grid,
            coordinate_offsets=coordinate_offsets,
            format_kwargs=format_kwargs,
        )
        # Add annotations to the figure
        if annotations:
            # Get existing annotations or create empty list
            existing_annotations = []
            try:
                scene = getattr(figure.layout, "scene", None)
                if scene and hasattr(scene, "annotations"):
                    existing_annotations = list(scene.annotations or [])
            except (AttributeError, TypeError):
                pass

            existing_annotations.extend(annotations)
            figure.update_layout(scene=dict(annotations=existing_annotations))

            # Debug information
            logger.debug(f"Applied {len(annotations)} labels to 3D plot")
        else:
            logger.debug(
                "No annotations to apply - labels may be filtered out or invisible"
            )

    def render_wells(
        self,
        figure: go.Figure,
        wells: typing.Optional[Wells[ThreeDimensions]] = None,
        cell_dimension: typing.Optional[typing.Tuple[float, float]] = None,
        depth_grid: typing.Optional[np.ndarray] = None,
        coordinate_offsets: typing.Optional[typing.Tuple[int, int, int]] = None,
        z_scale: float = 1.0,
        **kwargs: Unpack[WellKwargs],
    ) -> None:
        """
        Add visual representation of wells to 3D plot.

        This method renders a comprehensive 3D visualizations of wells on the figure including:
        - Wellbore trajectories as colored lines/tubes
        - Surface location markers with directional arrows
        - Perforated interval highlights (optional)
        - Interactive hover information

        :param figure: Plotly figure to add well visualizations to
        :param wells: Wells object containing injection and production wells
        :param cell_dimension: Physical size of each cell in x and y directions (feet).
            Required for physical coordinate conversion. If None, uses grid indices.
        :param depth_grid: 3D array with depth of each cell center (feet, positive downward).
            Required for physical coordinate conversion. If None, uses grid indices.
        :param coordinate_offsets: Coordinate offsets for sliced data (i_offset, j_offset, k_offset).
            Used to adjust well locations when visualizing a subset of the full grid.
        :param z_scale: Scale factor for Z-axis to match volume rendering (default: 1.0).
            Wells will be scaled to match the stretched/compressed volume visualization.
        :param show_wellbore: Whether to show wellbore trajectory as colored tubes (default: True)
        :param show_surface_marker: Whether to show arrows at surface location (default: True)
        :param show_perforations: Whether to highlight perforated intervals with thicker lines (default: False)
        :param injection_color: Color for injection wells (CSS color, hex, or rgb) (default: red)
        :param production_color: Color for production wells (CSS color, hex, or rgb) (default: green)
        :param shut_in_color: Color for shut-in/inactive wells (CSS color, hex, or rgb) (default: gray)
        :param wellbore_width: Width of wellbore line representation in pixels (default: 15.0)
        :param surface_marker_size: Size scaling factor for surface markers (default: 5.0)

        **Usage Example:**

        ```python
        # Basic usage with default styling
        renderer.render_wells(fig, model_state.wells)

        # Custom styling
        renderer.render_wells(
            fig,
            model_state.wells,
            cell_dimension=(100, 100),
            depth_grid=depth_grid,
            injection_color='#ff6b6b',
            production_color='#51cf66',
            wellbore_width=8.0,
            show_surface_marker=True
        )
        ```

        **Visual Elements:**
        - Injection wells: Red/warm colors with upward flow indication
        - Production wells: Green/cool colors with downward flow indication
        - Shut-in wells: Gray/muted colors, partially transparent
        - Surface markers: Cone/arrow shapes pointing into reservoir
        - Wellbore: Continuous line through all perforating intervals

        **Interactive Features:**
        - Hover over wellbore: Shows well name, type, BHP, radius, skin factor
        - Hover over surface marker: Shows well name and surface location
        - Legend groups: Wells can be toggled on/off in plot legend
        """
        if wells is None:
            logger.warning("render_wells called but wells parameter is None")
            return

        if not wells.exists():
            logger.warning("`render_wells` called but wells.exists() returned False")
            return

        logger.debug(
            f"Rendering {len(wells.injection_wells)} injection wells and {len(wells.production_wells)} production wells"
        )

        # Extract kwargs with defaults
        show_wellbore = kwargs.get("show_wellbore", True)
        show_surface_marker = kwargs.get("show_surface_marker", True)
        show_perforations = kwargs.get("show_perforations", False)
        injection_color = kwargs.get("injection_color", "#ff4444")
        production_color = kwargs.get("production_color", "#44dd44")
        shut_in_color = kwargs.get("shut_in_color", "#888888")
        wellbore_width = kwargs.get("wellbore_width", 15.0)
        surface_marker_size = kwargs.get("surface_marker_size", 5.0)

        def grid_to_physical(
            i: int, j: int, k: int
        ) -> typing.Optional[typing.Tuple[float, float, float]]:
            """
            Convert grid indices to physical coordinates.
            Returns None if coordinates contain NaN or are out of bounds.
            Note: Z-axis is negative (depth increases downward in Plotly convention)
            Z-scale is applied around z_mean to match volume rendering.
            """
            if cell_dimension is not None and depth_grid is not None:
                dx, dy = cell_dimension
                x_offset, y_offset, z_offset = coordinate_offsets or (0, 0, 0)

                # When using slices, the well coordinates are in the original grid space,
                # but `depth_grid` is in the sliced space. We need to adjust.
                # The `coordinate_offsets` tell us where the slice starts in the original grid.
                # So we need to convert original grid coords (i, j, k) to sliced coords.
                i_sliced = i - x_offset
                j_sliced = j - y_offset
                k_sliced = k - z_offset

                # Check if the well coordinates are within the sliced grid bounds
                if not (
                    0 <= i_sliced < depth_grid.shape[0]
                    and 0 <= j_sliced < depth_grid.shape[1]
                    and 0 <= k_sliced < depth_grid.shape[2]
                ):
                    # Well is outside the sliced region, skip it
                    return None

                # Physical X and Y (center wells in cells using +0.5 offset)
                x_phys = (i + 0.5) * dx
                y_phys = (j + 0.5) * dy

                # Physical Z (depth) - use depth_grid with sliced indices
                # Depth is positive downward, so negate for Z-axis (positive up)
                # depth_grid already includes structural dip
                depth_value = float(depth_grid[i_sliced, j_sliced, k_sliced])
                if np.isnan(depth_value):
                    return None

                z_phys = -depth_value  # Negate because Plotly Z-axis positive up

                return x_phys, y_phys, z_phys
            else:
                # Use grid indices directly (center wells in cells)
                return float(i) + 0.5, float(j) + 0.5, float(k) + 0.5

        # Process all injection wells
        for well in wells.injection_wells:
            # Determine color based on well state using is_shut_in() method
            if well.is_shut_in:
                color = shut_in_color
                well_type = "Injection (Shut-in)"
                opacity = 0.5
            else:
                color = injection_color
                well_type = "Injection"
                opacity = 1.0

            # Extract well trajectory points from perforating intervals
            if show_wellbore and well.perforating_intervals:
                x_points, y_points, z_points = [], [], []

                # Get the shallowest perforation (first one) for surface extension
                first_perf_start = well.perforating_intervals[0][0]
                first_perf_coords = grid_to_physical(*first_perf_start)

                # Add surface-to-perforation segment (neutral color)
                if first_perf_coords is not None and cell_dimension is not None:
                    x_surf, y_surf, z_perf = first_perf_coords
                    # Surface should be at z_offset (the top of the grid)
                    # coordinate_offsets contains the base Z coordinate
                    x_offset, y_offset, z_offset = coordinate_offsets or (0, 0, 0)
                    z_surface = float(z_offset)  # Top of volume

                    # Add neutral-colored segment from surface to perforation
                    figure.add_trace(
                        go.Scatter3d(
                            x=[x_surf, x_surf],
                            y=[y_surf, y_surf],
                            z=[z_surface, z_perf],
                            mode="lines",
                            line=dict(  # noqa
                                color="#999999",  # Neutral gray
                                width=wellbore_width * 0.7,  # Slightly thinner
                                dash="solid",
                            ),
                            opacity=0.6,
                            name=f"{well.name} casing",
                            showlegend=False,
                            legendgroup=well.name,
                            hovertemplate=(
                                f"<b>{well.name} Casing</b><br>"
                                f"Non-perforated section<br>"
                                "<extra></extra>"
                            ),
                        )
                    )

                for start_loc, end_loc in well.perforating_intervals:
                    # Convert start and end to physical coordinates
                    start_coords = grid_to_physical(*start_loc)
                    end_coords = grid_to_physical(*end_loc)

                    # Skip if coordinates contain NaN
                    if start_coords is None or end_coords is None:
                        continue

                    x_start, y_start, z_start = start_coords
                    x_end, y_end, z_end = end_coords

                    # Handle single-cell perforations (start == end)
                    # Create a small vertical line segment for visibility
                    if tuple(start_loc) == tuple(end_loc):
                        if cell_dimension is not None and depth_grid is not None:
                            # Use a small vertical extent for single-cell perforations
                            # Since depth_grid gives cell center, extend above and below
                            extension = 5.0  # 5 ft extension (physical, not scaled)
                            z_start -= extension  # Extend upward
                            z_end += extension  # Extend downward
                        else:
                            # For grid coordinates, extend by 0.5 in Z direction
                            z_start -= 0.5
                            z_end += 0.5

                    x_points.extend([x_start, x_end])
                    y_points.extend([y_start, y_end])
                    z_points.extend([z_start, z_end])

                # Add wellbore trajectory only if we have valid points
                if x_points:
                    figure.add_trace(
                        go.Scatter3d(
                            x=x_points,
                            y=y_points,
                            z=z_points,
                            mode="lines",
                            line=dict(  # noqa
                                color=color,
                                width=wellbore_width,
                                dash="solid" if well.is_open else "dash",
                            ),
                            opacity=opacity,
                            name=f"{well.name} ({well_type})",
                            showlegend=True,
                            legendgroup=well.name,
                            hovertemplate=(
                                f"<b>{well.name}</b><br>"
                                f"Type: {well_type}<br>"
                                f"Control: {str(well.control)[:20]}...<br>"
                                f"Radius: {well.radius:.2f} ft<br>"
                                f"Skin: {well.skin_factor:.2f}<br>"
                                "<extra></extra>"
                            ),
                        )
                    )

            # Add perforation highlights if requested (injection wells)
            if show_perforations and well.perforating_intervals and well.is_open:
                for start_loc, end_loc in well.perforating_intervals:
                    # Convert to physical coordinates
                    start_coords = grid_to_physical(*start_loc)
                    end_coords = grid_to_physical(*end_loc)

                    # Skip if coordinates contain NaN
                    if start_coords is None or end_coords is None:
                        continue

                    x_start, y_start, z_start = start_coords
                    x_end, y_end, z_end = end_coords

                    # Add perforation interval as thicker, more opaque line with markers
                    figure.add_trace(
                        go.Scatter3d(
                            x=[x_start, x_end],
                            y=[y_start, y_end],
                            z=[z_start, z_end],
                            mode="lines+markers",
                            line=dict(  # noqa
                                color=color,
                                width=wellbore_width * 1.5,  # 50% thicker
                            ),
                            marker=dict(  # noqa
                                size=4,
                                color=color,
                                symbol="diamond",
                            ),
                            opacity=min(opacity * 1.2, 1.0),  # More opaque
                            name=f"{well.name} perforations",
                            showlegend=False,  # Don't clutter legend
                            legendgroup=well.name,
                            hovertemplate=(
                                f"<b>{well.name} Perforation</b><br>"
                                f"Start: ({start_loc[0]}, {start_loc[1]}, {start_loc[2]})<br>"
                                f"End: ({end_loc[0]}, {end_loc[1]}, {end_loc[2]})<br>"
                                "<extra></extra>"
                            ),
                        )
                    )

            # Add surface marker with directional indicator
            if show_surface_marker and well.perforating_intervals:
                # Get surface location (X, Y from first perforation, but Z at surface)
                start_loc = well.perforating_intervals[0][0]
                surf_coords = grid_to_physical(*start_loc)

                # Only add surface marker if coordinates are valid
                if surf_coords is not None:
                    x_surf, y_surf, _ = surf_coords

                    # Surface should be at z_offset (the top of the grid)
                    # coordinate_offsets contains the base Z coordinate
                    x_offset, y_offset, z_offset = coordinate_offsets or (0, 0, 0)
                    z_surf = float(z_offset)  # Top of volume

                    # Add cone/arrow pointing into reservoir at the SURFACE
                    # Build fluid info for hover text
                    if well.injected_fluid is not None:
                        fluid_info = f"Injected Fluid: <b>{well.injected_fluid.name}</b>  ({well.injected_fluid.phase.value})"  # type: ignore
                    else:
                        fluid_info = "Injected Fluid: N/A"

                    status = "Open" if well.is_open else "Shut-in"

                    # Calculate appropriate cone size based on grid dimensions
                    # For aspectmode="data", scale relative to cell size
                    # NOTE: Do NOT include z_scale here — the cone direction vector
                    # (w) already contains z_scale via total_arrow_length, and Plotly
                    # computes cone size as sizeref * ||direction||, so including
                    # z_scale in both would square the scaling effect.
                    cone_size_multiplier = 1.0  # Default multiplier for cube mode
                    if cell_dimension is not None:
                        dx, dy = cell_dimension
                        cone_size_multiplier = max(dx, dy) * 0.04 / surface_marker_size

                    # Arrow sizing - base it on actual grid depth spacing
                    if depth_grid is not None:
                        avg_layer_depth = float(np.mean(np.diff(depth_grid, axis=2)))
                        base_arrow_length = avg_layer_depth * 0.25
                    else:
                        base_arrow_length = surface_marker_size * 0.5

                    # Apply z_scale (only to the direction/position, not sizeref)
                    total_arrow_length = base_arrow_length * z_scale

                    # Split: 60% stem, 40% cone
                    stem_length = total_arrow_length * 0.6
                    cone_length = total_arrow_length * 0.4

                    # For injection wells, arrow points DOWN but starts ABOVE surface
                    arrow_top = z_surf + total_arrow_length
                    arrow_stem_bottom = z_surf + cone_length

                    # Add cylindrical stem (arrow shaft) starting from above
                    figure.add_trace(
                        go.Scatter3d(
                            x=[x_surf, x_surf],
                            y=[y_surf, y_surf],
                            z=[arrow_top, arrow_stem_bottom],
                            mode="lines",
                            line=dict(  # noqa
                                color=color,
                                width=wellbore_width * 0.8,  # Make stem more visible
                            ),
                            opacity=opacity,
                            name=f"{well.name} arrow stem",
                            showlegend=False,
                            legendgroup=well.name,
                            hoverinfo="skip",
                        )
                    )

                    # Add cone (arrowhead) at the bottom of stem, pointing down to surface
                    figure.add_trace(
                        go.Cone(
                            x=[x_surf],
                            y=[y_surf],
                            z=[arrow_stem_bottom],  # Cone starts where stem ends
                            u=[0],
                            v=[0],
                            w=[-cone_length],  # Points down, tip reaches surface
                            sizemode="absolute",
                            sizeref=surface_marker_size * cone_size_multiplier,
                            colorscale=[[0, color], [1, color]],
                            showscale=False,
                            opacity=opacity,
                            name=f"{well.name} surface",
                            showlegend=False,
                            legendgroup=well.name,
                            hovertemplate=(
                                f"<b>{well.name} - Surface Location</b><br>"
                                f"Type: {well_type}<br>"
                                f"Status: {status}<br>"
                                f"{fluid_info}<br>"
                                f"Control: {str(well.control)[:100]}<br>"
                                f"Wellbore Radius: {well.radius:.2f} ft<br>"
                                f"Skin Factor: {well.skin_factor:.2f}<br>"
                                f"<br>"
                                f"Surface Coords:<br>"
                                f"  X: {x_surf:.1f} ft<br>"
                                f"  Y: {y_surf:.1f} ft<br>"
                                f"  Z: {z_surf:.1f} ft<br>"
                                "<extra></extra>"
                            ),
                        )
                    )

        # Process all production wells (similar logic)
        for well in wells.production_wells:
            logger.debug(
                f"Processing production well: {well.name}, is_active={well.is_active}, perforations={well.perforating_intervals}"
            )
            # Determine color based on well state using is_shut_in() method
            if well.is_shut_in:
                color = shut_in_color
                well_type = "Production (Shut-in)"
                opacity = 0.5
            else:
                color = production_color
                well_type = "Production"
                opacity = 1.0

            # Extract well trajectory points from perforating intervals
            if show_wellbore and well.perforating_intervals:
                x_points, y_points, z_points = [], [], []
                logger.debug(
                    f"  Extracting wellbore trajectory for {well.name}, show_wellbore={show_wellbore}"
                )

                # Get the shallowest perforation (first one) for surface extension
                first_perf_start = well.perforating_intervals[0][0]
                first_perf_coords = grid_to_physical(*first_perf_start)

                # Add surface-to-perforation segment (neutral color)
                if first_perf_coords is not None and cell_dimension is not None:
                    x_surf, y_surf, z_perf = first_perf_coords
                    # Surface should be at z_offset (the top of the grid)
                    # coordinate_offsets contains the base Z coordinate
                    x_offset, y_offset, z_offset = coordinate_offsets or (0, 0, 0)
                    z_surface = float(z_offset)  # Top of volume

                    # Add neutral-colored segment from surface to perforation
                    figure.add_trace(
                        go.Scatter3d(
                            x=[x_surf, x_surf],
                            y=[y_surf, y_surf],
                            z=[z_surface, z_perf],
                            mode="lines",
                            line=dict(  # noqa
                                color="#999999",  # Neutral gray
                                width=wellbore_width * 0.7,  # Slightly thinner
                                dash="solid",
                            ),
                            opacity=0.6,
                            name=f"{well.name} casing",
                            showlegend=False,
                            legendgroup=well.name,
                            hovertemplate=(
                                f"<b>{well.name} Casing</b><br>"
                                f"Non-perforated section<br>"
                                "<extra></extra>"
                            ),
                        )
                    )

                for start_loc, end_loc in well.perforating_intervals:
                    # Convert start and end to physical coordinates
                    start_coords = grid_to_physical(*start_loc)
                    end_coords = grid_to_physical(*end_loc)

                    logger.debug(
                        f"    Production perforation: {start_loc} -> {end_loc}, coords: {start_coords} -> {end_coords}"
                    )

                    # Skip if coordinates contain NaN
                    if start_coords is None or end_coords is None:
                        logger.debug("    Skipping perforation due to None coordinates")
                        continue

                    x_start, y_start, z_start = start_coords
                    x_end, y_end, z_end = end_coords

                    # Handle single-cell perforations (start == end)
                    # Create a small vertical line segment for visibility
                    if tuple(start_loc) == tuple(end_loc):
                        logger.debug(
                            f"    Single-cell perforation detected at {start_loc}, extending vertically"
                        )
                        if cell_dimension is not None and depth_grid is not None:
                            # Use a small vertical extent for single-cell perforations
                            extension = 5.0  # 5 ft extension (physical, not scaled)
                            z_start -= extension  # Extend upward
                            z_end += extension  # Extend downward
                        else:
                            # For grid coordinates, extend by 0.5 in Z direction
                            z_start -= 0.5
                            z_end += 0.5

                    x_points.extend([x_start, x_end])
                    y_points.extend([y_start, y_end])
                    z_points.extend([z_start, z_end])

                # Add wellbore trajectory only if we have valid points
                if x_points:
                    logger.debug(
                        f"  Adding production wellbore trace for {well.name} with {len(x_points)} points"
                    )
                    figure.add_trace(
                        go.Scatter3d(
                            x=x_points,
                            y=y_points,
                            z=z_points,
                            mode="lines",
                            line=dict(  # noqa
                                color=color,
                                width=wellbore_width,
                                dash="solid" if well.is_open else "dash",
                            ),
                            opacity=opacity,
                            name=f"{well.name} ({well_type})",
                            showlegend=True,
                            legendgroup=well.name,
                            hovertemplate=(
                                f"<b>{well.name}</b><br>"
                                f"Type: {well_type}<br>"
                                f"Control: {str(well.control)[:100]}<br>"
                                f"Radius: {well.radius:.2f} ft<br>"
                                f"Skin: {well.skin_factor:.2f}<br>"
                                "<extra></extra>"
                            ),
                        )
                    )
                else:
                    logger.warning(
                        f"  No valid points found for production well {well.name} wellbore trajectory"
                    )

            # Add perforation highlights if requested (production wells)
            if show_perforations and well.perforating_intervals and well.is_open:
                for start_loc, end_loc in well.perforating_intervals:
                    # Convert to physical coordinates
                    start_coords = grid_to_physical(*start_loc)
                    end_coords = grid_to_physical(*end_loc)

                    # Skip if coordinates contain NaN
                    if start_coords is None or end_coords is None:
                        continue

                    x_start, y_start, z_start = start_coords
                    x_end, y_end, z_end = end_coords

                    # Add perforation interval as thicker, more opaque line with markers
                    figure.add_trace(
                        go.Scatter3d(
                            x=[x_start, x_end],
                            y=[y_start, y_end],
                            z=[z_start, z_end],
                            mode="lines+markers",
                            line=dict(  # noqa
                                color=color,
                                width=wellbore_width * 1.5,  # 50% thicker
                            ),
                            marker=dict(  # noqa
                                size=4,
                                color=color,
                                symbol="diamond",
                            ),
                            opacity=min(opacity * 1.2, 1.0),  # More opaque
                            name=f"{well.name} perforations",
                            showlegend=False,  # Don't clutter legend
                            legendgroup=well.name,
                            hovertemplate=(
                                f"<b>{well.name} Perforation</b><br>"
                                f"Start: ({start_loc[0]}, {start_loc[1]}, {start_loc[2]})<br>"
                                f"End: ({end_loc[0]}, {end_loc[1]}, {end_loc[2]})<br>"
                                "<extra></extra>"
                            ),
                        )
                    )

            # Add surface marker with directional indicator (PRODUCTION WELLS)
            if show_surface_marker and well.perforating_intervals:
                # Get surface location (X, Y from first perforation, but Z at surface)
                start_loc = well.perforating_intervals[0][0]
                surf_coords = grid_to_physical(*start_loc)

                # Only add surface marker if coordinates are valid
                if surf_coords is not None:
                    x_surf, y_surf, _ = surf_coords

                    # Surface should be at z_offset (the top of the grid)
                    # `coordinate_offsets` contains the base Z coordinate
                    x_offset, y_offset, z_offset = coordinate_offsets or (0, 0, 0)
                    z_surf = float(z_offset)  # Top of volume

                    # Add cone/arrow pointing OUT OF reservoir at the SURFACE (production = outflow)
                    # Build fluid info for hover text (production wells can have multiple fluids)
                    fluid_names = []
                    if well.produced_fluids:
                        for fluid in well.produced_fluids:
                            fluid_name = f"<b>{fluid.name}</b> ({fluid.phase.value})"  # type: ignore
                            fluid_names.append(fluid_name)

                    fluid_info = (
                        "Produced Fluids: " + ", ".join(fluid_names)
                        if fluid_names
                        else "Produced Fluids: N/A"
                    )
                    status = "Open" if well.is_open else "Shut-in"

                    # Calculate appropriate cone size based on grid dimensions
                    # NOTE: Do NOT include z_scale here — the cone direction vector
                    # already contains z_scale via total_arrow_length, and Plotly
                    # computes cone size as sizeref * ||direction||.
                    cone_size_multiplier = 1.0  # Default multiplier for cube mode
                    if cell_dimension is not None:
                        dx, dy = cell_dimension
                        cone_size_multiplier = max(dx, dy) * 0.04 / surface_marker_size

                    # Arrow sizing - base it on actual grid depth spacing
                    if depth_grid is not None:
                        avg_layer_depth = float(np.mean(np.diff(depth_grid, axis=2)))
                        base_arrow_length = avg_layer_depth * 0.25
                    else:
                        base_arrow_length = surface_marker_size * 0.5

                    # Apply z_scale
                    total_arrow_length = base_arrow_length * z_scale

                    # Split: 60% stem, 40% cone
                    stem_length = total_arrow_length * 0.6
                    cone_length = total_arrow_length * 0.4

                    # Add cylindrical stem (arrow shaft) pointing up from surface
                    figure.add_trace(
                        go.Scatter3d(
                            x=[x_surf, x_surf],
                            y=[y_surf, y_surf],
                            z=[z_surf, z_surf + stem_length],
                            mode="lines",
                            line=dict(  # noqa
                                color=color,
                                width=wellbore_width * 0.8,  # Make stem more visible
                            ),
                            opacity=opacity,
                            name=f"{well.name} arrow stem",
                            showlegend=False,
                            legendgroup=well.name,
                            hoverinfo="skip",
                        )
                    )

                    # Add cone (arrowhead) at the end of stem pointing up
                    figure.add_trace(
                        go.Cone(
                            x=[x_surf],
                            y=[y_surf],
                            z=[z_surf + stem_length],  # Position at end of stem
                            u=[0],
                            v=[0],
                            w=[cone_length],  # Arrow pointing UP (production/outflow)
                            sizemode="absolute",
                            sizeref=surface_marker_size * cone_size_multiplier,
                            colorscale=[[0, color], [1, color]],
                            showscale=False,
                            opacity=opacity,
                            name=f"{well.name} surface",
                            showlegend=False,
                            legendgroup=well.name,
                            hovertemplate=(
                                f"<b>{well.name} - Surface Location</b><br>"
                                f"Type: {well_type}<br>"
                                f"Status: {status}<br>"
                                f"{fluid_info}<br>"
                                f"Control: {str(well.control)[:20]}...<br>"
                                f"Wellbore Radius: {well.radius:.2f} ft<br>"
                                f"Skin Factor: {well.skin_factor:.2f}<br>"
                                f"<br>"
                                f"Surface Coords:<br>"
                                f"  X: {x_surf:.1f} ft<br>"
                                f"  Y: {y_surf:.1f} ft<br>"
                                f"  Z: {z_surf:.1f} ft<br>"
                                "<extra></extra>"
                            ),
                        )
                    )

        # Position legend to avoid overlap with colorbar
        # Check if there's a colorbar (most volume/isosurface plots have one on the right)
        has_colorbar = any(
            hasattr(trace, "colorbar")
            and trace.colorbar is not None  # type: ignore[attr-defined]
            or (hasattr(trace, "showscale") and trace.showscale)  # type: ignore[attr-defined]
            for trace in figure.data
        )

        if has_colorbar:
            # Colorbar is typically on the right, position legend on the left
            figure.update_layout(
                legend=dict(  # noqa
                    x=0.01,  # Left side
                    y=0.99,  # Top
                    xanchor="left",
                    yanchor="top",
                    bgcolor="rgba(255, 255, 255, 0.8)",  # Semi-transparent white background
                    bordercolor="gray",
                    borderwidth=1,
                )
            )
        else:
            # No colorbar, use default right side position
            figure.update_layout(
                legend=dict(  # noqa
                    x=0.99,  # Right side
                    y=0.99,  # Top
                    xanchor="right",
                    yanchor="top",
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    bordercolor="gray",
                    borderwidth=1,
                )
            )

    def help(self) -> str:
        """
        Return a help string describing the renderer and its usage.

        :return: Help string
        """
        return f"{self.__class__.__name__} renderer\n{self.render.__doc__ or ''}"


class VolumeRenderer(BaseRenderer):
    """3D Volume renderer for scalar fields."""

    supports_physical_dimensions: typing.ClassVar[bool] = True

    def render(  # type: ignore[override]
        self,
        figure: go.Figure,
        data: ThreeDimensionalGrid,
        metadata: PropertyMeta,
        cell_dimension: typing.Optional[typing.Tuple[float, float]] = None,
        depth_grid: typing.Optional[ThreeDimensionalGrid] = None,
        surface_count: int = 50,
        opacity: typing.Optional[float] = None,
        isomin: typing.Optional[float] = None,
        isomax: typing.Optional[float] = None,
        cmin: typing.Optional[float] = None,
        cmax: typing.Optional[float] = None,
        use_opacity_scaling: typing.Optional[bool] = None,
        aspect_mode: typing.Optional[str] = "auto",
        z_scale: float = 1.0,
        labels: typing.Optional["Labels"] = None,
        auto_coarsen: bool = True,
        **kwargs,
    ) -> go.Figure:
        """
        Render a volume rendering plot.

        **Performance Considerations:**

        Volume rendering is computationally intensive. For optimal performance:

        - **Recommended grid size**: 80x80x80 cells (512K cells) or smaller
        - **Maximum grid size**: 100x100x100 cells (1M cells)
        - **Beyond 1M cells**: Automatic coarsening is applied (can be disabled with auto_coarsen=False)
        - **Memory usage**: ~4 MB per 1M cells (float32)

        For large grids (>1M cells), consider using:

        1. Manual coarsening with `bores.grids.base.coarsen_grid()` before rendering
        2. Slicing to visualize specific regions: `data[::2, ::2, ::2]` (every other cell)
        3. Using `IsosurfaceRenderer` instead (supports up to 2M cells)
        4. Using PyVista `pyvista3d.viz` with `PlotType.CELL_BLOCKS` for cell-level visualization

        :param figure: Plotly figure to add the volume rendering to
        :param data: 3D data array to render
        :param metadata: Property metadata for labeling and scaling
        :param cell_dimension: Physical size of each cell in x and y directions (feet)
        :param depth_grid: 3D array with depth of each cell center (feet, positive downward)
        :param surface_count: Number of isosurfaces to generate for volume rendering
        :param opacity: Opacity of the volume rendering (defaults to config value). Lower values allow better visualization of internal structures.
        :param isomin: Minimum isovalue for isosurface in ORIGINAL data units. Only values >= isomin will be shown.
            For log-scale properties, provide the original value (e.g., 0.1 cP), not log₁₀(0.1).
        :param isomax: Maximum isovalue for isosurface in ORIGINAL data units. Only values <= isomax will be shown.
            For log-scale properties, provide the original value (e.g., 1000 cP), not log₁₀(1000).
        :param cmin: Minimum data value for color mapping in ORIGINAL data units (defaults to data min).
            For log-scale properties, provide the original value (e.g., 0.1 cP), not log₁₀(0.1).
        :param cmax: Maximum data value for color mapping in ORIGINAL data units (defaults to data max).
            For log-scale properties, provide the original value (e.g., 1000 cP), not log₁₀(1000).

        Use cmin/cmax to control value to color mapping in the colorbar.

        :param use_opacity_scaling: Whether to use built-in opacity scaling for better depth perception (defaults to config)
        :param aspect_mode: Aspect mode for the 3D plot (default is "cube"). Could be any of "cube", "auto", or "data".
        :param z_scale: Scale factor for Z-axis (thickness) to make layers appear thicker.
            Values > 1.0 exaggerate vertical thickness, < 1.0 compress it. Default is 1.0 (true scale).
        :param labels: Optional collection of labels to add to the plot
        :param auto_coarsen: Whether to automatically coarsen grids larger than 1M cells for performance (default: True)
        :return: Plotly figure object with volume rendering
        """
        if data.ndim != 3:
            raise ValidationError("Volume rendering requires 3D data")

        # Automatic coarsening for large grids
        coordinate_offsets = kwargs.get("coordinate_offsets", None)
        original_shape = data.shape
        total_cells = np.prod(data.shape)

        if auto_coarsen and total_cells > MAX_VOLUME_CELLS_3D:
            # Calculate optimal coarsening factor
            target_cells = RECOMMENDED_VOLUME_CELLS_3D
            ratio = total_cells / target_cells
            coarsen_factor = int(np.ceil(ratio ** (1.0 / 3.0)))

            logger.warning(
                f"Grid size {data.shape} ({total_cells:,} cells) exceeds recommended limit of 1M cells. "
                f"Automatically coarsening by factor {coarsen_factor} for performance. "
                f"To disable, set auto_coarsen=False."
            )

            # Coarsen the data
            data = coarsen_grid(
                data, batch_size=(coarsen_factor, coarsen_factor, coarsen_factor)
            )

            # Coarsen depth_grid if provided
            if depth_grid is not None:
                depth_grid = coarsen_grid(
                    depth_grid,
                    batch_size=(coarsen_factor, coarsen_factor, coarsen_factor),
                )

            # Update cell dimensions if provided (cells are now larger)
            if cell_dimension is not None:
                cell_dimension = (
                    cell_dimension[0] * coarsen_factor,
                    cell_dimension[1] * coarsen_factor,
                )

            logger.info(
                f"Coarsened grid from {original_shape} to {data.shape} "
                f"({np.prod(data.shape):,} cells, {np.prod(data.shape) / total_cells * 100:.1f}% of original)"
            )

        use_opacity_scaling = (
            use_opacity_scaling
            if use_opacity_scaling is not None
            else self.config.use_opacity_scaling
        )

        # Store original user-provided values in original units
        original_cmin = cmin
        original_cmax = cmax
        original_isomin = isomin
        original_isomax = isomax

        # For volume rendering, we need both normalized (for volume) and display (for colorbar/hover)
        normalized_data, display_data = self.normalize_data(
            data, metadata, normalize_range=False
        )

        # Convert user-provided values to internal processed units
        if metadata.log_scale:
            # Validate that user provided positive values for log scale
            if original_cmin is not None and original_cmin <= 0:
                raise ValidationError("cmin must be > 0 for log scale data")
            if original_cmax is not None and original_cmax <= 0:
                raise ValidationError("cmax must be > 0 for log scale data")
            if original_isomin is not None and original_isomin <= 0:
                raise ValidationError("isomin must be > 0 for log scale data")
            if original_isomax is not None and original_isomax <= 0:
                raise ValidationError("isomax must be > 0 for log scale data")

            # Convert to log₁₀ space for internal use
            cmin = np.log10(original_cmin) if original_cmin is not None else None
            cmax = np.log10(original_cmax) if original_cmax is not None else None
            isomin = np.log10(original_isomin) if original_isomin is not None else None
            isomax = np.log10(original_isomax) if original_isomax is not None else None
        else:
            # For non-log scale, use values as-is
            cmin = original_cmin
            cmax = original_cmax
            isomin = original_isomin
            isomax = original_isomax

        # Get data range for colorbar mapping using ORIGINAL values
        data_min = (
            original_cmin
            if original_cmin is not None
            else float(np.nanmin(display_data))
        )
        data_max = (
            original_cmax
            if original_cmax is not None
            else float(np.nanmax(display_data))
        )

        # Create coordinate grids - use physical coordinates if available
        if cell_dimension is not None and depth_grid is not None:
            coordinate_offsets = kwargs.get("coordinate_offsets", None)
            x_coords, y_coords, z_coords = self.get_physical_coordinates(
                cell_dimension, depth_grid, coordinate_offsets
            )

            # For volume rendering, we need cell centers
            nx, ny, nz = data.shape
            x_centers = (x_coords[:-1] + x_coords[1:]) / 2
            y_centers = (y_coords[:-1] + y_coords[1:]) / 2
            # Calculate Z centers from Z boundaries
            z_centers = np.zeros((nx, ny, nz))
            for i, j, k in itertools.product(range(nx), range(ny), range(nz)):
                z_centers[i, j, k] = (z_coords[i, j, k] + z_coords[i, j, k + 1]) / 2

            # Note: Plotly `Volume` traces require regular rectangular grids
            # We average Z coordinates across X,Y to get a representative profile
            # For full dip visualization, use `pyvista3d.viz` instead
            z_profile = np.mean(z_centers, axis=(0, 1))  # Average across X,Y dimensions

            # Create meshgrid with averaged Z profile (reverse to show k=0 at top)
            x, y, z = np.meshgrid(x_centers, y_centers, z_profile[::-1], indexing="ij")

            x_title = "X Distance (ft)"
            y_title = "Y Distance (ft)"
            z_title = "Z Distance (ft) [averaged profile]"
        else:
            # Fallback to index-based coordinates - reverse Z to show k=0 at top
            nx, ny, nz = data.shape
            x, y, z = np.meshgrid(
                np.arange(nx), np.arange(ny), np.arange(nz)[::-1], indexing="ij"
            )
            x_title = "X Index"
            y_title = "Y Index"
            z_title = "Z Index"

        # Extract index offsets to show original dataset indices in hover text
        coordinate_offsets = kwargs.get("coordinate_offsets", None)
        x_index_offset, y_index_offset, z_index_offset = coordinate_offsets or (0, 0, 0)

        # Create custom hover text with ORIGINAL physical values
        display_values = display_data.flatten()

        # Create hover text
        hover_text = []
        for i, j, k in itertools.product(
            range(data.shape[0]), range(data.shape[1]), range(data.shape[2])
        ):
            flat_index = i * data.shape[1] * data.shape[2] + j * data.shape[2] + k
            # Apply index offsets to show original dataset positions
            absolute_i = x_index_offset + i
            absolute_j = y_index_offset + j
            absolute_k = z_index_offset + k
            # For hover text, map k to show k=0 at top (highest Z coordinate)
            display_k = data.shape[2] - 1 - k
            absolute_display_k = (
                z_index_offset + display_k
            )  # Apply offset to display k as well
            if cell_dimension is not None and depth_grid is not None:
                hover_text.append(
                    f"Cell: ({absolute_i}, {absolute_j}, {absolute_display_k})<br>"  # Show absolute indices
                    f"X: {x.flatten()[flat_index]:.1f} ft<br>"
                    f"Y: {y.flatten()[flat_index]:.1f} ft<br>"
                    f"Z: {z.flatten()[flat_index]:.1f} ft<br>"
                    f"{metadata.display_name}: {_format_value(display_values[flat_index], metadata)} {metadata.unit}"
                    + (" (log scale)" if metadata.log_scale else "")
                )
            else:
                hover_text.append(
                    f"Cell: ({absolute_i}, {absolute_j}, {absolute_k})<br>"  # Show absolute indices for fallback
                    f"X: {x.flatten()[flat_index]:.1f}<br>"
                    f"Y: {y.flatten()[flat_index]:.1f}<br>"
                    f"Z: {z.flatten()[flat_index]:.1f}<br>"
                    f"{metadata.display_name}: {_format_value(display_values[flat_index], metadata)} {metadata.unit}"
                    + (" (log scale)" if metadata.log_scale else "")
                )

        # Use the normalized data for the scale values
        # For log scale, create evenly-spaced values in LOG space for better visual distribution
        if metadata.log_scale and data_min > 0 and data_max > 0:
            # Create values evenly spaced in log space
            log_min = np.log10(data_min)
            log_max = np.log10(data_max)
            log_scale_values = np.linspace(log_min, log_max, num=6)
            scale_values = 10**log_scale_values  # Convert back to original units
        else:
            # For linear scale, use evenly spaced values
            scale_values = np.linspace(data_min, data_max, num=6)

        scale_text = [
            _format_value(value, metadata=metadata)  # type: ignore
            for value in scale_values
        ]
        scale_title = f"{metadata.display_name} ({metadata.unit})" + (
            " - Log Scale" if metadata.log_scale else ""
        )
        colorscale = self.get_colorscale(
            kwargs.get("color_scheme", metadata.color_scheme)
        )
        volume_opacity = opacity if opacity is not None else self.config.opacity
        opacity_scale = (
            self.config.opacity_scale_values if use_opacity_scaling else None
        )
        figure.add_trace(
            go.Volume(
                x=x.flatten(),
                y=y.flatten(),
                z=z.flatten(),
                value=normalized_data.flatten(),
                text=hover_text,
                hovertemplate="%{text}<extra></extra>",
                opacity=volume_opacity,
                surface_count=surface_count,
                colorscale=colorscale,
                showscale=self.config.show_colorbar,
                caps=dict(x_show=True, y_show=True, z_show=True),  # noqa  # Show all 6 faces
                opacityscale=opacity_scale,
                cmin=cmin,  # Use converted values for internal plotting
                cmax=cmax,  # Use converted values for internal plotting
                isomin=isomin,  # Use converted values for internal plotting
                isomax=isomax,  # Use converted values for internal plotting
                lighting=self.config.lighting,
                lightposition=self.config.light_position,
                colorbar=dict(  # noqa
                    title=scale_title,
                    tickmode="array",
                    tickvals=scale_values,
                    ticktext=scale_text,
                )
                if self.config.show_colorbar
                else None,
            )
        )

        # Calculate coordinate ranges for aspect ratio calculation
        x_range = (float(x.min()), float(x.max())) if x.size > 0 else None
        y_range = (float(y.min()), float(y.max())) if y.size > 0 else None
        z_range = (float(z.min()), float(z.max())) if z.size > 0 else None

        self.update_layout(
            figure,
            metadata=metadata,
            x_title=x_title,
            y_title=y_title,
            z_title=z_title,
            aspect_mode=aspect_mode,
            z_scale=z_scale,
            x_range=x_range,
            y_range=y_range,
            z_range=z_range,
        )
        if labels is not None:
            self.apply_labels(
                figure,
                labels,
                data=data,
                metadata=metadata,
                cell_dimension=cell_dimension,
                depth_grid=depth_grid,
                format_kwargs=kwargs.get("format_kwargs", None),
                coordinate_offsets=coordinate_offsets,
            )
        return figure

    def update_layout(
        self,
        figure: go.Figure,
        metadata: PropertyMeta,
        x_title: str = "X Index",
        y_title: str = "Y Index",
        z_title: str = "Z Index",
        aspect_mode: typing.Optional[str] = None,
        z_scale: float = 1.0,
        x_range: typing.Optional[typing.Tuple[float, float]] = None,
        y_range: typing.Optional[typing.Tuple[float, float]] = None,
        z_range: typing.Optional[typing.Tuple[float, float]] = None,
    ):
        """
        Update figure layout with dimensions and scene configuration.

        :param figure: Plotly figure to update
        :param metadata: Property metadata
        :param x_title: Title for X axis
        :param y_title: Title for Y axis
        :param z_title: Title for Z axis
        :param z_scale: Scale factor for Z-axis visual spacing (affects display only, not coordinate values)
        :param x_range: Tuple of (min, max) for x-axis data range
        :param y_range: Tuple of (min, max) for y-axis data range
        :param z_range: Tuple of (min, max) for z-axis data range
        """
        aspect_mode = aspect_mode or self.config.aspect_mode or "auto"

        scene_config = self.get_scene_config(
            x_title=x_title,
            y_title=y_title,
            z_title=z_title,
            aspect_mode=aspect_mode,
            z_scale=z_scale,
            x_range=x_range,
            y_range=y_range,
            z_range=z_range,
        )
        figure.update_layout(
            width=self.config.width,
            height=self.config.height,
            paper_bgcolor=self.config.paper_bgcolor,
            scene=scene_config,
        )


class IsosurfaceRenderer(BaseRenderer):
    """3D Isosurface renderer."""

    supports_physical_dimensions: typing.ClassVar[bool] = True

    def render(  # type: ignore[override]
        self,
        figure: go.Figure,
        data: ThreeDimensionalGrid,
        metadata: PropertyMeta,
        cell_dimension: typing.Optional[typing.Tuple[float, float]] = None,
        depth_grid: typing.Optional[ThreeDimensionalGrid] = None,
        isomin: typing.Optional[float] = None,
        isomax: typing.Optional[float] = None,
        cmin: typing.Optional[float] = None,
        cmax: typing.Optional[float] = None,
        surface_count: int = 50,
        opacity: typing.Optional[float] = None,
        aspect_mode: typing.Optional[str] = "auto",
        z_scale: float = 1.0,
        labels: typing.Optional["Labels"] = None,
        **kwargs,
    ) -> go.Figure:
        """
        Render an isosurface plot.

        :param figure: Plotly figure to add the isosurface to
        :param data: 3D data array to create isosurfaces from
        :param metadata: Property metadata for labeling and scaling
        :param cell_dimension: Physical size of each cell in x and y directions (feet)
        :param depth_grid: 3D array with depth of each cell center (feet, positive downward)
        :param isomin: Minimum isovalue for isosurface in ORIGINAL data units. Only values >= isomin will be shown.
            For log-scale properties, provide the original value (e.g., 0.1 cP), not log₁₀(0.1).
        :param isomax: Maximum isovalue for isosurface in ORIGINAL data units. Only values <= isomax will be shown.
            For log-scale properties, provide the original value (e.g., 1000 cP), not log₁₀(1000).
        :param cmin: Minimum data value for color mapping in ORIGINAL data units (defaults to data min).
            For log-scale properties, provide the original value (e.g., 0.1 cP), not log₁₀(0.1).
        :param cmax: Maximum data value for color mapping in ORIGINAL data units (defaults to data max).
            For log-scale properties, provide the original value (e.g., 1000 cP), not log₁₀(1000).

        Use cmin/cmax to control value to color mapping in the colorbar.

        :param surface_count: Number of isosurfaces to generate
        :param opacity: Opacity of the isosurface (defaults to config value). Lower values allow better visualization of internal structures.
        :param aspect_mode: Aspect mode for the 3D plot (default is "cube"). Could be any of "cube", "auto", or "data".
        :param z_scale: Scale factor for Z-axis (thickness) to make layers appear thicker.
            Values > 1.0 exaggerate vertical thickness, < 1.0 compress it. Default is 1.0 (true scale).
        :param labels: Optional collection of labels to add to the plot
        :return: Plotly figure object with isosurface plot
        """
        if data.ndim != 3:
            raise ValidationError("Isosurface plotting requires 3D data")

        # Store original user-provided values in original units
        original_cmin = cmin
        original_cmax = cmax
        original_isomin = isomin
        original_isomax = isomax

        # Normalized data for isosurface calculation
        normalized_data, display_data = self.normalize_data(
            data, metadata, normalize_range=False
        )

        # Convert user-provided values to internal processed units
        if metadata.log_scale:
            # Validate that user provided positive values for log scale
            if original_cmin is not None and original_cmin <= 0:
                raise ValidationError("cmin must be > 0 for log scale data")
            if original_cmax is not None and original_cmax <= 0:
                raise ValidationError("cmax must be > 0 for log scale data")
            if original_isomin is not None and original_isomin <= 0:
                raise ValidationError("isomin must be > 0 for log scale data")
            if original_isomax is not None and original_isomax <= 0:
                raise ValidationError("isomax must be > 0 for log scale data")

            # Convert to log₁₀ space for internal use
            cmin = np.log10(original_cmin) if original_cmin is not None else None
            cmax = np.log10(original_cmax) if original_cmax is not None else None
            isomin = np.log10(original_isomin) if original_isomin is not None else None
            isomax = np.log10(original_isomax) if original_isomax is not None else None
        else:
            # For non-log scale, use values as-is
            cmin = original_cmin
            cmax = original_cmax
            isomin = original_isomin
            isomax = original_isomax

        # Get data range for colorbar mapping using ORIGINAL values
        data_min = (
            original_cmin
            if original_cmin is not None
            else float(np.nanmin(display_data))
        )
        data_max = (
            original_cmax
            if original_cmax is not None
            else float(np.nanmax(display_data))
        )

        # Create coordinate grids - use physical coordinates if available
        if cell_dimension is not None and depth_grid is not None:
            coordinate_offsets = kwargs.get("coordinate_offsets", None)
            x_coords, y_coords, z_coords = self.get_physical_coordinates(
                cell_dimension=cell_dimension,
                depth_grid=depth_grid,
                coordinate_offsets=coordinate_offsets,
            )

            # For isosurface, we need cell centers
            nx, ny, nz = data.shape
            x_centers = (x_coords[:-1] + x_coords[1:]) / 2
            y_centers = (y_coords[:-1] + y_coords[1:]) / 2
            # Calculate Z centers from Z boundaries
            z_centers = np.zeros((nx, ny, nz))
            for i, j, k in itertools.product(range(nx), range(ny), range(nz)):
                z_centers[i, j, k] = (z_coords[i, j, k] + z_coords[i, j, k + 1]) / 2

            # Note: Plotly Isosurface traces require regular rectangular grids
            # We average Z coordinates across X,Y to get a representative profile
            # For full dip visualization, use pyvista3d.viz instead
            z_profile = np.mean(z_centers, axis=(0, 1))  # Average across X,Y dimensions

            # Create meshgrid with averaged Z profile (reverse to show k=0 at top)
            x, y, z = np.meshgrid(x_centers, y_centers, z_profile[::-1], indexing="ij")

            x_title = "X Distance (ft)"
            y_title = "Y Distance (ft)"
            z_title = "Z Distance (ft) [averaged profile]"
            x_flat = x.flatten()
            y_flat = y.flatten()
            z_flat = z.flatten()

            x_title = "X Distance (ft)"
            y_title = "Y Distance (ft)"
            z_title = "Z Distance (ft)"
        else:
            # Fallback to index-based coordinates
            x = np.arange(data.shape[0])
            y = np.arange(data.shape[1])
            z = np.arange(data.shape[2])[::-1]  # Reverse to put k=0 at top
            x_flat = np.repeat(x, data.shape[1] * data.shape[2])
            y_flat = np.tile(np.repeat(y, data.shape[2]), data.shape[0])
            z_flat = np.tile(z, data.shape[0] * data.shape[1])

            x_title = "X Index"
            y_title = "Y Index"
            z_title = "Z Index"

        # Extract index offsets to show original dataset indices in hover text
        coordinate_offsets = kwargs.get("coordinate_offsets", None)
        x_index_offset, y_index_offset, z_index_offset = coordinate_offsets or (0, 0, 0)

        display_values = display_data.flatten()
        hover_text = []
        for i in range(len(x_flat)):
            # Calculate original dataset indices for this flattened index
            # Need to map back to 3D indices first
            i_3d = i // (data.shape[1] * data.shape[2])
            j_3d = (i % (data.shape[1] * data.shape[2])) // data.shape[2]
            k_3d = i % data.shape[2]

            # Apply index offsets to show original dataset positions
            absolute_i = x_index_offset + i_3d
            absolute_j = y_index_offset + j_3d
            # For k, we need to reverse the index since we reversed Z for visualization
            # k_3d=0 corresponds to the highest Z (top layer in original data)
            # k_3d=nz-1 corresponds to the lowest Z (bottom layer in original data)
            absolute_k = z_index_offset + (data.shape[2] - 1 - k_3d)

            hover_text.append(
                f"Cell: ({absolute_i}, {absolute_j}, {absolute_k})<br>"  # Show absolute indices
                f"X: {x_flat[i]:.2f}<br>"
                f"Y: {y_flat[i]:.2f}<br>"
                f"Z: {z_flat[i]:.2f}<br>"
                f"{metadata.display_name}: {_format_value(display_values[i], metadata)} {metadata.unit}"
                + (" (log scale)" if metadata.log_scale else "")
            )

        # Use the normalized data for the scale values
        # For log scale, create evenly-spaced values in LOG space for better visual distribution
        if metadata.log_scale and data_min > 0 and data_max > 0:
            # Create values evenly spaced in log space
            log_min = np.log10(data_min)
            log_max = np.log10(data_max)
            log_scale_values = np.linspace(log_min, log_max, num=6)
            scale_values = 10**log_scale_values  # Convert back to original units
        else:
            # For linear scale, use evenly spaced values
            scale_values = np.linspace(data_min, data_max, num=6)

        scale_text = [
            _format_value(value, metadata=metadata)  # type: ignore
            for value in scale_values
        ]
        scale_title = f"{metadata.display_name} ({metadata.unit})" + (
            " - Log Scale" if metadata.log_scale else ""
        )
        colorscale = self.get_colorscale(
            kwargs.get("color_scheme", metadata.color_scheme)
        )
        isosurface_opacity = opacity if opacity is not None else self.config.opacity
        figure.add_trace(
            go.Isosurface(
                x=x_flat,
                y=y_flat,
                z=z_flat,
                value=normalized_data.flatten(),
                isomin=isomin,
                isomax=isomax,
                cmin=cmin,
                cmax=cmax,
                opacity=isosurface_opacity,
                surface_count=surface_count,
                colorscale=colorscale,
                showscale=self.config.show_colorbar,
                lighting=self.config.lighting,
                lightposition=self.config.light_position,
                text=hover_text,
                hovertemplate="%{text}<extra></extra>",
                colorbar=dict(  # noqa
                    title=scale_title,
                    tickmode="array",
                    tickvals=scale_values,
                    ticktext=scale_text,
                )
                if self.config.show_colorbar
                else None,
            )
        )

        # Calculate coordinate ranges for aspect ratio calculation
        x_range = (
            (float(x_flat.min()), float(x_flat.max())) if x_flat.size > 0 else None
        )
        y_range = (
            (float(y_flat.min()), float(y_flat.max())) if y_flat.size > 0 else None
        )
        z_range = (
            (float(z_flat.min()), float(z_flat.max())) if z_flat.size > 0 else None
        )

        self.update_layout(
            figure,
            metadata=metadata,
            x_title=x_title,
            y_title=y_title,
            z_title=z_title,
            aspect_mode=aspect_mode,
            z_scale=z_scale,
            x_range=x_range,
            y_range=y_range,
            z_range=z_range,
        )
        if labels is not None:
            self.apply_labels(
                figure,
                labels,
                data=data,
                metadata=metadata,
                cell_dimension=cell_dimension,
                depth_grid=depth_grid,
                format_kwargs=kwargs.get("format_kwargs", None),
                coordinate_offsets=coordinate_offsets,
            )
        return figure

    def update_layout(
        self,
        figure: go.Figure,
        metadata: PropertyMeta,
        x_title: str = "X Index",
        y_title: str = "Y Index",
        z_title: str = "Z Index",
        aspect_mode: typing.Optional[str] = None,
        z_scale: float = 1.0,
        x_range: typing.Optional[typing.Tuple[float, float]] = None,
        y_range: typing.Optional[typing.Tuple[float, float]] = None,
        z_range: typing.Optional[typing.Tuple[float, float]] = None,
    ) -> None:
        """
        Update figure layout with dimensions and scene configuration for isosurface plots.

        :param figure: Plotly figure to update
        :param metadata: Property metadata for title generation
        :param x_title: Title for X axis
        :param y_title: Title for Y axis
        :param z_title: Title for Z axis
        :param aspect_mode: Aspect mode for the 3D plot (default is "cube"). Could be any of "cube", "auto", or "data".
        :param z_scale: Scale factor for Z-axis visual spacing (affects display only, not coordinate values)
        :param x_range: Tuple of (min, max) for x-axis data range
        :param y_range: Tuple of (min, max) for y-axis data range
        :param z_range: Tuple of (min, max) for z-axis data range
        """
        aspect_mode = aspect_mode or self.config.aspect_mode or "auto"
        scene_config = self.get_scene_config(
            x_title=x_title,
            y_title=y_title,
            z_title=z_title,
            aspect_mode=aspect_mode,
            z_scale=z_scale,
            x_range=x_range,
            y_range=y_range,
            z_range=z_range,
        )
        figure.update_layout(
            width=self.config.width,
            height=self.config.height,
            paper_bgcolor=self.config.paper_bgcolor,
            scene=scene_config,
        )


def interpolate_opacity(
    x: np.ndarray, scale_values: typing.Sequence[typing.Sequence[float]]
) -> np.typing.NDArray[np.floating]:
    # scale_values: list of [fraction, opacity]
    fractions, opacities = zip(*scale_values)
    return np.interp(x, fractions, opacities)


class Scatter3DRenderer(BaseRenderer):
    """3D Scatter renderer for sparse data."""

    supports_physical_dimensions = True

    def render(  # type: ignore[override]
        self,
        figure: go.Figure,
        data: ThreeDimensionalGrid,
        metadata: PropertyMeta,
        cell_dimension: typing.Optional[typing.Tuple[float, float]] = None,
        depth_grid: typing.Optional[ThreeDimensionalGrid] = None,
        threshold: float = 0.0,
        sample_rate: float = 1.0,
        marker_size: int = 4,
        cmin: typing.Optional[float] = None,
        cmax: typing.Optional[float] = None,
        opacity: typing.Optional[float] = None,
        aspect_mode: typing.Optional[str] = "auto",
        z_scale: float = 1.0,
        labels: typing.Optional["Labels"] = None,
        **kwargs,
    ) -> go.Figure:
        """
        Render a 3D scatter plot for sparse data visualization.

        :param figure: Plotly figure to add the scatter plot to
        :param data: 3D data array to create scatter plot from
        :param metadata: Property metadata for labeling and scaling
        :param threshold: Value threshold for point inclusion (in normalized range 0.0-1.0)
        :param sample_rate: Fraction of points to sample for performance (0.0-1.0)

        Both threshold and sample_rate are applied to the normalized data to control point inclusion and rendering.
        This can be useful for large datasets where you want to visualize only significant points and reduce rendering load.

        :param marker_size: Size of scatter plot markers
        :param cmin: Minimum data value for color mapping in ORIGINAL data units (defaults to data min).
            For log-scale properties, provide the original value (e.g., 0.1 cP), not log₁₀(0.1).
        :param cmax: Maximum data value for color mapping in ORIGINAL data units (defaults to data max).
            For log-scale properties, provide the original value (e.g., 1000 cP), not log₁₀(1000).

        Use cmin/cmax to control value to color mapping in the colorbar.

        :param opacity: Opacity of the markers (defaults to config value)
        :param aspect_mode: Aspect mode for the 3D plot (default is "auto"). Could be any of "cube", "auto", or "data".
        :param z_scale: Scale factor for Z-axis (thickness) to make layers appear thicker.
            Values > 1.0 exaggerate vertical thickness, < 1.0 compress it. Default is 1.0 (true scale).
        :param labels: Optional collection of labels to add to the plot
        :return: Plotly figure object with 3D scatter plot
        """
        if data.ndim != 3:
            raise ValidationError("Scatter3D plotting requires 3D data")

        # Store original user-provided values in original units
        original_cmin = cmin
        original_cmax = cmax

        # Normalize data for consistent scaling and thresholding
        # Original data will be used for hover text and colorbar
        normalized_data, display_data = self.normalize_data(
            data, metadata, normalize_range=False
        )

        # Convert user-provided values to internal processed units
        if metadata.log_scale:
            # Validate that user provided positive values for log scale
            if original_cmin is not None and original_cmin <= 0:
                raise ValidationError("cmin must be > 0 for log scale data")
            if original_cmax is not None and original_cmax <= 0:
                raise ValidationError("cmax must be > 0 for log scale data")

            # Convert to log₁₀ space for internal use
            cmin = np.log10(original_cmin) if original_cmin is not None else None
            cmax = np.log10(original_cmax) if original_cmax is not None else None
        else:
            # For non-log scale, use values as-is
            cmin = original_cmin
            cmax = original_cmax

        # Get data range for colorbar mapping using ORIGINAL values
        data_min = (
            original_cmin
            if original_cmin is not None
            else float(np.nanmin(display_data))
        )
        data_max = (
            original_cmax
            if original_cmax is not None
            else float(np.nanmax(display_data))
        )

        # Subsample data for performance
        # Get indices where data exceeds threshold
        mask = normalized_data > threshold
        indices = np.where(mask)

        # Subsample
        n_points = len(indices[0])
        if n_points > 10000:  # Limit number of points for performance
            sample_indices = np.random.choice(
                n_points, int(n_points * sample_rate), replace=False
            )
            x_coords = indices[0][sample_indices]
            y_coords = indices[1][sample_indices]
            z_coords_raw = indices[2][sample_indices]
            values = display_data[
                x_coords, y_coords, z_coords_raw
            ]  # Use original values for display
        else:
            x_coords, y_coords, z_coords_raw = indices
            values = display_data[mask]  # Use original values for display

        if values.size == 0:
            raise ValidationError(
                "No data points selected for Scatter3D plot. "
                "Check your threshold, sample rate or data values."
            )

        # Apply numpy convention: z=0 should be at top, z increases downward
        z_coords = data.shape[2] - 1 - z_coords_raw

        if cell_dimension is not None and depth_grid is not None:
            # Convert indices to physical coordinates
            dx, dy = cell_dimension
            coordinate_offsets = kwargs.get("coordinate_offsets", None)
            x_offset, y_offset, _ = coordinate_offsets or (0, 0, 0)

            x_physical = x_offset * dx + x_coords * dx
            y_physical = y_offset * dy + y_coords * dy

            _, _, z_boundaries = self.get_physical_coordinates(
                cell_dimension=cell_dimension,
                depth_grid=depth_grid,
                coordinate_offsets=coordinate_offsets,
            )

            z_physical = np.array(
                [
                    (z_boundaries[x, y, z] + z_boundaries[x, y, z + 1]) / 2
                    for x, y, z in zip(x_coords, y_coords, z_coords)
                ]
            )

            # Extract index offsets to show original dataset indices in hover text
            x_index_offset, y_index_offset, z_index_offset = coordinate_offsets or (
                0,
                0,
                0,
            )

            hover_text = [
                f"Cell: ({x_index_offset + x_coords[i]}, {y_index_offset + y_coords[i]}, {z_index_offset + z_coords[i]})<br>"  # Show absolute indices
                f"X: {x_physical[i]:.1f} ft<br>"
                f"Y: {y_physical[i]:.1f} ft<br>"
                f"Z: {z_physical[i]:.1f} ft<br>"
                f"{metadata.display_name}: {_format_value(v, metadata)}"
                + (" (log scale)" if metadata.log_scale else "")
                for i, v in enumerate(values)
            ]
            # Use physical coordinates for axis titles
            x_title = "X Distance (ft)"
            y_title = "Y Distance (ft)"
            z_title = "Z Distance (ft)"
        else:
            # Use index coordinates directly
            x_physical = x_coords
            y_physical = y_coords
            z_physical = z_coords  # Use reversed z_coords, not z_coords_raw

            # Extract index offsets to show original dataset indices in hover text
            x_index_offset, y_index_offset, z_index_offset = kwargs.get(
                "coordinate_offsets", None
            ) or (0, 0, 0)

            hover_text = [
                f"Cell: ({x_index_offset + x_coords[i]}, {y_index_offset + y_coords[i]}, {z_index_offset + z_coords_raw[i]})<br>"  # Show absolute indices using raw z
                f"{metadata.display_name}: {_format_value(v, metadata)}"
                + (" (log scale)" if metadata.log_scale else "")
                for i, v in enumerate(values)
            ]

            x_title = "X Cell Index"
            y_title = "Y Cell Index"
            z_title = "Z Cell Index"

        # Use the normalized data for the scale values
        # For log scale, create evenly-spaced values in LOG space for better visual distribution
        if metadata.log_scale and data_min > 0 and data_max > 0:
            # Create values evenly spaced in log space
            log_min = np.log10(data_min)
            log_max = np.log10(data_max)
            log_scale_values = np.linspace(log_min, log_max, num=6)
            scale_values = 10**log_scale_values  # Convert back to original units
        else:
            # For linear scale, use evenly spaced values
            scale_values = np.linspace(data_min, data_max, num=6)

        scale_text = [
            _format_value(value, metadata=metadata)  # type: ignore
            for value in scale_values
        ]
        scale_title = f"{metadata.display_name} ({metadata.unit})" + (
            " - Log Scale" if metadata.log_scale else ""
        )
        colorscale = self.get_colorscale(
            kwargs.get("color_scheme", metadata.color_scheme)
        )
        scatter_opacity = opacity if opacity is not None else self.config.opacity
        figure.add_trace(
            go.Scatter3d(
                x=x_physical,
                y=y_physical,
                z=z_physical,
                mode="markers",
                marker=dict(  # noqa
                    size=marker_size,
                    color=values,
                    colorscale=colorscale,
                    opacity=scatter_opacity,
                    cmin=cmin,
                    cmax=cmax,
                    colorbar=dict(  # noqa
                        title=scale_title,
                        tickmode="array",
                        tickvals=scale_values,
                        ticktext=scale_text,
                    )
                    if self.config.show_colorbar
                    else None,
                ),
                text=hover_text,
                hovertemplate="%{text}<extra></extra>",
            )
        )

        # Calculate coordinate ranges for aspect ratio calculation
        x_range = (
            (float(np.min(x_physical)), float(np.max(x_physical)))
            if len(x_physical) > 0
            else None
        )
        y_range = (
            (float(np.min(y_physical)), float(np.max(y_physical)))
            if len(y_physical) > 0
            else None
        )
        z_range = (
            (float(np.min(z_physical)), float(np.max(z_physical)))
            if len(z_physical) > 0
            else None
        )

        self.update_layout(
            figure,
            metadata=metadata,
            aspect_mode=aspect_mode,
            x_title=x_title,
            y_title=y_title,
            z_title=z_title,
            z_scale=z_scale,
            x_range=x_range,
            y_range=y_range,
            z_range=z_range,
        )
        if labels is not None:
            self.apply_labels(
                figure,
                labels,
                data=data,
                metadata=metadata,
                cell_dimension=cell_dimension,
                depth_grid=depth_grid,
                coordinate_offsets=kwargs.get("coordinate_offsets", None),
                format_kwargs=kwargs.get("format_kwargs", None),
            )
        return figure

    def update_layout(
        self,
        figure: go.Figure,
        metadata: PropertyMeta,
        x_title: str = "X Cell Index",
        y_title: str = "Y Cell Index",
        z_title: str = "Z Cell Index",
        aspect_mode: typing.Optional[str] = None,
        z_scale: float = 1.0,
        x_range: typing.Optional[typing.Tuple[float, float]] = None,
        y_range: typing.Optional[typing.Tuple[float, float]] = None,
        z_range: typing.Optional[typing.Tuple[float, float]] = None,
    ) -> None:
        """
        Update figure layout with dimensions and scene configuration for scatter plots.

        :param figure: Plotly figure to update
        :param metadata: Property metadata for title generation
        :param x_title: Title for X axis
        :param y_title: Title for Y axis
        :param z_title: Title for Z axis
        :param aspect_mode: Aspect mode for the 3D plot (default is "auto"). Could be any of "cube", "auto", or "data".
        :param z_scale: Scale factor for Z-axis visual spacing (affects display only, not coordinate values)
        :param x_range: Tuple of (min, max) for x-axis data range
        :param y_range: Tuple of (min, max) for y-axis data range
        :param z_range: Tuple of (min, max) for z-axis data range
        """
        aspect_mode = aspect_mode or self.config.aspect_mode or "auto"
        scene_config = self.get_scene_config(
            x_title=x_title,
            y_title=y_title,
            z_title=z_title,
            aspect_mode=aspect_mode,
            z_scale=z_scale,
            x_range=x_range,
            y_range=y_range,
            z_range=z_range,
        )
        figure.update_layout(
            width=self.config.width,
            height=self.config.height,
            paper_bgcolor=self.config.paper_bgcolor,
            scene=scene_config,
        )


PLOT_TYPE_NAMES: typing.Dict[PlotType, str] = {
    PlotType.VOLUME: "3D Volume",
    PlotType.ISOSURFACE: "3D Isosurface",
    PlotType.SCATTER_3D: "3D Scatter",
    # PlotType.CELL_BLOCKS: "3D Cell Blocks",
}


class DataVisualizer:
    """
    Plotly-based 3D visualizer for three-dimensional (reservoir) data.
    """

    default_dashboard_title: typing.ClassVar[str] = "Model Properties"

    def __init__(
        self,
        config: typing.Optional[PlotConfig] = None,
        registry: typing.Optional[PropertyRegistry] = None,
    ) -> None:
        """
        Initialize the visualizer with optional configuration.

        :param config: Optional configuration for 3D rendering (uses defaults if None)
        :param registry: Optional fluid property registry (uses default if None)
        """
        self._config = config or PlotConfig()
        self._renderers: typing.Dict[PlotType, BaseRenderer] = {
            PlotType.VOLUME: VolumeRenderer(self._config),
            PlotType.ISOSURFACE: IsosurfaceRenderer(self._config),
            PlotType.SCATTER_3D: Scatter3DRenderer(self._config),
        }
        self.registry = registry or property_registry

    @property
    def config(self) -> PlotConfig:
        """Get the current plot configuration."""
        return self._config

    def add_renderer(
        self,
        plot_type: PlotType,
        renderer_type: typing.Type[BaseRenderer],
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> None:
        """
        Add a custom renderer for a specific plot type.

        :param plot_type: The type of plot to add (must be a valid PlotType)
        :param renderer_type: The class implementing the `BaseRenderer` interface
        :param args: Initialization arguments for the renderer class
        :param kwargs: Initialization keyword arguments for the renderer class
        :raises ValidationError: If plot_type is not a valid PlotType
        """
        if not isinstance(plot_type, PlotType):
            raise ValidationError(f"Invalid plot type: {plot_type}")
        self._renderers[plot_type] = renderer_type(self.config, *args, **kwargs)

    def get_renderer(self, plot_type: PlotType) -> BaseRenderer:
        """
        Get the renderer for a specific plot type.

        :param plot_type: The type of plot to get (must be a valid PlotType)
        :return: The renderer instance for the specified plot type
        :raises ValidationError: If plot_type is not a valid PlotType or no renderer is registered
        """
        if not isinstance(plot_type, PlotType):
            raise ValidationError(f"Invalid plot type: {plot_type}")
        renderer = self._renderers.get(plot_type, None)
        if renderer is None:
            raise ValidationError(f"No renderer registered for plot type: {plot_type}")
        return renderer

    def apply_slice(
        self,
        data: ThreeDimensionalGrid,
        x_slice: typing.Optional[
            typing.Union[int, slice, typing.Tuple[int, int]]
        ] = None,
        y_slice: typing.Optional[
            typing.Union[int, slice, typing.Tuple[int, int]]
        ] = None,
        z_slice: typing.Optional[
            typing.Union[int, slice, typing.Tuple[int, int]]
        ] = None,
    ) -> typing.Tuple[ThreeDimensionalGrid, typing.Tuple[slice, slice, slice]]:
        """
        Validate and apply slicing to 3D data, ensuring result is still 3D.

        :param data: Input 3D data array
        :param x_slice: X dimension slice specification
        :param y_slice: Y dimension slice specification
        :param z_slice: Z dimension slice specification
        :return: Tuple of (sliced_data, actual_slices_used)
        :raises ValidationError: If slicing would result in non-3D data
        """
        return slice_grid(data, x_slice=x_slice, y_slice=y_slice, z_slice=z_slice)

    def _get_data(
        self,
        source: typing.Union[ModelState[ThreeDimensions], ReservoirModel],
        name: str,
    ) -> ThreeDimensionalGrid:
        """
        Extract property data from model state or reservoir model.

        Delegates to shared utility function for consistent behavior
        across visualization modules.

        :param source: The model or model state containing the property
        :param name: Property name, supports dot notation (e.g., "permeability.x")
        :return: 3D numpy array containing the property data
        :raises ValidationError: If property name is invalid for the source type
        :raises AttributeError: If property is not found
        :raises TypeError: If property is not a 3-dimensional array
        """
        return get_data(source, name)

    def get_title(
        self,
        plot_type: PlotType,
        metadata: PropertyMeta,
        custom_title: typing.Optional[str] = None,
    ) -> str:
        """
        Smart title determination logic that includes plot type information.

        :param custom_title: Custom title provided by user
        :param plot_type: Type of plot being created
        :param metadata: Data metadata
        :return: Final title to use
        """
        # If the custom title is providd and it is not some kind of suffix (starts with "-"), use it directly
        if custom_title is not None and not custom_title.strip().startswith("-"):
            return custom_title
        if self.config.title:
            # If config has a title, add plot type as subtitle info
            plot_name = PLOT_TYPE_NAMES.get(plot_type, "3D Plot")
            return f"{self.config.title}{custom_title or ''}<br><sub>{plot_name}: {metadata.display_name}</sub>"
        plot_name = PLOT_TYPE_NAMES.get(plot_type, "3D Plot")
        return f"{plot_name}{custom_title or ''}: {metadata.display_name}<br><sub>Interactive 3D Visualization</sub>"

    def make_plot(
        self,
        source: typing.Union[
            ReservoirModel[ThreeDimensions],
            ModelState[ThreeDimensions],
            ThreeDimensionalGrid,
        ],
        property: typing.Optional[str] = None,
        plot_type: typing.Optional[typing.Union[PlotType, str]] = None,
        figure: typing.Optional[go.Figure] = None,
        title: typing.Optional[str] = None,
        width: typing.Optional[int] = None,
        height: typing.Optional[int] = None,
        x_slice: typing.Optional[
            typing.Union[int, slice, typing.Tuple[int, int]]
        ] = None,
        y_slice: typing.Optional[
            typing.Union[int, slice, typing.Tuple[int, int]]
        ] = None,
        z_slice: typing.Optional[
            typing.Union[int, slice, typing.Tuple[int, int]]
        ] = None,
        labels: typing.Optional["Labels"] = None,
        show_wells: bool = False,
        **kwargs: typing.Any,
    ) -> go.Figure:
        """
        Plot a specific model property or raw 3D grid data in 3D with optional data slicing.

        :param source: Either a `ReservoirModel` or `ModelState` containing reservoir model data, or a raw `ThreeDimensionalGrid`
        :param property: Name of the property to plot (from `PropertyRegistry`). Required when source is a ModelState,
            optional when source is a ThreeDimensionalGrid (will use generic metadata if not provided)
        :param plot_type: Type of 3D plot to create (volume, isosurface, scatter)
        :param figure: Optional existing Plotly figure to add to (creates new if None)
        :param title: Custom title for this plot (overrides config title)
        :param width: Custom width for this plot (overrides config width)
        :param height: Custom height for this plot (overrides config height)
        :param x_slice: X dimension slice specification:
            - int: Single index (e.g., 5 for cells[5:6, :, :])
            - tuple: Range (e.g., (10, 20) for cells[10:20, :, :])
            - slice: Full slice object (e.g., slice(10, 20, 2))
            - None: Use full dimension
        :param y_slice: Y dimension slice specification (same format as x_slice)
        :param z_slice: Z dimension slice specification (same format as x_slice)
        :param labels: Optional collection of labels to add to the plot
        :param show_wells: Whether to add well visualizations to the plot (default: False, only works with ModelState)
        :param kwargs: Additional plotting parameters specific to the plot type. Can also include
            well visualization kwargs: injection_color, production_color, shut_in_color,
            wellbore_width, surface_marker_size, show_wellbore, show_surface_marker, show_perforations.
            See WellKwargs TypedDict for details.
        :return: Plotly figure object containing the 3D visualization

        Usage Examples:

        ```python
        # Plot ModelState with property name
        viz.make_plot(state, "pressure", x_slice=(10, 20))

        # Plot raw ThreeDimensionalGrid directly
        grid_data = np.random.rand(10, 10, 5)
        viz.make_plot(grid_data)  # Uses generic metadata
        viz.make_plot(grid_data, property="custom_prop")  # Uses registered property metadata if available

        # Plot with string plot type (converted internally)
        viz.make_plot(state, "pressure", plot_type="volume")
        viz.make_plot(grid_data, plot_type="isosurface")

        # Plot single layer at Z index 5
        viz.make_plot(state, "oil_saturation", z_slice=5)

        # Plot corner section
        viz.make_plot(state, "temperature", x_slice=(0, 25), y_slice=(0, 25), z_slice=(0, 10))

        # Use slice objects for advanced slicing
        viz.make_plot(state, "viscosity", x_slice=slice(10, 50, 2))  # Every 2nd cell

        # Add well visualization with default styling (ModelState only)
        viz.make_plot(state, "pressure", show_wells=True)

        # Customize well visualization using kwargs (ModelState only)
        viz.make_plot(
            state,
            property="oil_saturation",
            show_wells=True,
            injection_color='#ff6b6b',
            production_color='#51cf66',
            wellbore_width=8.0,
            show_surface_marker=True
        )
        ```
        """
        if isinstance(plot_type, str):
            try:
                plot_type = PlotType(plot_type)
            except ValueError:
                raise ValidationError(
                    f"Invalid `plot_type` string: '{plot_type}'. "
                    f"Valid options are: {', '.join(pt.value for pt in PlotType)}"
                )

        # Extract data and metadata based on source type
        is_model_state = isinstance(source, ModelState)
        is_model = isinstance(source, ReservoirModel)

        if is_model_state or is_model:
            # When working with a model or model state, property is required
            if property is None:
                raise ValidationError(
                    "property parameter is required when source is a model or model state"
                )

            metadata = self.registry[property]
            data = self._get_data(source, metadata.name)  # type: ignore

            # Get original cell dimensions and depth grid from model
            if is_model_state:
                cell_dimension = source.model.cell_dimension  # type: ignore
                depth_grid = source.model.get_depth_grid(apply_dip=True)  # type: ignore
            else:
                cell_dimension = source.cell_dimension  # type: ignore
                depth_grid = source.get_depth_grid(apply_dip=True)  # type: ignore
        else:
            # Working with raw `ThreeDimensionalGrid`
            data = source

            # Create or retrieve metadata
            if property is not None and property in self.registry:
                metadata = self.registry[property]
            else:
                # Create generic metadata for raw grid data
                metadata = PropertyMeta(
                    name=property or "data",
                    display_name=property or "Data",
                    unit="",
                    color_scheme=ColorScheme.VIRIDIS,
                )

            # No physical dimensions available for raw grids
            cell_dimension = None
            depth_grid = None

        # Apply slicing if any slice parameters are provided
        coordinate_offsets = None
        if any(s is not None for s in [x_slice, y_slice, z_slice]):
            data, normalized_slices = self.apply_slice(data, x_slice, y_slice, z_slice)  # type: ignore
            x_slice_obj, y_slice_obj, z_slice_obj = normalized_slices

            # Update title to indicate slicing
            slice_info = []
            if x_slice is not None:
                slice_info.append(
                    f"X[{x_slice_obj.start or 0}:{x_slice_obj.stop or data.shape[0]}]"
                )
            if y_slice is not None:
                slice_info.append(
                    f"Y[{y_slice_obj.start or 0}:{y_slice_obj.stop or data.shape[1]}]"
                )
            if z_slice is not None:
                slice_info.append(
                    f"Z[{z_slice_obj.start or 0}:{z_slice_obj.stop or data.shape[2]}]"
                )

            slice_suffix = f" - Slice: {', '.join(slice_info)}"
            if title:
                title += slice_suffix
            else:
                title = slice_suffix  # Will be used by get_title

            # Calculate coordinate offsets for sliced data (cell index offsets)
            coordinate_offsets = (
                x_slice_obj.start or 0,
                y_slice_obj.start or 0,
                z_slice_obj.start or 0,
            )

            # If we sliced the data, we need to slice the depth_grid as well for physical coordinates
            if depth_grid is not None:
                depth_grid = depth_grid[x_slice_obj, y_slice_obj, z_slice_obj]  # type: ignore

        plot_type = plot_type or self.config.plot_type
        renderer = self.get_renderer(plot_type)

        # Pass coordinate offsets to renderers
        if coordinate_offsets is not None:
            kwargs["coordinate_offsets"] = coordinate_offsets

        # Pass labels to renderers that support them
        if labels is not None:
            kwargs["labels"] = labels

        fig = figure or go.Figure()
        # Pass physical dimensions to renderers that support them
        if renderer.supports_physical_dimensions:
            fig = renderer.render(
                fig,
                data,  # type: ignore
                metadata,
                cell_dimension=kwargs.pop("cell_dimension", cell_dimension),
                depth_grid=kwargs.pop("depth_grid", depth_grid),
                **kwargs,
            )
        else:
            fig = renderer.render(fig, data, metadata, **kwargs)  # type: ignore

        # Add well visualization if requested (only for `ModelState` data)
        if show_wells and is_model_state and source.wells_exists():  # type: ignore
            # Extract z_scale for well rendering (default to 1.0 if not specified)
            z_scale = kwargs.get("z_scale", 1.0)

            # Extract well visualization kwargs from the kwargs dict using TypedDict keys
            well_kwargs: WellKwargs = {}
            for key in WellKwargs.__annotations__:
                if key in kwargs:
                    well_kwargs[key] = kwargs.pop(key)  # type: ignore

            # Add wells to the figure using the renderer's method
            logger.debug(
                f"Rendering wells: {len(source.wells.injection_wells)} injection, "  # type: ignore
                f"{len(source.wells.production_wells)} production"  # type: ignore
            )
            renderer.render_wells(
                figure=fig,
                wells=source.wells,  # type: ignore
                cell_dimension=cell_dimension,
                depth_grid=depth_grid,
                coordinate_offsets=coordinate_offsets,
                z_scale=z_scale,
                **well_kwargs,
            )

        final_title = self.get_title(plot_type, metadata, title)

        # Apply final layout updates
        layout_updates: typing.Dict[str, typing.Any] = {"title": final_title}
        if width is not None:
            layout_updates["width"] = width
        if height is not None:
            layout_updates["height"] = height

        fig.update_layout(**layout_updates)
        _register_figure(fig)
        return fig

    def animate(
        self,
        sequence: typing.Union[
            typing.List[ReservoirModel[ThreeDimensions]],
            typing.Sequence[ModelState[ThreeDimensions]],
            typing.Sequence[ThreeDimensionalGrid],
        ],
        property: typing.Optional[str] = None,
        plot_type: typing.Optional[typing.Union[PlotType, str]] = None,
        frame_duration: int = 200,
        step_size: int = 1,
        width: typing.Optional[int] = None,
        height: typing.Optional[int] = None,
        title: typing.Optional[str] = None,
        x_slice: typing.Optional[
            typing.Union[int, slice, typing.Tuple[int, int]]
        ] = None,
        y_slice: typing.Optional[
            typing.Union[int, slice, typing.Tuple[int, int]]
        ] = None,
        z_slice: typing.Optional[
            typing.Union[int, slice, typing.Tuple[int, int]]
        ] = None,
        save: typing.Union[HtmlExporter, str, None] = None,
        labels: typing.Optional["Labels"] = None,
        **kwargs: typing.Any,
    ) -> go.Figure:
        """
        Create an animated plot showing property evolution over time.

        :param sequence: Sequence of `ReservoirModel`s, `ModelState`s or `ThreeDimensionalGrid`s representing time steps
        :param property: Name of the property to animate. Required when sequence contains ModelStates,
            optional when sequence contains raw grids
        :param plot_type: Type of 3D plot for animation frames
        :param frame_duration: Duration of each frame in milliseconds
        :param step_size: Step size to skip frames for performance (1 = every frame)
        :param width: Custom width for the animation
        :param height: Custom height for the animation
        :param title: Custom title for the animation
        :param x_slice: X dimension slice specification:
            - int: Single index (e.g., 5 for cells[5:6, :, :])
            - tuple: Range (e.g., (10, 20) for cells[10:20, :, :])
            - slice: Full slice object (e.g., slice(10, 20, 2))
            - None: Use full dimension
        :param y_slice: Y dimension slice specification (same format as x_slice)
        :param z_slice: Z dimension slice specification (same format as x_slice)
        :param save: Animation exporter. Can be an ``HtmlExporter`` instance or a
            string file path ending in ``.html`` (e.g. ``"animation.html"``).
            If provided, the figure is saved to disk after construction.
        :param labels: Optional collection of labels to add to the animation
        :param kwargs: Additional parameters for individual property plots.
        :return: Animated Plotly figure with time controls

        Examples:
        ```python
        from bores.visualization.plotly3d import viz

        # Save as interactive HTML via string path
        viz.animate(states, "pressure", save="pressure_animation.html")

        # Save with explicit HtmlExporter for fine control
        from bores.visualization.utils import HtmlExporter
        viz.animate(states, "pressure", save=HtmlExporter("out.html", auto_open=True))
        ```
        """
        if not sequence:
            raise ValidationError("No data provided")

        # Determine if we're working with models, model states or raw grids
        is_model_state_sequence = isinstance(sequence[0], ModelState)
        is_model_sequence = isinstance(sequence[0], ReservoirModel)

        if (is_model_state_sequence or is_model_sequence) and property is None:
            raise ValidationError(
                "property parameter is required when sequence contains models or model states"
            )

        # Convert string plot_type to PlotType enum if needed
        if isinstance(plot_type, str):
            try:
                plot_type = PlotType(plot_type)
            except ValueError:
                raise ValidationError(
                    f"Invalid plot_type string: '{plot_type}'. "
                    f"Valid options are: {', '.join(pt.value for pt in PlotType)}"
                )

        # Get metadata
        if is_model_state_sequence or is_model_sequence:
            metadata = self.registry[property]  # type: ignore
        else:
            # For raw grids, create or retrieve metadata
            if property is not None and property in self.registry:
                metadata = self.registry[property]
            else:
                metadata = PropertyMeta(
                    name=property or "data",
                    display_name=property or "Data",
                    unit="",
                    color_scheme=ColorScheme.VIRIDIS,
                )

        plot_type = plot_type or PlotType.VOLUME

        if "cmin" not in kwargs:
            # Add cmin/cmax to kwargs for consistent color mapping across frames
            if is_model_state_sequence or is_model_sequence:
                data_list: typing.List[ThreeDimensionalGrid] = [
                    self._get_data(source, metadata.name)  # type: ignore[arg-type]
                    for source in sequence
                ]
            else:
                data_list = typing.cast(
                    typing.List[ThreeDimensionalGrid], list(sequence)
                )

            cmin = float(np.nanmin([np.nanmin(data) for data in data_list]))
            cmax = float(np.nanmax([np.nanmax(data) for data in data_list]))
            kwargs["cmin"] = cmin
            kwargs["cmax"] = cmax
            logger.debug(f"Animation cmin: {cmin}, cmax: {cmax}")

        # Create base figure from first state/grid
        base_fig = self.make_plot(
            sequence[0],  # type: ignore
            property=property,
            plot_type=plot_type,
            width=width,
            height=height,
            x_slice=x_slice,
            y_slice=y_slice,
            z_slice=z_slice,
            labels=labels,
            **kwargs,
        )

        # Create frames for animation. Ensure we get all states
        frames = []
        state_count = len(sequence)
        for i in range(0, state_count, step_size):
            data_item = sequence[i]
            frame_fig = self.make_plot(
                data_item,  # type: ignore
                property=property,
                plot_type=plot_type,
                width=width,
                height=height,
                x_slice=x_slice,
                y_slice=y_slice,
                z_slice=z_slice,
                labels=labels,
                **kwargs,
            )

            # Create frame title with appropriate time units
            if is_model_state_sequence:
                state = typing.cast(ModelState[ThreeDimensions], data_item)
                if state.time >= 3600:  # More than 1 hour
                    time_str = f"t={state.time / 3600:.2f} hours"
                elif state.time >= 60:  # More than 1 minute
                    time_str = f"t={state.time / 60:.2f} minutes"
                else:
                    time_str = f"t={state.time:.2f} seconds"
            else:
                # For raw grids, just use frame index
                time_str = f"Frame {i}"

            # Extract annotations from the frame figure if they exist
            frame_layout: typing.Dict[str, typing.Any] = {
                "title": f"{metadata.display_name} at {time_str}"
            }

            if labels is not None:
                try:
                    # Extract annotations from the frame figure's scene layout
                    scene_layout = getattr(frame_fig.layout, "scene", None)
                    if scene_layout and hasattr(scene_layout, "annotations"):
                        annotations = list(scene_layout.annotations or [])
                        if annotations:
                            frame_layout["scene"] = {"annotations": annotations}
                            logger.debug(
                                f"Extracted {len(annotations)} annotations for frame {i}"
                            )
                        else:
                            frame_layout["scene"] = {"annotations": []}
                    else:
                        frame_layout["scene"] = {"annotations": []}
                except Exception as exc:
                    logger.warning(
                        f"Failed to extract annotations for frame {i}: {exc}"
                    )
                    frame_layout["scene"] = {"annotations": []}

            frame = go.Frame(
                data=frame_fig.data,
                name=f"frame_{i}",
                layout=frame_layout,
            )
            frames.append(frame)

        # Add frames to figure
        base_fig.frames = frames

        # Set initial annotations for the base figure (from first frame)
        if frames and labels is not None:
            try:
                if scene_layout := getattr(frames[0].layout, "scene", None) is not None:
                    initial_annotations = getattr(scene_layout, "annotations", [])
                    if initial_annotations:
                        base_fig.update_layout(
                            scene=dict(annotations=initial_annotations)  # noqa
                        )
                        logger.debug(
                            f"Set {len(initial_annotations)} initial annotations on base figure"
                        )
            except Exception as exc:
                logger.warning(f"Failed to set initial annotations: {exc}")

        # Set final title for animation
        if title:
            animation_title = title
        elif self.config.title:
            animation_title = self.config.title
        else:
            plot_name = PLOT_TYPE_NAMES.get(plot_type, "3D")
            animation_title = f"{plot_name} Animation: {metadata.display_name}"

        # Add animation controls
        base_fig.update_layout(
            title=animation_title,
            updatemenus=[
                {
                    "buttons": [
                        {
                            "args": [
                                None,
                                {
                                    "frame": {
                                        "duration": frame_duration,
                                        "redraw": True,
                                    },
                                    "fromcurrent": True,
                                },
                            ],
                            "label": "Play",
                            "method": "animate",
                        },
                        {
                            "args": [
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": True},
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                },
                            ],
                            "label": "Pause",
                            "method": "animate",
                        },
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 87},
                    "showactive": False,
                    "type": "buttons",
                    "x": 0.1,
                    "xanchor": "right",
                    "y": 0,
                    "yanchor": "top",
                }
            ],
            sliders=[
                {
                    "active": 0,
                    "yanchor": "top",
                    "xanchor": "left",
                    "currentvalue": {
                        "font": {"size": 18},
                        "prefix": "Time Step:",
                        "visible": True,
                        "xanchor": "right",
                    },
                    "transition": {"duration": 30, "easing": "cubic-in-out"},
                    "pad": {"b": 10, "t": 50},
                    "len": 0.9,
                    "x": 0.1,
                    "y": 0,
                    "steps": [
                        {
                            "args": [
                                [f"frame_{i}"],
                                {
                                    "frame": {
                                        "duration": frame_duration,
                                        "redraw": True,
                                    },
                                    "mode": "immediate",
                                    "transition": {"duration": 300},
                                },
                            ],
                            "label": f"{i}",
                            "method": "animate",
                        }
                        for i in range(len(sequence))
                    ],
                }
            ],
        )
        _register_figure(base_fig)

        if save is not None:
            if isinstance(save, str):
                save = HtmlExporter(save)
            save.write(base_fig)

        return base_fig

    def help(self, plot_type: typing.Optional[PlotType] = None) -> str:
        """
        Print help information about available plot types and their parameters.

        :param plot_type: Specific plot type to get help for (or None for all)
        :return: The help string

        Example:
        ```python
        from bores.visualization.plotly1d import viz, PlotType

        # Get help for all plot types
        print(viz.help())
        """
        if plot_type is not None:
            renderer = self.get_renderer(plot_type)
            return renderer.help()

        help_strings = []
        for pt, renderer in self._renderers.items():
            help_strings.append(f"=== {pt.value} Plot ===\n{renderer.help()}\n")
        return "\n".join(help_strings)


viz = DataVisualizer()
"""Global visualizer instance for 3D plots."""

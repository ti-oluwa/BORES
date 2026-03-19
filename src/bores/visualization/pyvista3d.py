"""
PyVista-based 3D Visualization Suite for Reservoir Simulation Data and Results.

Good drop-in replacement for plotly3d with VTK-powered rendering:
- Single-mesh cell-block rendering (vs. N Plotly traces)
- GPU-accelerated volume rendering (if available)
- Correct structural-dip geometry via ExplicitStructuredGrid
- Interactive slice planes and threshold widgets

Run the command below to add support for `pyvista`
```
pip install "bores-framework[pyvista]"
```
"""

import atexit
import logging
import typing
import weakref
from abc import ABC, abstractmethod
from enum import Enum

import attrs
import numpy as np

from bores._precision import get_dtype
from bores.errors import ValidationError
from bores.grids.utils import coarsen_grid
from bores.models import ReservoirModel
from bores.states import ModelState
from bores.types import ThreeDimensionalGrid, ThreeDimensions
from bores.visualization.base import (
    ColorScheme,
    PropertyMeta,
    PropertyRegistry,
    property_registry,
)
from bores.visualization.config import MAX_VOLUME_CELLS_3D, RECOMMENDED_VOLUME_CELLS_3D
from bores.visualization.plotly3d import (
    DEFAULT_CAMERA_POSITION,
    CameraPosition,
    WellKwargs,
)
from bores.visualization.utils import (
    FrameExporter,
    Labels,
    get_data,
    resolve_exporter,
    slice_grid,
)
from bores.wells import Wells

try:
    import pyvista as pv  # type: ignore[import]
except ImportError:
    pv = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)

if pv is None:
    raise ImportError(
        "PyVista is required for pyvista3d visualization.\n"
        "Install with: pip install 'bores-framework[pyvista]'\n"
        "Or: pip install pyvista trame imageio"
    )


_COLORSCHEME_TO_CMAP: typing.Dict[str, str] = {
    "viridis": "viridis",
    "plasma": "plasma",
    "inferno": "inferno",
    "magma": "magma",
    "cividis": "cividis",
    "turbo": "turbo",
    "rdylbu": "RdYlBu",
    "rdbu": "RdBu",
    "spectral": "Spectral",
    "balance": "RdBu",
    "earth": "gist_earth",
}


def _cmap(scheme: typing.Union[ColorScheme, str]) -> str:
    """
    Convert `ColorScheme` to PyVista colormap name.

    :param scheme: `ColorScheme` enum or string name
    :return: PyVista-compatible colormap name
    """
    return _COLORSCHEME_TO_CMAP.get(str(scheme).lower(), "viridis")


def _normalize_data(
    data: ThreeDimensionalGrid,
    metadata: PropertyMeta,
) -> typing.Tuple[ThreeDimensionalGrid, ThreeDimensionalGrid]:
    """
    Prepare data for visualization by clipping and handling zeros for log-scale.

    For log-scale properties, zeros/negatives are replaced with a small positive
    value so PyVista's ``log_scale=True`` can map colors correctly. The data is
    NOT log-transformed here — PyVista handles log color mapping internally and
    displays original values on the colorbar.

    Returns two arrays: `plot_data` (for coloring) and `display_data`
    (original units for labels/tooltips). Both remain in original scale.

    :param data: Raw 3D data array
    :param metadata: Property metadata specifying transformations
    :return: Tuple of (plot_data, display_data)
    """
    dtype = get_dtype()
    display = data.astype(dtype, copy=True)
    plot = data.astype(dtype, copy=True)

    if metadata.log_scale:
        # Replace zeros/negatives with small positive value for log color mapping
        pos_vals = data[data > 0]
        pos_min = float(np.nanmin(pos_vals)) if pos_vals.size > 0 else 1e-10
        plot = np.where(data <= 0, pos_min * 0.1, data).astype(dtype)

    if metadata.min_val is not None:
        plot = np.clip(plot, metadata.min_val, None).astype(dtype)
        display = np.clip(display, metadata.min_val, None).astype(dtype)
    if metadata.max_val is not None:
        plot = np.clip(plot, None, metadata.max_val).astype(dtype)
        display = np.clip(display, None, metadata.max_val).astype(dtype)

    return (
        typing.cast(ThreeDimensionalGrid, plot),
        typing.cast(ThreeDimensionalGrid, display),
    )


def _scalar_bar_args(
    metadata: PropertyMeta,
    extra: dict[str, typing.Any] | None = None,
) -> dict[str, typing.Any]:
    """
    Build PyVista scalar bar configuration dictionary.

    :param metadata: Property metadata containing display name, unit, and scale info
    :param extra: Optional dictionary to override default scalar bar settings
    :return: Complete scalar bar arguments dict for `pv.Plotter.add_mesh()`
    """
    label = f"{metadata.display_name} ({metadata.unit})"
    if metadata.log_scale:
        label += " (log scale)"

    base: dict[str, typing.Any] = {
        "title": label,
        "n_labels": 6,
        "fmt": "%.3g",
        "title_font_size": 14,
        "label_font_size": 12,
    }
    if extra:
        base.update(extra)
    return base


class PlotType(str, Enum):
    """Available plot types"""

    VOLUME = "volume"
    ISOSURFACE = "isosurface"
    SCATTER_3D = "scatter_3d"
    CELL_BLOCKS = "cell_blocks"


@attrs.frozen
class PlotConfig:
    """
    Configuration for PyVista 3D plots.
    """

    width: int = 1200
    """Plot width in pixels. Affects render window size and screenshot resolution."""

    height: int = 960
    """Plot height in pixels. Affects render window size and screenshot resolution."""

    plot_type: PlotType = PlotType.CELL_BLOCKS
    """Default plot type to use. Options: `VOLUME` (ray-cast), `ISOSURFACE` (contours),
    `SCATTER_3D` (point cloud), `CELL_BLOCKS` (voxel mesh)."""

    color_scheme: ColorScheme = ColorScheme.VIRIDIS
    """Default color scheme for data visualization. Maps scalar values to colors."""

    opacity: float = 1.0
    """Global opacity for rendered objects (0.0 = transparent, 1.0 = opaque)."""

    show_colorbar: bool = True
    """Whether to display scalar colorbar legend showing data value ranges."""

    show_axes: bool = True
    """Whether to display 3D coordinate axes with labels."""

    show_labels: bool = True
    """Whether to display point/cell labels when specified."""

    camera_position: CameraPosition = DEFAULT_CAMERA_POSITION
    """Camera eye position, look-at center, and up vector for initial view."""

    title: str = ""
    """Plot title displayed at top of render window."""

    show_cell_outlines: bool = True
    """Whether to draw cell edges/outlines for `CELL_BLOCKS` renderer."""

    cell_outline_color: str = "#404040"
    """Color for cell edges when show_cell_outlines or show_edges is True."""

    cell_outline_width: float = 1.0
    """Line width for cell edges in pixels."""

    use_opacity_scaling: bool = False
    """Whether to apply opacity scaling based on scalar values (advanced)."""

    aspect_mode: typing.Optional[typing.Literal["cube", "data", "auto"]] = None
    """Aspect ratio mode: 'cube' (equal axes), 'data' (match data scales),
    'auto' (VTK default), None (no aspect constraints)."""

    background_color: str = "white"
    """Background color for render window. Accepts CSS colors, hex, or RGB tuples."""

    off_screen: bool = False
    """Render off-screen without opening window. Required for headless servers/notebooks."""

    smooth_shading: bool = True
    """Enable smooth (Phong) shading for surface meshes. False uses flat shading."""

    show_edges: bool = True
    """Display mesh edges for all renderers. Overrides show_cell_outlines."""

    notebook: bool = False
    """Enable Jupyter notebook mode using trame backend for interactive widgets."""

    n_colors: int = 256
    """Number of discrete colors in colormap. Lower values create banded appearance."""

    enable_picking: bool = True
    """Enable cell/point picking to show information when clicking on mesh.
    Click on cells to see index, coordinates, and scalar values."""

    scalar_bar_args: typing.Optional[typing.Dict[str, typing.Any]] = None
    """
    Custom arguments passed to PyVista's scalar bar configuration.
    Overrides default title, position, formatting, etc.
    """

    enable_interactive: bool = True
    """
    Enable interactive widgets (opacity slider, orthogonal clip planes,
    keyboard shortcuts) for on-screen plots. Ignored for `off_screen` renders.
    """


def _make_image_data(
    data: ThreeDimensionalGrid,
    cell_dimension: typing.Optional[typing.Tuple[float, float]] = None,
    z_scale: float = 1.0,
) -> pv.ImageData:  # type: ignore
    """
    Build PyVista `ImageData` (uniform rectilinear grid) for flat reservoirs.

    :param data: 3D array of cell values with shape (nx, ny, nz)
    :param cell_dimension: Optional (dx, dy) cell spacing in physical units
    :param z_scale: Vertical exaggeration factor for Z-axis
    :return: `pv.ImageData` with `cell_data["values"]` populated
    """
    nx, ny, nz = data.shape
    dx = dy = 1.0
    if cell_dimension is not None:
        dx, dy = cell_dimension
    dz = 1.0 * z_scale

    grid = pv.ImageData()  # type: ignore
    grid.dimensions = (nx + 1, ny + 1, nz + 1)
    grid.origin = (0.0, 0.0, 0.0)
    grid.spacing = (dx, dy, dz)

    grid.cell_data["values"] = data.flatten(order="F")
    return grid


def _make_explicit_grid(
    data: ThreeDimensionalGrid,
    cell_dimension: typing.Tuple[float, float],
    depth_grid: ThreeDimensionalGrid,
    z_scale: float = 1.0,
    coordinate_offsets: typing.Optional[typing.Tuple[int, int, int]] = None,
) -> pv.StructuredGrid:  # type: ignore
    """
    Build PyVista `StructuredGrid` for reservoirs with structural dip.

    Corner-point coordinates computed via NumPy broadcasting (no Python loops).
    Supports sliced grids via `coordinate_offsets` for proper spatial positioning.

    :param data: 3D array of cell values with shape (nx, ny, nz)
    :param cell_dimension: (dx, dy) cell spacing in physical units
    :param depth_grid: 3D array of cell center depths matching data.shape
    :param z_scale: Vertical exaggeration factor for Z-axis
    :param coordinate_offsets: Optional (x_offset, y_offset, z_offset) for sliced grids
    :return: `pv.StructuredGrid` with `cell_data["values"]` populated
    :raises ValidationError: If depth_grid.shape != data.shape
    """
    if depth_grid.shape != data.shape:
        raise ValidationError(
            f"depth_grid shape {depth_grid.shape} != data shape {data.shape}"
        )

    nx, ny, nz = data.shape
    dx, dy = cell_dimension
    x_off, y_off, _ = coordinate_offsets or (0, 0, 0)

    x_bounds = (x_off + np.arange(nx + 1)) * dx
    y_bounds = (y_off + np.arange(ny + 1)) * dy

    z_centers = -depth_grid * z_scale
    z_bounds = np.empty((nx, ny, nz + 1), dtype=z_centers.dtype)

    if nz > 1:
        z_bounds[:, :, 1:-1] = 0.5 * (z_centers[:, :, :-1] + z_centers[:, :, 1:])
        z_bounds[:, :, 0] = z_centers[:, :, 0] - 0.5 * (
            z_centers[:, :, 1] - z_centers[:, :, 0]
        )
        z_bounds[:, :, -1] = z_centers[:, :, -1] + 0.5 * (
            z_centers[:, :, -1] - z_centers[:, :, -2]
        )
    else:
        z_bounds[:, :, 0] = z_centers[:, :, 0] - 0.5
        z_bounds[:, :, 1] = z_centers[:, :, 0] + 0.5

    ic = np.clip(np.arange(nx + 1) - 1, 0, nx - 1)
    jc = np.clip(np.arange(ny + 1) - 1, 0, ny - 1)

    # Map cell-center Z values to corner nodes using nearest-cell approach
    z_corner = z_bounds[
        ic[:, None, None],
        jc[None, :, None],
        np.arange(nz + 1)[None, None, :],
    ]

    # Create meshgrid for X, Y coordinates
    x_grid, y_grid = np.meshgrid(x_bounds, y_bounds, indexing="ij")

    # Broadcast to 3D (replicate for each Z layer)
    x_3d = np.repeat(x_grid[:, :, np.newaxis], nz + 1, axis=2)
    y_3d = np.repeat(y_grid[:, :, np.newaxis], nz + 1, axis=2)

    grid = pv.StructuredGrid(x_3d, y_3d, z_corner)  # type: ignore
    grid.cell_data["values"] = data.flatten(order="F")
    return grid


def _render_wells(
    plotter: pv.Plotter,  # type: ignore
    wells: Wells[ThreeDimensions],
    cell_dimension: typing.Optional[typing.Tuple[float, float]],
    depth_grid: typing.Optional[ThreeDimensionalGrid],
    coordinate_offsets: typing.Optional[typing.Tuple[int, int, int]],
    z_scale: float = 1.0,
    **kwargs: typing.Any,
) -> None:
    """
    Render all wells with wellbore trajectories, casing, and surface markers on the plotter.

    Renders complete well visualization including:
    - Wellbore trajectories through perforations (colored by well type)
    - Casing segments from surface to first perforation (dotted gray)
    - Surface location markers with directional arrows pointing into reservoir (both point downward)

    :param plotter: PyVista plotter to add well meshes to
    :param wells: Collection of injection and production wells
    :param cell_dimension: Optional (dx, dy) for physical coordinates
    :param depth_grid: Optional depth array for structural dip
    :param coordinate_offsets: Optional (x_off, y_off, z_off) for sliced grids
    :param z_scale: Vertical exaggeration factor
    :param kwargs: Well rendering options:
        - show_wellbore: Show perforated intervals (default: True)
        - show_surface_marker: Show surface arrows (default: True)
        - show_well_labels: Show well name labels at surface (default: True)
        - injection_color: Injection well color (default: "#ff4444")
        - production_color: Production well color (default: "#44dd44")
        - shut_in_color: Shut-in well color (default: "#888888")
        - wellbore_width: Line width (default: 11.0)
        - surface_marker_size: Arrow thickness multiplier (default: 2.4)
    """
    if wells is None or not wells.exists():
        return

    # Extract kwargs with defaults
    show_wellbore = kwargs.get("show_wellbore", True)
    show_surface_marker = kwargs.get("show_surface_marker", True)
    show_well_labels = kwargs.get("show_well_labels", True)
    injection_color = _normalize_hex_color(kwargs.get("injection_color", "#ff4444"))
    production_color = _normalize_hex_color(kwargs.get("production_color", "#44dd44"))
    shut_in_color = _normalize_hex_color(kwargs.get("shut_in_color", "#888888"))
    line_width = float(kwargs.get("wellbore_width", 11.0))
    surface_marker_size = float(kwargs.get("surface_marker_size", 2.4))

    x_off, y_off, z_off = coordinate_offsets or (0, 0, 0)

    def _to_physical(
        i: int, j: int, k: int
    ) -> typing.Optional[typing.Tuple[float, float, float]]:
        """Convert grid indices to physical coordinates (centered in cells)."""
        if cell_dimension is None or depth_grid is None:
            return float(i) + 0.5, float(j) + 0.5, float(k) + 0.5
        dx, dy = cell_dimension
        i_sl, j_sl, k_sl = i - x_off, j - y_off, k - z_off
        if not (
            0 <= i_sl < depth_grid.shape[0]
            and 0 <= j_sl < depth_grid.shape[1]
            and 0 <= k_sl < depth_grid.shape[2]
        ):
            return None
        d = float(depth_grid[i_sl, j_sl, k_sl])
        return (
            None
            if np.isnan(d)
            else (float((i + 0.5) * dx), float((j + 0.5) * dy), float(-d * z_scale))
        )

    # Calculate surface Z coordinate
    z_surface = float(-z_off) if depth_grid is not None else 0.0

    # Calculate arrow length for surface markers (proportional to grid scale)
    if depth_grid is not None and depth_grid.size > 0:
        avg_layer_depth = float(np.mean(np.abs(np.diff(depth_grid, axis=2))))
        arrow_length = avg_layer_depth * 3.0 * z_scale
    else:
        arrow_length = 15.0
    # Ensure a visible minimum
    arrow_length = max(arrow_length, 5.0)

    # Process injection wells
    for well in wells.injection_wells:
        color = shut_in_color if well.is_shut_in else injection_color
        opacity = 0.5 if well.is_shut_in else 1.0

        if not well.perforating_intervals:
            continue

        # Get first perforation for surface connection
        first_perf_start = well.perforating_intervals[0][0]
        first_perf_coords = _to_physical(*first_perf_start)

        if first_perf_coords is None:
            continue

        x_surf, y_surf, z_perf = first_perf_coords

        # Render casing (surface to first perforation)
        if show_wellbore and cell_dimension is not None:
            casing_pts = np.array(
                [[x_surf, y_surf, z_surface], [x_surf, y_surf, z_perf]]
            )
            casing = pv.PolyData(casing_pts)  # type: ignore
            casing.lines = np.array([2, 0, 1])
            plotter.add_mesh(
                casing,
                color="#999999",
                line_width=line_width * 0.5,
                style="wireframe",
                opacity=0.6,
            )

        # Render perforating intervals
        if show_wellbore:
            perforation_points = []
            perforation_cells = []

            for (si, sj, sk), (ei, ej, ek) in well.perforating_intervals:
                sc = _to_physical(si, sj, sk)
                ec = _to_physical(ei, ej, ek)
                if sc is None or ec is None:
                    continue

                # Extend single-cell perforations vertically
                if (si, sj, sk) == (ei, ej, ek):
                    ext = 5.0 * z_scale if cell_dimension is not None else 0.5
                    sc = (sc[0], sc[1], sc[2] + ext)
                    ec = (ec[0], ec[1], ec[2] - ext)

                base = len(perforation_points)
                perforation_points.extend([sc, ec])
                perforation_cells.extend([2, base, base + 1])

            if perforation_points:
                perforation_mesh = pv.PolyData(  # type: ignore
                    np.array(perforation_points, dtype=float)
                )
                perforation_mesh.lines = np.array(perforation_cells, dtype=int)
                plotter.add_mesh(
                    perforation_mesh,
                    color=color,
                    line_width=line_width,
                    render_lines_as_tubes=True,
                    opacity=opacity,
                )

        # Render surface marker (arrow pointing down)
        if show_surface_marker:
            arrow_top = z_surface + arrow_length
            arrow = pv.Arrow(  # type: ignore
                start=(x_surf, y_surf, arrow_top),
                direction=(0.0, 0.0, -1.0),
                scale=arrow_length,
                shaft_radius=0.04 * surface_marker_size,
                tip_radius=0.08 * surface_marker_size,
                tip_length=0.25,
            )
            plotter.add_mesh(arrow, color=color, opacity=opacity, smooth_shading=False)

        # Render well name label with info
        if show_well_labels and hasattr(well, "name") and well.name:
            label_z = z_surface + arrow_length * 1.2
            well_type = "Injection (Shut-in)" if well.is_shut_in else "Injection"
            parts = [well.name, well_type]
            if hasattr(well, "radius"):
                parts.append(f"Radius={well.radius:.2f} ft")
            if hasattr(well, "skin_factor"):
                parts.append(f"Skin={well.skin_factor:.2f}")
            if hasattr(well, "control") and well.control is not None:
                parts.append(str(well.control)[:100])
            label_text = "\n".join(parts)
            plotter.add_point_labels(
                [(x_surf, y_surf, label_z)],
                [label_text],
                font_size=10,
                text_color=color,
                bold=True,
                show_points=False,
                always_visible=True,
            )

    # Process production wells
    for well in wells.production_wells:
        color = shut_in_color if well.is_shut_in else production_color
        opacity = 0.5 if well.is_shut_in else 1.0

        if not well.perforating_intervals:
            continue

        # Get first perforation for surface connection
        first_perf_start = well.perforating_intervals[0][0]
        first_perf_coords = _to_physical(*first_perf_start)

        if first_perf_coords is None:
            continue

        x_surf, y_surf, z_perf = first_perf_coords

        # Render casing (surface to first perforation)
        if show_wellbore and cell_dimension is not None:
            casing_pts = np.array(
                [[x_surf, y_surf, z_surface], [x_surf, y_surf, z_perf]]
            )
            casing = pv.PolyData(casing_pts)  # type: ignore
            casing.lines = np.array([2, 0, 1])
            plotter.add_mesh(
                casing,
                color="#999999",
                line_width=line_width * 0.5,
                style="wireframe",
                opacity=0.6,
            )

        # Render perforating intervals
        if show_wellbore:
            perforation_points: typing.List = []
            perforation_cells: typing.List[int] = []

            for (si, sj, sk), (ei, ej, ek) in well.perforating_intervals:
                sc = _to_physical(si, sj, sk)
                ec = _to_physical(ei, ej, ek)
                if sc is None or ec is None:
                    continue

                # Extend single-cell perforations vertically
                if (si, sj, sk) == (ei, ej, ek):
                    ext = 5.0 * z_scale if cell_dimension is not None else 0.5
                    sc = (sc[0], sc[1], sc[2] + ext)
                    ec = (ec[0], ec[1], ec[2] - ext)

                base = len(perforation_points)
                perforation_points.extend([sc, ec])
                perforation_cells.extend([2, base, base + 1])

            if perforation_points:
                perforation_mesh = pv.PolyData(  # type: ignore
                    np.array(perforation_points, dtype=float)
                )
                perforation_mesh.lines = np.array(perforation_cells, dtype=int)
                plotter.add_mesh(
                    perforation_mesh,
                    color=color,
                    line_width=line_width,
                    render_lines_as_tubes=True,
                    opacity=opacity,
                )

        # Render surface marker (arrow outward of reservoir)
        if show_surface_marker:
            arrow_top = z_surface + arrow_length
            arrow = pv.Arrow(  # type: ignore
                start=(x_surf, y_surf, z_surface),
                direction=(0.0, 0.0, 1.0),
                scale=arrow_length,
                shaft_radius=0.04 * surface_marker_size,
                tip_radius=0.08 * surface_marker_size,
                tip_length=0.25,
            )
            plotter.add_mesh(arrow, color=color, opacity=opacity, smooth_shading=False)

        # Render well name label with info
        if show_well_labels and hasattr(well, "name") and well.name:
            label_z = z_surface + arrow_length * 1.2
            well_type = "Production (Shut-in)" if well.is_shut_in else "Production"
            parts = [well.name, well_type]
            if hasattr(well, "radius"):
                parts.append(f"Radius={well.radius:.2f} ft")
            if hasattr(well, "skin_factor"):
                parts.append(f"Skin={well.skin_factor:.2f}")
            if hasattr(well, "control") and well.control is not None:
                parts.append(str(well.control)[:150])
            label_text = "\n".join(parts)
            plotter.add_point_labels(
                [(x_surf, y_surf, label_z)],
                [label_text],
                font_size=10,
                text_color=color,
                bold=True,
                show_points=False,
                always_visible=True,
            )


def _normalize_hex_color(color: str) -> str:
    """
    Normalize hex color to 6-digit format for PyVista compatibility.

    PyVista requires full 6-digit hex colors (#RRGGBB), but doesn't support
    CSS-style 3-digit shorthand (#RGB). This function expands 3-digit codes.

    :param color: Color string (may be 3-digit hex, 6-digit hex, or named color)
    :return: Normalized color string (6-digit hex if input was hex)

    Examples:
    ```python
    _normalize_hex_color("#333")      # Returns "#333333"
    _normalize_hex_color("#abc")      # Returns "#aabbcc"
    _normalize_hex_color("#FF5500")   # Returns "#FF5500" (unchanged)
    _normalize_hex_color("white")     # Returns "white" (unchanged)
    ```
    """
    if isinstance(color, str) and color.startswith("#"):
        # Remove the '#' prefix
        hex_code = color[1:]

        # If it's a 3-digit hex code, expand to 6 digits
        if len(hex_code) == 3:
            return f"#{hex_code[0]}{hex_code[0]}{hex_code[1]}{hex_code[1]}{hex_code[2]}{hex_code[2]}"

    # Return as-is for 6-digit hex or named colors
    return color


_active_plotters: typing.List[weakref.ref] = []


def _register_plotter(plotter: pv.Plotter) -> None:  # type: ignore
    """
    Track active plotter for automatic GPU resource cleanup.

    Uses weakref to avoid preventing garbage collection.

    :param plotter: PyVista Plotter instance to track
    """
    _active_plotters.append(weakref.ref(plotter, _on_plotter_deleted))


def _on_plotter_deleted(weak_reference: weakref.ref) -> None:
    """
    Weakref callback to remove dead reference from registry.

    :param weak_reference: `weakref.ref` object being deleted
    """
    try:
        _active_plotters.remove(weak_reference)
    except ValueError:
        pass


def cleanup_resources() -> None:
    """
    Close all active plotters and free GPU/VTK resources.

    Safe to call even if plotters are already closed. Useful for cleaning up
    after batch operations or before terminating interactive sessions.

    Example:
    ```python
    from bores.visualization.pyvista3d import cleanup_resources

    # Create many plots...
    cleanup_resources()  # Free all GPU memory
    ```
    """
    for weak_reference in _active_plotters[:]:
        plotter = weak_reference()
        if plotter is not None:
            try:
                plotter.close()
            except (RuntimeError, AttributeError) as exc:
                logger.debug(f"Failed to close plotter: {exc}")
    _active_plotters.clear()


atexit.register(cleanup_resources)


class BaseRenderer(ABC):
    """
    Abstract base class for PyVista 3D renderers.

    Provides common infrastructure for plotter creation, grid building,
    and rendering pipeline. Subclasses implement specific visualization
    techniques (volume rendering, isosurfaces, etc.).
    """

    supports_physical_dimensions: typing.ClassVar[bool] = False

    def __init__(self, config: PlotConfig) -> None:
        """
        Initialize renderer with configuration.

        :param config: Plot configuration specifying dimensions, colors, options
        """
        self.config = config

    def _plotter(self, title: str = "") -> pv.Plotter:  # type: ignore
        """
        Create and configure PyVista Plotter instance.

        :param title: Window title (uses config.title if empty)
        :return: Configured `pv.Plotter` ready for mesh addition
        """
        if self.config.notebook:
            pv.set_jupyter_backend("trame")  # type: ignore

        plotter = pv.Plotter(  # type: ignore
            off_screen=self.config.off_screen,
            window_size=[self.config.width, self.config.height],
            title=title or self.config.title or "BORES 3D Visualizer",
        )
        plotter.background_color = _normalize_hex_color(self.config.background_color)  # type: ignore
        if self.config.show_axes:
            plotter.add_axes()  # type: ignore

        _register_plotter(plotter)
        return plotter

    def _build_grid(
        self,
        plot_data: ThreeDimensionalGrid,
        cell_dimension: typing.Optional[typing.Tuple[float, float]],
        depth_grid: typing.Optional[ThreeDimensionalGrid],
        z_scale: float,
        coordinate_offsets: typing.Optional[typing.Tuple[int, int, int]],
    ) -> typing.Union[pv.ImageData, pv.StructuredGrid]:  # type: ignore
        """
        Build appropriate PyVista grid type based on geometry.

        Automatically selects `StructuredGrid` for dipping structures
        or `ImageData` for flat/uniform grids.

        :param plot_data: Normalized 3D data array
        :param cell_dimension: Optional (dx, dy) cell spacing
        :param depth_grid: Optional depth array.
        :param z_scale: Vertical exaggeration factor
        :param coordinate_offsets: Optional (x, y, z) offsets for sliced grids
        :return: `pv.ImageData` or `pv.StructuredGrid`
        """
        if depth_grid is not None and cell_dimension is not None:
            return _make_explicit_grid(
                plot_data,
                cell_dimension,
                depth_grid,
                z_scale=z_scale,
                coordinate_offsets=coordinate_offsets,
            )
        return _make_image_data(plot_data, cell_dimension, z_scale=z_scale)

    @abstractmethod
    def render(
        self,
        data: ThreeDimensionalGrid,
        metadata: PropertyMeta,
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> pv.Plotter:  # type: ignore
        """Render data and return configured `Plotter`."""
        ...

    def help(self) -> str:
        """
        Return help text for this renderer.

        :return: Renderer name and render method docstring
        """
        doc = self.render.__doc__ or ""
        return f"{self.__class__.__name__}\n{doc}"


class VolumeRenderer(BaseRenderer):
    """
    GPU ray-cast volume rendering via `pv.Plotter.add_volume(...)`.

    Auto-coarsens grids exceeding `MAX_VOLUME_CELLS_3D`.

    Falls back to `fixed_point` CPU mapper if GPU unavailable.
    """

    supports_physical_dimensions = True

    def render(  # type: ignore[override]
        self,
        data: ThreeDimensionalGrid,
        metadata: PropertyMeta,
        cell_dimension: typing.Optional[typing.Tuple[float, float]] = None,
        depth_grid: typing.Optional[ThreeDimensionalGrid] = None,
        opacity: typing.Optional[float] = None,
        cmin: typing.Optional[float] = None,
        cmax: typing.Optional[float] = None,
        z_scale: float = 1.0,
        auto_coarsen: bool = True,
        volume_mapper: str = "smart",
        shade: bool = False,
        title: str = "",
        coordinate_offsets: typing.Optional[typing.Tuple[int, int, int]] = None,
        **kwargs: typing.Any,
    ) -> pv.Plotter:  # type: ignore
        """
        GPU-accelerated ray-cast volume rendering.

        Uses VTK volume mapper for direct volume rendering. Automatically coarsens
        large grids to maintain interactive performance.

        **Performance Notes:**
        - **GPU required**: Volume rendering is very GPU-intensive. Without a dedicated
          GPU, interaction will be laggy regardless of settings.
        - **CPU alternatives**: For CPU-only systems, use `CELL_BLOCKS` or `ISOSURFACE`
          renderers instead for better performance.
        - **Mapper options**:
            - `"smart"` (default): Auto-selects best available (GPU if available)
            - `"gpu"`: Force GPU ray-casting (fails if GPU unavailable)
            - `"fixed_point"`: CPU ray-casting (slower but works everywhere)
        - **Coarsening**: Set `auto_coarsen=True` (default) to reduce grid size
        - **Shading**: Disable shading (`shade=False`, default) for faster rendering

        :param data: 3D array of scalar values
        :param metadata: Property metadata (name, unit, color scheme, log scale)
        :param cell_dimension: Optional (dx, dy) cell spacing for physical units
        :param depth_grid: Optional depth array for structural dip
        :param opacity: Override global opacity (0.0-1.0)
        :param cmin: Minimum value for colormap in original data units
        :param cmax: Maximum value for colormap in original data units
        :param z_scale: Vertical exaggeration factor (default 1.0)
        :param auto_coarsen: Coarsen grids exceeding `MAX_VOLUME_CELLS_3D` (default: True)
        :param volume_mapper: Volume mapper type: "smart", "gpu", or "fixed_point" (default: "smart")
        :param shade: Enable shading (slower but prettier) (default: False)
        :param title: Plot title (overrides config.title)
        :param coordinate_offsets: (x, y, z) offsets for sliced grids
        :param kwargs: Additional rendering options (color_scheme, n_colors, etc.)
        :return: Configured `pv.Plotter` ready for `.show()` or `.screenshot()`
        :raises ValidationError: If data is not 3-D or invalid `volume_mapper`
        """
        if data.ndim != 3:
            raise ValidationError("Volume renderer requires 3-D data")

        valid_mappers = ("smart", "gpu", "fixed_point")
        if volume_mapper not in valid_mappers:
            raise ValidationError(
                f"Invalid volume_mapper '{volume_mapper}'. "
                f"Must be one of: {', '.join(valid_mappers)}"
            )

        total = int(np.prod(data.shape))
        if auto_coarsen and total > MAX_VOLUME_CELLS_3D:
            factor = int(np.ceil((total / RECOMMENDED_VOLUME_CELLS_3D) ** (1 / 3)))
            logger.warning(
                "Grid %s (%d cells) — coarsening x%d for volume rendering",
                data.shape,
                total,
                factor,
            )
            t = (factor,) * 3
            data = coarsen_grid(data, batch_size=t)
            if depth_grid is not None:
                depth_grid = coarsen_grid(depth_grid, batch_size=t)
            if cell_dimension is not None:
                cell_dimension = (
                    cell_dimension[0] * factor,
                    cell_dimension[1] * factor,
                )

        plot_data, _ = _normalize_data(data, metadata)
        grid = self._build_grid(
            plot_data, cell_dimension, depth_grid, z_scale, coordinate_offsets
        )

        plotter = self._plotter(title)
        clim = (cmin, cmax) if cmin is not None and cmax is not None else None

        # Use fewer colors for volume rendering by default (faster)
        n_colors = kwargs.get("n_colors", min(self.config.n_colors, 128))

        add_kwargs: typing.Dict[str, typing.Any] = {
            "scalars": "values",
            "opacity": opacity if opacity is not None else self.config.opacity,
            "cmap": _cmap(kwargs.get("color_scheme", metadata.color_scheme)),
            "show_scalar_bar": self.config.show_colorbar,
            "scalar_bar_args": _scalar_bar_args(metadata, self.config.scalar_bar_args),
            "n_colors": n_colors,
            "shade": shade,
            "log_scale": metadata.log_scale,
        }
        if clim is not None:
            add_kwargs["clim"] = clim

        # Apply volume mapper selection
        if volume_mapper == "smart":
            # Try GPU first, fallback to CPU if unavailable
            add_kwargs["mapper"] = "smart_gpu"
            try:
                plotter.add_volume(grid, **add_kwargs)
                logger.debug("Volume rendering using smart GPU mapper")
            except (RuntimeError, ValueError):
                logger.info(
                    "GPU volume rendering unavailable, falling back to CPU ('fixed_point'). "
                    "Performance will be limited without GPU acceleration."
                )
                add_kwargs["mapper"] = "fixed_point"
                plotter.add_volume(grid, **add_kwargs)
        elif volume_mapper == "gpu":
            add_kwargs["mapper"] = "gpu"
            plotter.add_volume(grid, **add_kwargs)
            logger.debug("Volume rendering using GPU mapper")
        else:  # fixed_point
            add_kwargs["mapper"] = "fixed_point"
            plotter.add_volume(grid, **add_kwargs)
            logger.info(
                "Volume rendering using CPU (fixed_point) - may be slow without GPU"
            )

        return plotter


class IsosurfaceRenderer(BaseRenderer):
    """
    Isosurface extraction via VTK Marching-Cubes (mesh.contour()).

    `cell_data_to_point_data` is a single VTK filter pass.
    """

    supports_physical_dimensions = True

    def render(  # type: ignore[override]
        self,
        data: ThreeDimensionalGrid,
        metadata: PropertyMeta,
        cell_dimension: typing.Optional[typing.Tuple[float, float]] = None,
        depth_grid: typing.Optional[ThreeDimensionalGrid] = None,
        isomin: typing.Optional[float] = None,
        isomax: typing.Optional[float] = None,
        cmin: typing.Optional[float] = None,
        cmax: typing.Optional[float] = None,
        surface_count: int = 20,
        opacity: typing.Optional[float] = None,
        z_scale: float = 1.0,
        title: str = "",
        coordinate_offsets: typing.Optional[typing.Tuple[int, int, int]] = None,
        **kwargs: typing.Any,
    ) -> pv.Plotter:  # type: ignore
        """
        Extract and render isosurface contours via VTK Marching Cubes.

        Creates smooth surfaces at specified data values. Ideal for visualizing
        boundaries, fronts, or specific property thresholds.

        :param data: 3D array of scalar values
        :param metadata: Property metadata (name, unit, color scheme, log scale)
        :param cell_dimension: Optional (dx, dy) cell spacing
        :param depth_grid: Optional depth array for structural dip
        :param isomin: Minimum isosurface value (original units, defaults to data min)
        :param isomax: Maximum isosurface value (original units, defaults to data max)
        :param cmin: Colormap min value (can differ from isomin)
        :param cmax: Colormap max value (can differ from isomax)
        :param surface_count: Number of evenly-spaced isosurface levels
        :param opacity: Override global opacity (0.0-1.0)
        :param z_scale: Vertical exaggeration factor
        :param title: Plot title
        :param coordinate_offsets: (x, y, z) offsets for sliced grids
        :param kwargs: Additional options (color_scheme, etc.)
        :return: Configured `pv.Plotter`
        :raises ValidationError: If data is not 3-D
        """
        if data.ndim != 3:
            raise ValidationError("Isosurface renderer requires 3-D data")

        plot_data, _ = _normalize_data(data, metadata)
        grid = self._build_grid(
            plot_data, cell_dimension, depth_grid, z_scale, coordinate_offsets
        )
        grid_pts = grid.cell_data_to_point_data()

        low = isomin if isomin is not None else float(np.nanmin(plot_data))
        high = isomax if isomax is not None else float(np.nanmax(plot_data))

        contours = grid_pts.contour(
            isosurfaces=np.linspace(low, high, surface_count),  # type: ignore
            scalars="values",
        )

        plotter = self._plotter(title)
        clim = (cmin, cmax) if cmin is not None and cmax is not None else None

        add_kwargs: typing.Dict[str, typing.Any] = {
            "scalars": "values",
            "opacity": opacity if opacity is not None else self.config.opacity,
            "cmap": _cmap(kwargs.get("color_scheme", metadata.color_scheme)),
            "smooth_shading": self.config.smooth_shading,
            "show_scalar_bar": self.config.show_colorbar,
            "scalar_bar_args": _scalar_bar_args(metadata, self.config.scalar_bar_args),
            "n_colors": self.config.n_colors,
            "log_scale": metadata.log_scale,
        }
        if clim is not None:
            add_kwargs["clim"] = clim

        plotter.add_mesh(contours, **add_kwargs)
        return plotter


class CellBlockRenderer(BaseRenderer):
    """
    Renders entire grid as single voxel mesh.
    """

    supports_physical_dimensions = True

    def render(  # type: ignore[override]
        self,
        data: ThreeDimensionalGrid,
        metadata: PropertyMeta,
        cell_dimension: typing.Tuple[float, float] = (1.0, 1.0),
        depth_grid: typing.Optional[ThreeDimensionalGrid] = None,
        cmin: typing.Optional[float] = None,
        cmax: typing.Optional[float] = None,
        subsampling_factor: int = 1,
        opacity: typing.Optional[float] = None,
        show_edges: typing.Optional[bool] = None,
        z_scale: float = 1.0,
        title: str = "",
        threshold_percentile: typing.Optional[float] = None,
        coordinate_offsets: typing.Optional[typing.Tuple[int, int, int]] = None,
        **kwargs: typing.Any,
    ) -> pv.Plotter:  # type: ignore
        """
        Render grid as voxel blocks (single structured mesh).

        Each cell is visualized as a colored voxel with edge outlines to show
        individual cells as distinct units. Similar to Petrel/ResInsight cell-based
        views. Supports subsampling and thresholding for large models.

        :param data: 3D array of scalar values
        :param metadata: Property metadata (name, unit, color scheme)
        :param cell_dimension: (dx, dy) cell spacing in physical units
        :param depth_grid: Optional depth array for structural dip
        :param cmin: Colormap minimum value (original units)
        :param cmax: Colormap maximum value (original units)
        :param subsampling_factor: Sample every Nth cell per axis (1=all, 2=half, etc.)
        :param opacity: Override global opacity (0.0-1.0)
        :param show_edges: Show cell edges (default: True for cell blocks, set False to disable)
        :param z_scale: Vertical exaggeration factor
        :param title: Plot title
        :param threshold_percentile: Hide cells below this percentile (0-100)
        :param coordinate_offsets: (x, y, z) offsets for sliced grids
        :param kwargs: Additional options (color_scheme, etc.)
        :return: Configured `pv.Plotter`
        :raises ValidationError: If data not 3-D or subsampling_factor < 1
        """
        if data.ndim != 3:
            raise ValidationError("Cell block renderer requires 3-D data")
        if subsampling_factor < 1:
            raise ValidationError("`subsampling_factor` must be >= 1")

        if subsampling_factor > 1:
            s = slice(None, None, subsampling_factor)
            data = data[s, s, s]  # type: ignore[assignment]
            if depth_grid is not None:
                depth_grid = depth_grid[s, s, s]  # type: ignore[assignment]
            if cell_dimension is not None:
                cell_dimension = (
                    cell_dimension[0] * subsampling_factor,
                    cell_dimension[1] * subsampling_factor,
                )

        plot_data, _ = _normalize_data(data, metadata)
        grid = self._build_grid(
            plot_data, cell_dimension, depth_grid, z_scale, coordinate_offsets
        )

        if threshold_percentile is not None:
            lo = float(np.nanpercentile(plot_data, threshold_percentile))
            grid = grid.threshold(lo, scalars="values")

        plotter = self._plotter(title)
        # For cell blocks renderer, default to showing edges to make cells visually distinct
        draw_edges = (
            show_edges
            if show_edges is not None
            else (self.config.show_edges or self.config.show_cell_outlines or True)
        )
        clim = (cmin, cmax) if cmin is not None and cmax is not None else None

        add_kwargs: typing.Dict[str, typing.Any] = {
            "scalars": "values",
            "opacity": opacity if opacity is not None else self.config.opacity,
            "cmap": _cmap(kwargs.get("color_scheme", metadata.color_scheme)),
            "show_edges": draw_edges,
            "edge_color": _normalize_hex_color(self.config.cell_outline_color),
            "line_width": self.config.cell_outline_width,
            "smooth_shading": self.config.smooth_shading,
            "show_scalar_bar": self.config.show_colorbar,
            "scalar_bar_args": _scalar_bar_args(metadata, self.config.scalar_bar_args),
            "n_colors": self.config.n_colors,
            "log_scale": metadata.log_scale,
        }
        if clim is not None:
            add_kwargs["clim"] = clim

        plotter.add_mesh(grid, **add_kwargs)
        plotter._bores_mesh = grid  # type: ignore[attr-defined]
        return plotter


class Scatter3DRenderer(BaseRenderer):
    """Renders above-threshold cells as 3D point cloud."""

    supports_physical_dimensions = True

    def render(  # type: ignore[override]
        self,
        data: ThreeDimensionalGrid,
        metadata: PropertyMeta,
        cell_dimension: typing.Optional[typing.Tuple[float, float]] = None,
        depth_grid: typing.Optional[ThreeDimensionalGrid] = None,
        threshold: float = 0.0,
        sample_rate: float = 1.0,
        point_size: int = 5,
        cmin: typing.Optional[float] = None,
        cmax: typing.Optional[float] = None,
        opacity: typing.Optional[float] = None,
        z_scale: float = 1.0,
        title: str = "",
        coordinate_offsets: typing.Optional[typing.Tuple[int, int, int]] = None,
        **kwargs: typing.Any,
    ) -> pv.Plotter:  # type: ignore
        """
        Render above-threshold cells as 3D point cloud.

        Filters cells below threshold and optionally subsamples for performance.
        Useful for sparse data or highlighting specific regions.

        :param data: 3D array of scalar values
        :param metadata: Property metadata (name, unit, color scheme)
        :param cell_dimension: Optional (dx, dy) cell spacing
        :param depth_grid: Optional depth array for structural dip
        :param threshold: Display only cells with values > threshold (normalized units)
        :param sample_rate: Fraction of qualifying points to display (0.0-1.0)
        :param point_size: Point rendering size in pixels
        :param cmin: Colormap minimum value
        :param cmax: Colormap maximum value
        :param opacity: Override global opacity (0.0-1.0)
        :param z_scale: Vertical exaggeration factor
        :param title: Plot title
        :param coordinate_offsets: (x, y, z) offsets for sliced grids
        :param kwargs: Additional options (color_scheme, etc.)
        :return: Configured `pv.Plotter`
        :raises ValidationError: If data not 3-D or no points above threshold
        """
        if data.ndim != 3:
            raise ValidationError("Scatter3D renderer requires 3-D data")

        plot_data, _ = _normalize_data(data, metadata)

        xi, yi, zi = np.where(plot_data > threshold)
        vals = plot_data[xi, yi, zi]

        if vals.size == 0:
            raise ValidationError(
                "No data points above threshold for Scatter3D renderer"
            )

        if sample_rate < 1.0:
            n = max(1, int(vals.size * sample_rate))
            idx = np.random.choice(vals.size, n, replace=False)
            xi, yi, zi, vals = xi[idx], yi[idx], zi[idx], vals[idx]

        x_off, y_off, _ = coordinate_offsets or (0, 0, 0)

        if cell_dimension is not None:
            dx, dy = cell_dimension
            px = (x_off + xi + 0.5) * dx
            py = (y_off + yi + 0.5) * dy
            pz = (
                -depth_grid[xi, yi, zi] * z_scale
                if depth_grid is not None
                else (zi + 0.5) * z_scale
            )
        else:
            px = xi + 0.5
            py = yi + 0.5
            pz = zi + 0.5

        cloud = pv.PolyData(np.column_stack([px, py, pz]))  # type: ignore
        cloud["values"] = vals

        plotter = self._plotter(title)
        clim = (cmin, cmax) if cmin is not None and cmax is not None else None

        add_kwargs: typing.Dict[str, typing.Any] = {
            "scalars": "values",
            "point_size": point_size,
            "opacity": opacity if opacity is not None else self.config.opacity,
            "cmap": _cmap(kwargs.get("color_scheme", metadata.color_scheme)),
            "show_scalar_bar": self.config.show_colorbar,
            "scalar_bar_args": _scalar_bar_args(metadata, self.config.scalar_bar_args),
            "n_colors": self.config.n_colors,
            "log_scale": metadata.log_scale,
        }
        if clim is not None:
            add_kwargs["clim"] = clim

        plotter.add_points(cloud, **add_kwargs)
        plotter._bores_mesh = cloud  # type: ignore[attr-defined]
        return plotter


_PLOT_TYPE_NAMES: typing.Dict[PlotType, str] = {
    PlotType.VOLUME: "3D Volume",
    PlotType.ISOSURFACE: "3D Isosurface",
    PlotType.SCATTER_3D: "3D Scatter",
    PlotType.CELL_BLOCKS: "3D Cell Blocks",
}


def _apply_slider_theme() -> None:
    """
    Apply a dark, minimal slider theme to both PyVista slider styles.

    Should be called once before any `add_slider_widget` calls so every slider in the plotter picks it up.

    Design:
    - Thin flat track (tube_width 0.006) in muted steel-blue  #4a90d9
    - Slim pill handle (slider_width 0.022) in bright white
    - Zero-opacity caps so only the handle/track are visible
    - Applies to both "classic" and "modern" named styles
    """
    ss = pv.global_theme.slider_styles  # type: ignore
    for style in (ss.classic, ss.modern):
        style.slider_length = 0.015  # short pill handle
        style.slider_width = 0.022  # slim — not a fat disk
        style.slider_color = (1.0, 1.0, 1.0)  # white handle
        style.tube_width = 0.006  # thin track
        style.tube_color = (0.29, 0.56, 0.85)  # #4a90d9 steel-blue
        style.cap_opacity = 0.0  # invisible end-caps
        style.cap_length = 0.008
        style.cap_width = 0.02


def _styled_slider(
    plotter: pv.Plotter,  # type: ignore
    callback: typing.Callable,
    rng: typing.Tuple[float, float],
    value: float,
    title: str,
    pointa: typing.Tuple[float, float],
    pointb: typing.Tuple[float, float],
    fmt: str = "%.3g",
) -> None:
    """
    Add a styled vertical slider and apply post-creation VTK tweaks.

    PyVista exposes geometry/colour via `global_theme.slider_styles` but
    font size and label colour require reaching into the VTK representation
    directly after the widget is created.

    :param plotter: Active PyVista Plotter
    :param callback: Single-argument callable receiving the slider value
    :param rng: (min, max) value range
    :param value: Initial value
    :param title: Label shown alongside the slider
    :param pointa: Normalised (x, y) start position in the viewport
    :param pointb: Normalised (x, y) end position in the viewport
    :param fmt: `printf`-style format string for the live value readout
    """
    widget = plotter.add_slider_widget(
        callback,
        rng=rng,
        value=value,
        title=title,
        pointa=pointa,
        pointb=pointb,
        style="modern",
        title_height=0.022,
        title_opacity=0.92,
        title_color="#e0e8f4",  # cool off-white, readable on dark/light BG
        color="#4a90d9",  # handle colour echoes the track
        fmt=fmt,
        interaction_event="always",
    )

    # VTK post-creation tweaks, things PyVista doesn't expose in the call sig
    rep = widget.GetRepresentation()

    # Title (static label)
    title_prop = rep.GetTitleProperty()
    title_prop.SetColor(0.88, 0.91, 0.96)  # #e0e8f4 in float RGB
    title_prop.SetFontFamilyToArial()
    title_prop.BoldOff()
    title_prop.ItalicOff()
    title_prop.ShadowOff()

    # Live value label
    label_prop = rep.GetLabelProperty()
    label_prop.SetColor(0.55, 0.75, 0.98)  # brighter accent blue for the number
    label_prop.SetFontFamilyToArial()
    label_prop.BoldOff()
    label_prop.ShadowOff()

    rep.SetLabelFormat(fmt)


def _setup_interactive_widgets(
    plotter: pv.Plotter,  # type: ignore
    config: PlotConfig,
    metadata: PropertyMeta,
) -> None:
    """
    Add interactive widgets and keyboard shortcuts to a PyVista plotter.

    **Sliders** (left side, vertical):
        - Opacity
        - Threshold    (grid meshes only)
        - Z-scale      (vertical exaggeration)
        - C-min        (colormap lower bound, grid meshes only)
        - C-max        (colormap upper bound, grid meshes only)

    **Keyboard shortcuts** (press `h` to show/hide help overlay):
        - `0`       - reset camera
        - `s`       - save screenshot
        - `a`       - toggle axes
        - `g`       - toggle grid/cell edges
        - `k`       - toggle colorbar
        - `1/2/3`   - add/remove X/Y/Z slice plane → drag the orange handle
        - `b`       - toggle box-crop widget
        - `v`       - cycle view presets (iso / top / front / right)
        - `h`       - toggle help overlay

    **Slicing how-to:**
        Press `1`, `2`, or `3` to insert a cutting plane along that axis.
        An orange rectangular handle appears on the mesh — click and drag it to
        move the plane through the grid.  Press the same key again to remove it.
        Multiple planes can be active simultaneously.

    Note: `e` (exit), `w` (wireframe), `f` (fly-to), `p` (pick) are
    reserved by VTK and are not overridden.
    """
    _apply_slider_theme()

    mesh = getattr(plotter, "_bores_mesh", None)
    has_grid = mesh is not None and hasattr(mesh, "threshold")

    cmap = _cmap(metadata.color_scheme)
    mesh_kwargs: dict[str, typing.Any] = {
        "scalars": "values",
        "cmap": cmap,
        "show_scalar_bar": False,
        "log_scale": metadata.log_scale,
    }

    # Slider: Opacity
    def _set_opacity(value: float) -> None:
        for actor in plotter.renderer.actors.values():
            prop = getattr(actor, "GetProperty", None)
            if prop is not None:
                p = prop()
                if hasattr(p, "SetOpacity"):
                    p.SetOpacity(value)

    _styled_slider(
        plotter,
        _set_opacity,
        rng=(0.0, 1.0),
        value=config.opacity,
        title="Opacity",
        pointa=(0.06, 0.95),
        pointb=(0.06, 0.82),
        fmt="%.2f",
    )

    # Slider: Threshold  (grid meshes only)
    if has_grid:
        scalars = mesh.active_scalars  # type: ignore
        data_min = float(scalars.min()) if scalars is not None else 0.0
        data_max = float(scalars.max()) if scalars is not None else 1.0

        _thresh: dict[str, typing.Any] = {"actor": None}

        def _apply_threshold(value: float) -> None:
            if _thresh["actor"] is not None:
                plotter.remove_actor(_thresh["actor"])
                _thresh["actor"] = None
            if value <= data_min:
                return
            try:
                threshed = mesh.threshold(value, scalars="values")  # type: ignore
                if threshed.n_cells > 0:
                    _thresh["actor"] = plotter.add_mesh(
                        threshed, opacity=config.opacity, **mesh_kwargs
                    )
            except Exception as exc:
                logger.error(exc, exc_info=True)

        _styled_slider(
            plotter,
            _apply_threshold,
            rng=(data_min, data_max),
            value=data_min,
            title="Threshold",
            pointa=(0.06, 0.76),
            pointb=(0.06, 0.63),
            fmt="%.3g",
        )

    # Slider: Z-scale  (vertical exaggeration)
    # Live-rescales the mesh by adjusting the renderer's Z-scale rather
    # than rebuilding geometry, so it works for any renderer type.
    _zscale_state = {"current": 1.0}

    def _set_zscale(value: float) -> None:
        # Clamp to avoid zero/negative which would invert the grid
        value = max(value, 0.05)
        factor = value / _zscale_state["current"]
        _zscale_state["current"] = value
        # Scale every actor's Z; avoids rebuilding the mesh entirely
        for actor in plotter.renderer.actors.values():
            if hasattr(actor, "SetScale"):
                sx, sy, sz = actor.GetScale()  # type: ignore
                actor.SetScale(sx, sy, sz * factor)  # type: ignore
        plotter.render()

    _styled_slider(
        plotter,
        _set_zscale,
        rng=(0.1, 20.0),
        value=1.0,
        title="Z-scale",
        pointa=(0.06, 0.57),
        pointb=(0.06, 0.44),
        fmt="%.1fx",
    )

    # Sliders: C-min / C-max  (colormap range, grid meshes only)
    # Two stacked sliders.  Guards prevent min > max and max < min.
    if has_grid:
        scalars = mesh.active_scalars  # type: ignore
        _cdata_min = float(scalars.min()) if scalars is not None else 0.0
        _cdata_max = float(scalars.max()) if scalars is not None else 1.0
        _clim: dict[str, float] = {"lo": _cdata_min, "hi": _cdata_max}

        def _apply_clim() -> None:
            """Push current lo/hi to every scalar-bar-linked actor."""
            for name in list(plotter.scalar_bars.keys()):
                mapper = plotter.scalar_bars[name].GetMapper()
                if mapper is not None:
                    mapper.SetScalarRange(_clim["lo"], _clim["hi"])
            plotter.render()

        def _set_cmin(value: float) -> None:
            _clim["lo"] = min(value, _clim["hi"] - 1e-9)
            _apply_clim()

        def _set_cmax(value: float) -> None:
            _clim["hi"] = max(value, _clim["lo"] + 1e-9)
            _apply_clim()

        _styled_slider(
            plotter,
            _set_cmin,
            rng=(_cdata_min, _cdata_max),
            value=_cdata_min,
            title="C-min",
            pointa=(0.06, 0.38),
            pointb=(0.06, 0.27),
            fmt="%.3g",
        )

        _styled_slider(
            plotter,
            _set_cmax,
            rng=(_cdata_min, _cdata_max),
            value=_cdata_max,
            title="C-max",
            pointa=(0.06, 0.21),
            pointb=(0.06, 0.10),
            fmt="%.3g",
        )

    # Slice planes  (1/2/3 keys)
    # Inserts a draggable orange cutting plane.  User clicks the handle
    # and drags it through the volume.  Same key removes it.
    _slice_actors: dict[str, bool] = {}
    _slice_status: dict[str, typing.Any] = {"actor": None}

    def _update_slice_status() -> None:
        """Show/hide a small HUD line telling the user what to do."""
        if _slice_status["actor"] is not None:
            plotter.remove_actor(_slice_status["actor"])
            _slice_status["actor"] = None

        active = sorted(_slice_actors.keys())
        if not active:
            return

        axes_label = "/".join(a.upper() for a in active)
        _slice_status["actor"] = plotter.add_text(
            f"Slice active ({axes_label}). Click the orange handle and drag to move",
            position=(10, 30),
            font_size=8,
            color="#4a90d9",
            name="_bores_slice_hint",
        )

    def _toggle_slice(axis: str, normal: str) -> None:
        if not has_grid:
            return
        if axis in _slice_actors:
            # Remove this plane: rebuild remaining ones from scratch
            active = {k: v for k, v in _slice_actors.items() if k != axis}
            plotter.clear_plane_widgets()
            _slice_actors.clear()
            for ax in active:
                plotter.add_mesh_slice(
                    mesh,
                    normal=ax,
                    interaction_event="always",
                    **mesh_kwargs,
                )
                _slice_actors[ax] = True
            logger.info("Removed %s slice plane", axis.upper())
        else:
            plotter.add_mesh_slice(
                mesh,
                normal=normal,
                interaction_event="always",
                **mesh_kwargs,
            )
            _slice_actors[axis] = True
            logger.info(
                "Added %s slice. Click the orange handle on the mesh and drag",
                axis.upper(),
            )
        _update_slice_status()

    plotter.add_key_event("1", lambda: _toggle_slice("x", "x"))  # type: ignore
    plotter.add_key_event("2", lambda: _toggle_slice("y", "y"))  # type: ignore
    plotter.add_key_event("3", lambda: _toggle_slice("z", "z"))  # type: ignore

    # Box-crop widget  (b key)
    _box_state = {"active": False}

    def _toggle_box() -> None:
        if not has_grid:
            return
        if _box_state["active"]:
            plotter.clear_box_widgets()
            _box_state["active"] = False
            logger.info("Box crop removed")
        else:
            plotter.add_mesh_clip_box(
                mesh,
                interaction_event="always",
                color="grey",
                **mesh_kwargs,
            )
            _box_state["active"] = True
            logger.info("Box crop enabled. Drag the grey handles to crop")

    plotter.add_key_event("b", _toggle_box)  # type: ignore

    # View presets  (v key)
    _VIEWS = ["isometric", "xy", "xz", "yz"]
    _view_idx = {"i": 0}

    def _cycle_view() -> None:
        _view_idx["i"] = (_view_idx["i"] + 1) % len(_VIEWS)
        view = _VIEWS[_view_idx["i"]]
        if view == "isometric":
            plotter.view_isometric()  # type: ignore
        else:
            plotter.view_vector(
                {"xy": (0, 0, 1), "xz": (0, -1, 0), "yz": (1, 0, 0)}[view]  # type: ignore
            )
        logger.info("View: %s", view)

    plotter.add_key_event("v", _cycle_view)  # type: ignore

    # Misc shortcuts
    def _screenshot() -> None:
        fname = "bores_screenshot.png"
        plotter.screenshot(fname)
        logger.info("Screenshot saved to %s", fname)

    plotter.add_key_event("s", _screenshot)  # type: ignore
    plotter.add_key_event("0", lambda: plotter.reset_camera())  # type: ignore

    _axes_vis = {"v": config.show_axes}

    def _toggle_axes() -> None:
        (plotter.hide_axes if _axes_vis["v"] else plotter.show_axes)()  # type: ignore
        _axes_vis["v"] = not _axes_vis["v"]

    plotter.add_key_event("a", _toggle_axes)  # type: ignore

    # Snapshot original edge visibility per actor, keyed by actor id.
    # Captured after all meshes have been added so arrows / tubes / wells
    # are already present. Toggle-off restores this instead of forcing True
    # on every actor (which would add outlines to surface markers etc.).
    _original_edge_vis: dict[int, bool] = {}
    for actor in plotter.renderer.actors.values():
        prop = getattr(actor, "GetProperty", None)
        if prop is not None:
            p = prop()
            if hasattr(p, "GetEdgeVisibility"):
                _original_edge_vis[id(actor)] = bool(p.GetEdgeVisibility())

    _edge_vis = {"on": True}  # tracks whether edges are currently shown

    def _toggle_edges() -> None:
        turning_off = _edge_vis["on"]
        _edge_vis["on"] = not turning_off
        for actor in plotter.renderer.actors.values():
            prop = getattr(actor, "GetProperty", None)
            if prop is not None:
                p = prop()
                if hasattr(p, "SetEdgeVisibility"):
                    if turning_off:
                        p.SetEdgeVisibility(False)
                    else:
                        # Restore to whatever the actor originally had —
                        # arrows/tubes/wells had False, grid mesh had True/False
                        # per config. Never force True on actors that didn't have it.
                        original = _original_edge_vis.get(id(actor), False)
                        p.SetEdgeVisibility(original)
        plotter.render()

    plotter.add_key_event("g", _toggle_edges)  # type: ignore

    _cbar_vis = {"v": config.show_colorbar}

    def _toggle_colorbar() -> None:
        _cbar_vis["v"] = not _cbar_vis["v"]
        for name in list(plotter.scalar_bars.keys()):
            actor = plotter.scalar_bars[name]
            actor.SetVisibility(_cbar_vis["v"])
        plotter.render()

    plotter.add_key_event("k", _toggle_colorbar)  # type: ignore

    # Help overlay  (h key)
    _HELP_LINES = (
        "  ── Keyboard shortcuts ──────────────────\n"
        "  '1' '2' '3' - Add X / Y / Z slice plane\n"
        "               → click the orange handle\n"
        "                 and drag to move it\n"
        "               → press same key to remove\n"
        "  'b'         - Box crop (drag grey handles)\n"
        "  'v'         - Cycle views (iso/top/front/side)\n"
        "  '0'         - Reset camera\n"
        "  's'         - Save screenshot\n"
        "  'a'         - Toggle axes\n"
        "  'g'         - Toggle cell edges\n"
        "  'k'         - Toggle colorbar\n"
        "  'h'         - Toggle this help\n"
        "\n"
        "  ── Sliders (left panel) ─────────────────\n"
        "  Opacity     - Fade the entire mesh\n"
        "  Threshold   - Hide cells below value\n"
        "  Z-scale     - Vertical exaggeration\n"
        "  C-min/C-max - Narrow colormap range"
    )
    _help_state: typing.Dict[str, typing.Any] = {"actor": None, "visible": False}

    def _toggle_help() -> None:
        if _help_state["visible"] and _help_state["actor"] is not None:
            plotter.remove_actor(_help_state["actor"])
            _help_state["actor"] = None
            _help_state["visible"] = False
        else:
            _help_state["actor"] = plotter.add_text(
                _HELP_LINES,
                position="upper_right",
                font_size=7,
                color="#2c3e50",
                name="_bores_help",
            )
            _help_state["visible"] = True

    plotter.add_key_event("h", _toggle_help)  # type: ignore
    plotter.add_text(
        "Press 'h' for help | '1/2/3' to slice",
        position=(10, 10),
        font_size=8,
        color="#aaaaaa",
    )


class DataVisualizer:
    """
    PyVista-based 3D visualizer for three-dimensional (reservoir) data.

    ```python
    # Plotly
    from bores.visualization.plotly3d import DataVisualizer, PlotConfig

    # PyVista
    from bores.visualization.pyvista3d import DataVisualizer, PlotConfig
    ```

    Call `.show()` to open window or `.screenshot("out.png")` for off-screen capture.
    """

    def __init__(
        self,
        config: typing.Optional[PlotConfig] = None,
        registry: typing.Optional[PropertyRegistry] = None,
    ) -> None:
        """
        Initialize PyVista 3D visualizer.

        :param config: Optional plot configuration (uses defaults if None)
        :param registry: Optional property metadata registry (uses global if None)
        """
        self._config = config or PlotConfig()
        self._renderers: typing.Dict[PlotType, BaseRenderer] = {
            PlotType.VOLUME: VolumeRenderer(self._config),
            PlotType.ISOSURFACE: IsosurfaceRenderer(self._config),
            PlotType.SCATTER_3D: Scatter3DRenderer(self._config),
            PlotType.CELL_BLOCKS: CellBlockRenderer(self._config),
        }
        self.registry = registry or property_registry

    @property
    def config(self) -> PlotConfig:
        """
        Get current plot configuration.

        :return: `PlotConfig` instance with rendering settings
        """
        return self._config

    def get_renderer(self, plot_type: PlotType) -> BaseRenderer:
        """
        Retrieve renderer instance for specified plot type.

        :param plot_type: Type of renderer to retrieve
        :return: `BaseRenderer` instance configured with current config
        :raises ValidationError: If `plot_type` has no registered renderer
        """
        renderer = self._renderers.get(plot_type)
        if renderer is None:
            raise ValidationError(f"No renderer for {plot_type!r}")
        return renderer

    def add_renderer(
        self,
        plot_type: PlotType,
        renderer_type: typing.Type[BaseRenderer],
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> None:
        """
        Register custom renderer for a plot type.

        Allows extending with custom visualization techniques.

        :param plot_type: Plot type to associate with renderer
        :param renderer_type: `BaseRenderer` subclass to instantiate
        :param args: Positional arguments for renderer constructor
        :param kwargs: Keyword arguments for renderer constructor
        """
        self._renderers[plot_type] = renderer_type(self._config, *args, **kwargs)

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
        Apply slicing operations to 3D grid data.

        :param data: 3D grid data to slice
        :param x_slice: X-axis slice (int, slice, or (start, end) tuple)
        :param y_slice: Y-axis slice (int, slice, or (start, end) tuple)
        :param z_slice: Z-axis slice (int, slice, or (start, end) tuple)
        :return: Tuple of (sliced_data, (x_slice_obj, y_slice_obj, z_slice_obj))
        :raises ValidationError: If slice specifications are invalid
        """
        return slice_grid(data, x_slice, y_slice, z_slice)

    def make_plot(
        self,
        source: typing.Union[
            ReservoirModel[ThreeDimensions],
            ModelState[ThreeDimensions],
            ThreeDimensionalGrid,
        ],
        property: typing.Optional[str] = None,
        plot_type: typing.Optional[typing.Union[PlotType, str]] = None,
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
        labels: typing.Optional[Labels] = None,
        show_wells: bool = False,
        **kwargs: typing.Any,
    ) -> typing.Any:
        """
        Create interactive 3D visualization of reservoir data.

        Returns a PyVista `Plotter` instance ready for display or export.
        Call `.show()` to open interactive window, `.screenshot("file.png")`
        for image export, or use in Jupyter notebooks with `notebook=True`.

        :param source: Data source - `ReservoirModel`, `ModelState`, or raw 3D array
        :param property: Property name to visualize (required for model/state sources).
            Supports dot notation like "permeability.x"
        :param plot_type: Visualization type - "volume", "isosurface", "scatter_3d",
            "cell_blocks", or `PlotType` enum. Uses config default if None
        :param title: Plot title (auto-generated from property if None)
        :param width: Override window width in pixels
        :param height: Override window height in pixels
        :param x_slice: X-axis slice specification (int index, slice object, or (start, end) tuple)
        :param y_slice: Y-axis slice specification
        :param z_slice: Z-axis slice specification
        :param labels: Optional Labels collection for point/region annotations
        :param show_wells: Whether to render wells from `ModelState` (if available)
        :param kwargs: Renderer-specific options:
            - cell_dimension: (dx, dy) physical cell spacing
            - depth_grid: 3D depth array for structural dip
            - z_scale: Vertical exaggeration factor (default 1.0)
            - opacity: Override global opacity (0.0-1.0)
            - cmin/cmax: Colormap value range
            - auto_coarsen: Auto-coarsen large grids (VolumeRenderer)
            - isomin/isomax/surface_count: Isosurface options
            - threshold_percentile/subsampling_factor: CellBlock options
            - threshold/sample_rate/point_size: Scatter3D options
            - injection_color/production_color: Well colors
        :return: Configured `pv.Plotter` instance. Call `.show()` to display
        :raises ValidationError: If inputs are invalid or incompatible

        Examples:
        ```python
        from bores.visualization.pyvista3d import viz, PlotType

        # Volume render pressure from model state
        plotter = viz.make_plot(state, "pressure", plot_type=PlotType.VOLUME)
        plotter.show()

        # Isosurface with custom colors and slicing
        plotter = viz.make_plot(
            state, "oil-saturation",
            plot_type="isosurface",
            z_slice=(0, 10),
            isomin=0.3, isomax=0.9,
            surface_count=5
        )

        # Cell blocks with wells
        plotter = viz.make_plot(
            state, "permeability.x",
            plot_type="cell_blocks",
            show_wells=True,
            show_edges=True,
            threshold_percentile=10
        )
        ```
        """
        if isinstance(plot_type, str):
            try:
                plot_type = PlotType(plot_type)
            except ValueError:
                raise ValidationError(
                    f"Invalid plot_type '{plot_type}'. Valid: {[pt.value for pt in PlotType]}"
                )

        is_model_state = isinstance(source, ModelState)
        is_model = isinstance(source, ReservoirModel)

        if is_model_state or is_model:
            if property is None:
                raise ValidationError(
                    "property is required for model / model state sources"
                )
            metadata = self.registry[property]
            data = self._get_data(source, metadata.name)  # type: ignore
            cell_dimension = (
                source.model.cell_dimension if is_model_state else source.cell_dimension  # type: ignore
            )
            depth_grid = (
                source.model.get_depth_grid(apply_dip=True)  # type: ignore
                if is_model_state
                else source.get_depth_grid(apply_dip=True)  # type: ignore
            )
        else:
            data = source
            metadata = (
                self.registry[property]
                if property is not None and property in self.registry
                else PropertyMeta(
                    name=property or "data",
                    display_name=property or "Data",
                    unit="",
                    color_scheme=ColorScheme.VIRIDIS,
                )
            )
            cell_dimension = None
            depth_grid = None

        coordinate_offsets = None
        if any(s is not None for s in [x_slice, y_slice, z_slice]):
            data, (sx, sy, sz) = self.apply_slice(data, x_slice, y_slice, z_slice)  # type: ignore[arg-type]
            coordinate_offsets = (sx.start or 0, sy.start or 0, sz.start or 0)
            if depth_grid is not None:
                depth_grid = depth_grid[sx, sy, sz]  # type: ignore

        plot_type = plot_type or self._config.plot_type

        config = self._config
        if width is not None or height is not None:
            config = attrs.evolve(
                config, width=width or config.width, height=height or config.height
            )

        renderer = type(self.get_renderer(plot_type))(config)

        plot_title = title or (
            f"{_PLOT_TYPE_NAMES.get(plot_type, '3D')}: {metadata.display_name}"
        )
        kwargs["title"] = plot_title
        if coordinate_offsets is not None:
            kwargs["coordinate_offsets"] = coordinate_offsets

        if renderer.supports_physical_dimensions:
            plotter = renderer.render(
                data,  # type: ignore[arg-type]
                metadata,
                cell_dimension=kwargs.pop("cell_dimension", cell_dimension),
                depth_grid=kwargs.pop("depth_grid", depth_grid),
                **kwargs,
            )
        else:
            plotter = renderer.render(data, metadata, **kwargs)  # type: ignore[arg-type]

        if show_wells and is_model_state and source.wells_exists():  # type: ignore
            well_kwargs = {
                k: kwargs[k] for k in WellKwargs.__annotations__ if k in kwargs
            }
            _render_wells(
                plotter,
                source.wells,  # type: ignore
                cell_dimension=cell_dimension,
                depth_grid=depth_grid,  # type: ignore
                coordinate_offsets=coordinate_offsets,
                z_scale=kwargs.get("z_scale", 1.0),
                **well_kwargs,
            )

        if labels is not None and self._config.show_labels:
            z_sc = kwargs.get("z_scale", 1.0)
            for label in labels.visible():
                if cell_dimension is not None and depth_grid is not None:
                    try:
                        xp, yp, zp = label.position.as_physical(
                            cell_dimension,
                            depth_grid,  # type: ignore
                            coordinate_offsets,
                        )
                        # as_physical returns -depth; apply z_scale to match mesh
                        zp = zp * z_sc
                        position = [[xp, yp, zp]]
                        text = [
                            label.name
                            or f"({label.position.x},{label.position.y},{label.position.z})"
                        ]
                    except Exception as exc:
                        logger.debug(f"Label physical coordinates failed: {exc}")
                        continue
                else:
                    position = [[label.position.x, label.position.y, label.position.z]]
                    text = [
                        label.name
                        or str((label.position.x, label.position.y, label.position.z))
                    ]

                plotter.add_point_labels(
                    position,
                    text,
                    font_size=label.font_size,
                    text_color=_normalize_hex_color(label.font_color),
                    show_points=True,
                    point_size=8,
                    always_visible=True,
                )

        plotter.add_title(plot_title, font_size=8)

        # Enable cell/point picking for interactive data inspection
        if self._config.enable_picking and not self._config.off_screen:

            def _on_pick(picked: typing.Any) -> None:
                if picked is None or picked.n_cells == 0:
                    return
                center = picked.center
                vals = picked.active_scalars
                val = float(vals[0]) if vals is not None and len(vals) > 0 else None
                parts = [
                    f"Cell center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})"
                ]
                if val is not None:
                    parts.append(f"Value: {val:.4g}")
                logger.info(" | ".join(parts))

            try:
                plotter.enable_cell_picking(
                    callback=_on_pick,
                    show_message=False,
                    through=False,
                    style="surface",
                )
            except Exception as exc:
                logger.error(
                    f"Cell picking unavailable in this PyVista version: \n{exc}",
                    exc_info=True,
                )

        # Add interactive widgets
        if config.enable_interactive and not config.off_screen:
            _setup_interactive_widgets(plotter, config, metadata)

        return plotter

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
        save: typing.Union[FrameExporter, str, None] = None,
        output_gif: typing.Optional[str] = None,
        labels: typing.Optional[Labels] = None,
        **kwargs: typing.Any,
    ) -> typing.List[typing.Any]:
        """
        Create animation from time-series data with optional export.

        Renders each frame independently using `make_plot(...)`. When a `save` is provided
        (or `output_gif`), frames are captured as images,
        passed to the save for export, and `Plotter`s are closed to free GPU memory.
        Without a save, returns list of `Plotter` instances for manual inspection.

        :param sequence: Time-ordered collection of `ReservoirModel`s, `ModelState`s, or 3D arrays
        :param property: Property name to visualize (required for model/state sequences).
            Supports dot notation like "permeability.x"
        :param plot_type: Visualization type - "volume", "isosurface", "scatter_3d",
            "cell_blocks", or `PlotType` enum
        :param frame_duration: Duration per frame in milliseconds
        :param step_size: Sample every Nth frame (1=all frames, 2=every other, etc.)
        :param width: Override window width in pixels for all frames
        :param height: Override window height in pixels for all frames
        :param title: Base title (frame number appended automatically)
        :param x_slice: X-axis slice specification applied to all frames
        :param y_slice: Y-axis slice specification applied to all frames
        :param z_slice: Z-axis slice specification applied to all frames
        :param save: Animation exporter. Can be a `FrameExporter` instance, or a string
            file path whose extension determines the format (e.g. "out.mp4", "out.gif",
            "out.webp"). If None and `output_gif` is also None, no export is performed.
        :param output_gif: Equivalent to `save=GifExporter(output_gif)`.
        :param labels: Optional Labels collection for annotations (same for all frames)
        :param kwargs: Additional renderer options (passed to make_plot):
            - cmin/cmax: Fixed colormap range across all frames (recommended)
            - opacity/z_scale/etc.: Other rendering options
        :return: List of `pv.Plotter` instances (one per frame after step_size sampling).
            If exporting, `Plotter`s are closed but still returned
        :raises ValidationError: If sequence is empty or property required but missing

        Examples:
        ```python
        from bores.visualization.pyvista3d import viz
        from bores.visualization.utils import Mp4Exporter

        # Export as MP4 via string path
        viz.animate(states, "oil-saturation", save="oil_saturation.mp4")

        # Export as GIF via string path
        viz.animate(states, "pressure", save="pressure.gif")

        # Export with explicit save for fine control
        viz.animate(states, "pressure", save=Mp4Exporter("out.mp4", quality=10))

        # Backward compatible GIF export
        viz.animate(states, "pressure", output_gif="out.gif")
        ```

        Notes:
        - Automatically computes global cmin/cmax from all frames if not specified
        - For large sequences, use `step_size > 1` to reduce frame count
        - Each frame uses same rendering options for visual consistency
        """
        if not sequence:
            raise ValidationError("Empty sequence")

        resolved_exporter = resolve_exporter(save, output_gif)

        is_model_sequence = isinstance(sequence[0], (ModelState, ReservoirModel))
        if is_model_sequence and property is None:
            raise ValidationError(
                "`property` required for model / model state sequences"
            )

        if isinstance(plot_type, str):
            plot_type = PlotType(plot_type)
        plot_type = plot_type or self._config.plot_type

        if "cmin" not in kwargs:
            if is_model_sequence:
                meta = self.registry[property]  # type: ignore
                arrays = [self._get_data(s, meta.name) for s in sequence]  # type: ignore
            else:
                arrays = list(sequence)  # type: ignore
            kwargs["cmin"] = float(np.nanmin([np.nanmin(a) for a in arrays]))  # type: ignore
            kwargs["cmax"] = float(np.nanmax([np.nanmax(a) for a in arrays]))  # type: ignore

        plotters: typing.List[typing.Any] = []
        frames: typing.List[np.ndarray] = []

        for i, item in enumerate(sequence[::step_size]):
            frame_title = title or (
                f"{_PLOT_TYPE_NAMES.get(plot_type, '3D')}: {property or 'Data'} - frame {i}"
            )
            plotter = self.make_plot(
                item,  # type: ignore
                property=property,
                plot_type=plot_type,
                title=frame_title,
                width=width,
                height=height,
                x_slice=x_slice,
                y_slice=y_slice,
                z_slice=z_slice,
                labels=labels,
                **kwargs,
            )
            plotters.append(plotter)

            if resolved_exporter is not None:
                frames.append(plotter.screenshot(return_img=True))
                plotter.close()

        if resolved_exporter is not None and frames:
            fps = 1000.0 / frame_duration
            resolved_exporter.write(frames, fps=fps)  # type: ignore

        return plotters

    def help(self, plot_type: typing.Optional[PlotType] = None) -> str:
        """
        Get help text for renderers.

        :param plot_type: Optional specific renderer to get help for
        :return: Help text describing renderer(s) usage
        """
        if plot_type is not None:
            return self.get_renderer(plot_type).help()
        return "\n".join(
            f"=== {pt.value} ===\n{r.help()}" for pt, r in self._renderers.items()
        )


viz = DataVisualizer()
"""Global `PyVista` visualizer instance. Has a similar API to `plotly3d.viz`."""

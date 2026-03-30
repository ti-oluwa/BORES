"""
Plotly-based 2D Visualization Suite for 2 Dimensional Reservoir Simulation Data and Results.
"""

import collections.abc
import logging
import typing
import weakref
from abc import ABC, abstractmethod
from enum import Enum

import attrs
import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go  # type: ignore[import-untyped]
from plotly.subplots import make_subplots  # type: ignore[import-untyped]

from bores.errors import ValidationError
from bores.types import TwoDimensionalGrid
from bores.visualization.base import ColorScheme, PropertyMeta

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
    visualizations or multiple figures. Call when figures are no longer needed.

    Example:
    ```python
    from bores.visualization.plotly2d import cleanup_figures

    # Create many figures
    for i in range(100):
        plotter.make_plot(...)

    # Free memory when done
    cleanup_figures()
    ```
    """
    for ref in _active_figures[:]:
        fig = ref()
        if fig is not None:
            try:
                fig.data = []
            except (AttributeError, RuntimeError) as exc:
                logger.debug(f"Failed to cleanup figure: {exc}")
    _active_figures.clear()


@attrs.frozen
class PlotConfig:
    """Configuration for 2D plots."""

    # Dimensions
    width: int = 800
    """Figure width in pixels"""

    height: int = 600
    """Figure height in pixels"""

    # Display options
    show_colorbar: bool = True
    """Whether to display color scale bar"""

    title: typing.Optional[str] = None
    """Optional title for plots"""

    # Styling
    color_scheme: str = "viridis"
    """Default colorscale for plots (viridis, plasma, inferno, etc.)"""

    opacity: float = 0.8
    """Default opacity for plot elements"""

    # Grid and axes
    show_grid: bool = True
    """Whether to show grid lines"""

    grid_color: str = "lightgray"
    """Color of grid lines (use 'lightgray', 'gray', 'white', etc. for contrast control)"""

    xaxis_grid_color: typing.Optional[str] = None
    """Color of x-axis grid lines (if None, uses grid_color)"""

    yaxis_grid_color: typing.Optional[str] = None
    """Color of y-axis grid lines (if None, uses grid_color)"""

    axis_line_color: str = "black"
    """Color of axis lines"""

    axis_line_width: float = 1.0
    """Width of axis lines"""

    # Text and labels
    font_family: str = "Arial, sans-serif"
    """Font family for text elements"""

    font_size: int = 12
    """Default font size"""

    title_font_size: int = 16
    """Font size for titles"""

    axis_title_font_size: int = 14
    """Font size for axis titles"""

    # Colorbar settings
    colorbar_thickness: int = 20
    """Thickness of colorbar in pixels"""

    colorbar_len: float = 0.8
    """Length of colorbar as fraction of plot height"""

    # Plot-specific defaults
    contour_line_width: float = 1.5
    """Default width for contour lines"""

    contour_levels: int = 20
    """Default number of contour levels"""

    scatter_marker_size: int = 6
    """Default size for scatter plot markers"""

    line_width: float = 2.0
    """Default width for line plots"""

    # Layout margins
    margin_left: int = 80
    """Left margin in pixels"""

    margin_right: int = 80
    """Right margin in pixels"""

    margin_top: int = 80
    """Top margin in pixels"""

    margin_bottom: int = 80
    """Bottom margin in pixels"""

    # Background
    plot_bgcolor: str = "#f8f9fa"
    """Background color of plot area (light gray for subtle contrast)"""

    paper_bgcolor: str = "#ffffff"
    """Background color of entire figure"""


class PlotType(str, Enum):
    """Enumeration of available 2D plot types."""

    HEATMAP = "heatmap"
    CONTOUR = "contour"
    CONTOUR_FILLED = "contour_filled"
    SCATTER = "scatter"
    LINE = "line"
    SURFACE = "surface"


class BaseRenderer(ABC):
    """
    Abstract base class for 2D renderers.

    All 2D renderers must implement the render method to create
    Plotly visualizations for 2D reservoir data.
    """

    def __init__(self, config: PlotConfig) -> None:
        """
        Initialize the renderer with configuration.

        :param config: Plot configuration settings
        """
        self.config = config

    @abstractmethod
    def render(
        self,
        figure: go.Figure,
        data: TwoDimensionalGrid,
        metadata: PropertyMeta,
        x_coords: typing.Optional[np.ndarray] = None,
        y_coords: typing.Optional[np.ndarray] = None,
        x_label: str = "X",
        y_label: str = "Y",
        **kwargs: typing.Any,
    ) -> go.Figure:
        """
        Render the 2D data onto the provided figure.

        :param figure: Plotly figure to add the plot to
        :param data: 2D data array to visualize
        :param metadata: Property metadata for labeling and scaling
        :param x_coords: Optional x-coordinate array (defaults to indices)
        :param y_coords: Optional y-coordinate array (defaults to indices)
        :param x_label: Label for the x-axis
        :param y_label: Label for the y-axis
        :param kwargs: Additional plotting parameters
        :return: Updated Plotly figure
        """
        ...

    def normalize_data(
        self,
        data: TwoDimensionalGrid,
        metadata: PropertyMeta,
        normalize_range: bool = True,
    ) -> typing.Tuple[TwoDimensionalGrid, TwoDimensionalGrid]:
        """
        Normalize data for consistent visualization.

        :param data: Input 2D data array
        :param metadata: Property metadata containing scaling information
        :param normalize_range: Whether to normalize to 0-1 range
        :return: Tuple of (normalized_data, display_data)
        """
        display_data = data.copy()
        if metadata.log_scale:
            # Apply log transformation, handling zeros and negatives
            display_data = np.where(display_data > 0, display_data, np.nan)  # type: ignore[assignment]
            display_data = np.log10(display_data)  # type: ignore[assignment]

        normalized_data = display_data.copy()
        if normalize_range:
            data_min = np.nanmin(display_data)
            data_max = np.nanmax(display_data)
            if data_max > data_min:
                normalized_data = (display_data - data_min) / (data_max - data_min)  # type: ignore[assignment]
            else:
                normalized_data = np.zeros_like(display_data)

        return typing.cast(TwoDimensionalGrid, normalized_data), typing.cast(
            TwoDimensionalGrid, display_data
        )

    def get_colorscale(self, color_scheme: typing.Optional[str] = None) -> str:
        """
        Get the appropriate colorscale for the plot.

        :param color_scheme: Optional color scheme override
        :return: Plotly colorscale name
        """
        if color_scheme:
            return ColorScheme(color_scheme).value
        return ColorScheme(self.config.color_scheme).value or "viridis"

    def format_value(
        self, value: float, metadata: PropertyMeta, precision: int = 3
    ) -> str:
        """
        Format a data value for display.

        :param value: Value to format
        :param metadata: Property metadata for formatting context
        :param precision: Number of decimal places
        :return: Formatted string
        """
        if np.isnan(value):
            return "NaN"

        if metadata.log_scale:
            # Convert back from log space for display
            original_value = 10**value
            return f"{original_value:.{precision}g}"
        return f"{value:.{precision}g}"

    def help(self) -> str:
        """
        Return a help string describing the renderer and its usage.

        :return: Help string
        """
        return f"{self.__class__.__name__} renderer\n{self.render.__doc__ or ''}"


class HeatmapRenderer(BaseRenderer):
    """2D Heatmap renderer using Plotly's heatmap trace."""

    def render(
        self,
        figure: go.Figure,
        data: TwoDimensionalGrid,
        metadata: PropertyMeta,
        x_coords: typing.Optional[np.ndarray] = None,
        y_coords: typing.Optional[np.ndarray] = None,
        x_label: str = "X",
        y_label: str = "Y",
        cmin: typing.Optional[float] = None,
        cmax: typing.Optional[float] = None,
        **kwargs: typing.Any,
    ) -> go.Figure:
        """
        Render a heatmap visualization.

        :param figure: Plotly figure to add the heatmap to
        :param data: 2D data array to visualize
        :param metadata: Property metadata for labeling and scaling
        :param x_coords: Optional x-coordinate array
        :param y_coords: Optional y-coordinate array
        :param x_label: Label for the x-axis
        :param y_label: Label for the y-axis
        :param cmin: Minimum value for color mapping
        :param cmax: Maximum value for color mapping
        :return: Updated figure
        """
        if data.ndim != 2:
            raise ValidationError("Heatmap plotting requires 2D data")

        _, display_data = self.normalize_data(
            data=data, metadata=metadata, normalize_range=False
        )

        # Handle coordinate arrays
        if x_coords is None:
            x_coords = np.arange(data.shape[1])
        if y_coords is None:
            y_coords = np.arange(data.shape[0])

        # Create hover text
        hover_text = []
        for i in range(data.shape[0]):
            hover_row = []
            for j in range(data.shape[1]):
                hover_row.append(  # noqa
                    f"{x_label}: {x_coords[j]:.4f}<br>"
                    f"{y_label}: {y_coords[i]:.4f}<br>"
                    f"{metadata.display_name}: {self.format_value(display_data[i, j], metadata)} {metadata.unit}"
                )
            hover_text.append(hover_row)

        colorscale = self.get_colorscale(
            kwargs.get("color_scheme", metadata.color_scheme)
        )
        figure.add_trace(
            go.Heatmap(
                z=display_data,
                x=x_coords,
                y=y_coords,
                colorscale=colorscale,
                zmin=cmin,
                zmax=cmax,
                showscale=self.config.show_colorbar,
                hovertemplate="%{text}<extra></extra>",
                text=hover_text,
                colorbar=dict(  # noqa
                    title=f"{metadata.display_name} ({metadata.unit})"
                    + (" - Log Scale" if metadata.log_scale else "")
                )
                if self.config.show_colorbar
                else None,
            )
        )
        self.update_layout(
            figure=figure,
            x_label=x_label,
            y_label=y_label,
            title=metadata.display_name,
        )
        return figure

    def update_layout(
        self,
        figure: go.Figure,
        x_label: str,
        y_label: str,
        title: str,
    ) -> None:
        """Update figure layout for heatmap plots."""
        figure.update_layout(
            width=self.config.width,
            height=self.config.height,
            xaxis_title=x_label,
            yaxis_title=y_label,
            title=title,
        )


class ContourRenderer(BaseRenderer):
    """2D Contour renderer for contour and filled contour plots."""

    def __init__(self, config: PlotConfig, filled: bool = False) -> None:
        """
        Initialize contour renderer.

        :param config: Plot configuration
        :param filled: Whether to create filled contours
        """
        super().__init__(config)
        self.filled = filled

    def render(
        self,
        figure: go.Figure,
        data: TwoDimensionalGrid,
        metadata: PropertyMeta,
        x_coords: typing.Optional[np.ndarray] = None,
        y_coords: typing.Optional[np.ndarray] = None,
        x_label: str = "X",
        y_label: str = "Y",
        contour_levels: typing.Optional[int] = None,
        cmin: typing.Optional[float] = None,
        cmax: typing.Optional[float] = None,
        **kwargs: typing.Any,
    ) -> go.Figure:
        """
        Render a contour visualization.

        :param figure: Plotly figure to add the contour to
        :param data: 2D data array to visualize
        :param metadata: Property metadata for labeling and scaling
        :param x_coords: Optional x-coordinate array
        :param y_coords: Optional y-coordinate array
        :param x_label: Label for the x-axis
        :param y_label: Label for the y-axis
        :param contour_levels: Number of contour levels
        :param cmin: Minimum value for color mapping
        :param cmax: Maximum value for color mapping
        :return: Updated figure
        """
        if data.ndim != 2:
            raise ValidationError("Contour plotting requires 2D data")

        _, display_data = self.normalize_data(
            data=data, metadata=metadata, normalize_range=False
        )

        # Handle coordinate arrays
        if x_coords is None:
            x_coords = np.arange(data.shape[1])
        if y_coords is None:
            y_coords = np.arange(data.shape[0])

        colorscale = self.get_colorscale(
            kwargs.get("color_scheme", metadata.color_scheme)
        )
        levels = contour_levels or 20

        # Calculate proper contour levels
        data_min = cmin if cmin is not None else np.nanmin(display_data)
        data_max = cmax if cmax is not None else np.nanmax(display_data)
        level_step = (data_max - data_min) / levels if data_max > data_min else 1.0

        contour_trace = go.Contour(
            z=display_data,
            x=x_coords,
            y=y_coords,
            colorscale=colorscale,
            zmin=data_min,
            zmax=data_max,
            showscale=self.config.show_colorbar,
            contours=dict(  # noqa
                start=data_min,
                end=data_max,
                size=level_step,
                showlabels=True,
                labelfont=dict(size=10),  # noqa
            ),
            line=dict(width=self.config.contour_line_width),  # noqa
            colorbar=dict(  # noqa
                title=f"{metadata.display_name} ({metadata.unit})"
                + (" - Log Scale" if metadata.log_scale else "")
            )
            if self.config.show_colorbar
            else None,
        )

        if self.filled:
            contour_trace.update(contours_coloring="fill")

        figure.add_trace(contour_trace)
        self.update_layout(
            figure=figure,
            x_label=x_label,
            y_label=y_label,
            title=metadata.display_name,
        )
        return figure

    def update_layout(
        self,
        figure: go.Figure,
        x_label: str,
        y_label: str,
        title: str,
    ) -> None:
        """Update figure layout for contour plots."""
        figure.update_layout(
            width=self.config.width,
            height=self.config.height,
            xaxis_title=x_label,
            yaxis_title=y_label,
            title=title,
        )


class ScatterRenderer(BaseRenderer):
    """2D Scatter renderer for sparse data visualization."""

    def render(
        self,
        figure: go.Figure,
        data: TwoDimensionalGrid,
        metadata: PropertyMeta,
        x_coords: typing.Optional[np.ndarray] = None,
        y_coords: typing.Optional[np.ndarray] = None,
        x_label: str = "X",
        y_label: str = "Y",
        threshold: float = 0.0,
        marker_size: int = 6,
        cmin: typing.Optional[float] = None,
        cmax: typing.Optional[float] = None,
        **kwargs: typing.Any,
    ) -> go.Figure:
        """
        Render a scatter plot visualization.

        :param figure: Plotly figure to add the scatter plot to
        :param data: 2D data array to visualize
        :param metadata: Property metadata for labeling and scaling
        :param x_coords: Optional x-coordinate array
        :param y_coords: Optional y-coordinate array
        :param x_label: Label for the x-axis
        :param y_label: Label for the y-axis
        :param threshold: Minimum value threshold for point inclusion
        :param marker_size: Size of scatter markers
        :param cmin: Minimum value for color mapping
        :param cmax: Maximum value for color mapping
        :return: Updated figure
        """
        if data.ndim != 2:
            raise ValidationError("Scatter plotting requires 2D data")

        _, display_data = self.normalize_data(
            data=data, metadata=metadata, normalize_range=False
        )

        # Handle coordinate arrays
        if x_coords is None:
            x_coords = np.arange(data.shape[1])  # type: ignore
        if y_coords is None:
            y_coords = np.arange(data.shape[0])  # type: ignore

        # Create coordinate meshgrid
        X, Y = np.meshgrid(x_coords, y_coords)

        # Apply threshold filter
        mask = display_data > threshold
        if not np.any(mask):
            raise ValidationError("No data points above threshold for scatter plot")

        x_scatter = X[mask]
        y_scatter = Y[mask]
        values = display_data[mask]

        # Create hover text
        hover_text = [
            f"{x_label}: {x:.4f}<br>"
            f"{y_label}: {y:.4f}<br>"
            f"{metadata.display_name}: {self.format_value(v, metadata)} {metadata.unit}"
            for x, y, v in zip(x_scatter, y_scatter, values)
        ]

        colorscale = self.get_colorscale(
            kwargs.get("color_scheme", metadata.color_scheme)
        )

        figure.add_trace(
            go.Scatter(
                x=x_scatter,
                y=y_scatter,
                mode="markers",
                marker=dict(  # noqa
                    size=marker_size,
                    color=values,
                    colorscale=colorscale,
                    cmin=cmin,
                    cmax=cmax,
                    opacity=self.config.opacity,
                    colorbar=dict(  # noqa
                        title=f"{metadata.display_name} ({metadata.unit})"
                        + (" - Log Scale" if metadata.log_scale else "")
                    )
                    if self.config.show_colorbar
                    else None,
                ),
                text=hover_text,
                hovertemplate="%{text}<extra></extra>",
            )
        )
        self.update_layout(
            figure=figure,
            x_label=x_label,
            y_label=y_label,
            title=metadata.display_name,
        )
        return figure

    def update_layout(
        self,
        figure: go.Figure,
        x_label: str,
        y_label: str,
        title: str,
    ) -> None:
        """Update figure layout for scatter plots."""
        figure.update_layout(
            width=self.config.width,
            height=self.config.height,
            xaxis_title=x_label,
            yaxis_title=y_label,
            title=title,
        )


class LineRenderer(BaseRenderer):
    """2D Line renderer for line plots and cross-sections."""

    def render(
        self,
        figure: go.Figure,
        data: TwoDimensionalGrid,
        metadata: PropertyMeta,
        x_coords: typing.Optional[np.ndarray] = None,
        y_coords: typing.Optional[np.ndarray] = None,
        x_label: str = "X",
        y_label: str = "Y",
        line_mode: typing.Literal["horizontal", "vertical", "both"] = "horizontal",
        line_indices: typing.Optional[typing.Union[int, typing.List[int]]] = None,
        **kwargs: typing.Any,
    ) -> go.Figure:
        """
        Render line plots (cross-sections) of 2D data.

        :param figure: Plotly figure to add the line plots to
        :param data: 2D data array to visualize
        :param metadata: Property metadata for labeling and scaling
        :param x_coords: Optional x-coordinate array
        :param y_coords: Optional y-coordinate array
        :param x_label: Label for the x-axis
        :param y_label: Label for the y-axis
        :param line_mode: Direction of lines to plot
        :param line_indices: Specific indices to plot (defaults to middle)
        :return: Updated figure
        """
        if data.ndim != 2:
            raise ValidationError("Line plotting requires 2D data")

        _, display_data = self.normalize_data(
            data=data, metadata=metadata, normalize_range=False
        )

        # Handle coordinate arrays
        if x_coords is None:
            x_coords = np.arange(data.shape[1])
        if y_coords is None:
            y_coords = np.arange(data.shape[0])

        # Determine line indices
        if line_indices is None:
            line_indices = [data.shape[0] // 2, data.shape[1] // 2]
        elif isinstance(line_indices, int):
            line_indices = [line_indices]

        # Plot horizontal lines (constant y, varying x)
        if (
            line_mode in ["horizontal", "both"]
            and len(line_indices) >= 1
            and 0 <= line_indices[0] < data.shape[0]
        ):
            idx = line_indices[0]
            figure.add_trace(
                go.Scatter(
                    x=x_coords,
                    y=display_data[idx, :],
                    mode="lines+markers",
                    name=f"{y_label}={y_coords[idx]:.4f}",
                    line=dict(width=self.config.line_width),  # noqa
                    marker=dict(  # noqa
                        size=8,
                        symbol="circle",
                        line=dict(width=1, color="white"),
                    ),
                )
            )

        # Plot vertical lines (constant x, varying y)
        if line_mode in ["vertical", "both"]:
            if len(line_indices) >= 2 and 0 <= line_indices[1] < data.shape[1]:
                idx = line_indices[1]
                figure.add_trace(
                    go.Scatter(
                        x=y_coords,
                        y=display_data[:, idx],
                        mode="lines+markers",
                        name=f"{x_label}={x_coords[idx]:.4f}",
                        line=dict(width=self.config.line_width),  # noqa
                        marker=dict(  # noqa
                            size=8,
                            symbol="circle",
                            line=dict(width=1, color="white"),  # noqa
                        ),
                    )
                )
            elif (
                line_mode == "vertical"
                and len(line_indices) >= 1
                and 0 <= line_indices[0] < data.shape[1]
            ):
                idx = line_indices[0]
                figure.add_trace(
                    go.Scatter(
                        x=y_coords,
                        y=display_data[:, idx],
                        mode="lines+markers",
                        name=f"{x_label}={x_coords[idx]:.4f}",
                        line=dict(width=self.config.line_width),  # noqa
                        marker=dict(  # noqa
                            size=8,
                            symbol="circle",
                            line=dict(width=1, color="white"),  # noqa
                        ),
                    )
                )

        self.update_layout(
            figure=figure,
            x_label=x_label,
            y_label=f"{metadata.display_name} ({metadata.unit})",
            title=metadata.display_name,
        )
        return figure

    def update_layout(
        self,
        figure: go.Figure,
        x_label: str,
        y_label: str,
        title: str,
    ) -> None:
        """Update figure layout for line plots."""
        figure.update_layout(
            width=self.config.width,
            height=self.config.height,
            xaxis_title=x_label,
            yaxis_title=y_label,
            title=title,
            showlegend=True,
        )


class SurfaceRenderer(BaseRenderer):
    """3D Surface renderer for 2D data."""

    def render(
        self,
        figure: go.Figure,
        data: TwoDimensionalGrid,
        metadata: PropertyMeta,
        x_coords: typing.Optional[np.ndarray] = None,
        y_coords: typing.Optional[np.ndarray] = None,
        x_label: str = "X",
        y_label: str = "Y",
        cmin: typing.Optional[float] = None,
        cmax: typing.Optional[float] = None,
        **kwargs: typing.Any,
    ) -> go.Figure:
        """
        Render a 3D surface plot of 2D data.

        :param figure: Plotly figure to add the surface to
        :param data: 2D data array to visualize
        :param metadata: Property metadata for labeling and scaling
        :param x_coords: Optional x-coordinate array
        :param y_coords: Optional y-coordinate array
        :param x_label: Label for the x-axis
        :param y_label: Label for the y-axis
        :param cmin: Minimum value for color mapping
        :param cmax: Maximum value for color mapping
        :return: Updated figure
        """
        if data.ndim != 2:
            raise ValidationError("Surface plotting requires 2D data")

        _, display_data = self.normalize_data(
            data=data, metadata=metadata, normalize_range=False
        )

        # Handle coordinate arrays
        if x_coords is None:
            x_coords = np.arange(data.shape[1])
        if y_coords is None:
            y_coords = np.arange(data.shape[0])

        colorscale = self.get_colorscale(
            kwargs.get("color_scheme", metadata.color_scheme)
        )
        figure.add_trace(
            go.Surface(
                z=display_data,
                x=x_coords,
                y=y_coords,
                colorscale=colorscale,
                cmin=cmin,
                cmax=cmax,
                showscale=self.config.show_colorbar,
                colorbar=dict(  # noqa
                    title=f"{metadata.display_name} ({metadata.unit})"
                    + (" - Log Scale" if metadata.log_scale else "")
                )
                if self.config.show_colorbar
                else None,
            )
        )
        self.update_layout(
            figure=figure,
            x_label=x_label,
            y_label=y_label,
            title=metadata.display_name,
        )
        return figure

    def update_layout(
        self,
        figure: go.Figure,
        x_label: str,
        y_label: str,
        title: str,
    ) -> None:
        """Update figure layout for surface plots."""
        figure.update_layout(
            width=self.config.width,
            height=self.config.height,
            scene=dict(  # noqa
                xaxis_title=x_label,
                yaxis_title=y_label,
                zaxis_title=title,
                aspectmode="auto",
            ),
            title=title,
        )


PLOT_TYPE_NAMES: typing.Dict[PlotType, str] = {
    PlotType.HEATMAP: "Heatmap",
    PlotType.CONTOUR: "Contour",
    PlotType.CONTOUR_FILLED: "Filled Contour",
    PlotType.SCATTER: "Scatter",
    PlotType.LINE: "Line Plot",
    PlotType.SURFACE: "3D Surface",
}


class DataVisualizer:
    """
    2D visualizer for two-dimensional (reservoir) data.
    """

    def __init__(self, config: typing.Optional[PlotConfig] = None) -> None:
        """
        Initialize the visualizer with optional configuration.

        :param config: Optional configuration for 2D rendering (uses defaults if None)
        """
        self._config = config or PlotConfig()
        self._renderers: typing.Dict[PlotType, BaseRenderer] = {
            PlotType.HEATMAP: HeatmapRenderer(self._config),
            PlotType.CONTOUR: ContourRenderer(self._config, filled=False),
            PlotType.CONTOUR_FILLED: ContourRenderer(self._config, filled=True),
            PlotType.SCATTER: ScatterRenderer(self._config),
            PlotType.LINE: LineRenderer(self._config),
            PlotType.SURFACE: SurfaceRenderer(self._config),
        }

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

        :param plot_type: The type of plot to add
        :param renderer_type: The class implementing the BaseRenderer interface
        :param args: Initialization arguments for the renderer class
        :param kwargs: Initialization keyword arguments for the renderer class
        """
        if not isinstance(plot_type, PlotType):
            raise ValidationError(f"Invalid plot type: {plot_type}")
        self._renderers[plot_type] = renderer_type(self.config, *args, **kwargs)

    def get_renderer(self, plot_type: typing.Union[PlotType, str]) -> BaseRenderer:
        """
        Get the renderer for a specific plot type.

        :param plot_type: The type of plot to get (PlotType enum or string)
        :return: The renderer instance for the specified plot type
        """
        # Convert string `plot_type` to `PlotType` enum if needed
        if isinstance(plot_type, str):
            try:
                plot_type = PlotType(plot_type)
            except ValueError:
                raise ValidationError(
                    f"Invalid plot_type string: '{plot_type}'. "
                    f"Valid options are: {', '.join(pt.value for pt in PlotType)}"
                )

        if not isinstance(plot_type, PlotType):
            raise ValidationError(f"Invalid plot type: {plot_type}")

        renderer = self._renderers.get(plot_type, None)
        if renderer is None:
            raise ValidationError(f"No renderer registered for plot type: {plot_type}")
        return renderer

    def make_plot(
        self,
        data: typing.Union[TwoDimensionalGrid, npt.NDArray[np.floating]],
        plot_type: typing.Union[PlotType, str] = PlotType.HEATMAP,
        metadata: typing.Optional[PropertyMeta] = None,
        figure: typing.Optional[go.Figure] = None,
        title: typing.Optional[str] = None,
        x_coords: typing.Optional[npt.NDArray] = None,
        y_coords: typing.Optional[npt.NDArray] = None,
        x_label: str = "X",
        y_label: str = "Y",
        width: typing.Optional[int] = None,
        height: typing.Optional[int] = None,
        **kwargs: typing.Any,
    ) -> go.Figure:
        """
        Create a 2D plot of the provided data.

        :param data: 2D numpy array to visualize
        :param plot_type: Type of 2D plot to create
        :param metadata: Optional property metadata for labeling and scaling
        :param figure: Optional existing figure to add to
        :param title: Custom title for the plot
        :param x_coords: Optional x-coordinate array (defaults to indices)
        :param y_coords: Optional y-coordinate array (defaults to indices)
        :param x_label: Label for the x-axis
        :param y_label: Label for the y-axis
        :param width: Custom width for the plot
        :param height: Custom height for the plot
        :param kwargs: Additional plotting parameters specific to the plot type
        :return: Plotly figure object containing the 2D visualization
        """
        if data.ndim != 2:
            raise ValidationError("2D visualization requires 2D data array")

        if isinstance(plot_type, str):
            try:
                plot_type = PlotType(plot_type)
            except ValueError:
                raise ValidationError(f"Invalid plot type: {plot_type}")

        data = typing.cast(TwoDimensionalGrid, data)

        # Create default metadata if not provided
        if metadata is None:
            metadata = PropertyMeta(
                name="data",
                display_name="Data",
                unit="",
                log_scale=False,
                color_scheme=ColorScheme.VIRIDIS,
            )

        # Get renderer and create plot
        renderer = self.get_renderer(plot_type)
        fig = figure or go.Figure()
        fig = renderer.render(
            fig,
            data,
            metadata,
            x_coords=x_coords,
            y_coords=y_coords,
            x_label=x_label,
            y_label=y_label,
            **kwargs,
        )

        # Apply final layout updates
        layout_updates: typing.Dict[str, typing.Any] = {}

        if title:
            layout_updates["title"] = title
        elif not title and self.config.title:
            plot_name = PLOT_TYPE_NAMES.get(plot_type, "2D Plot")
            layout_updates["title"] = (
                f"{self.config.title}<br><sub>{plot_name}: {metadata.display_name}</sub>"
            )
        elif not title:
            plot_name = PLOT_TYPE_NAMES.get(plot_type, "2D Plot")
            layout_updates["title"] = f"{plot_name}: {metadata.display_name}"

        if width is not None:
            layout_updates["width"] = width
        if height is not None:
            layout_updates["height"] = height

        if layout_updates:
            fig.update_layout(**layout_updates)
        _register_figure(fig)
        return fig

    def make_plots(
        self,
        data_list: typing.Sequence[
            typing.Union[TwoDimensionalGrid, npt.NDArray[np.floating]]
        ],
        plot_types: typing.Union[
            PlotType, str, typing.Sequence[typing.Union[PlotType, str]]
        ],
        metadata_list: typing.Optional[typing.Sequence[PropertyMeta]] = None,
        titles: typing.Optional[typing.Sequence[str]] = None,
        rows: int = 1,
        cols: int = 1,
        x_coords: typing.Optional[np.ndarray] = None,
        y_coords: typing.Optional[np.ndarray] = None,
        x_label: str = "X",
        y_label: str = "Y",
        subplot_titles: typing.Optional[typing.Sequence[str]] = None,
        shared_xaxes: bool = True,
        shared_yaxes: bool = True,
        vertical_spacing: typing.Optional[float] = None,
        horizontal_spacing: typing.Optional[float] = None,
        **kwargs: typing.Any,
    ) -> go.Figure:
        """
        Create subplots with multiple 2D visualizations.

        :param data_list: Sequence of 2D data arrays to visualize
        :param plot_types: Plot type(s) to use (single type or list)
        :param metadata_list: Optional list of property metadata
        :param titles: Optional list of plot titles
        :param rows: Number of subplot rows
        :param cols: Number of subplot columns
        :param x_coords: Optional x-coordinate array
        :param y_coords: Optional y-coordinate array
        :param x_label: Label for x-axes
        :param y_label: Label for y-axes
        :param subplot_titles: Optional titles for each subplot
        :param shared_xaxes: Whether to share x-axes across subplots
        :param shared_yaxes: Whether to share y-axes across subplots
        :param vertical_spacing: Vertical spacing between subplots (0.0 to 1.0)
        :param horizontal_spacing: Horizontal spacing between subplots (0.0 to 1.0)
        :param kwargs: Additional plotting parameters
        :return: Plotly figure with subplots
        """
        if len(data_list) > rows * cols:
            raise ValidationError(
                f"Too many plots ({len(data_list)}) for {rows}x{cols} subplot grid"
            )

        if isinstance(plot_types, str):
            try:
                plot_types = PlotType(plot_types)
            except ValueError:
                raise ValidationError(f"Invalid plot type: {plot_types}")

        # Handle plot types
        if isinstance(plot_types, PlotType):
            plot_types = [plot_types] * len(data_list)
        elif isinstance(plot_types, collections.abc.Sequence):
            plot_types = [
                PlotType(pt) if isinstance(pt, str) else pt for pt in plot_types
            ]

        if len(plot_types) != len(data_list):
            raise ValidationError(
                "Number of plot types must match number of data arrays"
            )

        # Create subplot figure
        subplot_kwargs: typing.Dict[str, typing.Any] = {
            "rows": rows,
            "cols": cols,
            "subplot_titles": subplot_titles,
            "shared_xaxes": shared_xaxes,
            "shared_yaxes": shared_yaxes,
        }
        if vertical_spacing is not None:
            subplot_kwargs["vertical_spacing"] = vertical_spacing
        if horizontal_spacing is not None:
            subplot_kwargs["horizontal_spacing"] = horizontal_spacing

        fig = make_subplots(**subplot_kwargs)

        # Add each plot to its subplot
        for idx, (data, plot_type) in enumerate(zip(data_list, plot_types)):
            row = (idx // cols) + 1
            col = (idx % cols) + 1

            if data.ndim != 2:
                raise ValidationError("2D visualization requires 2D data array")
            data = typing.cast(TwoDimensionalGrid, data)

            # Create temporary figure for this subplot
            temp_fig = go.Figure()

            # Get metadata
            metadata = (
                metadata_list[idx]
                if metadata_list
                else PropertyMeta(
                    name=f"data_{idx}",
                    display_name=f"Data {idx}",
                    unit="",
                    log_scale=False,
                    color_scheme=ColorScheme.VIRIDIS,
                )
            )

            # Render on temporary figure
            renderer = self.get_renderer(plot_type)
            temp_fig = renderer.render(
                temp_fig,
                data,
                metadata,
                x_coords=x_coords,
                y_coords=y_coords,
                x_label=x_label,
                y_label=y_label,
                **kwargs,
            )

            # Add traces to main figure
            for trace in temp_fig.data:
                fig.add_trace(trace, row=row, col=col)

        # Update layout
        fig.update_layout(
            width=self.config.width,
            height=self.config.height,
            title=self.config.title or "2D Visualization Subplots",
        )
        _register_figure(fig)
        return fig

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
"""Global visualizer instance for 2D plots."""

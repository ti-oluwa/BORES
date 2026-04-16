"""Boundary condition implementations for 2D/3D reservoir grids."""

import enum
import functools
import threading
import typing

import attrs
import numpy as np
import numpy.typing as npt
from typing_extensions import ParamSpec, Self

from bores.constants import c
from bores.errors import DeserializationError, SerializationError, ValidationError
from bores.grids.base import CapillaryPressureGrids, RelativeMobilityGrids, RelPermGrids
from bores.models import FluidProperties, RockProperties
from bores.serialization import Serializable, make_serializable_type_registrar
from bores.stores import StoreSerializable
from bores.transmissibility import FaceTransmissibilities
from bores.types import NDimension, NDimensionalGrid, ThreeDimensions

__all__ = [
    "Boundary",
    "BoundaryCondition",
    "BoundaryConditions",
    "BoundaryMetadata",
    "BoundaryType",
    "CarterTracyAquifer",
    "DirichletBoundary",
    "GhostCellIndex",
    "NeumannBoundary",
    "ParameterizedBoundaryFunction",
    "RobinBoundary",
    "boundary_function",
    "get_boundary_function",
    "list_boundary_functions",
]

GhostCellIndex = typing.Tuple[int, int, int]
"""Padded-grid (i, j, k) index identifying a single ghost cell."""


# Boundary-function registry
_BOUNDARY_FUNCTIONS: typing.Dict[str, typing.Callable] = {}
_boundary_function_lock = threading.Lock()

P = ParamSpec("P")
R = typing.TypeVar("R")


@typing.overload
def boundary_function(func: typing.Callable[P, R]) -> typing.Callable[P, R]: ...


@typing.overload
def boundary_function(
    func: None = None,
    name: typing.Optional[str] = None,
    override: bool = False,
) -> typing.Callable[[typing.Callable[P, R]], typing.Callable[P, R]]: ...


def boundary_function(
    func: typing.Optional[typing.Callable[P, R]] = None,
    name: typing.Optional[str] = None,
    override: bool = False,
) -> typing.Union[
    typing.Callable[P, R],
    typing.Callable[[typing.Callable[P, R]], typing.Callable[P, R]],
]:
    """
    Register a callable as a named boundary function for serialization.

    A boundary function is any callable that computes values (pressure, flux,
    or a production-index array) to support a boundary condition. Registering
    a function allows `BoundaryCondition` subclasses that store it as an
    attribute to be serialized and deserialized by name.

    Usage:

    ```python
    @boundary_function
    def linear_pressure_gradient(x, y):
        return 2000 - 0.5 * x

    @boundary_function(name="custom_gradient")
    def my_gradient(x, y):
        return 2500 + 0.1 * x
    ```

    :param func: The function to register. When the decorator is used
        without arguments this is supplied automatically by Python.
    :param name: Optional registration key. Defaults to `func.__name__`.
    :param override: If *True*, an existing registration under the same name
        is silently replaced.
    :return: The (unmodified) function, or a decorator when called without
        a function argument.
    :raises ValidationError: If the function has no usable name or the name
        is already taken and *override* is *False*.
    """

    def decorator(fn: typing.Callable[P, R]) -> typing.Callable[P, R]:
        key = name or getattr(fn, "__name__", None)
        if not key:
            raise ValidationError(
                "Boundary function must have a `__name__` attribute or a name "
                "must be provided via the `name` argument."
            )
        with _boundary_function_lock:
            if not override and key in _BOUNDARY_FUNCTIONS:
                raise ValidationError(
                    f"Boundary function '{key}' is already registered. "
                    f"Use `override=True` or choose a different name."
                )
            _BOUNDARY_FUNCTIONS[key] = fn
        return fn

    if func is not None:
        return decorator(func)
    return decorator


def list_boundary_functions() -> typing.List[str]:
    """Return the names of all registered boundary functions."""
    with _boundary_function_lock:
        return list(_BOUNDARY_FUNCTIONS.keys())


def get_boundary_function(name: str) -> typing.Callable:
    """Return a registered boundary function by name.

    :param name: Registration key used when the function was decorated.
    :return: The callable.
    :raises ValidationError: If *name* is not found in the registry.
    """
    with _boundary_function_lock:
        if name not in _BOUNDARY_FUNCTIONS:
            raise ValidationError(
                f"Boundary function '{name}' is not registered. "
                f"Use `@boundary_function` to register it. "
                f"Available: {list(_BOUNDARY_FUNCTIONS.keys())}"
            )
        return _BOUNDARY_FUNCTIONS[name]


def _serialize_boundary_function(
    func: typing.Callable[..., typing.Any], recurse: bool = True
) -> typing.Dict[str, typing.Any]:
    """
    Serialize a boundary function to a JSON-compatible dict.

    Supports registered functions, `functools.partial` wrappers, and
    `ParameterizedBoundaryFunction` instances.

    :param func: The callable to serialize.
    :param recurse: Passed through to `ParameterizedBoundaryFunction.dump`.
    :return: A dict with a `"type"` discriminator key.
    :raises SerializationError: If the function cannot be identified.
    """
    with _boundary_function_lock:
        for reg_name, registered in _BOUNDARY_FUNCTIONS.items():
            if func is registered:
                return {"type": "registered", "name": reg_name}

    if isinstance(func, functools.partial):
        return {
            "type": "partial",
            "func": _serialize_boundary_function(func.func, recurse),
            "args": list(func.args),
            "kwargs": dict(func.keywords),
        }

    if isinstance(func, ParameterizedBoundaryFunction):
        return {"type": "parameterized", "data": func.dump(recurse)}

    raise SerializationError(
        f"Cannot serialize boundary function {func!r}. "
        f"Register it with `@boundary_function` first. "
        f"Available: {list(_BOUNDARY_FUNCTIONS.keys())}"
    )


def _deserialize_boundary_function(
    data: typing.Mapping[str, typing.Any],
) -> typing.Callable[..., typing.Any]:
    """
    Deserialize a boundary function from a dict produced by `_serialize_boundary_function`.

    :param data: Serialized representation.
    :return: The reconstructed callable.
    :raises DeserializationError: On unknown or malformed data.
    """
    func_type = data.get("type")

    if func_type == "registered":
        if "name" not in data:
            raise DeserializationError(
                "Missing 'name' key for registered boundary function."
            )
        return get_boundary_function(data["name"])

    if func_type == "partial":
        if "func" not in data:
            raise DeserializationError(
                "Missing 'func' key for partial boundary function."
            )
        base = _deserialize_boundary_function(data["func"])
        return functools.partial(base, *data.get("args", []), **data.get("kwargs", {}))

    if func_type == "parameterized":
        if "data" not in data:
            raise DeserializationError(
                "Missing 'data' key for parameterized boundary function."
            )
        return ParameterizedBoundaryFunction.load(data["data"])

    raise DeserializationError(
        f"Unknown boundary function type: '{func_type}'. "
        f"Valid types are 'registered', 'partial', 'parameterized'."
    )


class ParameterizedBoundaryFunction(
    Serializable,
    fields={"func_name": str, "params": typing.Dict[str, typing.Any]},
):
    """
    A fully serializable alternative to `functools.partial` for boundary functions.

    Stores a reference to a registered boundary function by name together with
    a fixed parameter dict. On call, the stored parameters are merged with any
    additional keyword arguments supplied by the caller.

    Usage:

    ```python
    @boundary_function
    def linear_pi(metadata, slope=1.0, intercept=0.0):
        # returns a production-index array shaped like the boundary slice
        ...

    alpha = ParameterizedBoundaryFunction(
        func_name="linear_pi",
        params={"slope": 2.5, "intercept": 50.0},
    )

    robin = RobinBoundary(pressure=2000.0, alpha=alpha)
    ```

    :param func_name: Name under which the base function was registered with
        `@boundary_function`.
    :param params: Fixed keyword parameters merged into every call.
    """

    def __init__(
        self,
        func_name: str,
        params: typing.Dict[str, typing.Any],
    ) -> None:
        self.func_name = func_name
        self.params = params
        self._func = get_boundary_function(func_name)

    def __call__(self, *args: typing.Any, **kwargs: typing.Any) -> np.ndarray:
        """
        Invoke the underlying function with stored parameters.

        :param args: Positional arguments forwarded to the base function.
        :param kwargs: Keyword arguments forwarded to the base function; these
            take precedence over the stored *params*.
        :return: Whatever the base function returns (typically an `np.ndarray`).
        """
        return self._func(*args, **{**self.params, **kwargs})

    def __dump__(self, recurse: bool = True) -> typing.Dict[str, typing.Any]:
        return {"func_name": self.func_name, "params": self.params}

    @classmethod
    def __load__(cls, data: typing.Mapping[str, typing.Any]) -> Self:
        return cls(func_name=data["func_name"], params=dict(data["params"]))


# Built-in registered alpha functions for `RobinBoundary`


@boundary_function(name="constant_productivity_index")
def _constant_productivity_index(
    metadata: "BoundaryMetadata",
    boundary_slice: typing.Tuple[slice, ...],
    direction: "Boundary",
    value: float = 1.0,
) -> np.ndarray:
    """
    Return a uniform production-index array equal to *value* over the boundary slice.

    This is the simplest alpha function for `RobinBoundary`: every ghost cell
    has the same productivity index. Use `ParameterizedBoundaryFunction` to
    set the actual numeric value:

    ```python
    alpha = ParameterizedBoundaryFunction(
        func_name="constant_productivity_index",
        params={"value": 25.0},
    )
    ```

    :param metadata: Boundary metadata (unused here, kept for API consistency).
    :param boundary_slice: Slice tuple identifying the boundary cells.
    :param direction: Face direction (unused here).
    :param value: Scalar productivity-index value (ft³/day/psi).
    :return: Array of shape matching `metadata.pressure_grid[boundary_slice]` filled with *value*.
    """
    assert metadata.fluid_properties is not None
    shape = metadata.fluid_properties.pressure_grid[boundary_slice].shape
    return np.full(shape, value, dtype=metadata.dtype)


@boundary_function(name="transmissibility_weighted_pi")
def _transmissibility_weighted_pi(
    metadata: "BoundaryMetadata",
    boundary_slice: typing.Tuple[slice, ...],
    direction: "Boundary",
    mobility_scale: float = 1.0,
) -> np.ndarray:
    """
    Return a production-index array derived from face transmissibilities and total mobility.

    The productivity index at each boundary cell is:

        PI[i,j,k] = T_face[i,j,k] * lambda_total[i,j,k] * mobility_scale

    where *T_face* is the geometric transmissibility of the face adjacent to
    the ghost cell and *lambda_total* is the sum of the three-phase relative
    mobilities at that cell.

    :param metadata: Boundary metadata. Must carry `face_transmissibilities`
        and `relative_mobility_grids`.
    :param boundary_slice: Slice tuple identifying the boundary cells.
    :param direction: Face direction; determines which transmissibility array
        (x, y, or z) is sampled.
    :param mobility_scale: Optional multiplicative scale applied to the result.
    :return: Array shaped like the boundary slice.
    :raises ValidationError: If required metadata fields are absent.
    """
    if metadata.face_transmissibilities is None:
        raise ValidationError(
            "'transmissibility_weighted_pi' requires `face_transmissibilities` "
            "in BoundaryMetadata."
        )
    if metadata.relative_mobility_grids is None:
        raise ValidationError(
            "'transmissibility_weighted_pi' requires `relative_mobility_grids` "
            "in BoundaryMetadata."
        )

    transmissibilities = metadata.face_transmissibilities
    if direction in (Boundary.LEFT, Boundary.RIGHT):
        T_face = transmissibilities.x[boundary_slice]
    elif direction in (Boundary.FRONT, Boundary.BACK):
        T_face = transmissibilities.y[boundary_slice]
    else:
        T_face = transmissibilities.z[boundary_slice]

    water_relative_mobility, oil_relative_mobility, gas_relative_mobility = (
        metadata.relative_mobility_grids
    )
    total_relative_mobility = (
        water_relative_mobility[boundary_slice]
        + oil_relative_mobility[boundary_slice]
        + gas_relative_mobility[boundary_slice]
    )
    return T_face * total_relative_mobility * mobility_scale


class Boundary(enum.Enum):
    """Enumeration of possible boundary face directions."""

    LEFT = "left"
    """The negative-x direction (west face)."""
    RIGHT = "right"
    """The positive-x direction (east face)."""
    FRONT = "front"
    """The negative-y direction (south face)."""
    BACK = "back"
    """The positive-y direction (north face)."""
    BOTTOM = "bottom"
    """The negative-z direction (shallowest face, k=0 convention)."""
    TOP = "top"
    """The positive-z direction (deepest face, k=-1 convention)."""


class BoundaryType(enum.Enum):
    """
    Discriminator that controls how a ghost-cell value is applied by the solver.

    - **FLUX** - the ghost-cell value is a volumetric flow rate (ft³/day) into
    the boundary face. The solver uses it as a Neumann source term.

    - **PRESSURE** - the ghost-cell value is a pressure (psi). The solver
    treats it as a Dirichlet constraint and computes the face flux from the
    pressure difference and transmissibility.
    """

    FLUX = "flux"
    PRESSURE = "pressure"


@attrs.frozen
class BoundaryMetadata:
    """
    Metadata bundle supplied to every boundary condition call.

    Every field is *Optional* so that boundary conditions that do not need
    the full context can be constructed cheaply. A boundary condition that
    requires a specific field should validate its presence at call time and
    raise `ValidationError` with a clear message.

    :param fluid_properties: Full fluid property object for the padded grid,
        carrying pressure, saturation, viscosity, density and PVT arrays.
    :param rock_properties: Rock property object for the padded grid, carrying
        porosity, permeability, compressibility and saturation endpoint grids.
    :param relative_permeability_grids: Three-phase relative permeability
        grids (krw, kro, krg) on the padded grid.
    :param relative_mobility_grids: Three-phase relative mobility grids
        (lambda_w, lambda_o, lambda_g) on the padded grid.
    :param capillary_pressure_grids: Oil-water and gas-oil capillary pressure
        grids on the padded grid.
    :param face_transmissibilities: Precomputed geometric face transmissibilities
        (x, y, z) for the padded grid in mD·ft.
    :param time: Current simulation time in seconds.
    :param grid_shape: Original (un-padded) grid shape as `(nx, ny, nz)`.
    :param cell_dimension: Physical cell dimensions `(dx, dy)` in feet.
    :param thickness_grid: Un-padded cell thickness array (ft).
    """

    fluid_properties: typing.Optional[FluidProperties] = None
    """Full `FluidProperties` object for the padded grid."""

    rock_properties: typing.Optional[RockProperties] = None
    """Full `RockProperties` object for the padded grid."""

    relative_permeability_grids: typing.Optional[RelPermGrids] = None
    """Three-phase relative permeability grids (krw, kro, krg)."""

    relative_mobility_grids: typing.Optional[RelativeMobilityGrids] = None
    """Three-phase relative mobility grids (lambda_w, lambda_o, lambda_g)."""

    capillary_pressure_grids: typing.Optional[CapillaryPressureGrids] = None
    """Oil-water and gas-oil capillary pressure grids."""

    face_transmissibilities: typing.Optional[FaceTransmissibilities] = None
    """Precomputed geometric face transmissibilities (x, y, z)."""

    time: typing.Optional[float] = None
    """Current simulation time (seconds)."""

    grid_shape: typing.Optional[typing.Tuple[int, ...]] = None
    """Original un-padded grid shape ``(nx, ny, nz)``."""

    cell_dimension: typing.Optional[typing.Tuple[float, float]] = None
    """Physical cell dimensions ``(dx, dy)`` in feet."""

    thickness_grid: typing.Optional[NDimensionalGrid] = None
    """Un-padded cell thickness array (ft)."""

    dtype: npt.DTypeLike = np.float64
    """Data type for any arrays returned by boundary conditions."""


def build_boundary_metadata(
    fluid_properties: FluidProperties[ThreeDimensions],
    rock_properties: RockProperties[ThreeDimensions],
    relperm_grids: RelPermGrids[ThreeDimensions],
    relative_mobility_grids: RelativeMobilityGrids[ThreeDimensions],
    capillary_pressure_grids: CapillaryPressureGrids[ThreeDimensions],
    face_transmissibilities: FaceTransmissibilities,
    grid_shape: ThreeDimensions,
    cell_dimension: typing.Tuple[float, float],
    thickness_grid: NDimensionalGrid[ThreeDimensions],
    time: float,
    dtype: npt.DTypeLike = np.float64,
) -> BoundaryMetadata:
    """
    Build a `BoundaryMetadata` bundle from the current simulation state.

    :param fluid_properties: Current fluid property object for the grid.
    :param rock_properties: Current rock property object for the grid.
    :param relperm_grids: Three-phase relative permeability grids (krw, kro, krg).
    :param relative_mobility_grids: Three-phase relative mobility grids.
    :param capillary_pressure_grids: Oil-water and gas-oil capillary pressure grids.
    :param face_transmissibilities: Precomputed geometric face transmissibilities.
    :param grid_shape: Un-padded grid shape `(nx, ny, nz)`.
    :param cell_dimension: Physical cell dimensions `(dx, dy)` in feet.
    :param thickness_grid: Un-padded cell thickness array (ft).
    :param time: Current simulation time (seconds).
    :return: Populated `BoundaryMetadata` instance.
    """
    return BoundaryMetadata(
        fluid_properties=fluid_properties,
        rock_properties=rock_properties,
        relative_permeability_grids=relperm_grids,
        relative_mobility_grids=relative_mobility_grids,
        capillary_pressure_grids=capillary_pressure_grids,
        face_transmissibilities=face_transmissibilities,
        time=time,
        grid_shape=grid_shape,
        cell_dimension=cell_dimension,
        thickness_grid=thickness_grid,
        dtype=dtype,
    )


class BoundaryCondition(
    StoreSerializable,
    typing.Generic[NDimension],
    serializers={"func": _serialize_boundary_function},
    deserializers={"func": _deserialize_boundary_function},
):
    """
    Abstract base for all boundary conditions.

    A boundary condition is a *callable* object. Concrete subclasses must
    implement `__call__` to return ghost-cell values and `get_type` to declare
    whether those values are fluxes or pressures.

    Serialization is inherited from `StoreSerializable`; each subclass that
    wants to participate in the registry must be decorated with
    `@boundary_condition` and implement `__dump__` / `__load__`.
    """

    __abstract_serializable__ = True

    def __call__(
        self,
        boundary_slice: typing.Tuple[slice, ...],
        direction: Boundary,
        metadata: BoundaryMetadata,
    ) -> np.ndarray:
        """
        Compute and return ghost-cell values for the specified boundary face.

        The returned array has the same shape as
        `metadata.fluid_properties.pressure_grid[boundary_slice]` (or any
        other grid on the padded domain sliced with *boundary_slice*).

        :param boundary_slice: Tuple of `slice` objects that, when applied to a
            padded grid, selects exactly the ghost-cell layer for this face.
            For example, the left boundary of a 3-D padded grid is
            `(slice(0, 1), slice(None), slice(None))`.
        :param direction: Which face of the domain this call covers.
        :param metadata: Simulation context bundle.
        :return: Array of ghost-cell values shaped like the boundary slice.
        """
        raise NotImplementedError

    def get_type(self) -> BoundaryType:
        """
        Return the `BoundaryType` that describes how the returned values are used.

        :return: `BoundaryType.FLUX` or `BoundaryType.PRESSURE`.
        """
        raise NotImplementedError

    def is_static(self) -> bool:
        """
        Return *True* if this condition produces the same values every time step.

        `BoundaryConditions.get_boundaries` uses this hint to avoid redundant
        recomputation: faces whose condition returns *True* here are evaluated
        once and cached; all others are re-evaluated on every call.

        The default implementation returns *False* (safe but conservative).
        Subclasses that know they are time-independent should override this.

        :return: *True* if the ghost-cell values are independent of time and
            the current field state.
        """
        return False


_BOUNDARY_CONDITIONS: typing.Dict[str, typing.Type[BoundaryCondition]] = {}
boundary_condition = make_serializable_type_registrar(
    base_cls=BoundaryCondition,
    registry=_BOUNDARY_CONDITIONS,
    lock=threading.Lock(),
    key_attr="__type__",
    override=False,
    auto_register_serializer=True,
    auto_register_deserializer=True,
)
"""Decorator that registers a `BoundaryCondition` subclass for serialization."""


def get_neighbour_slice(
    boundary_slice: typing.Tuple[slice, ...], direction: Boundary
) -> typing.Tuple[slice, ...]:
    """
    Return the slice of interior cells adjacent to a ghost-cell layer.

    :param boundary_slice: Slice that selects the ghost-cell layer.
    :param direction: Face direction.
    :return: Slice that selects the immediately adjacent interior cells.
    :raises ValidationError: If a z-direction is requested on a 2-D grid.
    """
    nbr = list(boundary_slice)
    ndim = len(boundary_slice)

    if direction in (Boundary.BOTTOM, Boundary.TOP) and ndim < 3:
        raise ValidationError(
            f"Cannot use {direction.name} boundary on a {ndim}D grid. "
            "Z-direction boundaries require 3-D grids."
        )

    if direction == Boundary.LEFT:
        nbr[0] = slice(1, 2)
    elif direction == Boundary.RIGHT:
        nbr[0] = slice(-2, -1)
    elif direction == Boundary.FRONT:
        nbr[1] = slice(1, 2)
    elif direction == Boundary.BACK:
        nbr[1] = slice(-2, -1)
    elif direction == Boundary.BOTTOM:
        nbr[2] = slice(1, 2)
    elif direction == Boundary.TOP:
        nbr[2] = slice(-2, -1)

    return tuple(nbr)


@boundary_condition
@attrs.frozen
class NeumannBoundary(BoundaryCondition[NDimension]):
    """
    Constant-flux (Neumann) boundary condition.

    Returns a uniform array equal to `flux` (ft³/day) over the ghost-cell
    layer. The default flux is **0 ft³/day**, which is equivalent to a
    no-flow (sealed) boundary.

    A positive value represents flow *into* the reservoir; a negative value
    represents flow *out of* the reservoir. Users who specify rates in
    bbl/day must convert to ft³/day (1 bbl = 5.614583 ft³) before
    constructing this object (or use `c.BARRELS_TO_CUBIC_FEET` multiplier).

    :param flux: Volumetric flow rate (ft³/day) applied uniformly across
        the boundary face. Default is *0.0* (no-flow).

    Example:
    ``python
    # No-flow (sealed) boundary - default
    sealed = NeumannBoundary()

    # Water injection at 500 ft³/day
    injector = NeumannBoundary(flux=500.0)

    # Production at 200 bbl/day converted to ft³/day
    producer = NeumannBoundary(flux=-200.0 * 5.614583)
    ```
    """

    __type__ = "neumann_boundary"

    flux: float = 0.0
    """Volumetric flow rate (ft³/day). Positive = into reservoir."""

    def __call__(
        self,
        boundary_slice: typing.Tuple[slice, ...],
        direction: Boundary,
        metadata: BoundaryMetadata,
    ) -> np.ndarray:
        """
        Return a uniform array filled with `flux`.

        :param boundary_slice: Ghost-cell layer slice.
        :param direction: Face direction (unused; flux is spatially uniform).
        :param metadata: Simulation context. Must carry `fluid_properties` so
            that the returned array shape can be determined.
        :return: Array shaped like the boundary slice, every element equal to `self.flux`.
        :raises ValidationError: If `metadata.fluid_properties` is *None*.
        """
        if metadata.fluid_properties is None:
            raise ValidationError(
                f"{self.__class__.__name__} requires `metadata.fluid_properties` to determine "
                "the boundary slice shape."
            )
        shape = metadata.fluid_properties.pressure_grid[boundary_slice].shape
        return np.full(shape, self.flux, dtype=metadata.dtype)

    def get_type(self) -> BoundaryType:
        """
        Return `BoundaryType.FLUX`.

        :return: `BoundaryType.FLUX`
        """
        return BoundaryType.FLUX

    def is_static(self) -> bool:
        """
        Return *True*; a constant flux never changes between time steps.

        :return: *True*
        """
        return True

    def is_noflow(self) -> bool:
        return self.flux == 0

    def __dump__(self, recurse: bool = True) -> typing.Dict[str, typing.Any]:
        return {"flux": self.flux}

    @classmethod
    def __load__(cls, data: typing.Mapping[str, typing.Any]) -> Self:
        return cls(flux=float(data.get("flux", 0.0)))


@boundary_condition
@attrs.frozen
class DirichletBoundary(BoundaryCondition[NDimension]):
    """
    Constant-pressure (Dirichlet) boundary condition.

    Returns a uniform array equal to `pressure` (psi) over the ghost-cell
    layer. The solver uses this value directly as the ghost-cell pressure,
    driving flow across the face according to the local transmissibility.

    :param pressure: Prescribed boundary pressure (psi).

    Example:

    ```python
    # Fixed pressure inlet at 3 000 psi
    inlet = DirichletBoundary(pressure=3000.0)

    # Fixed pressure outlet at 1 200 psi
    outlet = DirichletBoundary(pressure=1200.0)
    ```
    """

    __type__ = "dirichlet_boundary"

    pressure: float
    """Boundary pressure (psi)."""

    def __call__(
        self,
        boundary_slice: typing.Tuple[slice, ...],
        direction: Boundary,
        metadata: BoundaryMetadata,
    ) -> np.ndarray:
        """
        Return a uniform array filled with `pressure`.

        :param boundary_slice: Ghost-cell layer slice.
        :param direction: Face direction (unused; pressure is spatially uniform).
        :param metadata: Simulation context. Must carry `fluid_properties`.
        :return: Array shaped like the boundary slice, every element equal to `self.pressure`.
        :raises ValidationError: If `metadata.fluid_properties` is *None*.
        """
        if metadata.fluid_properties is None:
            raise ValidationError(
                f"{self.__class__.__name__} requires `metadata.fluid_properties` to determine "
                "the boundary slice shape."
            )
        shape = metadata.fluid_properties.pressure_grid[boundary_slice].shape
        return np.full(shape, self.pressure, dtype=metadata.dtype)

    def get_type(self) -> BoundaryType:
        """
        Return `BoundaryType.PRESSURE`.

        :return: `BoundaryType.PRESSURE`
        """
        return BoundaryType.PRESSURE

    def is_static(self) -> bool:
        """
        Return *True*; a constant pressure never changes between time steps.

        :return: *True*
        """
        return True

    def __dump__(self, recurse: bool = True) -> typing.Dict[str, typing.Any]:
        return {"pressure": self.pressure}

    @classmethod
    def __load__(cls, data: typing.Mapping[str, typing.Any]) -> Self:
        return cls(pressure=float(data["pressure"]))


@boundary_condition
@attrs.frozen
class RobinBoundary(BoundaryCondition[NDimension]):
    """
    Robin (mixed) boundary condition that returns a volumetric flux (ft³/day).

    The boundary flux is computed as a productivity-index formulation:

    ```
    q[i,j,k] = alpha(metadata, boundary_slice, direction)[i,j,k]
                * (pressure - P_interior[i,j,k])
    ```

    where:

    - *pressure* is the user-specified reference boundary pressure (psi),
    - *P_interior* is the pressure of the adjacent interior cell (psi), and
    - *alpha* is a user-supplied callable that returns a production-index-like
      array (ft³/day/psi) shaped like the boundary slice.

    A positive result means flow *into* the reservoir; a negative result means
    flow *out of* the reservoir, consistent with the sign convention used
    throughout the simulator.

    The `alpha` callable must accept the same positional arguments as
    `BoundaryCondition.__call__`, i.e. `(metadata, boundary_slice, direction)`,
    and must return an `np.ndarray` shaped like the boundary slice. You can use a registered
    `@boundary_function`, a `ParameterizedBoundaryFunction`, or any callable
    that satisfies this contract.

    **Built-in alpha functions**:

    Two ready-to-use alpha functions are registered by default:

    - `"constant_productivity_index"` - uniform PI equal to a scalar
      *value* parameter (ft³/day/psi).
    - `"transmissibility_weighted_pi"` - `T_face * lambda_total`, where
      *T_face* is the geometric face transmissibility and *lambda_total* is
      the sum of three-phase relative mobilities.

    :param pressure: Reference boundary pressure (psi).  Typically the
        aquifer pressure, injection manifold pressure, or any target pressure
        at the face.
    :param alpha: Callable `(metadata, boundary_slice, direction) -> np.ndarray`
        returning a production-index array (ft³/day/psi). Must be a registered
        boundary function or a `ParameterizedBoundaryFunction` so that the
        condition can be serialized.

    Example::

        alpha = ParameterizedBoundaryFunction(
            func_name="constant_productivity_index",
            params={"value": 10.0},
        )
        robin = RobinBoundary(pressure=2500.0, alpha=alpha)
    """

    __type__ = "robin_boundary"

    pressure: float
    """Reference boundary pressure (psi)."""

    alpha: typing.Callable[
        ["BoundaryMetadata", typing.Tuple[slice, ...], Boundary],
        np.ndarray,
    ]
    """Production-index callable (ft³/day/psi) shaped like the boundary slice."""

    def __call__(
        self,
        boundary_slice: typing.Tuple[slice, ...],
        direction: Boundary,
        metadata: BoundaryMetadata,
    ) -> np.ndarray:
        """
        Compute the boundary flux array.

        :param boundary_slice: Ghost-cell layer slice.
        :param direction: Face direction.
        :param metadata: Simulation context. Must carry `fluid_properties` so
            that the interior neighbour pressure can be read.
        :return: Array of volumetric fluxes (ft³/day) shaped like the boundary
            slice. Positive = flow into the reservoir.
        :raises ValidationError: If `metadata.fluid_properties` is *None*.
        """
        if metadata.fluid_properties is None:
            raise ValidationError(
                f"{self.__class__.__name__} requires `metadata.fluid_properties` to read "
                "interior neighbour pressures."
            )

        nbr_slice = get_neighbour_slice(boundary_slice, direction)
        p_interior = metadata.fluid_properties.pressure_grid[nbr_slice].astype(
            metadata.dtype, copy=False
        )
        pi_array = np.asarray(
            self.alpha(metadata, boundary_slice, direction), dtype=metadata.dtype
        )
        return pi_array * (self.pressure - p_interior)

    def get_type(self) -> BoundaryType:
        """Return `BoundaryType.FLUX`.

        :return: `BoundaryType.FLUX`
        """
        return BoundaryType.FLUX

    def is_static(self) -> bool:
        """Return *False*; flux depends on the current interior pressure field.

        :return: *False*
        """
        return False

    def __dump__(self, recurse: bool = True) -> typing.Dict[str, typing.Any]:
        return {
            "pressure": self.pressure,
            "alpha": _serialize_boundary_function(self.alpha, recurse),
        }

    @classmethod
    def __load__(cls, data: typing.Mapping[str, typing.Any]) -> Self:
        return cls(
            pressure=float(data["pressure"]),
            alpha=_deserialize_boundary_function(data["alpha"]),
        )


@boundary_condition
@attrs.define
class CarterTracyAquifer(BoundaryCondition[NDimension]):
    """
    Carter-Tracy finite aquifer boundary condition.

    Computes transient water influx from a finite radial aquifer using the
    Van Everdingen-Hurst (1949) superposition approach as reformulated by
    Carter and Tracy (1960). The condition returns a volumetric flux array
    (ft³/day) - positive means water flows *into* the reservoir.

    **Two construction modes**

    *Physical properties (recommended)* - supply `aquifer_permeability`,
    `aquifer_porosity`, `aquifer_compressibility`, `water_viscosity`,
    `inner_radius`, `outer_radius`, and `aquifer_thickness`. The aquifer
    constant *B* and hydraulic diffusivity are derived from first principles.

    *Calibrated constant (legacy)* - supply a history-matched
    `aquifer_constant` (bbl/psi) directly, together with `dimensionless_radius_ratio`.

    **Sign convention**

    The returned flux follows the simulator-wide convention: positive values
    represent influx (injection) into the reservoir, negative values represent
    withdrawal.

    **Units**

    All flux values are in ft³/day. Internal pressure drops are in psi.
    Time is in days (metadata carries seconds; conversion is applied internally).

    :param initial_pressure: Initial aquifer / reservoir pressure (psi).
    :param aquifer_permeability: Aquifer horizontal permeability (mD).
    :param aquifer_porosity: Aquifer porosity (fraction).
    :param aquifer_compressibility: Total aquifer compressibility, rock + water (psi⁻¹).
    :param water_viscosity: Water viscosity at reservoir conditions (cP).
    :param inner_radius: Reservoir-aquifer contact radius (ft).
    :param outer_radius: Aquifer outer extent (ft).
    :param aquifer_thickness: Aquifer thickness (ft).
    :param aquifer_constant: Pre-computed aquifer constant *B* (bbl/psi) for calibrated-constant mode.
    :param dimensionless_radius_ratio: r_outer / r_inner used in calibrated-constant mode. Default is *10.0*.
    :param angle: Aquifer encroachment angle in degrees. *360* = full contact;
        *180* = half-circle edge drive.  Default is *360*.

    **References**:

    Carter, R.D. and Tracy, G.W. (1960). *An Improved Method for Calculating
    Water Influx.* JPT, 12(5), 415-417.

    Van Everdingen, A.F. and Hurst, W. (1949). *The Application of the Laplace
    Transformation to Flow Problems in Reservoirs.* Trans. AIME, 186, 305-324.
    """

    __type__ = "carter_tracy_aquifer"

    initial_pressure: float
    """Initial aquifer / reservoir pressure (psi)."""

    aquifer_permeability: typing.Optional[float] = attrs.field(default=None)
    """Aquifer horizontal permeability (mD)."""

    aquifer_porosity: typing.Optional[float] = attrs.field(default=None)
    """Aquifer porosity (fraction)."""

    aquifer_compressibility: typing.Optional[float] = attrs.field(default=None)
    """Total aquifer compressibility, rock + water (psi⁻¹)."""

    water_viscosity: typing.Optional[float] = attrs.field(default=None)
    """Water viscosity at reservoir conditions (cP)."""

    inner_radius: typing.Optional[float] = attrs.field(default=None)
    """Reservoir-aquifer contact radius (ft)."""

    outer_radius: typing.Optional[float] = attrs.field(default=None)
    """Aquifer outer extent (ft)."""

    aquifer_thickness: typing.Optional[float] = attrs.field(default=None)
    """Aquifer thickness (ft)."""

    aquifer_constant: typing.Optional[float] = attrs.field(default=None)
    """Pre-computed aquifer constant *B* (bbl/psi) - calibrated-constant mode."""

    dimensionless_radius_ratio: float = attrs.field(default=10.0)
    """r_outer / r_inner - used only in calibrated-constant mode."""

    angle: float = attrs.field(default=360.0)
    """Aquifer encroachment angle (degrees).  360 = full contact."""

    _pressure_history: typing.List[typing.Tuple[float, float]] = attrs.field(
        factory=list, init=False
    )
    """History of (time_days, pressure_drop_psi) pairs for superposition."""

    _cumulative_influx: float = attrs.field(default=0.0, init=False)
    """Cumulative water influx (ft³ or bbl, consistent with *B* units)."""

    _computed_aquifer_constant: typing.Optional[float] = attrs.field(
        default=None, init=False
    )
    """Derived or user-supplied aquifer constant *B*."""

    _computed_dimensionless_radius_ratio: typing.Optional[float] = attrs.field(
        default=None, init=False
    )
    """Derived or user-supplied dimensionless radius ratio r_D."""

    _hydraulic_diffusivity: typing.Optional[float] = attrs.field(
        default=None, init=False
    )
    """Hydraulic diffusivity η (ft²/day) - *None* in calibrated-constant mode."""

    def __attrs_post_init__(self) -> None:
        has_physical = all(
            x is not None
            for x in (
                self.aquifer_permeability,
                self.aquifer_porosity,
                self.aquifer_compressibility,
                self.water_viscosity,
                self.inner_radius,
                self.outer_radius,
                self.aquifer_thickness,
            )
        )
        has_constant = self.aquifer_constant is not None

        if not (has_physical or has_constant):
            raise ValidationError(
                f"{type(self).__name__!r} requires either:\n"
                "  Physical properties: aquifer_permeability, aquifer_porosity, "
                "aquifer_compressibility, water_viscosity, inner_radius, "
                "outer_radius, aquifer_thickness.\n"
                "  Or a calibrated constant: aquifer_constant."
            )

        if has_physical:
            assert self.inner_radius is not None and self.outer_radius is not None
            assert self.aquifer_permeability is not None
            assert self.aquifer_porosity is not None
            assert self.aquifer_compressibility is not None
            assert self.water_viscosity is not None
            assert self.aquifer_thickness is not None

            r_D = self.outer_radius / self.inner_radius
            object.__setattr__(self, "_computed_dimensionless_radius_ratio", r_D)

            angle_fraction = self.angle / 360.0
            r_sq_diff = self.outer_radius**2 - self.inner_radius**2
            B = (
                1.119
                * self.aquifer_porosity
                * self.aquifer_compressibility
                * angle_fraction
                * r_sq_diff
                * self.aquifer_thickness
                / self.water_viscosity
            )
            object.__setattr__(self, "_computed_aquifer_constant", B)

            eta = (
                0.006328
                * self.aquifer_permeability
                / (
                    self.aquifer_porosity
                    * self.water_viscosity
                    * self.aquifer_compressibility
                )
            )
            object.__setattr__(self, "_hydraulic_diffusivity", eta)

        else:
            object.__setattr__(
                self, "_computed_aquifer_constant", self.aquifer_constant
            )
            object.__setattr__(
                self,
                "_computed_dimensionless_radius_ratio",
                self.dimensionless_radius_ratio,
            )
            object.__setattr__(self, "_hydraulic_diffusivity", None)

    def __call__(
        self,
        boundary_slice: typing.Tuple[slice, ...],
        direction: Boundary,
        metadata: BoundaryMetadata,
    ) -> np.ndarray:
        """
        Compute the aquifer water influx flux array.

        If `metadata.time` is *None* the aquifer falls back to a constant
        influx rate of zero (no support) rather than raising an error,
        which is the safest option when timing information is unavailable.

        :param boundary_slice: Ghost-cell layer slice.
        :param direction: Face direction the aquifer is attached to.
        :param metadata: Simulation context. Must carry `fluid_properties`
            so that boundary-face pressures can be read, and `time`
            (seconds) for the superposition computation.
        :return: Array of volumetric influx rates (ft³/day) shaped like the
            boundary slice. Positive = water flows into the reservoir.
        :raises ValidationError: If `metadata.fluid_properties` is *None*.
        """
        if metadata.fluid_properties is None:
            raise ValidationError(
                f"{self.__class__.__name__} requires `metadata.fluid_properties`."
            )

        shape = metadata.fluid_properties.pressure_grid[boundary_slice].shape
        if metadata.time is None:
            return np.zeros(shape, dtype=metadata.dtype)

        current_time_days = metadata.time * c.DAYS_PER_SECOND

        nbr_slice = get_neighbour_slice(boundary_slice, direction)
        avg_pressure = float(
            np.mean(metadata.fluid_properties.pressure_grid[nbr_slice])
        )
        pressure_drop = self.initial_pressure - avg_pressure

        history = self._pressure_history
        if not history or current_time_days > history[-1][0]:
            history.append((current_time_days, pressure_drop))

        influx_rate = self._compute_influx_rate(current_time_days)

        if len(history) > 1:
            dt = current_time_days - history[-2][0]
            object.__setattr__(
                self, "_cumulative_influx", self._cumulative_influx + influx_rate * dt
            )

        # Distribute the scalar influx uniformly across the boundary face cells.
        n_cells = int(np.prod(shape)) if shape else 1
        per_cell_flux = influx_rate / n_cells if n_cells > 0 else 0.0
        return np.full(shape, per_cell_flux, dtype=metadata.dtype)

    def get_type(self) -> BoundaryType:
        """
        Return `BoundaryType.FLUX`.

        :return: `BoundaryType.FLUX`
        """
        return BoundaryType.FLUX

    def is_static(self) -> bool:
        """Return *False*; aquifer influx changes every time step.

        :return: *False*
        """
        return False

    def _compute_influx_rate(self, current_time_days: float) -> float:
        """
        Compute water influx rate using Van Everdingen-Hurst superposition.

        :param current_time_days: Current simulation time (days).
        :return: Total influx rate (bbl/day or ft³/day consistent with *B* units).
        """
        if not self._pressure_history:
            return 0.0

        assert self._computed_aquifer_constant is not None
        assert self._computed_dimensionless_radius_ratio is not None

        total = 0.0
        for t_i, dp_i in self._pressure_history:
            if dp_i <= 0.0:
                continue
            dt = current_time_days - t_i
            if dt <= 0.0:
                continue

            if self._hydraulic_diffusivity is not None:
                assert self.inner_radius is not None
                t_D = self._hydraulic_diffusivity * dt / (self.inner_radius**2)
            else:
                t_D = dt

            W_D_prime = self._van_everdingen_hurst_derivative(
                t_D, self._computed_dimensionless_radius_ratio
            )
            total += dp_i * W_D_prime

        if self._hydraulic_diffusivity is not None:
            return self._computed_aquifer_constant * total
        else:
            return self._computed_aquifer_constant * (self.angle / 360.0) * total

    @staticmethod
    def _van_everdingen_hurst_derivative(t_D: float, r_D: float) -> float:
        """
        Evaluate the dimensionless water influx derivative W_D'(t_D, r_D).

        Uses a Chatas-style approximation with a sigmoid blend between the
        early-time (infinite-acting) and late-time (boundary-dominated)
        regimes.

        :param t_D: Dimensionless time.
        :param r_D: Dimensionless radius ratio (r_outer / r_inner).
        :return: W_D'(t_D, r_D) - dimensionless.
        """
        if t_D <= 0.0:
            return 0.0

        early = float(np.sqrt(t_D / np.pi))

        if t_D < 0.1:
            return early

        r_D_sq = r_D * r_D
        denom = r_D_sq - 1.0
        if denom < 1e-6:
            return early

        beta = (np.pi**2) / denom
        late = (2.0 * r_D_sq / denom) * np.exp(-beta * t_D)

        weight = 1.0 / (1.0 + np.exp(-10.0 * (t_D - 0.5)))
        return float((1.0 - weight) * early + weight * late)

    def __dump__(self, recurse: bool = True) -> typing.Dict[str, typing.Any]:
        data: typing.Dict[str, typing.Any] = {
            "initial_pressure": self.initial_pressure,
            "angle": self.angle,
            "pressure_history": list(self._pressure_history),
            "cumulative_influx": self._cumulative_influx,
        }
        if self._hydraulic_diffusivity is not None:
            data.update(
                {
                    "aquifer_permeability": self.aquifer_permeability,
                    "aquifer_porosity": self.aquifer_porosity,
                    "aquifer_compressibility": self.aquifer_compressibility,
                    "water_viscosity": self.water_viscosity,
                    "inner_radius": self.inner_radius,
                    "outer_radius": self.outer_radius,
                    "aquifer_thickness": self.aquifer_thickness,
                }
            )
        else:
            data.update(
                {
                    "aquifer_constant": self.aquifer_constant,
                    "dimensionless_radius_ratio": self.dimensionless_radius_ratio,
                }
            )
        return data

    @classmethod
    def __load__(cls, data: typing.Mapping[str, typing.Any]) -> Self:
        if "aquifer_permeability" in data:
            instance = cls(
                initial_pressure=data["initial_pressure"],
                aquifer_permeability=data["aquifer_permeability"],
                aquifer_porosity=data["aquifer_porosity"],
                aquifer_compressibility=data["aquifer_compressibility"],
                water_viscosity=data["water_viscosity"],
                inner_radius=data["inner_radius"],
                outer_radius=data["outer_radius"],
                aquifer_thickness=data["aquifer_thickness"],
                angle=data.get("angle", 360.0),
            )
        else:
            instance = cls(
                initial_pressure=data["initial_pressure"],
                aquifer_constant=data["aquifer_constant"],
                dimensionless_radius_ratio=data.get("dimensionless_radius_ratio", 10.0),
                angle=data.get("angle", 360.0),
            )
        # Restore mutable history
        object.__setattr__(
            instance, "_pressure_history", list(data.get("pressure_history", []))
        )
        object.__setattr__(
            instance, "_cumulative_influx", float(data.get("cumulative_influx", 0.0))
        )
        return instance


# Face-descriptor helpers
_FACE_SLICES_2D: typing.Dict[
    Boundary, typing.Tuple[typing.Tuple[slice, ...], typing.Tuple[slice, ...]]
] = {
    Boundary.LEFT: (
        (slice(0, 1), slice(None)),
        (slice(1, 2), slice(None)),
    ),
    Boundary.RIGHT: (
        (slice(-1, None), slice(None)),
        (slice(-2, -1), slice(None)),
    ),
    Boundary.FRONT: (
        (slice(None), slice(0, 1)),
        (slice(None), slice(1, 2)),
    ),
    Boundary.BACK: (
        (slice(None), slice(-1, None)),
        (slice(None), slice(-2, -1)),
    ),
}

_FACE_SLICES_3D: typing.Dict[
    Boundary, typing.Tuple[typing.Tuple[slice, ...], typing.Tuple[slice, ...]]
] = {
    Boundary.LEFT: (
        (slice(0, 1), slice(None), slice(None)),
        (slice(1, 2), slice(None), slice(None)),
    ),
    Boundary.RIGHT: (
        (slice(-1, None), slice(None), slice(None)),
        (slice(-2, -1), slice(None), slice(None)),
    ),
    Boundary.FRONT: (
        (slice(None), slice(0, 1), slice(None)),
        (slice(None), slice(1, 2), slice(None)),
    ),
    Boundary.BACK: (
        (slice(None), slice(-1, None), slice(None)),
        (slice(None), slice(-2, -1), slice(None)),
    ),
    Boundary.BOTTOM: (
        (slice(None), slice(None), slice(0, 1)),
        (slice(None), slice(None), slice(1, 2)),
    ),
    Boundary.TOP: (
        (slice(None), slice(None), slice(-1, None)),
        (slice(None), slice(None), slice(-2, -1)),
    ),
}


def _face_slices(
    ndim: int,
) -> typing.Dict[
    Boundary, typing.Tuple[typing.Tuple[slice, ...], typing.Tuple[slice, ...]]
]:
    """
    Return the ghost-slice / neighbour-slice pair dict for a given grid dimensionality.

    :param ndim: Grid dimensionality (2 or 3).
    :return: Dict mapping each `Boundary` face to a 2-tuple of `(ghost_slice, neighbour_slice)`.
    :raises ValidationError: If *ndim* is not 2 or 3.
    """
    if ndim == 2:
        return _FACE_SLICES_2D
    if ndim == 3:
        return _FACE_SLICES_3D
    raise ValidationError(f"Grid must be 2-D or 3-D, got {ndim}-D.")


@typing.final
@attrs.frozen
class BoundaryConditions(Serializable, typing.Generic[NDimension]):
    """
    Container that assigns a `BoundaryCondition` to every face of the grid.

    For a 2-D grid the active faces are left, right, front, and back.
    For a 3-D grid all six faces are active (bottom and top are added).

    Defaults to `NeumannBoundary()` (zero flux, no-flow) on every face.

    The central method is `get_boundaries`, which evaluates each face's
    condition and assembles two sparse boundary grids. One for
    `FLUX` ghost cells and one for `PRESSURE` ghost cells.

    A per-face *static cache* avoids recomputing constant conditions
    (e.g. `NeumannBoundary` or `DirichletBoundary`) on every time step.
    Only faces whose condition returns `is_static() == False` are
    re-evaluated on subsequent calls to `get_boundaries`.

    :param left: Condition for the x- face (west).
    :param right: Condition for the x+ face (east).
    :param front: Condition for the y- face (south).
    :param back: Condition for the y+ face (north).
    :param bottom: Condition for the z- face (shallowest layer).
    :param top: Condition for the z+ face (deepest layer).

    Example:

    ```python
    from bores.boundary_conditions import (
        BoundaryConditions, NeumannBoundary, DirichletBoundary,
        RobinBoundary, ParameterizedBoundaryFunction,
    )

    alpha = ParameterizedBoundaryFunction(
        func_name="constant_productivity_index",
        params={"value": 5.0},
    )
    bc = BoundaryConditions(
        left=DirichletBoundary(pressure=3000.0),
        right=RobinBoundary(pressure=1500.0, alpha=alpha),
        front=NeumannBoundary(),   # no-flow
        back=NeumannBoundary(),    # no-flow
    )

    flux_grid, pressure_grid = bc.get_boundaries(
        grid_shape=model.grid_shape,
        metadata=metadata,
        pad_width=1,
    )
    ```
    """

    left: BoundaryCondition = attrs.field(factory=NeumannBoundary)
    """Condition for the x- face (west)."""
    right: BoundaryCondition = attrs.field(factory=NeumannBoundary)
    """Condition for the x+ face (east)."""
    front: BoundaryCondition = attrs.field(factory=NeumannBoundary)
    """Condition for the y- face (south)."""
    back: BoundaryCondition = attrs.field(factory=NeumannBoundary)
    """Condition for the y+ face (north)."""
    bottom: BoundaryCondition = attrs.field(factory=NeumannBoundary)
    """Condition for the z- face (shallowest layer)."""
    top: BoundaryCondition = attrs.field(factory=NeumannBoundary)
    """Condition for the z+ face (deepest layer)."""

    _flux_cache: typing.Optional[NDimensionalGrid] = attrs.field(
        default=None, init=False, repr=False
    )
    """Cached sparse flux boundary grid (NaN where no flux BC)."""

    _pressure_cache: typing.Optional[NDimensionalGrid] = attrs.field(
        default=None, init=False, repr=False
    )
    """Cached sparse pressure boundary grid (NaN where no pressure BC)."""

    _cache_initialised: bool = attrs.field(default=False, init=False, repr=False)
    """*True* once the first `get_boundaries` call has populated the caches."""

    _all_static: typing.Optional[bool] = attrs.field(
        default=None, init=False, repr=False
    )
    """Cache indicating whether all boundary conditions are static (determined on first call)."""

    def _face_conditions(
        self, ndim: int
    ) -> typing.List[typing.Tuple[Boundary, BoundaryCondition]]:
        """
        Return the (direction, condition) pairs active for a grid of *ndim* dimensions.

        :param ndim: Grid dimensionality (2 or 3).
        :return: List of ``(Boundary, BoundaryCondition)`` pairs.
        """
        pairs: typing.List[typing.Tuple[Boundary, BoundaryCondition]] = [
            (Boundary.LEFT, self.left),
            (Boundary.RIGHT, self.right),
            (Boundary.FRONT, self.front),
            (Boundary.BACK, self.back),
        ]
        if ndim == 3:
            pairs.extend(
                [
                    (Boundary.BOTTOM, self.bottom),
                    (Boundary.TOP, self.top),
                ]
            )
        return pairs

    def get_boundaries(
        self,
        grid_shape: typing.Tuple[int, ...],
        metadata: BoundaryMetadata,
        pad_width: int = 1,
    ) -> typing.Tuple[NDimensionalGrid, NDimensionalGrid]:
        """
        Evaluate all face boundary conditions and return two sparse boundary grids.

        On the *first* call every face is evaluated and the results are stored
        in the internal caches. On *subsequent* calls only faces whose
        condition reports `is_static() == False` are re-evaluated; static
        faces reuse their cached values. This means that for a model with
        only no-flow or constant-pressure boundaries the boundary evaluation
        cost after the first time step is essentially zero.

        The returned grids are sparse arrays of shape `(nx+2*pad_width, ny+2*pad_width, nz+2*pad_width)`
        where boundary cells contain the boundary value and all other cells are NaN.

        Both arrays are the *same* Python objects that back the cache, so
        callers must not mutate them; treat them as read-only views.

        :param grid_shape: Un-padded grid shape `(nx, ny)` or `(nx, ny, nz)`.
        :param metadata: Simulation context bundle passed through to each
            boundary condition.
        :param pad_width: Ghost-cell layer width. Must match the padding used
            when the grids were created. Default is *1*.
        :return: A 2-tuple `(flux_grid, pressure_grid)` where each element is
            a sparse NDimensionalGrid with shape matching the padded grid.
            *flux_grid* - boundary cells controlled by a FLUX-type condition,
            values in ft³/day, NaN elsewhere.
            *pressure_grid* - boundary cells controlled by a PRESSURE-type
            condition, values in psi, NaN elsewhere.
        :raises ValidationError: If *grid_shape* has unsupported dimensionality
            or required metadata fields are absent for a given condition.
        """
        ndim = len(grid_shape)
        padded_shape = tuple(s + 2 * pad_width for s in grid_shape)
        faces = _face_slices(ndim)
        face_conditions = self._face_conditions(ndim)

        first_call = not self._cache_initialised

        # Initialize caches on first call
        if first_call:
            flux_cache = np.full(padded_shape, np.nan, dtype=metadata.dtype)
            pressure_cache = np.full(padded_shape, np.nan, dtype=metadata.dtype)
            object.__setattr__(self, "_flux_cache", flux_cache)
            object.__setattr__(self, "_pressure_cache", pressure_cache)
            # Determine and cache whether all boundaries are static
            all_static = all(
                condition.is_static() for direction, condition in face_conditions
            )
            object.__setattr__(self, "_all_static", all_static)
        else:
            flux_cache = self._flux_cache
            pressure_cache = self._pressure_cache
            assert flux_cache is not None and pressure_cache is not None
            # If all conditions are static, skip re-evaluation entirely
            if self._all_static:
                return flux_cache, pressure_cache

        for direction, condition in face_conditions:
            # Skip static faces after the first initialisation pass.
            if not first_call and condition.is_static():
                continue

            ghost_slice, _ = faces[direction]
            values = condition(ghost_slice, direction, metadata)
            bc_type = condition.get_type()

            # `ghost_slice` addresses the padded cache but the BC returns values
            # shaped for the real grid face (no corners/edges).
            # So we build an inset slice: on the face-normal axis keep ghost_slice as-is,
            # on all other axes shift inward by pad_width to skip corner ghost cells.
            inset = []
            for _, s in enumerate(ghost_slice):
                if isinstance(s, slice) and s.start is None and s.stop is None:
                    # This is a slice(None) axis. It spans the full padded dimension.
                    # Inset by pad_width on each end to reach only real-cell-facing ghosts.
                    inset.append(slice(pad_width, -pad_width))
                else:
                    # This is the face-normal axis so we keep as-is (the single ghost layer).
                    inset.append(s)
            inset_slice = tuple(inset)

            # Write values to the appropriate cache and clear the opposite cache
            if bc_type == BoundaryType.FLUX:
                flux_cache[inset_slice] = values
                pressure_cache[inset_slice] = np.nan
            else:
                pressure_cache[inset_slice] = values
                flux_cache[inset_slice] = np.nan

        if first_call:
            object.__setattr__(self, "_cache_initialised", True)

        return flux_cache, pressure_cache

    def refresh_boundaries(
        self,
        metadata: BoundaryMetadata,
        pad_width: int = 1,
    ) -> typing.Tuple[NDimensionalGrid, NDimensionalGrid]:
        """
        Re-evaluate (only dynamic or time-dependent) boundary conditions.

        On entry, the caches must already be initialized (via a prior call to
        `get_boundaries`). This method updates only the cache entries for
        boundaries that report `is_static() == False`, leaving all others untouched.

        **Precondition**: `get_boundaries` must have been called at least once.

        :param metadata: Simulation context bundle. Should typically have
            an updated `fluid_properties` reflecting the latest pressure state.
        :param pad_width: Ghost-cell layer width (must match the value used in
            the initial `get_boundaries` call).
        :return: A 2-tuple `(flux_grid, pressure_grid)` containing the updated cached arrays.
        :raises RuntimeError: If called before `get_boundaries` initialization.
        """
        if not self._cache_initialised:
            raise RuntimeError(
                "`refresh_boundaries` requires prior initialization via `get_boundaries`."
            )

        flux_cache = self._flux_cache
        pressure_cache = self._pressure_cache
        assert flux_cache is not None and pressure_cache is not None

        # If all conditions are static, nothing to refresh
        if self._all_static:
            return flux_cache, pressure_cache

        ndim = flux_cache.ndim
        faces = _face_slices(ndim)
        face_conditions = self._face_conditions(ndim)

        for direction, condition in face_conditions:
            # Skip static faces. Their cache entries are already correct
            if condition.is_static():
                continue

            ghost_slice, _ = faces[direction]
            values = condition(ghost_slice, direction, metadata)
            bc_type = condition.get_type()

            # Build inset slice (same logic as in get_boundaries)
            inset = []
            for _, s in enumerate(ghost_slice):
                if isinstance(s, slice) and s.start is None and s.stop is None:
                    inset.append(slice(pad_width, -pad_width))
                else:
                    inset.append(s)
            inset_slice = tuple(inset)

            # Update only the cache entries for this dynamic face
            if bc_type == BoundaryType.FLUX:
                flux_cache[inset_slice] = values
                pressure_cache[inset_slice] = np.nan
            else:
                pressure_cache[inset_slice] = values
                flux_cache[inset_slice] = np.nan

        return flux_cache, pressure_cache

    def __dump__(self, recurse: bool = True) -> typing.Dict[str, typing.Any]:
        return {
            "left": self.left.dump(recurse),
            "right": self.right.dump(recurse),
            "front": self.front.dump(recurse),
            "back": self.back.dump(recurse),
            "bottom": self.bottom.dump(recurse),
            "top": self.top.dump(recurse),
        }

    @classmethod
    def __load__(cls, data: typing.Mapping[str, typing.Any]) -> Self:
        def _load_face(key: str) -> BoundaryCondition:
            face_data = data.get(key)
            if face_data is None:
                return NeumannBoundary()
            return BoundaryCondition.load(face_data)  # type: ignore[return-value]

        return cls(
            left=_load_face("left"),
            right=_load_face("right"),
            front=_load_face("front"),
            back=_load_face("back"),
            bottom=_load_face("bottom"),
            top=_load_face("top"),
        )

import typing
import warnings
from contextvars import ContextVar, Token

import attrs
import numpy as np
from typing_extensions import NamedTuple, Self

from bores.types import (
    IntArray,
    Integer,
    IntOrArray,
    NDimension,
    NumberArray,
    OneDimension,
    T,
    ThreeDimensions,
    TwoDimensions,
    UnitConversionTable,
    UnitSystem,
)
from bores.units import get_conversion_factors
from bores.utils import scale

DType = typing.TypeVar("DType", float, np.floating)
ShapeT = typing.TypeVar("ShapeT", bound=tuple[int, ...])


__all__ = ["ContextFlag", "GridDimensions", "MapAxes"]


class GridDimensions(NamedTuple):
    """
    Structured grid extent, as declared by `SPECGRID` (or `DIMENS`).

    Needed by every per-cell array keyword (for expected length / reshape)
    and by `BOX` / operator resolution (for IJK -> flat index mapping).
    """

    nx: Integer
    ny: Integer
    nz: Integer

    @property
    def n_cells(self) -> Integer:
        """Total number of cells `nx * ny * nz`."""
        return self.nx * self.ny * self.nz

    @typing.overload
    def flat_index(self, i: Integer, j: Integer, k: Integer) -> Integer: ...
    @typing.overload
    def flat_index(
        self, i: IntArray[NDimension], j: IntOrArray[NDimension], k: IntOrArray[NDimension]
    ) -> IntArray[NDimension]: ...
    @typing.overload
    def flat_index(
        self, i: IntOrArray[NDimension], j: IntArray[NDimension], k: IntOrArray[NDimension]
    ) -> IntArray[NDimension]: ...
    @typing.overload
    def flat_index(
        self, i: IntOrArray[NDimension], j: IntOrArray[NDimension], k: IntArray[NDimension]
    ) -> IntArray[NDimension]: ...

    def flat_index(self, i, j, k):
        """
        Convert 0-based `(i, j, k)` to a flat index in Eclipse's natural
        ordering (`i` fastest, `k` slowest):

        `index = i + j*nx + k*nx*ny`.

        Works on single indices and on arrays alike. When any of the three is an array, the
        result is an array of the same shape.

        :param i: 0-based x index or indices.
        :param j: 0-based y index or indices.
        :param k: 0-based z index or indices.
        :returns: Flat cell index or indices.
        """
        return i + j * self.nx + k * self.nx * self.ny

    def ijk_index(self, flat: Integer) -> tuple[int, int, int]:
        """
        Convert a flat index to  0-based `(i, j, k)`.

        Given the flat index was generated Eclipse's natural
        ordering (`i` fastest, `k` slowest):

        `index = i + j*nx + k*nx*ny`.

        :param flat: Flat cell index.
        :returns: 0-based `(i, j, k)` cell index.
        """
        i = int(flat % self.nx)
        j = int((flat // self.nx) % self.ny)
        k = int(flat // (self.nx * self.ny))
        return i, j, k

    def ijk_indices(
        self, flat: IntArray[NDimension]
    ) -> tuple[IntArray[NDimension], IntArray[NDimension], IntArray[NDimension]]:
        """
        Convert flat indices to 0-based `(i, j, k)` indices, for any number of cells at once.

        Given the flat indices were generated in Eclipse's natural ordering
        (`i` fastest, `k` slowest): `index = i + j*nx + k*nx*ny`.

        :param flat: Flat cell indices, of any shape.
        :returns: Arrays `(i, j, k)`, each with the shape of `flat`.
        """
        # Integer arithmetic keeps the shape of `flat`, which NumPy's type stubs do not express
        return typing.cast(
            "tuple[IntArray[NDimension], IntArray[NDimension], IntArray[NDimension]]",
            (flat % self.nx, (flat // self.nx) % self.ny, flat // (self.nx * self.ny)),
        )


@attrs.frozen(slots=True)
class MapAxes:
    """
    Map coordinate system axes parsed from the GRDECL `MAPAXES` keyword.

    Stores the map origin and two unit-vector axis points so that grid
    coordinates can be rotated into the map CRS.

    :param origin: Shape `(2,)` - `(X, Y)` map coordinate origin.
    :param map_x_axis_point: Shape `(2,)` - point on the map X-axis.
    :param map_y_axis_point: Shape `(2,)` - point on the map Y-axis.
    :param unit_system: Unit system in which the axis coordinates are
        expressed.
    """

    origin: NumberArray[OneDimension]
    map_x_axis_point: NumberArray[OneDimension]
    map_y_axis_point: NumberArray[OneDimension]
    unit_system: UnitSystem = UnitSystem.FIELD
    rotation_matrix: NumberArray[ThreeDimensions] = attrs.field(init=False)

    def __attrs_post_init__(self) -> None:
        object.__setattr__(self, "rotation_matrix", self._compute_rotation_matrix())

    def _compute_rotation_matrix(self) -> NumberArray[TwoDimensions]:
        origin = self.origin
        x_vec = self.map_x_axis_point - origin
        y_vec = self.map_y_axis_point - origin
        x_norm = np.linalg.norm(x_vec)
        y_norm = np.linalg.norm(y_vec)
        if x_norm < 1e-14 or y_norm < 1e-14:
            warnings.warn(
                "`MAPAXES` has a degenerate (zero-length) axis vector. "
                "The map coordinate rotation will be skipped.",
                stacklevel=3,
            )
            return np.eye(2, dtype=np.float64)  # type: ignore[return-value]

        x_dir = x_vec / x_norm
        y_dir = y_vec / y_norm
        return np.array(  # type: ignore[return-value]
            [[x_dir[0], y_dir[0]], [x_dir[1], y_dir[1]]],
            dtype=np.float64,
        )

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: UnitConversionTable | None = None,
    ) -> Self:
        """
        Return a new `MapAxes` with all coordinates expressed in *target*.

        :param target: Target `UnitSystem`.
        :returns: New `MapAxes` in the target unit system, or `self`
            if already in the target system.
        """
        if self.unit_system == target:
            return self

        factor = get_conversion_factors(self.unit_system, target, table=table)["length"]
        return self.__class__(
            origin=scale(self.origin, factor),
            map_x_axis_point=scale(self.map_x_axis_point, factor),
            map_y_axis_point=scale(self.map_y_axis_point, factor),
            unit_system=target,
        )


class ContextFlag(typing.Generic[T]):
    """
    Context-local flag.

    Uses `contextvars.ContextVar` for isolation across concurrent contexts.
    Each context maintains its own flag value.
    """

    __slots__ = ("_token", "_var")

    def __init__(self, id: str, /, initial: T) -> None:
        """
        Initialize the context flag with an initial value.

        :param id: Unique identifier for the context variable
        :param initial: Initial value for new contexts
        """
        self._var: ContextVar[T] = ContextVar(id, default=initial)
        self._token: ContextVar[Token | None] = ContextVar(f"{id}:token", default=None)

    def get(self) -> T:
        """
        Get the flag value in the current context.

        :return: Current value (defaults to initial if never set in this context)
        """
        return self._var.get()

    def set(self, value: T) -> None:
        """
        Set the flag to a new value in the current context.

        :param value: New value
        """
        self._token.set(self._var.set(value))

    def reset(self) -> None:
        """Reset the flag to its initial value in the current context."""
        if (token := self._token.get()) is not None:
            self._var.reset(token)

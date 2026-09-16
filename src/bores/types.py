import enum
import typing
from pathlib import Path
from typing import TypeAlias

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_array, csr_matrix  # type: ignore[import-untyped]
from scipy.sparse.linalg import LinearOperator  # type: ignore[import-untyped]
from typing_extensions import Self, TypedDict

__all__ = [
    "ArrayLike",
    "CapillaryPressures",
    "Coordinates",
    "DiscretizationMethod",
    "FluidPhase",
    "Interpolator",
    "MiscibilityModel",
    "MixingRuleFunc",
    "NDimension",
    "OneDimension",
    "OneDimensionalGrid",
    "Orientation",
    "Preconditioner",
    "RelativePermeabilities",
    "Solver",
    "SolverFunc",
    "ThreeDimensionalGrid",
    "ThreeDimensions",
    "TwoDimensionalGrid",
    "TwoDimensions",
    "UnitSystem",
    "WellFluidType",
    "Wettability",
]

T = typing.TypeVar("T")
Tco = typing.TypeVar("Tco", covariant=True)
S = typing.TypeVar("S")


class UnitSystem(enum.Enum):
    """
    Declared unit system.

    Systems follow Eclipse/OPM conventions:

    ```
    =========  =======  ======  =======  =========  =======  ========  ========
    System     Length   Area    Volume   Perm.      Press.   Temp.     Rates (/day)
    =========  =======  ======  =======  =========  =======  ========  ========
    FIELD      ft       ft²     ft³      mD         psi      °F        STB/SCF
    METRIC     m        m²      m³       mD         bar      °C        Sm³
    LAB        cm       cm²     cm³      mD         atm      °C        cm³ or cc
    SI         m        m²      m³       m²         Pa       K         m³
    =========  =======  ======  =======  =========  =======  ========  ========
    ```
    """

    FIELD = "field"
    METRIC = "metric"
    LAB = "lab"
    SI = "si"

    @classmethod
    def _missing_(cls, value: object) -> Self | None:
        lowered = str(value).lower()
        for member in cls:
            if member.value == lowered:
                return member
        return None


NDimension = typing.TypeVar("NDimension", bound=tuple[int, ...])
Coordinates = typing.TypeVar("Coordinates", bound=tuple[int, ...])

ThreeDimensions: TypeAlias = tuple[int, int, int]
"""3D indices"""
TwoDimensions: TypeAlias = tuple[int, int]
"""2D indices"""
OneDimension: TypeAlias = tuple[int]
"""1D index"""

Integer: TypeAlias = int | np.integer
Float: TypeAlias = float | np.floating
Number: TypeAlias = int | float | np.floating | np.integer
Boolean: TypeAlias = bool | np.bool_
FloatArray: TypeAlias = np.ndarray[NDimension, np.dtype[np.floating]]
IntArray: TypeAlias = np.ndarray[NDimension, np.dtype[np.integer]]
NumberArray: TypeAlias = FloatArray[NDimension] | IntArray[NDimension]
BooleanArray: TypeAlias = np.ndarray[NDimension, np.dtype[np.bool_]]
IntOrArray: TypeAlias = Integer | IntArray[NDimension]
FloatOrArray: TypeAlias = Float | FloatArray[NDimension]
NumberOrArray: TypeAlias = Number | FloatArray[NDimension] | IntArray[NDimension]


CellArray: typing.TypeAlias = NumberArray[OneDimension]
"""Shape (n_cells,) per-cell integer or floating array."""

IntCellArray: typing.TypeAlias = IntArray[OneDimension]
"""Shape (n_cells,) per-cell integer array."""

BooleanCellArray: typing.TypeAlias = BooleanArray[OneDimension]
"""Shape (n_cells,) per-cell boolean array."""


NDimensionalGrid: TypeAlias = np.ndarray[NDimension, np.dtype[np.floating]]
ThreeDimensionalGrid = NDimensionalGrid[ThreeDimensions]
"""3D grid type for simulation data, represented as a 3D NumPy array of floats"""
TwoDimensionalGrid = NDimensionalGrid[TwoDimensions]
"""2D grid type for simulation data, represented as a 2D NumPy array of floats"""
OneDimensionalGrid = NDimensionalGrid[OneDimension]
"""1D grid type for simulation data, represented as a 1D NumPy array of floats"""


TableQuery = NumberOrArray[NDimension] | npt.ArrayLike
TableResult = NumberOrArray[NDimension]
InterpolationMethod = typing.Literal["linear", "cubic"]


class Orientation(enum.Enum):
    """
    Enum representing directional orientation in a 2D/3D simulation.
    """

    X = "x"
    Y = "y"
    Z = "z"
    UNSET = "unset"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def _missing_(cls, value: object) -> Self | None:
        lowered = str(value).lower()
        for member in cls:
            if member.value == lowered:
                return member
        return None


class FluidPhase(enum.Enum):
    """Enum representing the phase of the fluid in the reservoir."""

    WATER = "water"
    GAS = "gas"
    OIL = "oil"
    DISSOLVED_GAS = "disgas"
    LIQUID_OIL = "lio"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def _missing_(cls, value: object) -> Self | None:
        lowered = str(value).lower()
        for member in cls:
            if member.value == lowered:
                return member
            if lowered == "wat":
                return cls("water")
        return None


FrictionMethod = typing.Literal["simplified", "colebrook"]


class Side(enum.Enum):
    """
    One of the six axis-aligned flanks of a grid's bounding box/domain
    boundary, classified by dominant outward-normal axis and sign.

    `WEST`/`EAST`: the X axis (negative / positive outward normal).
    `SOUTH`/`NORTH`: the Y axis (negative / positive).
    `TOP`/`BOTTOM`: the Z axis. Depth increases downward throughout BORES
    (positive-down convention), so `TOP` is the shallowest (min-Z, negative
    outward normal) flank and `BOTTOM` is the deepest (max-Z, positive
    outward normal) flank.
    """

    WEST = "west"
    EAST = "east"
    SOUTH = "south"
    NORTH = "north"
    TOP = "top"
    BOTTOM = "bottom"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def _missing_(cls, value: object) -> Self | None:
        lowered = str(value).lower()
        for member in cls:
            if member.value == lowered:
                return member
        return None


WellFluidType = typing.Literal["water", "oil", "gas"]
"""Types of fluids that can be injected in the simulation"""

DiscretizationMethod = typing.Literal["impes", "implicit"]
"""
Discretization methods for numerical simulations

- `"impes"`: Implicit pressure, Explicit saturation
- `"implicit"`: Both pressure and saturation are treated implicitly
"""

MiscibilityModel = typing.Literal["immiscible", "todd-longstaff"]
"""Miscibility models for fluid interactions in the simulation"""

Spacing = typing.Literal["cosine", "linear"]


class ArrayLike(typing.Protocol[Tco]):
    """
    Protocol for an array-like object that supports
    basic operations like length, indexing, iteration, and containment checks.
    """

    def __len__(self) -> int:
        """Returns the length of the array-like object."""
        ...

    def __getitem__(self, index: int, /) -> Tco:
        """Returns the item at the specified index."""
        ...

    def __iter__(self) -> typing.Iterator[Tco]:
        """Returns an iterator over the items in the array-like object."""
        ...

    def __contains__(self, obj: typing.Any, /) -> bool:
        """Checks if the object is in the array-like object."""
        ...


Interpolator = typing.Callable[[float], float]


PreconditionerStr = (
    typing.Literal["cpr", "ilu", "amg", "block_jacobi", "polynomial", "diagonal"] | str
)
PreconditionerFactory = typing.Callable[[csr_array | csr_matrix], LinearOperator]
Preconditioner = LinearOperator | PreconditionerStr | PreconditionerFactory

SolverStr = (
    typing.Literal[
        "gmres",
        "lgmres",
        "bicgstab",
        "tfqmr",
        "cg",
        "cgs",
        "minres",
        "bicg",
        "qmr",
        "gcrotmk",
        "direct",
    ]
    | str
)


class SolverFunc(typing.Protocol):
    """
    Protocol for a solver function compatible with SciPy's linear solvers.
    """

    def __call__(
        self,
        A: typing.Any,
        b: typing.Any,
        x0: typing.Any | None,
        *,
        rtol: float,
        atol: float,
        maxiter: int | None,
        M: typing.Any | None,
        callback: typing.Callable[[npt.NDArray], None] | None,
    ) -> npt.NDArray: ...


Solver = SolverFunc | SolverStr


class MixingRuleFunc(typing.Protocol):
    """
    Protocol for a mixing rule function that combines two-phase oil relative
    permeabilities into a three-phase estimate.

    Parameters that a simple rules does not need can simply be ignored in the implementation.
    """

    def __call__(
        self,
        *,
        kro_w: NumberOrArray[NDimension],
        kro_g: NumberOrArray[NDimension],
        krw: NumberOrArray[NDimension],
        krg: NumberOrArray[NDimension],
        kr_max: NumberOrArray[NDimension],
        water_saturation: NumberOrArray[NDimension],
        oil_saturation: NumberOrArray[NDimension],
        gas_saturation: NumberOrArray[NDimension],
    ) -> NumberOrArray[NDimension]:
        """
        Compute the three-phase oil relative permeability.

        :param kro_w: Two-phase oil relative permeability from the oil-water
            table at the current water saturation.
        :param kro_g: Two-phase oil relative permeability from the gas-oil
            table at the current gas (or oil) saturation.
        :param krw: Two-phase water relative permeability from the oil-water
            table at the current water saturation.
        :param krg: Two-phase gas relative permeability from the gas-oil
            table at the current gas saturation.
        :param kr_max: Usually oil relative permeability at connate water
            saturation (kro at Sw=swc, Sg=0).
        :param water_saturation: Current water saturation.
        :param oil_saturation: Current oil saturation.
        :param gas_saturation: Current gas saturation.
        :return: Three-phase oil relative permeability.
        """
        ...


class MixingRulePartialDerivatives(TypedDict):
    """
    The seven partial derivatives of a three-phase oil relative permeability
    mixing rule with respect to each of its seven saturation-dependent
    arguments.

    The mixing rule signature is:

        kro = rule(kro_w, kro_g, krw, krg, kr_max,
                   water_saturation, oil_saturation, gas_saturation)

    The derivatives here are taken with respect to the *inputs that vary
    with saturation* in a way that is explicit in the mixing rule itself.
    The chain-rule terms through `kro_w`, `kro_g`, `krw`, and `krg`
    back to physical saturations are assembled at the call site in the table
    derivative methods.

    Fields:

    dkro_dkro_w :
        ∂kro / ∂kro_w  - sensitivity to the oil-water two-phase oil kr.
    dkro_dkro_g :
        ∂kro / ∂kro_g  - sensitivity to the gas-oil two-phase oil kr.
    dkro_dkrw :
        ∂kro / ∂krw  - sensitivity to the two-phase water kr.
        Zero for rules that do not use `krw` directly (most simple rules).
    dkro_dkrg :
        ∂kro / ∂krg  - sensitivity to the two-phase gas kr.
        Zero for rules that do not use `krg` directly (most simple rules).
    dkro_dsw_explicit :
        ∂kro / ∂Sw  through the explicit water-saturation argument of the
        mixing rule (e.g. saturation weighting in `eclipse_rule`).
        Zero for rules that do not depend directly on saturation.
    dkro_dso_explicit :
        ∂kro / ∂So  through the explicit oil-saturation argument.
    dkro_dsg_explicit :
        ∂kro / ∂Sg  through the explicit gas-saturation argument.
    """

    dkro_dkro_w: NumberOrArray
    dkro_dkro_g: NumberOrArray
    dkro_dkrw: NumberOrArray
    dkro_dkrg: NumberOrArray
    dkro_dsw_explicit: NumberOrArray
    dkro_dso_explicit: NumberOrArray
    dkro_dsg_explicit: NumberOrArray


class MixingRuleDFunc(typing.Protocol):
    """
    Protocol for a mixing rule partial derivatives function.
    """

    def __call__(
        self,
        *,
        kro_w: NumberOrArray[NDimension],
        kro_g: NumberOrArray[NDimension],
        krw: NumberOrArray[NDimension],
        krg: NumberOrArray[NDimension],
        kr_max: NumberOrArray[NDimension],
        water_saturation: NumberOrArray[NDimension],
        oil_saturation: NumberOrArray[NDimension],
        gas_saturation: NumberOrArray[NDimension],
    ) -> (
        MixingRulePartialDerivatives
        | tuple[
            NumberOrArray[NDimension],
            NumberOrArray[NDimension],
            NumberOrArray[NDimension],
            NumberOrArray[NDimension],
            NumberOrArray[NDimension],
            NumberOrArray[NDimension],
            NumberOrArray[NDimension],
        ]
    ):
        """
        Return the seven partial derivatives of the mixing rule.

        The tuple order (when not returning a `MixingRulePartialDerivatives` dict) is:

        `(dkro_dkro_w, dkro_dkro_g, dkro_dkrw, dkro_dkrg,
           dkro_dsw_explicit, dkro_dso_explicit, dkro_dsg_explicit)`

        :param kro_w: Two-phase oil kr from oil-water table.
        :param kro_g: Two-phase oil kr from gas-oil table.
        :param krw: Two-phase water kr from oil-water table.
        :param krg: Two-phase gas kr from gas-oil table.
        :param kr_max: Oil kr at connate water (normalization reference).
        :param water_saturation: Current water saturation.
        :param oil_saturation: Current oil saturation.
        :param gas_saturation: Current gas saturation.
        :return: The seven partial derivatives, either as a
            `MixingRulePartialDerivatives` dict or a 7-tuple.
        """
        ...


class RelativePermeabilities(TypedDict):
    """Dictionary holding relative permeabilities for different phases."""

    water: NumberOrArray
    oil: NumberOrArray
    gas: NumberOrArray


class RelativePermeabilityDerivatives(TypedDict):
    """Dictionary holding relative permeabilities derivatives."""

    # w.r.t water
    dkrw_dsw: NumberOrArray
    dkro_dsw: NumberOrArray
    dkrg_dsw: NumberOrArray
    # w.r.t oil
    dkrw_dso: NumberOrArray
    dkro_dso: NumberOrArray
    dkrg_dso: NumberOrArray
    # w.r.t gas
    dkrw_dsg: NumberOrArray
    dkro_dsg: NumberOrArray
    dkrg_dsg: NumberOrArray


class CapillaryPressures(TypedDict):
    """Dictionary containing capillary pressures for different phase pairs."""

    oil_water: NumberOrArray  # Pcow = Po - Pw
    gas_oil: NumberOrArray  # Pcgo = Pg - Po


class CapillaryPressureDerivatives(TypedDict):
    """Dictionary containing capillary pressure derivatives for different phase pairs."""

    dpcow_dsw: NumberOrArray
    dpcow_dso: NumberOrArray
    dpcgo_dsg: NumberOrArray
    dpcgo_dso: NumberOrArray


class Wettability(enum.Enum):
    """Enum representing the wettability type of the reservoir rock."""

    WATER_WET = "water_wet"
    OIL_WET = "oil_wet"
    MIXED_WET = "mixed_wet"

    def __str__(self) -> str:
        return self.value


Kcon = typing.TypeVar("Kcon", contravariant=True)
Vcon = typing.TypeVar("Vcon", contravariant=True)


ZFactorMethod = typing.Literal["papay", "hall-yarborough", "dak"]

GridIntersectionMethod = typing.Literal["aabb", "exact"]


class UnitConversionFactors(TypedDict):
    """
    Multiplicative conversion factors from one `UnitSystem` to another.

    Every factor converts a value expressed in the source unit system into
    the corresponding value in the target unit system by simple multiplication.
    Temperature is the only exception, requiring both a multiplicative factor
    and an additive offset:

        T_target = T_source * temperature + temperature_offset

    All factors are dimensionless ratios; negative values are never valid.
    """

    # Fundamental units

    pressure: Number
    """
    Pressure conversion factor.

    FIELD : psi
    METRIC: bar
    LAB   : atm
    SI    : Pa
    """

    length: Number
    """
    Length conversion factor.

    FIELD : ft
    METRIC: m
    LAB   : cm
    SI    : m
    """

    area: Number
    """
    Area conversion factor (= length²).

    FIELD : ft²
    METRIC: m²
    LAB   : cm²
    SI    : m²
    """

    volume: Number
    """
    Reservoir (in-situ) volume conversion factor (= length³).

    FIELD : ft³  (reservoir cubic feet)
    METRIC: m³   (reservoir cubic metres)
    LAB   : cm³  (reservoir cubic centimetres)
    SI    : m³   (reservoir cubic metres)
    """

    time: Number
    """
    Time conversion factor.

    FIELD : day
    METRIC: day
    LAB   : hour
    SI    : second
    """

    mass: Number
    """
    Mass conversion factor.

    FIELD : lbm
    METRIC: kg
    LAB   : g
    SI    : kg
    """

    temperature: Number
    """
    Multiplicative temperature conversion factor.

    Applied as: `T_target = T_source * temperature + temperature_offset`.
    """

    temperature_offset: Number
    """
    Additive temperature conversion offset (in target units).

    Applied after the multiplicative factor:
    `T_target = T_source * temperature + temperature_offset`.
    """

    # Derived units

    density: Number
    """
    Density conversion factor (= mass / volume = mass / length³).

    FIELD : lbm/ft³
    METRIC: kg/m³
    LAB   : g/cm³
    SI    : kg/m³
    """

    viscosity: Number
    """
    Dynamic viscosity conversion factor.

    FIELD : cP
    METRIC: cP
    LAB   : cP
    SI    : Pa·s
    """

    permeability: Number
    """
    Permeability conversion factor.

    FIELD : mD
    METRIC: mD
    LAB   : mD
    SI    : m²
    """

    compressibility: Number
    """
    Compressibility conversion factor (= 1 / pressure).

    FIELD : 1/psi
    METRIC: 1/bar
    LAB   : 1/atm
    SI    : 1/Pa
    """

    # Surface volumes

    liquid_surface_volume: Number
    """
    Surface liquid volume conversion factor.

    FIELD : STB  (stock-tank barrel)
    METRIC: Sm³  (standard cubic metre)
    LAB   : scc  (standard cubic centimetre)
    SI    : Sm³
    """

    gas_surface_volume: Number
    """
    Surface gas volume conversion factor.

    FIELD : SCF  (standard cubic foot)
    METRIC: Sm³
    LAB   : scc
    SI    : Sm³
    """

    # Reservoir-volume ratios (formation volume factors)

    liquid_fvf: Number
    """
    Liquid formation volume factor conversion factor.

    Converts the ratio reservoir-volume / surface-volume:

    FIELD  -> METRIC: rb/STB  -> rm³/Sm³  (ratio stays 1.0 since both sides
                                          use the same volume system)
    FIELD  -> LAB:   rb/STB  -> rcc/scc
    FIELD  -> SI:    rb/STB  -> rm³/Sm³
    """

    gas_fvf: Number
    """
    Gas formation volume factor conversion factor.

    Converts the ratio reservoir-volume / surface-volume:

    FIELD  -> METRIC: rcf/SCF -> rm³/Sm³
    FIELD  -> LAB:   rcf/SCF -> rcc/scc
    FIELD  -> SI:    rcf/SCF -> rm³/Sm³
    """

    # Solution ratios

    gas_oil_ratio: Number
    """
    Solution gas-oil ratio (GOR) conversion factor.

    FIELD : SCF/STB
    METRIC: Sm³/Sm³
    LAB   : scc/scc
    SI    : Sm³/Sm³
    """

    oil_gas_ratio: Number
    """
    Vaporised oil ratio (OGR / Rv) conversion factor.

    FIELD : STB/SCF
    METRIC: Sm³/Sm³
    LAB   : scc/scc
    SI    : Sm³/Sm³
    """

    # Flow rates

    liquid_surface_rate: Number
    """
    Surface liquid volumetric flow rate conversion factor.

    FIELD : STB/day
    METRIC: Sm³/day
    LAB   : scc/hour
    SI    : Sm³/second
    """

    gas_surface_rate: Number
    """
    Surface gas volumetric flow rate conversion factor.

    FIELD : SCF/day
    METRIC: Sm³/day
    LAB   : scc/hour
    SI    : Sm³/second
    """

    reservoir_rate: Number
    """
    Reservoir (in-situ) volumetric flow rate conversion factor.

    FIELD : ft³/day
    METRIC: m³/day
    LAB   : cm³/hour
    SI    : m³/second
    """


UnitConversionTable = dict[tuple[UnitSystem, UnitSystem], UnitConversionFactors]
"""Mapping of unit system pairs `(from, target)` to unit conversion factors"""


class SupportsUnitSystem(typing.Protocol):
    """Protocol defining `UnitSystem` support."""

    unit_system: UnitSystem

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: UnitConversionTable | None = None,
    ) -> Self: ...


PathOrStr = str | Path
TextOrPath = str | bytes | Path
